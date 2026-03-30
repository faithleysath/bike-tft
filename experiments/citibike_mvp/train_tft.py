#!/usr/bin/env python3
import argparse
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import lightning.pytorch as pl
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, LitLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_REAL_COLUMNS = ["station_lat", "station_lng"]
KNOWN_REAL_COLUMNS = [
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]
UNKNOWN_REAL_COLUMNS = [
    "dep_count",
    "arr_count",
    "net_flow",
    "dep_classic_count",
    "dep_electric_count",
    "arr_classic_count",
    "arr_electric_count",
]


def project_path(value: str | Path) -> Path:
    """Resolve repo-relative paths no matter where the script is launched from."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def as_frame(value: object) -> pd.DataFrame:
    """Help pyright treat pandas filtering operations as DataFrames."""
    return cast(pd.DataFrame, value)


def parse_limit_batches(value: str) -> int | float:
    """Parse Lightning limit_*_batches arguments as int or fraction."""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("batch limits must be positive")
    if parsed.is_integer():
        return int(parsed)
    if parsed <= 1:
        return parsed
    raise argparse.ArgumentTypeError(
        "batch limits must be a positive integer or a float fraction in (0, 1]"
    )


class EpochTimingCallback(Callback):
    """Print lightweight epoch timing so short benchmark runs are easier to compare."""

    def __init__(self, train_batch_size: int) -> None:
        self.train_batch_size = train_batch_size
        self._train_epoch_start: float | None = None
        self._full_epoch_start: float | None = None
        self._train_loop_reported = False

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        now = perf_counter()
        self._train_epoch_start = now
        self._full_epoch_start = now
        self._train_loop_reported = False

    def _print_train_loop_timing(self, trainer: pl.Trainer) -> None:
        if self._train_epoch_start is None:
            return
        elapsed = perf_counter() - self._train_epoch_start
        if elapsed <= 0:
            return
        batches = trainer.num_training_batches
        samples_per_second = batches * self.train_batch_size / elapsed
        print(
            f"Epoch {trainer.current_epoch} train loop: {elapsed:.1f}s, "
            f"{batches} batches, ~{samples_per_second:.1f} window samples/s"
        )
        self._train_loop_reported = True

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        self._print_train_loop_timing(trainer)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking or self._full_epoch_start is None:
            return
        elapsed = perf_counter() - self._full_epoch_start
        print(f"Epoch {trainer.current_epoch} full loop (train+val): {elapsed:.1f}s")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._train_loop_reported:
            self._print_train_loop_timing(trainer)


def load_data(path: str | Path) -> pd.DataFrame:
    df = as_frame(pd.read_parquet(project_path(path)))
    df["station_id"] = df["station_id"].astype(str)
    df["time_idx"] = df["time_idx"].astype("int32")
    # static fields cannot be NA
    df["station_lat"] = df["station_lat"].fillna(df["station_lat"].median()).astype("float32")
    df["station_lng"] = df["station_lng"].fillna(df["station_lng"].median()).astype("float32")
    for column in KNOWN_REAL_COLUMNS + UNKNOWN_REAL_COLUMNS:
        df[column] = df[column].astype("float32")
    return df


def make_datasets(
    df: pd.DataFrame, args: argparse.Namespace
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    max_time_idx = int(df["time_idx"].max())
    training_cutoff = max_time_idx - args.validation_horizon
    print(f"max_time_idx={max_time_idx}, training_cutoff={training_cutoff}")

    train_frame = as_frame(df[df.time_idx <= training_cutoff])
    training = TimeSeriesDataSet(
        train_frame,
        time_idx="time_idx",
        target=args.target,
        group_ids=["station_id"],
        min_encoder_length=args.max_encoder_length // 2,
        max_encoder_length=args.max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=args.max_prediction_length,
        static_categoricals=["station_id"],
        static_reals=STATIC_REAL_COLUMNS,
        time_varying_known_reals=KNOWN_REAL_COLUMNS,
        time_varying_unknown_reals=UNKNOWN_REAL_COLUMNS,
        target_normalizer=GroupNormalizer(groups=["station_id"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        df,
        min_prediction_idx=training_cutoff + 1,
        stop_randomization=True,
        predict=False,
    )
    return training, validation


def save_dataset_artifacts(
    dataset: TimeSeriesDataSet, outdir: Path, save_full_dataset: bool
) -> None:
    """Persist lightweight dataset parameters and optionally the full dataset snapshot."""
    params_path = outdir / "timeseries_dataset_params.pt"
    torch.save(dataset.get_parameters(), params_path.as_posix())
    print(f"Saved dataset parameters: {params_path}")

    if not save_full_dataset:
        return

    full_dataset_path = outdir / "timeseries_dataset.pkl"
    dataset.save(full_dataset_path.as_posix())
    print(f"Saved full TimeSeriesDataSet snapshot: {full_dataset_path}")


def build_loggers(args: argparse.Namespace, outdir: Path) -> list[CSVLogger | LitLogger]:
    """Create local and optional Lightning.ai experiment loggers."""
    loggers: list[CSVLogger | LitLogger] = [CSVLogger(save_dir=outdir.as_posix(), name="logs")]
    if not args.litlogger:
        return loggers

    try:
        lit_logger = LitLogger(
            root_dir=(outdir / "lightning_logs").as_posix(),
            name=args.litlogger_name or outdir.name,
            teamspace=args.litlogger_teamspace,
            metadata={
                "target": args.target,
                "precision": args.precision,
                "batch_size": str(args.batch_size),
                "max_encoder_length": str(args.max_encoder_length),
                "max_prediction_length": str(args.max_prediction_length),
            },
            log_model=args.litlogger_log_model,
            save_logs=args.litlogger_save_logs,
            checkpoint_name="best",
        )
        print(f"Lightning.ai experiment URL: {lit_logger.url}")
        loggers.append(lit_logger)
    except Exception as exc:
        print(
            "LitLogger unavailable; continuing with local CSV logs only. "
            f"Reason: {exc}"
        )
    return loggers


def main():
    parser = argparse.ArgumentParser(description="Train TFT on station-time panel")
    parser.add_argument("--data", required=True, help="Path to station_hour_panel.parquet")
    parser.add_argument("--output-dir", required=True, help="Directory to store logs/checkpoints")
    parser.add_argument("--target", default="dep_count", help="Prediction target column")
    parser.add_argument("--max-encoder-length", type=int, default=24 * 7, help="Past context length")
    parser.add_argument("--max-prediction-length", type=int, default=6, help="Forecast horizon")
    parser.add_argument("--validation-horizon", type=int, default=24 * 7, help="Holdout length in time_idx units")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--hidden-continuous-size", type=int, default=16)
    parser.add_argument("--attention-head-size", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes for training")
    parser.add_argument(
        "--val-num-workers",
        type=int,
        default=None,
        help="DataLoader worker processes for validation; defaults to --num-workers",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable pinned host memory for DataLoaders; mainly useful on CUDA",
    )
    parser.add_argument(
        "--precision",
        default="32-true",
        help="Lightning precision mode, e.g. 32-true or 16-mixed",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=parse_limit_batches,
        default=1.0,
        help="Lightning limit_train_batches value; integer batches or fraction of an epoch",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=parse_limit_batches,
        default=1.0,
        help="Lightning limit_val_batches value; integer batches or fraction of validation",
    )
    parser.add_argument(
        "--num-sanity-val-steps",
        type=int,
        default=2,
        help="Lightning num_sanity_val_steps; set to 0 for cleaner throughput benchmarks",
    )
    parser.add_argument(
        "--profiler",
        choices=["simple", "advanced"],
        default=None,
        help="Optional Lightning profiler for debugging throughput bottlenecks",
    )
    parser.add_argument(
        "--ckpt-path",
        default=None,
        help="Resume training from a Lightning checkpoint path",
    )
    parser.add_argument(
        "--save-full-dataset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save the full TimeSeriesDataSet snapshot in addition to lightweight "
            "dataset parameters. This can be much larger because it includes "
            "encoded training data."
        ),
    )
    parser.add_argument(
        "--litlogger",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Lightning.ai experiment tracking when credentials are available",
    )
    parser.add_argument(
        "--litlogger-name",
        default=None,
        help="Experiment name shown in Lightning.ai; defaults to the output directory name",
    )
    parser.add_argument(
        "--litlogger-teamspace",
        default=None,
        help="Lightning.ai teamspace to upload the run into",
    )
    parser.add_argument(
        "--litlogger-log-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload saved checkpoints to Lightning.ai when LitLogger is enabled",
    )
    parser.add_argument(
        "--litlogger-save-logs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Capture terminal stdout/stderr in Lightning.ai; may re-exec the process under a recorder",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.val_num_workers is not None and args.val_num_workers < 0:
        parser.error("--val-num-workers must be non-negative")
    if args.num_sanity_val_steps < 0:
        parser.error("--num-sanity-val-steps must be non-negative")
    ckpt_path = None
    if args.ckpt_path is not None:
        ckpt_path = project_path(args.ckpt_path)
        if not ckpt_path.exists():
            parser.error(f"--ckpt-path does not exist: {ckpt_path}")

    pl.seed_everything(args.seed)
    # Better Tensor Core utilization on recent NVIDIA GPUs.
    torch.set_float32_matmul_precision("high")
    outdir = project_path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    train_ds, val_ds = make_datasets(df, args)

    val_num_workers = args.num_workers if args.val_num_workers is None else args.val_num_workers
    train_loader = train_ds.to_dataloader(
        train=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=args.pin_memory,
    )
    val_loader = val_ds.to_dataloader(
        train=False,
        batch_size=args.batch_size * 2,
        num_workers=val_num_workers,
        persistent_workers=val_num_workers > 0,
        pin_memory=args.pin_memory,
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, verbose=True, mode="min")
    lr_logger = LearningRateMonitor(logging_interval="epoch")
    epoch_timing_callback = EpochTimingCallback(train_batch_size=args.batch_size)
    checkpoint_callback = ModelCheckpoint(
        dirpath=outdir / "checkpoints",
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    loggers = build_loggers(args, outdir)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        precision=args.precision,
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback, checkpoint_callback, epoch_timing_callback],
        logger=cast(Any, loggers),
        log_every_n_steps=20,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        num_sanity_val_steps=args.num_sanity_val_steps,
        enable_model_summary=True,
        profiler=args.profiler,
    )

    tft = cast(
        TemporalFusionTransformer,
        TemporalFusionTransformer.from_dataset(
            train_ds,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            attention_head_size=args.attention_head_size,
            dropout=args.dropout,
            hidden_continuous_size=args.hidden_continuous_size,
            loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
            log_interval=10,
            reduce_on_plateau_patience=2,
            # PyTorch Forecasting defaults this to -1e9, which overflows in float16 AMP.
            mask_bias=-float("inf"),
        ),
    )
    n_parameters = sum(parameter.numel() for parameter in tft.parameters())
    print(f"Number of parameters: {n_parameters / 1e3:.1f}k")

    if ckpt_path is not None:
        print(f"Resuming from checkpoint: {ckpt_path}")
    trainer.fit(
        tft,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path.as_posix() if ckpt_path is not None else None,
    )
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")

    # Save dataset configuration needed to rebuild compatible datasets later.
    save_dataset_artifacts(train_ds, outdir, save_full_dataset=args.save_full_dataset)


if __name__ == "__main__":
    main()
