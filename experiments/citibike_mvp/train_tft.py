#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import cast

import lightning.pytorch as pl
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    outdir = project_path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    train_ds, val_ds = make_datasets(df, args)

    train_loader = train_ds.to_dataloader(train=True, batch_size=args.batch_size, num_workers=0)
    val_loader = val_ds.to_dataloader(train=False, batch_size=args.batch_size * 2, num_workers=0)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, verbose=True, mode="min")
    lr_logger = LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = ModelCheckpoint(
        dirpath=outdir / "checkpoints",
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    logger = CSVLogger(save_dir=outdir.as_posix(), name="logs")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
        logger=logger,
        log_every_n_steps=20,
        enable_model_summary=True,
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
        ),
    )
    n_parameters = sum(parameter.numel() for parameter in tft.parameters())
    print(f"Number of parameters: {n_parameters / 1e3:.1f}k")

    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")

    # Save dataset parameters so later data can reuse same schema
    train_ds.save((outdir / "timeseries_dataset.pkl").as_posix())


if __name__ == "__main__":
    main()
