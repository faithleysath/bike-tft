#!/usr/bin/env python3
import argparse
import html
from pathlib import Path
import sys
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

plt.style.use("dark_background")
ACCENT = "#f4a261"
ACCENT_COOL = "#7fd1b9"
ACCENT_RED = "#e76f51"
BG = "#0b1320"
PANEL = "#131f33"
GRID = "#33415c"
TEXT = "#eef2ff"
MUTED = "#94a3b8"


def load_training_helpers():
    from stages.stage_01_citibike_mvp.train_tft import load_data, make_datasets

    return load_data, make_datasets


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a visual report for a TFT training run")
    parser.add_argument("--run-dir", required=True, help="Run directory containing logs and checkpoints")
    parser.add_argument("--data", required=True, help="Path to station_hour_panel.parquet")
    parser.add_argument(
        "--validation-horizon",
        type=int,
        default=168,
        help="Validation holdout length in time_idx units; defaults to 168 hours",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write the report into; defaults to <run-dir>/report",
    )
    parser.add_argument(
        "--forecast-start-time-idx",
        type=int,
        default=None,
        help="Validation prediction start to visualize; defaults to the first holdout step",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "font.size": 11,
            "font.sans-serif": [
                "Hiragino Sans GB",
                "PingFang HK",
                "Songti SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
        }
    )


def load_metrics(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = run_dir / "logs" / "version_0" / "metrics.csv"
    metrics = pd.read_csv(metrics_path)

    train_step = metrics.loc[metrics["train_loss_step"].notna(), ["epoch", "step", "train_loss_step"]].copy()
    train_step["train_loss_smooth"] = train_step["train_loss_step"].rolling(window=200, min_periods=1).mean()

    epoch_train = metrics.loc[metrics["train_loss_epoch"].notna(), ["epoch", "train_loss_epoch"]].copy()
    epoch_val = metrics.loc[
        metrics["val_loss"].notna(),
        ["epoch", "val_loss", "val_MAE", "val_RMSE", "val_SMAPE", "val_MAPE"],
    ].copy()
    epoch_metrics = epoch_val.merge(epoch_train, on="epoch", how="left").sort_values("epoch").reset_index(drop=True)
    return train_step, epoch_metrics


def load_best_checkpoint(run_dir: Path) -> Path:
    best_matches = sorted((run_dir / "checkpoints").glob("best-*.ckpt"))
    if best_matches:
        return best_matches[0]
    return run_dir / "checkpoints" / "last.ckpt"


def load_checkpoint_metadata(checkpoint_path: Path) -> dict:
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def compute_validation_summary(df: pd.DataFrame, validation_horizon: int) -> tuple[pd.DataFrame, dict[str, float | int]]:
    max_time_idx = int(df["time_idx"].max())
    cutoff = max_time_idx - validation_horizon
    val = df.loc[df["time_idx"] > cutoff, ["time_idx", "station_id", "dep_count"]].copy()

    summary = {
        "max_time_idx": max_time_idx,
        "cutoff": cutoff,
        "val_rows": int(len(val)),
        "val_stations": int(val["station_id"].nunique()),
        "val_mean": float(val["dep_count"].mean()),
        "val_median": float(val["dep_count"].median()),
        "val_zero_rate": float((val["dep_count"] == 0).mean()),
    }
    return val, summary


def compute_baselines(df: pd.DataFrame, validation_horizon: int) -> pd.DataFrame:
    ordered = df.loc[:, ["time_idx", "station_id", "dep_count"]].sort_values(["station_id", "time_idx"]).copy()
    max_time_idx = int(ordered["time_idx"].max())
    cutoff = max_time_idx - validation_horizon

    for lag in (1, 24, 168):
        ordered[f"lag{lag}"] = ordered.groupby("station_id")["dep_count"].shift(lag)

    val = ordered.loc[ordered["time_idx"] > cutoff].copy()
    rows: list[dict[str, float | str]] = []
    for name in ("lag1", "lag24", "lag168"):
        subset = val.dropna(subset=[name]).copy()
        error = subset["dep_count"] - subset[name]
        rows.append(
            {
                "model": name,
                "mae": float(error.abs().mean()),
                "rmse": float(np.sqrt(np.square(error).mean())),
            }
        )
    return pd.DataFrame(rows)


def build_model(train_ds, checkpoint: dict) -> TemporalFusionTransformer:
    hp = checkpoint["hyper_parameters"]
    model = TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=hp["learning_rate"],
        hidden_size=hp["hidden_size"],
        attention_head_size=hp["attention_head_size"],
        dropout=hp["dropout"],
        hidden_continuous_size=hp["hidden_continuous_size"],
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        log_interval=10,
        reduce_on_plateau_patience=2,
        mask_bias=-float("inf"),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    return model


def make_forecast_subset(val_ds, forecast_start_time_idx: int, max_prediction_length: int):
    last_time_idx = forecast_start_time_idx + max_prediction_length - 1
    return val_ds.filter(
        lambda x: (x.time_idx_first_prediction == forecast_start_time_idx)
        & (x.time_idx_last == last_time_idx)
    )


def predict_fixed_windows(
    model: TemporalFusionTransformer,
    subset,
) -> pd.DataFrame:
    prediction = model.predict(
        subset,
        mode="quantiles",
        mode_kwargs={"quantiles": [0.1, 0.5, 0.9]},
        return_y=True,
        return_index=True,
        trainer_kwargs={
            "accelerator": "cpu",
            "devices": 1,
            "logger": False,
            "enable_progress_bar": False,
        },
    )
    predicted = prediction.output.detach().cpu().numpy()
    actual = prediction.y[0].detach().cpu().numpy()
    index_df = prediction.index.reset_index(drop=True)

    rows: list[dict[str, float | int | str]] = []
    for row_idx in range(len(index_df)):
        station_id = str(index_df.loc[row_idx, "station_id"])
        base_time_idx = int(index_df.loc[row_idx, "time_idx"])
        for horizon_idx in range(predicted.shape[1]):
            rows.append(
                {
                    "station_id": station_id,
                    "prediction_time_idx": base_time_idx,
                    "forecast_time_idx": base_time_idx + horizon_idx,
                    "horizon": horizon_idx + 1,
                    "predicted_p10": float(predicted[row_idx, horizon_idx, 0]),
                    "predicted": float(predicted[row_idx, horizon_idx, 1]),
                    "predicted_p90": float(predicted[row_idx, horizon_idx, 2]),
                    "actual": float(actual[row_idx, horizon_idx]),
                }
            )
    return pd.DataFrame(rows)


def compute_interpretation(model: TemporalFusionTransformer, subset) -> dict[str, torch.Tensor]:
    raw_output = model.predict(
        subset,
        mode="raw",
        trainer_kwargs={
            "accelerator": "cpu",
            "devices": 1,
            "logger": False,
            "enable_progress_bar": False,
        },
    )
    return model.interpret_output(raw_output, reduction="mean")


def compute_station_metadata(df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    coords = (
        df.loc[:, ["station_id", "station_lat", "station_lng"]]
        .drop_duplicates(subset=["station_id"])
        .copy()
    )
    val_station = (
        val_df.groupby("station_id", as_index=False)
        .agg(
            val_mean=("dep_count", "mean"),
            val_sum=("dep_count", "sum"),
        )
        .copy()
    )
    return coords.merge(val_station, on="station_id", how="left")


def compute_station_slice_metrics(
    prediction_df: pd.DataFrame, station_metadata: pd.DataFrame
) -> pd.DataFrame:
    metrics = (
        prediction_df.assign(
            abs_error=lambda d: (d["actual"] - d["predicted"]).abs(),
            bias=lambda d: d["predicted"] - d["actual"],
            interval_width=lambda d: d["predicted_p90"] - d["predicted_p10"],
            covered=lambda d: ((d["actual"] >= d["predicted_p10"]) & (d["actual"] <= d["predicted_p90"])).astype(int),
        )
        .groupby("station_id", as_index=False)
        .agg(
            actual_mean=("actual", "mean"),
            predicted_mean=("predicted", "mean"),
            mae=("abs_error", "mean"),
            bias=("bias", "mean"),
            interval_width=("interval_width", "mean"),
            coverage=("covered", "mean"),
        )
    )
    return metrics.merge(station_metadata, on="station_id", how="left")


def compute_top_station_ids(station_metrics: pd.DataFrame, count: int = 6) -> list[str]:
    return (
        station_metrics.sort_values(["val_mean", "actual_mean"], ascending=False)
        .head(count)["station_id"]
        .astype(str)
        .tolist()
    )


def build_variable_importance_frames(
    checkpoint: dict, interpretation: dict[str, torch.Tensor]
) -> dict[str, pd.DataFrame]:
    hp = checkpoint["hyper_parameters"]
    static_names = hp["static_categoricals"] + hp["static_reals"]
    encoder_names = hp["time_varying_categoricals_encoder"] + hp["time_varying_reals_encoder"]
    decoder_names = hp["time_varying_categoricals_decoder"] + hp["time_varying_reals_decoder"]

    frames = {
        "static": pd.DataFrame(
            {
                "variable": static_names,
                "importance": interpretation["static_variables"].detach().cpu().numpy(),
            }
        ).sort_values("importance", ascending=False),
        "encoder": pd.DataFrame(
            {
                "variable": encoder_names,
                "importance": interpretation["encoder_variables"].detach().cpu().numpy(),
            }
        ).sort_values("importance", ascending=False),
        "decoder": pd.DataFrame(
            {
                "variable": decoder_names,
                "importance": interpretation["decoder_variables"].detach().cpu().numpy(),
            }
        ).sort_values("importance", ascending=False),
        "attention": pd.DataFrame(
            {
                "lag_hour": np.arange(-len(interpretation["attention"]), 0),
                "attention": interpretation["attention"].detach().cpu().numpy(),
            }
        ),
    }
    return frames


def plot_training_overview(train_step: pd.DataFrame, epoch_metrics: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=170)

    axes[0].plot(train_step["step"], train_step["train_loss_step"], color="#3c4f76", alpha=0.25, linewidth=1)
    axes[0].plot(train_step["step"], train_step["train_loss_smooth"], color=ACCENT, linewidth=2.4)
    axes[0].set_title("训练步损失")
    axes[0].set_xlabel("全局步数")
    axes[0].set_ylabel("分位数损失")
    axes[0].grid(alpha=0.35)

    axes[1].plot(epoch_metrics["epoch"], epoch_metrics["train_loss_epoch"], marker="o", color=ACCENT_COOL, linewidth=2.2, label="训练损失")
    axes[1].plot(epoch_metrics["epoch"], epoch_metrics["val_loss"], marker="o", color=ACCENT, linewidth=2.2, label="验证损失")
    best_row = epoch_metrics.loc[epoch_metrics["val_loss"].idxmin()]
    axes[1].scatter([best_row["epoch"]], [best_row["val_loss"]], color=ACCENT_RED, s=70, zorder=3)
    axes[1].annotate(
        f"最佳 e{int(best_row['epoch'])}\n{best_row['val_loss']:.3f}",
        xy=(best_row["epoch"], best_row["val_loss"]),
        xytext=(12, 12),
        textcoords="offset points",
        color=TEXT,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": BG, "edgecolor": GRID},
    )
    axes[1].set_title("按 Epoch 的训练/验证对比")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("损失")
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].grid(alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_validation_metrics(epoch_metrics: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=170)
    columns = [
        ("val_MAE", "验证集 MAE", ACCENT),
        ("val_RMSE", "验证集 RMSE", ACCENT_COOL),
        ("val_SMAPE", "验证集 SMAPE", "#cdb4db"),
    ]
    for axis, (column, title, color) in zip(axes, columns, strict=True):
        axis.plot(epoch_metrics["epoch"], epoch_metrics[column], marker="o", linewidth=2.2, color=color)
        best_idx = epoch_metrics[column].idxmin()
        axis.scatter([epoch_metrics.loc[best_idx, "epoch"]], [epoch_metrics.loc[best_idx, column]], color=ACCENT_RED, s=60, zorder=3)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_comparison(
    baselines: pd.DataFrame, best_mae: float, best_rmse: float, output_path: Path
) -> pd.DataFrame:
    comparison = pd.concat(
        [
            pd.DataFrame([{"model": "TFT", "mae": best_mae, "rmse": best_rmse}]),
            baselines.copy(),
        ],
        ignore_index=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), dpi=170)
    for axis, metric, title in zip(
        axes,
        ("mae", "rmse"),
        ("验证集 MAE 与朴素基线对比", "验证集 RMSE 与朴素基线对比"),
        strict=True,
    ):
        order = comparison.sort_values(metric, ascending=True).reset_index(drop=True)
        colors = [ACCENT if model_name == "TFT" else ACCENT_COOL for model_name in order["model"]]
        axis.barh(order["model"], order[metric], color=colors, alpha=0.92)
        for row_idx, row in order.iterrows():
            axis.text(row[metric] + 0.02, row_idx, f"{row[metric]:.2f}", va="center", color=TEXT, fontsize=10)
        axis.set_title(title)
        axis.set_xlabel(metric.upper())
        axis.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return comparison


def plot_validation_distribution(val_df: pd.DataFrame, output_path: Path) -> None:
    clipped = val_df["dep_count"].clip(upper=30)
    fig, ax = plt.subplots(figsize=(13.5, 4.5), dpi=170)
    bins = np.arange(0, 31 + 1) - 0.5
    ax.hist(clipped, bins=bins, color=ACCENT_COOL, alpha=0.85, edgecolor=BG)
    ax.set_title("验证目标分布（借出量大于 30 的部分已截断）")
    ax.set_xlabel("每小时借出量 dep_count")
    ax.set_ylabel("样本数")
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_horizon_error(prediction_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    grouped = (
        prediction_df.assign(
            abs_error=lambda d: (d["actual"] - d["predicted"]).abs(),
            covered=lambda d: ((d["actual"] >= d["predicted_p10"]) & (d["actual"] <= d["predicted_p90"])).astype(int),
        )
        .groupby("horizon", as_index=False)
        .agg(
            mae=("abs_error", "mean"),
            mean_actual=("actual", "mean"),
            mean_predicted=("predicted", "mean"),
            coverage=("covered", "mean"),
        )
    )

    fig, ax1 = plt.subplots(figsize=(13.5, 4.8), dpi=170)
    ax2 = ax1.twinx()

    ax1.bar(grouped["horizon"], grouped["mae"], color=ACCENT, alpha=0.85, width=0.55, label="MAE")
    ax2.plot(grouped["horizon"], grouped["mean_actual"], color=ACCENT_COOL, marker="o", linewidth=2.2, label="真实均值")
    ax2.plot(grouped["horizon"], grouped["mean_predicted"], color="#f7d794", marker="o", linewidth=2.2, label="预测均值")
    ax2.plot(
        grouped["horizon"],
        grouped["coverage"] * grouped["mean_actual"].max(),
        color="#cdb4db",
        linestyle="--",
        linewidth=1.8,
        label="区间覆盖率（缩放）",
    )

    ax1.set_title("固定起点预测切片的步长误差")
    ax1.set_xlabel("预测步长")
    ax1.set_ylabel("MAE", color=ACCENT)
    ax2.set_ylabel("平均借出量", color=ACCENT_COOL)
    ax1.set_xticks(grouped["horizon"])
    ax1.set_xticklabels([f"H+{value}" for value in grouped["horizon"]])
    ax1.grid(axis="y", alpha=0.28)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return grouped


def plot_station_examples(
    prediction_df: pd.DataFrame, top_station_ids: list[str], output_path: Path
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=170, sharex=True, sharey=False)
    horizons = np.arange(1, 7)

    axes_flat = list(axes.flat)
    for axis, station_id in zip(axes_flat, top_station_ids):
        station_slice = prediction_df.loc[prediction_df["station_id"] == station_id].sort_values("horizon")
        axis.plot(horizons, station_slice["actual"], color=ACCENT_COOL, marker="o", linewidth=2.2, label="真实值")
        axis.plot(horizons, station_slice["predicted"], color=ACCENT, marker="o", linewidth=2.2, label="预测中位数")
        axis.fill_between(
            horizons,
            station_slice["predicted_p10"],
            station_slice["predicted_p90"],
            color=ACCENT,
            alpha=0.16,
            label="P10-P90 区间",
        )
        axis.set_title(f"站点 {station_id}")
        axis.set_xticks(horizons)
        axis.set_xticklabels([f"H+{value}" for value in horizons])
        axis.grid(alpha=0.28)

    for axis in axes_flat[len(top_station_ids) :]:
        axis.axis("off")

    for axis in axes[-1]:
        axis.set_xlabel("预测步长")
    for axis in axes[:, 0]:
        axis.set_ylabel("借出量")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncols=3, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("验证集起点切片下的高活跃站点预测示例", y=1.02, fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_scatter(prediction_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=170)
    palette = plt.cm.plasma(np.linspace(0.18, 0.92, prediction_df["horizon"].nunique()))

    max_value = max(float(prediction_df["actual"].max()), float(prediction_df["predicted"].max())) + 0.5
    for horizon, color in zip(sorted(prediction_df["horizon"].unique()), palette, strict=True):
        subset = prediction_df.loc[prediction_df["horizon"] == horizon]
        axes[0].scatter(
            subset["actual"],
            subset["predicted"],
            s=30,
            alpha=0.7,
            color=color,
            label=f"H+{horizon}",
            edgecolors="none",
        )
    axes[0].plot([0, max_value], [0, max_value], color=MUTED, linestyle="--", linewidth=1.2)
    axes[0].set_xlim(0, max_value)
    axes[0].set_ylim(0, max_value)
    axes[0].set_title("各预测步长下的真实值与预测值散点")
    axes[0].set_xlabel("真实借出量")
    axes[0].set_ylabel("预测借出量")
    axes[0].legend(frameon=False, ncols=3, fontsize=9, loc="upper left")
    axes[0].grid(alpha=0.28)

    residuals = prediction_df.assign(residual=lambda d: d["predicted"] - d["actual"])
    box_data = [
        residuals.loc[residuals["horizon"] == horizon, "residual"]
        for horizon in sorted(residuals["horizon"].unique())
    ]
    box = axes[1].boxplot(
        box_data,
        patch_artist=True,
        medianprops={"color": BG, "linewidth": 1.5},
        boxprops={"linewidth": 1.2},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for patch, color in zip(box["boxes"], palette, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(color)
    axes[1].axhline(0, color=MUTED, linestyle="--", linewidth=1.1)
    axes[1].set_title("各预测步长的残差分布")
    axes[1].set_xlabel("预测步长")
    axes[1].set_ylabel("预测值 - 真实值")
    axes[1].set_xticks(np.arange(1, prediction_df["horizon"].nunique() + 1))
    axes[1].set_xticklabels([f"H+{value}" for value in sorted(prediction_df["horizon"].unique())])
    axes[1].grid(axis="y", alpha=0.28)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_station_maps(station_metrics: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6), dpi=170)
    map_specs = [
        ("val_mean", "验证周站点平均借出量", "magma", 14.0),
        ("mae", "固定预测切片的站点 MAE", "viridis", 130.0),
    ]

    for axis, (column, title, cmap, size_scale) in zip(axes, map_specs, strict=True):
        scatter = axis.scatter(
            station_metrics["station_lng"],
            station_metrics["station_lat"],
            c=station_metrics[column],
            s=18 + station_metrics[column].fillna(0) * size_scale,
            cmap=cmap,
            alpha=0.85,
            linewidths=0.4,
            edgecolors="#dbe4ff22",
        )
        top_rows = station_metrics.nlargest(4, column)
        for _, row in top_rows.iterrows():
            axis.text(
                row["station_lng"] + 0.002,
                row["station_lat"] + 0.0015,
                str(row["station_id"]),
                fontsize=8,
                color=TEXT,
            )
        axis.set_title(title)
        axis.set_xlabel("经度")
        axis.set_ylabel("纬度")
        axis.grid(alpha=0.18)
        fig.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_station_rankings(station_metrics: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=170)

    top_mae = station_metrics.nlargest(10, "mae").sort_values("mae")
    axes[0].barh(top_mae["station_id"], top_mae["mae"], color=ACCENT_RED, alpha=0.86)
    axes[0].set_title("固定预测切片中 MAE 最大的站点")
    axes[0].set_xlabel("MAE")
    axes[0].grid(axis="x", alpha=0.28)

    top_bias = station_metrics.iloc[
        station_metrics["bias"].abs().sort_values(ascending=False).index
    ].head(10).sort_values("bias")
    colors = [ACCENT_COOL if value < 0 else ACCENT for value in top_bias["bias"]]
    axes[1].barh(top_bias["station_id"], top_bias["bias"], color=colors, alpha=0.86)
    axes[1].axvline(0, color=MUTED, linestyle="--", linewidth=1.1)
    axes[1].set_title("平均偏差绝对值最大的站点")
    axes[1].set_xlabel("偏差（预测值 - 真实值）")
    axes[1].grid(axis="x", alpha=0.28)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_interpretation_overview(
    importance_frames: dict[str, pd.DataFrame], output_path: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=170)

    static_df = importance_frames["static"].head(6).sort_values("importance")
    encoder_df = importance_frames["encoder"].head(10).sort_values("importance")
    decoder_df = importance_frames["decoder"].head(10).sort_values("importance")
    attention_df = importance_frames["attention"].tail(72)

    axes[0, 0].barh(static_df["variable"], static_df["importance"], color=ACCENT_COOL, alpha=0.9)
    axes[0, 0].set_title("静态变量重要性")
    axes[0, 0].grid(axis="x", alpha=0.26)

    axes[0, 1].barh(encoder_df["variable"], encoder_df["importance"], color=ACCENT, alpha=0.9)
    axes[0, 1].set_title("Encoder 侧最重要变量")
    axes[0, 1].grid(axis="x", alpha=0.26)

    axes[1, 0].barh(decoder_df["variable"], decoder_df["importance"], color="#cdb4db", alpha=0.9)
    axes[1, 0].set_title("Decoder 侧最重要变量")
    axes[1, 0].grid(axis="x", alpha=0.26)

    axes[1, 1].plot(attention_df["lag_hour"], attention_df["attention"], color=ACCENT_COOL, linewidth=2.2)
    axes[1, 1].fill_between(attention_df["lag_hour"], attention_df["attention"], color=ACCENT_COOL, alpha=0.16)
    axes[1, 1].set_title("预测起点前 72 小时的平均注意力")
    axes[1, 1].set_xlabel("距离预测起点的历史小时")
    axes[1, 1].set_ylabel("注意力权重")
    axes[1, 1].grid(alpha=0.28)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def table_to_html(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, classes="metric-table", border=0, justify="left")


def build_html_report(
    output_path: Path,
    run_dir: Path,
    checkpoint_path: Path,
    epoch_metrics: pd.DataFrame,
    validation_summary: dict[str, float | int],
    baseline_comparison: pd.DataFrame,
    horizon_metrics: pd.DataFrame,
    top_station_ids: list[str],
    station_metrics: pd.DataFrame,
    forecast_start_time_idx: int,
) -> None:
    best_row = epoch_metrics.loc[epoch_metrics["val_loss"].idxmin()]
    last_row = epoch_metrics.iloc[-1]

    improve_vs_lag1 = baseline_comparison.loc[baseline_comparison["model"] == "lag1", "mae"].iloc[0]
    improve_pct = (improve_vs_lag1 - best_row["val_MAE"]) / improve_vs_lag1 * 100
    coverage_mean = float(station_metrics["coverage"].mean())
    interval_width_mean = float(station_metrics["interval_width"].mean())
    worst_station = station_metrics.sort_values("mae", ascending=False).iloc[0]
    top_mae_table = table_to_html(
        station_metrics.sort_values("mae", ascending=False)
        .head(8)[["station_id", "mae", "bias", "val_mean"]]
        .rename(columns={"station_id": "站点", "mae": "MAE", "bias": "偏差", "val_mean": "验证周均值"})
        .round({"mae": 3, "bias": 3, "val_mean": 3})
    )
    bias_table = table_to_html(
        station_metrics.iloc[station_metrics["bias"].abs().sort_values(ascending=False).index]
        .head(8)[["station_id", "bias", "mae", "coverage"]]
        .rename(columns={"station_id": "站点", "bias": "偏差", "mae": "MAE", "coverage": "覆盖率"})
        .round({"bias": 3, "mae": 3, "coverage": 3})
    )

    def img(name: str, alt: str) -> str:
        return f'<img src="assets/{name}" alt="{html.escape(alt)}">'

    body = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TFT 训练报告</title>
  <style>
    :root {{
      --bg: #07111f;
      --panel: #101b2f;
      --line: #24334d;
      --text: #eef2ff;
      --muted: #9fb0c8;
      --accent: #f4a261;
      --cool: #7fd1b9;
      --danger: #e76f51;
      --shadow: 0 28px 80px rgba(0, 0, 0, 0.28);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Hiragino Sans GB", "PingFang HK", "Songti SC", "Avenir Next", "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(244,162,97,0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(127,209,185,0.12), transparent 26%),
        linear-gradient(180deg, #08101d 0%, #07111f 100%);
    }}
    .shell {{
      width: min(1200px, calc(100vw - 40px));
      margin: 0 auto;
      padding: 32px 0 64px;
    }}
    .hero {{
      padding: 28px 30px 24px;
      border: 1px solid rgba(159,176,200,0.16);
      background: linear-gradient(180deg, rgba(16,27,47,0.94), rgba(10,18,31,0.94));
      box-shadow: var(--shadow);
      overflow: hidden;
      position: relative;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      inset: auto -12% -35% auto;
      width: 340px;
      height: 340px;
      background: radial-gradient(circle, rgba(244,162,97,0.22), transparent 68%);
      pointer-events: none;
    }}
    .eyebrow {{
      color: var(--cool);
      letter-spacing: 0.12em;
      text-transform: uppercase;
      font-size: 12px;
      margin-bottom: 14px;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(34px, 5vw, 60px);
      line-height: 0.95;
      max-width: 10ch;
    }}
    .lede {{
      margin: 18px 0 0;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.65;
      max-width: 70ch;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 26px;
    }}
    .stat {{
      padding: 14px 16px;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(159,176,200,0.14);
      min-height: 96px;
    }}
    .stat .label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
      margin-bottom: 12px;
    }}
    .stat .value {{
      font-size: 30px;
      font-weight: 700;
      line-height: 1;
    }}
    .stat .sub {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }}
    .section {{
      margin-top: 26px;
      padding: 22px 24px 26px;
      border: 1px solid rgba(159,176,200,0.12);
      background: rgba(12, 21, 37, 0.86);
      box-shadow: var(--shadow);
    }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: end;
      margin-bottom: 18px;
    }}
    h2 {{
      margin: 0;
      font-size: 28px;
    }}
    .section-copy {{
      color: var(--muted);
      max-width: 68ch;
      line-height: 1.65;
      font-size: 15px;
    }}
    .two-up {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 18px;
      align-items: start;
    }}
    .notes {{
      display: grid;
      gap: 10px;
    }}
    .note {{
      padding: 14px 16px;
      border-left: 2px solid var(--accent);
      background: rgba(255,255,255,0.03);
      color: var(--muted);
      line-height: 1.6;
    }}
    .visual-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .visual-full {{
      margin-top: 18px;
    }}
    .table-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      margin-top: 18px;
    }}
    figure {{
      margin: 0;
      padding: 0;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
      border: 1px solid rgba(159,176,200,0.12);
      background: #0d1728;
    }}
    figcaption {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.55;
    }}
    .footer {{
      margin-top: 26px;
      color: var(--muted);
      font-size: 14px;
    }}
    code {{
      font-family: "SFMono-Regular", "JetBrains Mono", monospace;
      color: var(--cool);
    }}
    .metric-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      background: rgba(255,255,255,0.02);
      border: 1px solid rgba(159,176,200,0.12);
    }}
    .metric-table th,
    .metric-table td {{
      padding: 10px 12px;
      border-bottom: 1px solid rgba(159,176,200,0.08);
      text-align: left;
    }}
    .metric-table th {{
      color: var(--cool);
      font-weight: 600;
      background: rgba(127,209,185,0.05);
    }}
    .metric-table tr:last-child td {{
      border-bottom: 0;
    }}
    @media (max-width: 900px) {{
      .stats, .visual-grid, .two-up, .table-grid {{
        grid-template-columns: 1fr;
      }}
      .shell {{
        width: min(100vw - 24px, 1200px);
      }}
      .hero, .section {{
        padding-left: 18px;
        padding-right: 18px;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="eyebrow">Citi Bike TFT 训练报告</div>
      <h1>{html.escape(run_dir.name)}</h1>
      <p class="lede">
        这份报告总结了复制到本地的训练结果，最佳 checkpoint 为
        <code>{html.escape(checkpoint_path.name)}</code>。相较朴素的 <code>lag1</code> 基线，
        模型把验证集 MAE 提升了 <strong>{improve_pct:.1f}%</strong>；同时这段验证窗口本身比较稀疏，
        零值占比达到 <strong>{validation_summary["val_zero_rate"]:.1%}</strong>。
      </p>
      <div class="stats">
        <div class="stat">
          <div class="label">最佳验证损失</div>
          <div class="value">{best_row["val_loss"]:.3f}</div>
          <div class="sub">最佳点在 epoch {int(best_row["epoch"])} · 最后一轮为 {last_row["val_loss"]:.3f}</div>
        </div>
        <div class="stat">
          <div class="label">最佳验证 MAE</div>
          <div class="value">{best_row["val_MAE"]:.3f}</div>
          <div class="sub">RMSE {best_row["val_RMSE"]:.3f} · SMAPE {best_row["val_SMAPE"]:.3f}</div>
        </div>
        <div class="stat">
          <div class="label">验证窗口</div>
          <div class="value">{validation_summary["val_rows"]:,}</div>
          <div class="sub">{validation_summary["val_stations"]} 个站点 · 平均借出量 {validation_summary["val_mean"]:.2f}</div>
        </div>
        <div class="stat">
          <div class="label">区间覆盖 / 宽度</div>
          <div class="value">{coverage_mean:.1%}</div>
          <div class="sub">P10-P90 平均宽度 {interval_width_mean:.2f} · 切片起点 time_idx={forecast_start_time_idx}</div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>训练走势</h2>
          <div class="section-copy">
            训练 step loss 下降得很快，验证损失则较早找到最佳点，随后一直在较窄范围内波动。
            这更像是“快速收敛后进入平台期”，而不是“越训越坏”。
          </div>
        </div>
      </div>
      <div class="two-up">
        <figure>
          {img("training_overview.png", "训练概览")}
          <figcaption>
            左图是 step 级分位数损失及其平滑曲线，右图是按 epoch 聚合后的训练/验证损失，并标出了最终保存的最佳 checkpoint。
          </figcaption>
        </figure>
        <div class="notes">
          <div class="note">
            验证损失在 epoch <strong>{int(best_row["epoch"])}</strong> 达到最低点，之后虽然没有继续刷新，但最后一轮也只比最佳点高
            <strong>{((last_row["val_loss"] - best_row["val_loss"]) / best_row["val_loss"]) * 100:.1f}%</strong>。
          </div>
          <div class="note">
            这次训练不需要跑满所有设定 epoch 就已经给出足够稳定的信号，说明当前配置在这个数据切分上收敛很快。
          </div>
          <div class="note">
            由于验证窗口零值很多，百分比误差会被严重扭曲，所以这份报告有意不把 MAPE 当成核心指标来展示。
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>验证指标</h2>
          <div class="section-copy">
            这里展示的是这次训练里最稳定、也最值得读的验证指标。对当前这个 holdout 来说，MAE 和 RMSE 的参考价值最高。
          </div>
        </div>
      </div>
      <div class="visual-grid">
        <figure>
          {img("validation_metrics.png", "验证指标趋势")}
          <figcaption>
            MAE、RMSE 和 SMAPE 随 epoch 的变化。可以看到最佳 MAE 与按验证损失选出的 checkpoint 基本一致。
          </figcaption>
        </figure>
        <figure>
          {img("baseline_comparison.png", "基线对比")}
          <figcaption>
            TFT 与几个简单 lag 基线的对比。即便是最强的简单延续型基线，在 MAE 和 RMSE 上也都落后于模型。
          </figcaption>
        </figure>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>验证窗口特征</h2>
          <div class="section-copy">
            这段 holdout 周期比全量数据稀疏得多，所以必须结合分布背景来看误差。下面这组图展示了目标值的稀疏性，以及误差如何随预测步长变化。
          </div>
        </div>
      </div>
      <div class="visual-grid">
        <figure>
          {img("validation_distribution.png", "验证分布")}
          <figcaption>
            验证期需求整体偏低，而且零值占比高，这也是为什么绝对误差类指标会比百分比误差更可信。
          </figcaption>
        </figure>
        <figure>
          {img("forecast_horizon.png", "步长误差")}
          <figcaption>
            固定起点预测切片上的分步长误差。它能帮助我们判断模型是在第几步开始明显偏离，还是 6 小时内整体都比较稳。
          </figcaption>
        </figure>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>校准与偏差</h2>
          <div class="section-copy">
            这一部分关注的不只是“平均来看准不准”，而是模型在站点层面能不能把幅度和偏差也一起对上。
          </div>
        </div>
      </div>
      <div class="visual-grid">
        <figure>
          {img("prediction_scatter.png", "预测散点与残差")}
          <figcaption>
            左图是按预测步长着色的真实值/预测值散点，右图是按步长聚合的残差分布，可以直接看出模型是偏保守还是偏激进。
          </figcaption>
        </figure>
        <figure>
          {img("station_rankings.png", "站点排行")}
          <figcaption>
            切片里最难预测的站点并不是随机冒出来的。有些站点即便整体 MAE 看着不错，依然会持续低估或高估。
          </figcaption>
        </figure>
      </div>
      <div class="table-grid">
        <div>
          <div class="section-copy">固定预测切片中 MAE 最大的站点</div>
          {top_mae_table}
        </div>
        <div>
          <div class="section-copy">平均偏差绝对值最大的站点</div>
          {bias_table}
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>空间视角</h2>
          <div class="section-copy">
            把站点放回地图后，能更容易区分两件事：哪些地方在验证周本来就更忙，以及哪些地方在这次固定预测切片里依然很难拟合。
          </div>
        </div>
      </div>
      <figure class="visual-full">
        {img("station_maps.png", "站点地图")}
        <figcaption>
          左图是验证周的站点平均借出量，右图是这次固定预测切片下的站点 MAE。当前切片里最难的站点是 <strong>{html.escape(str(worst_station["station_id"]))}</strong>，MAE 为 <strong>{worst_station["mae"]:.2f}</strong>。
        </figcaption>
      </figure>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>TFT 可解释性快照</h2>
          <div class="section-copy">
            这里用一页图浓缩了模型在固定验证切片上最依赖的信号：静态站点信息、encoder 侧历史特征、decoder 侧日历特征，以及最近一段历史上的注意力分布。
          </div>
        </div>
      </div>
      <figure class="visual-full">
        {img("interpretation_overview.png", "解释性概览")}
        <figcaption>
          变量重要性是在当前预测切片上聚合得到的，并分别展示了静态、encoder、decoder 三条路径；注意力面板为了更易读，只保留了最近 72 小时。
        </figcaption>
      </figure>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>站点预测示例</h2>
          <div class="section-copy">
            这些例子挑的是验证周里最活跃的站点，并固定使用 holdout 起点的那一条 6 小时预测序列来展示。
          </div>
        </div>
      </div>
      <figure class="visual-full">
        {img("station_examples.png", "站点预测示例")}
        <figcaption>
          示例站点：{", ".join(html.escape(station_id) for station_id in top_station_ids)}。
          橙色半透明区域表示 P10-P90 区间，所以这组图不只是单线拟合对比，也能顺手看一下不确定性范围。
        </figcaption>
      </figure>
    </section>

    <div class="footer">
      报告生成自 <code>{html.escape(str(run_dir))}</code>。如果你想单独查看图片，可以直接打开 <code>assets/</code> 目录；如果想继续做分析，这个报告目录里还附带了站点级和步长级的 CSV 摘要。
    </div>
  </main>
</body>
</html>
"""
    output_path.write_text(body, encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    load_stage_data, make_stage_datasets = load_training_helpers()

    run_dir = project_path(args.run_dir)
    output_dir = project_path(args.output_dir) if args.output_dir else run_dir / "report"
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    train_step, epoch_metrics = load_metrics(run_dir)
    checkpoint_path = load_best_checkpoint(run_dir)
    checkpoint = load_checkpoint_metadata(checkpoint_path)

    df = load_stage_data(args.data)
    val_df, validation_summary = compute_validation_summary(df, validation_horizon=args.validation_horizon)
    baselines = compute_baselines(df, validation_horizon=args.validation_horizon)
    station_metadata = compute_station_metadata(df, val_df)

    plot_training_overview(train_step, epoch_metrics, assets_dir / "training_overview.png")
    plot_validation_metrics(epoch_metrics, assets_dir / "validation_metrics.png")
    best_row = epoch_metrics.loc[epoch_metrics["val_loss"].idxmin()]
    baseline_comparison = plot_baseline_comparison(
        baselines,
        best_mae=float(best_row["val_MAE"]),
        best_rmse=float(best_row["val_RMSE"]),
        output_path=assets_dir / "baseline_comparison.png",
    )
    plot_validation_distribution(val_df, assets_dir / "validation_distribution.png")

    train_ds, val_ds = make_stage_datasets(
        df,
        argparse.Namespace(
            target=checkpoint["dataset_parameters"]["target"],
            max_encoder_length=int(checkpoint["hyper_parameters"]["max_encoder_length"]),
            max_prediction_length=int(checkpoint["dataset_parameters"]["max_prediction_length"]),
            validation_horizon=args.validation_horizon,
        ),
    )
    max_prediction_length = int(train_ds.max_prediction_length)
    forecast_start_time_idx = args.forecast_start_time_idx
    if forecast_start_time_idx is None:
        forecast_start_time_idx = validation_summary["cutoff"] + 1

    subset = make_forecast_subset(val_ds, forecast_start_time_idx, max_prediction_length)
    model = build_model(train_ds, checkpoint)
    prediction_df = predict_fixed_windows(model, subset)
    station_metrics = compute_station_slice_metrics(prediction_df, station_metadata)
    interpretation = compute_interpretation(model, subset)
    importance_frames = build_variable_importance_frames(checkpoint, interpretation)

    horizon_metrics = plot_horizon_error(prediction_df, assets_dir / "forecast_horizon.png")
    plot_prediction_scatter(prediction_df, assets_dir / "prediction_scatter.png")
    plot_station_maps(station_metrics, assets_dir / "station_maps.png")
    plot_station_rankings(station_metrics, assets_dir / "station_rankings.png")
    plot_interpretation_overview(importance_frames, assets_dir / "interpretation_overview.png")

    top_station_ids = compute_top_station_ids(station_metrics, count=6)
    plot_station_examples(prediction_df, top_station_ids, assets_dir / "station_examples.png")

    prediction_df.to_csv(output_dir / "forecast_slice_predictions.csv", index=False)
    horizon_metrics.to_csv(output_dir / "horizon_metrics.csv", index=False)
    station_metrics.to_csv(output_dir / "station_slice_metrics.csv", index=False)
    importance_frames["static"].to_csv(output_dir / "static_variable_importance.csv", index=False)
    importance_frames["encoder"].to_csv(output_dir / "encoder_variable_importance.csv", index=False)
    importance_frames["decoder"].to_csv(output_dir / "decoder_variable_importance.csv", index=False)
    importance_frames["attention"].to_csv(output_dir / "attention_profile.csv", index=False)

    build_html_report(
        output_path=output_dir / "index.html",
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        epoch_metrics=epoch_metrics,
        validation_summary=validation_summary,
        baseline_comparison=baseline_comparison,
        horizon_metrics=horizon_metrics,
        top_station_ids=top_station_ids,
        station_metrics=station_metrics,
        forecast_start_time_idx=int(forecast_start_time_idx),
    )

    summary_path = output_dir / "README.txt"
    summary_path.write_text(
        dedent(
            f"""\
            TFT 中文报告已生成。

            打开：
            - {output_dir / 'index.html'}

            图表资源：
            - {assets_dir / 'training_overview.png'}
            - {assets_dir / 'validation_metrics.png'}
            - {assets_dir / 'baseline_comparison.png'}
            - {assets_dir / 'validation_distribution.png'}
            - {assets_dir / 'forecast_horizon.png'}
            - {assets_dir / 'prediction_scatter.png'}
            - {assets_dir / 'station_maps.png'}
            - {assets_dir / 'station_rankings.png'}
            - {assets_dir / 'interpretation_overview.png'}
            - {assets_dir / 'station_examples.png'}

            CSV 摘要：
            - {output_dir / 'forecast_slice_predictions.csv'}
            - {output_dir / 'horizon_metrics.csv'}
            - {output_dir / 'station_slice_metrics.csv'}
            - {output_dir / 'static_variable_importance.csv'}
            - {output_dir / 'encoder_variable_importance.csv'}
            - {output_dir / 'decoder_variable_importance.csv'}
            - {output_dir / 'attention_profile.csv'}
            """
        ),
        encoding="utf-8",
    )
    print(f"报告已写入 {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
