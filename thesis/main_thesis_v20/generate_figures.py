from __future__ import annotations

import shutil
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "figures"
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
PLATFORM_SCREENSHOT_SOURCE = ROOT / "visualization_platform" / "artifacts" / "bike_viz_dashboard_real.png"


def configure_plotting() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    if FONT_PATH.exists():
        font_manager.fontManager.addfont(str(FONT_PATH))
        font_name = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
        plt.rcParams["font.family"] = [font_name, "DejaVu Sans"]
        plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 220


def wrap_label(text: str, width: int = 13) -> str:
    if "\n" in text:
        return text
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def add_box(ax, xy, wh, text, face="#eef4ff", edge="#3867a8", fontsize=10, lw=1.5):
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#172033",
        linespacing=1.28,
    )
    return box


def add_arrow(ax, start, end, color="#4b5563", lw=1.8, rad=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)


def add_group(ax, xy, wh, label, face="#f8fafc", edge="#cbd5e1", lw=1.1):
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.018,
        y + h - 0.028,
        label,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color="#334155",
    )
    return box


def finish_diagram(fig, ax, title: str, out_name: str):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.savefig(OUT_DIR / out_name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_data_pipeline():
    fig, ax = plt.subplots(figsize=(13.5, 7.3))
    colors = ["#dbeafe", "#dcfce7", "#ffedd5", "#f5e8ff", "#e0f2fe", "#fee2e2"]
    boxes = {
        "orders": ((0.04, 0.73), (0.18, 0.12), "Citi Bike 2022\n原始订单\n30,689,921 行", colors[0]),
        "weather": ((0.04, 0.51), (0.18, 0.12), "Open-Meteo\n小时天气\n温度/降水/风速", colors[1]),
        "poi": ((0.04, 0.29), (0.18, 0.12), "OpenStreetMap\n500m POI\n7 类设施", colors[2]),
        "aggregate": ((0.31, 0.62), (0.18, 0.13), "小时级站点面板\ndep/arr/net-flow\n库存代理", colors[3]),
        "select": ((0.31, 0.39), (0.18, 0.13), "站点筛选\nTop 883\n约覆盖 90% 流量", colors[4]),
        "features": ((0.58, 0.55), (0.18, 0.16), "多源特征融合\n时间/天气/历史\nPOI/OD 图/容量", colors[1]),
        "bundle": ((0.58, 0.31), (0.18, 0.14), "训练 bundle\n12 小时输入\n12 小时输出", colors[0]),
        "models": ((0.82, 0.55), (0.14, 0.15), "预测模型\nGraph WaveNet\nTFT-style", colors[5]),
        "rebalance": ((0.82, 0.29), (0.14, 0.15), "调度回测\nmin-cost flow\n可视化平台", colors[2]),
    }
    for xy, wh, text, color in boxes.values():
        add_box(ax, xy, wh, text, face=color, fontsize=10)
    add_arrow(ax, (0.22, 0.79), (0.31, 0.69))
    add_arrow(ax, (0.22, 0.57), (0.31, 0.66))
    add_arrow(ax, (0.22, 0.35), (0.31, 0.45))
    add_arrow(ax, (0.49, 0.685), (0.58, 0.63))
    add_arrow(ax, (0.49, 0.455), (0.58, 0.60))
    add_arrow(ax, (0.67, 0.55), (0.67, 0.45))
    add_arrow(ax, (0.76, 0.63), (0.82, 0.63))
    add_arrow(ax, (0.76, 0.38), (0.82, 0.37))
    add_arrow(ax, (0.89, 0.55), (0.89, 0.44))
    ax.text(0.58, 0.22, "数据划分：train / val / test = 0.7 / 0.1 / 0.2，测试窗口 1749 个", fontsize=10, color="#374151")
    finish_diagram(fig, ax, "图3-1 数据处理与特征工程流程", "fig3_1_data_pipeline.png")


def fig_forecast_architecture():
    fig, ax = plt.subplots(figsize=(13.5, 7.3))
    add_group(ax, (0.03, 0.19), (0.22, 0.62), "输入特征", face="#f8fafc")
    add_group(ax, (0.29, 0.16), (0.40, 0.68), "双分支预测主体", face="#ffffff", edge="#c7d2fe")
    add_group(ax, (0.73, 0.19), (0.24, 0.62), "输出结果", face="#f8fafc")
    add_box(ax, (0.05, 0.57), (0.18, 0.13), "多源历史特征\npast 12h", "#e0f2fe", fontsize=9.5)
    add_box(ax, (0.05, 0.31), (0.18, 0.13), "未来已知特征\nfuture covariates", "#dcfce7", fontsize=9.5)
    add_box(ax, (0.33, 0.58), (0.14, 0.13), "Graph WaveNet\n时序卷积 + 图卷积", "#dbeafe", fontsize=9.1)
    add_box(ax, (0.51, 0.58), (0.14, 0.13), "TFT-style\nLSTM + attention", "#f5e8ff", fontsize=9.3)
    add_box(ax, (0.33, 0.31), (0.14, 0.13), "共享编码\n时空表征", "#eef2ff", fontsize=9.3)
    add_box(ax, (0.51, 0.31), (0.14, 0.13), "分位数头\nq10 / q50 / q90", "#ffedd5", fontsize=9.3)
    add_box(ax, (0.76, 0.56), (0.18, 0.12), "dep / arr\n点预测", "#fee2e2", fontsize=9.6)
    add_box(ax, (0.76, 0.38), (0.18, 0.12), "q10 / q50 / q90\n区间预测", "#dcfce7", fontsize=9.4)
    add_box(ax, (0.76, 0.20), (0.18, 0.12), "调度输入\nfuture net-flow", "#e0f2fe", fontsize=9.4)
    for s, e in [
        ((0.23, 0.64), (0.33, 0.64)),
        ((0.23, 0.38), (0.33, 0.39)),
        ((0.23, 0.64), (0.33, 0.39)),
        ((0.23, 0.38), (0.33, 0.64)),
        ((0.47, 0.64), (0.51, 0.64)),
        ((0.65, 0.64), (0.76, 0.62)),
        ((0.47, 0.38), (0.51, 0.38)),
        ((0.65, 0.38), (0.76, 0.44)),
        ((0.65, 0.64), (0.76, 0.44)),
    ]:
        add_arrow(ax, s, e)
    ax.text(0.03, 0.09, "统一任务：过去 12 小时多源特征 + 未来已知时间特征 -> 未来 12 小时站点级租还预测", fontsize=10.2, color="#475569")
    finish_diagram(fig, ax, "图4-1 预测模型总体结构", "fig4_1_forecast_architecture.png")


def fig_gwnet_time_netloss():
    fig, ax = plt.subplots(figsize=(13.5, 7.3))
    add_group(ax, (0.03, 0.20), (0.20, 0.60), "输入与约束", face="#f8fafc")
    add_group(ax, (0.27, 0.24), (0.46, 0.54), "Graph WaveNet time + net-loss", face="#ffffff", edge="#bfdbfe")
    add_group(ax, (0.77, 0.20), (0.20, 0.60), "输出与辅助目标", face="#f8fafc")
    add_box(ax, (0.05, 0.54), (0.16, 0.14), "输入张量\n[B,12,N,F]", "#e0f2fe", fontsize=9.4)
    add_box(ax, (0.05, 0.30), (0.16, 0.14), "未来时间特征\nhour / dow / month", "#dcfce7", fontsize=9.3)
    add_box(ax, (0.31, 0.56), (0.14, 0.12), "Temporal Conv\n1,2,4 × 2", "#dbeafe", fontsize=9.2)
    add_box(ax, (0.48, 0.56), (0.14, 0.12), "Graph Conv\nadaptive + OD top-k20", "#eef2ff", fontsize=9.0)
    add_box(ax, (0.65, 0.56), (0.06, 0.12), "Skip\nEnd", "#f5e8ff", fontsize=9.0)
    add_box(ax, (0.31, 0.30), (0.40, 0.14), "Time-conditioned Horizon Readout\nper-step time embedding + shared state", "#fee2e2", fontsize=9.1)
    add_box(ax, (0.80, 0.54), (0.14, 0.13), "dep / arr\n12-step forecast", "#dcfce7", fontsize=9.3)
    add_box(ax, (0.80, 0.34), (0.14, 0.13), "net-flow MAE\nλ = 0.10", "#e0f2fe", fontsize=9.3)
    add_box(ax, (0.80, 0.18), (0.14, 0.10), "反变换评价", "#ffedd5", fontsize=9.3)
    for s, e in [
        ((0.21, 0.61), (0.31, 0.62)),
        ((0.45, 0.62), (0.48, 0.62)),
        ((0.62, 0.62), (0.65, 0.62)),
        ((0.68, 0.56), (0.80, 0.60)),
        ((0.68, 0.34), (0.80, 0.40)),
        ((0.68, 0.34), (0.80, 0.23)),
        ((0.21, 0.37), (0.31, 0.37)),
        ((0.21, 0.61), (0.31, 0.37)),
    ]:
        add_arrow(ax, s, e)
    ax.text(0.03, 0.09, "核心收益：目标时间特征缓解 horizon 相位偏移；辅助 net-flow 损失让输出更贴近调度目标。", fontsize=10.2, color="#475569")
    finish_diagram(fig, ax, "图4-2 Graph WaveNet time + net-loss 模型结构", "fig4_2_gwnet_time_netloss.png")


def fig_tft_quantile():
    fig, ax = plt.subplots(figsize=(13.5, 7.3))
    steps = [
        ((0.04, 0.62), (0.16, 0.13), "多源历史输入\n88 features\npast 12h", "#e0f2fe"),
        ((0.25, 0.62), (0.16, 0.13), "Gated Feature\nProjection\n特征融合", "#dcfce7"),
        ((0.46, 0.62), (0.16, 0.13), "LSTM Encoder\n历史时序状态", "#dbeafe"),
        ((0.25, 0.33), (0.16, 0.13), "未来已知时间\nhorizon queries", "#ffedd5"),
        ((0.46, 0.33), (0.16, 0.13), "Multi-head\nTemporal Attention\n对齐历史滞后", "#f5e8ff"),
        ((0.67, 0.47), (0.16, 0.14), "Station Embedding\n静态上下文\n+ attention context", "#fee2e2"),
        ((0.87, 0.47), (0.10, 0.14), "Quantile Head\nq10/q50/q90", "#dcfce7"),
    ]
    for xy, wh, text, color in steps:
        add_box(ax, xy, wh, text, face=color, fontsize=9.5)
    for s, e in [
        ((0.20, 0.685), (0.25, 0.685)),
        ((0.41, 0.685), (0.46, 0.685)),
        ((0.54, 0.62), (0.54, 0.46)),
        ((0.41, 0.395), (0.46, 0.395)),
        ((0.62, 0.395), (0.67, 0.52)),
        ((0.62, 0.685), (0.67, 0.55)),
        ((0.83, 0.54), (0.87, 0.54)),
    ]:
        add_arrow(ax, s, e)
    ax.text(0.04, 0.20, "正式结果：q50 Avg MAE 1.5899，PICP80 0.8107，平均区间宽度 4.8970。", fontsize=10.5, color="#374151")
    finish_diagram(fig, ax, "图4-3 TFT-style 分位数预测模块结构", "fig4_3_tft_quantile.png")


def fig_rebalancing_pipeline():
    fig, ax = plt.subplots(figsize=(13.5, 7.3))
    steps = [
        ((0.04, 0.60), (0.16, 0.13), "当前库存\ncapacity_hat\ninventory_hat", "#e0f2fe"),
        ((0.04, 0.33), (0.16, 0.13), "未来净流量预测\npoint / quantile\n12 horizons", "#ffedd5"),
        ((0.27, 0.47), (0.17, 0.15), "Rolling Horizon\n库存递推\nI + cumulative flow", "#dcfce7"),
        ((0.51, 0.47), (0.16, 0.15), "安全库存带\n20% - 80%\ndonor / receiver", "#f5e8ff"),
        ((0.73, 0.47), (0.16, 0.15), "Min-cost Flow\n距离费用\ncap = 200 bikes/h", "#dbeafe"),
        ((0.73, 0.22), (0.16, 0.13), "Transfer Plan\n取放任务\nbike-km", "#fee2e2"),
        ((0.51, 0.22), (0.16, 0.13), "真实流量回放\n库存模拟\n边界小时", "#dcfce7"),
    ]
    for xy, wh, text, color in steps:
        add_box(ax, xy, wh, text, face=color, fontsize=9.8)
    for s, e in [
        ((0.20, 0.665), (0.27, 0.55)),
        ((0.20, 0.395), (0.27, 0.51)),
        ((0.44, 0.545), (0.51, 0.545)),
        ((0.67, 0.545), (0.73, 0.545)),
        ((0.81, 0.47), (0.81, 0.35)),
        ((0.73, 0.285), (0.67, 0.285)),
    ]:
        add_arrow(ax, s, e)
    ax.text(0.04, 0.17, "评价指标：empty / full / below lower / above upper / total bike-km / transfer actions", fontsize=10.5, color="#374151")
    finish_diagram(fig, ax, "图5-1 预测驱动调度流程", "fig5_1_rebalancing_pipeline.png")


def fig_platform_architecture():
    fig, ax = plt.subplots(figsize=(13.5, 7.3))
    add_group(ax, (0.03, 0.61), (0.24, 0.14), "前端展示层", face="#f8fafc")
    add_group(ax, (0.30, 0.34), (0.40, 0.42), "FastAPI 后端服务", face="#ffffff", edge="#c7d2fe")
    add_group(ax, (0.74, 0.20), (0.21, 0.58), "离线数据与结果缓存", face="#f8fafc")
    add_box(ax, (0.05, 0.63), (0.20, 0.09), "React + TypeScript\nMapLibre / ECharts", "#e0f2fe", fontsize=9.2)
    add_box(ax, (0.33, 0.58), (0.34, 0.08), "GET /api/meta   GET /api/decision   GET /api/station", "#dcfce7", fontsize=9.0)
    add_box(ax, (0.33, 0.46), (0.15, 0.10), "DataRepository\n站点面板", "#dbeafe", fontsize=9.1)
    add_box(ax, (0.50, 0.46), (0.15, 0.10), "ForecastService\n预测输出", "#f5e8ff", fontsize=9.1)
    add_box(ax, (0.33, 0.32), (0.15, 0.10), "RebalancingService\n调度计算", "#ffedd5", fontsize=9.1)
    add_box(ax, (0.50, 0.32), (0.15, 0.10), "CacheService\n结果复用", "#fee2e2", fontsize=9.1)
    add_box(ax, (0.76, 0.56), (0.16, 0.10), "站点小时面板", "#dbeafe", fontsize=9.2)
    add_box(ax, (0.76, 0.43), (0.16, 0.10), "静态特征 / 图关系", "#dcfce7", fontsize=9.0)
    add_box(ax, (0.76, 0.30), (0.16, 0.10), "预测缓存 / 模型输出", "#f5e8ff", fontsize=9.0)
    add_box(ax, (0.76, 0.17), (0.16, 0.10), "调度结果 / run summary", "#fee2e2", fontsize=9.0)
    for s, e in [
        ((0.25, 0.68), (0.33, 0.62)),
        ((0.67, 0.62), (0.76, 0.61)),
        ((0.67, 0.58), (0.76, 0.49)),
        ((0.67, 0.54), (0.76, 0.36)),
        ((0.67, 0.36), (0.76, 0.23)),
        ((0.33, 0.51), (0.33, 0.42)),
        ((0.48, 0.51), (0.48, 0.42)),
        ((0.56, 0.36), (0.56, 0.42)),
    ]:
        add_arrow(ax, s, e)
    finish_diagram(fig, ax, "图6-1 可视化平台系统架构", "fig6_1_platform_architecture.png")


def fig_dashboard_mock():
    if not PLATFORM_SCREENSHOT_SOURCE.exists():
        raise FileNotFoundError(f"Missing real platform screenshot: {PLATFORM_SCREENSHOT_SOURCE}")
    shutil.copyfile(PLATFORM_SCREENSHOT_SOURCE, OUT_DIR / "fig6_2_dashboard_map.png")


def fig_model_mae():
    df = pd.DataFrame(
        [
            {"model": "AGCRN\nbaseline", "avg_mae": 2.2096},
            {"model": "AGCRN\nlog1p+OD", "avg_mae": 2.1073},
            {"model": "Graph\nWaveNet v1", "avg_mae": 1.7666},
            {"model": "GWNet time\n+ net-loss", "avg_mae": 1.6238},
            {"model": "GWNet\n+ POI", "avg_mae": 1.7233},
            {"model": "TFT-style\nquantile", "avg_mae": 1.5899},
        ]
    ).sort_values("avg_mae", ascending=True)
    palette = sns.color_palette("Blues_r", len(df))
    fig, ax = plt.subplots(figsize=(11.8, 6.4))
    sns.barplot(
        data=df,
        y="model",
        x="avg_mae",
        hue="model",
        order=df["model"],
        palette=palette,
        legend=False,
        ax=ax,
        edgecolor="#334155",
        linewidth=0.6,
    )
    ax.set_xlabel("测试集 Avg MAE（原始订单计数尺度）")
    ax.set_ylabel("")
    ax.set_title("图7-1 预测模型 Avg MAE 对比", fontsize=16, fontweight="bold", pad=14)
    ax.grid(axis="x", linestyle="--", alpha=0.22)
    ax.set_xlim(0, 2.35)
    for patch, value in zip(ax.patches, df["avg_mae"].tolist()):
        ax.text(
            patch.get_width() + 0.02,
            patch.get_y() + patch.get_height() / 2,
            f"{value:.4f}",
            va="center",
            ha="left",
            fontsize=9.8,
            color="#0f172a",
        )
    ax.text(
        0.985,
        -0.12,
        "按平均 MAE 从低到高排序",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.2,
        color="#64748b",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig7_1_model_mae_comparison.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_horizon_metric_curves():
    model_specs = [
        (
            "Graph WaveNet v1",
            ROOT / "forecasting_models/agcrn_nyc_gwnet_v1/runs/gwnet_top883_log1p_topk20_b64/test_horizon_metrics.csv",
        ),
        (
            "Graph WaveNet time + net-loss",
            ROOT / "forecasting_models/agcrn_nyc_gwnet_time_netloss_v1/runs/gwnet_time_netloss_top883_log1p_topk20_b64/test_horizon_metrics.csv",
        ),
        (
            "TFT q50",
            ROOT / "forecasting_models/tft_quantile_calibrator_v1/runs/tft_quantile_top883_poi_v1_b16_e8/test_q50_horizon_metrics.csv",
        ),
    ]
    frames = []
    for model_name, path in model_specs:
        df = pd.read_csv(path, usecols=["horizon", "target", "mae"])
        df["model"] = model_name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def fig_horizon_mae_curve():
    df = load_horizon_metric_curves()
    df["target_label"] = df["target"].map({"dep": "出发量 dep", "arr": "到达量 arr"})
    palette = {
        "Graph WaveNet v1": "#94a3b8",
        "Graph WaveNet time + net-loss": "#22c55e",
        "TFT q50": "#3b82f6",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6), sharey=True)
    for ax, target, title in zip(axes, ["dep", "arr"], ["出发量 dep", "到达量 arr"]):
        sub = df[df["target"] == target].copy()
        sns.lineplot(
            data=sub,
            x="horizon",
            y="mae",
            hue="model",
            style="model",
            markers=True,
            dashes=False,
            linewidth=2.0,
            palette=palette,
            ax=ax,
        )
        ax.set_title(title, fontsize=13.5, fontweight="bold", pad=10)
        ax.set_xlabel("预测步长 horizon")
        ax.set_ylabel("MAE" if target == "dep" else "")
        ax.set_xticks(range(1, 13))
        ax.grid(True, linestyle="--", alpha=0.22)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:3], labels[:3], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("图7-5 代表性模型分步长 MAE 曲线", fontsize=16, fontweight="bold", y=1.07)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "fig7_5_horizon_mae_curve.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_quantile_example():
    forecast_path = ROOT / "forecasting_models/tft_quantile_calibrator_v1/runs/tft_quantile_top883_poi_v1_b16_e8/test_quantile_forecasts_for_rebalancing.parquet"
    panel_path = ROOT / "dataset/preprocessing/processed/nyc_top883_v2/nyc_station_hour_panel.parquet"
    decision_ts = pd.Timestamp("2022-10-19 21:00:00")
    fds = ds.dataset(forecast_path, format="parquet")
    ftab = fds.to_table(
        columns=["target_ts", "node_idx", "net_flow_q10", "net_flow_q50", "net_flow_q90"],
        filter=ds.field("decision_ts") == np.datetime64(decision_ts, "ns"),
    )
    f = ftab.to_pandas()
    start = f["target_ts"].min()
    end = f["target_ts"].max()
    pds = ds.dataset(panel_path, format="parquet")
    ptab = pds.to_table(
        columns=["ts", "node_idx", "station_name", "net_flow"],
        filter=(ds.field("ts") >= np.datetime64(start, "us")) & (ds.field("ts") <= np.datetime64(end, "us")),
    )
    actual = ptab.to_pandas().rename(columns={"ts": "target_ts", "net_flow": "actual_net_flow"})
    merged = f.merge(actual, on=["target_ts", "node_idx"], how="left")
    node_scores = merged.groupby("node_idx")["actual_net_flow"].apply(lambda s: float(np.abs(s).sum()))
    node_idx = int(node_scores.idxmax())
    station_name = str(merged.loc[merged["node_idx"] == node_idx, "station_name"].dropna().iloc[0])
    example = merged[merged["node_idx"] == node_idx].sort_values("target_ts")
    return decision_ts, node_idx, station_name, example


def fig_quantile_interval():
    decision_ts, node_idx, station_name, df = load_quantile_example()
    x = np.arange(1, len(df) + 1)
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    sns.lineplot(x=x, y=df["net_flow_q50"], ax=ax, color="#2563eb", marker="o", linewidth=2.3, label="TFT q50")
    sns.lineplot(x=x, y=df["actual_net_flow"], ax=ax, color="#111827", marker="s", linewidth=2.1, label="真实净流量")
    ax.fill_between(x, df["net_flow_q10"], df["net_flow_q90"], color="#93c5fd", alpha=0.30, label="q10-q90 区间")
    ax.axhline(0, color="#64748b", lw=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xlabel("预测步长 horizon（小时）")
    ax.set_ylabel("站点净流量（辆）")
    ax.set_title(
        f"图7-2 TFT q10-q90 站点净流量预测区间\n决策时刻 {decision_ts}，node {node_idx}，{station_name}",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig7_2_quantile_interval_example.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_attention_heatmap():
    path = ROOT / "forecasting_models/tft_quantile_calibrator_v1/runs/tft_quantile_top883_poi_v1_b16_e8/interpretability_v1/attention_horizon_lag_matrix.csv"
    df = pd.read_csv(path)
    heatmap_df = df.set_index("horizon")
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    sns.heatmap(
        heatmap_df,
        ax=ax,
        cmap="YlGnBu",
        cbar_kws={"label": "平均注意力权重", "fraction": 0.035, "pad": 0.02},
        linewidths=0.4,
        linecolor="#ffffff",
    )
    ax.set_title("图7-3 TFT-style 注意力滞后热力图", fontsize=16, fontweight="bold", pad=16)
    ax.set_xlabel("历史相对时间")
    ax.set_ylabel("预测步长 horizon")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig7_3_attention_heatmap.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_rebalancing_boundary():
    labels = [
        "无调度",
        "oracle\nmin-cost",
        "GWNet v1\nforecast",
        "GWNet time\n+ net-loss",
        "TFT q50",
        "TFT q10",
        "TFT q90",
    ]
    below = [412308, 96889, 93842, 99860, 106077, 219667, 65415]
    above = [631700, 13331, 16378, 16956, 18424, 75693, 349587]
    data = pd.DataFrame(
        {
            "strategy": np.repeat(labels, 2),
            "violation_type": ["below lower", "above upper"] * len(labels),
            "hours": np.array(sum(zip(below, above), ())),
        }
    )
    data["strategy"] = pd.Categorical(data["strategy"], categories=labels, ordered=True)
    fig, ax = plt.subplots(figsize=(12.8, 6.9))
    sns.barplot(
        data=data,
        y="strategy",
        x="hours",
        hue="violation_type",
        orient="h",
        palette={"below lower": "#60a5fa", "above upper": "#f97316"},
        ax=ax,
        edgecolor="#1f2937",
        linewidth=0.4,
    )
    totals = np.array(below) + np.array(above)
    for yi, total in enumerate(totals):
        ax.text(total + 18000, yi, f"{total:,}", ha="left", va="center", fontsize=9.2)
    ax.set_title("图7-4 调度结果安全库存带违规小时对比", fontsize=16, fontweight="bold", pad=16)
    ax.set_xlabel("安全库存带违规站点-小时")
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", alpha=0.22)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig7_4_rebalancing_boundary_comparison.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_inventory_case_study():
    inventory_path = ROOT / "rebalancing_algorithms/nyc_rebalancing_mincost_v1/runs/forecast_gwnet_time_netloss_mincost_h12_top883_v2_cap200/inventory_simulation.parquet"
    static_path = ROOT / "dataset/preprocessing/processed/nyc_top883_v2/nyc_station_static_features.csv"
    df = pd.read_parquet(inventory_path)
    static = pd.read_csv(static_path, usecols=["node_idx", "station_name"])
    df = df.merge(static, on="node_idx", how="left")
    df["base_violate"] = (
        (df["baseline_inventory_end_next_hour"] < df["lower_target_inventory"])
        | (df["baseline_inventory_end_next_hour"] > df["upper_target_inventory"])
    ).astype(int)
    df["reb_violate"] = (
        (df["inventory_end_next_hour"] < df["lower_target_inventory"])
        | (df["inventory_end_next_hour"] > df["upper_target_inventory"])
    ).astype(int)
    df["abs_delta"] = (df["inventory_after_rebalance"] - df["inventory_before_rebalance"]).abs()
    station_stats = (
        df.groupby(["node_idx", "station_name"])
        .agg(
            base_violate=("base_violate", "sum"),
            reb_violate=("reb_violate", "sum"),
            abs_delta=("abs_delta", "sum"),
        )
        .assign(reduction=lambda x: x["base_violate"] - x["reb_violate"])
        .sort_values(["reduction", "abs_delta"], ascending=False)
    )
    candidates = station_stats.head(10).sort_values(["abs_delta", "reduction"], ascending=False)
    selected_key = candidates.index[0]
    selected_node = int(selected_key[0])
    station = df[df["node_idx"] == selected_node].sort_values("decision_ts").copy()
    peak_idx = station["matched_transfer_delta"].abs().idxmax()
    peak_pos = station.index.get_loc(peak_idx)
    left = max(0, peak_pos - 18)
    right = min(len(station), peak_pos + 19)
    window = station.iloc[left:right].copy()
    return window, candidates.loc[selected_key]


def fig_inventory_case_study():
    window, summary = load_inventory_case_study()
    station_name = window["station_name"].dropna().iloc[0]
    node_idx = int(window["node_idx"].iloc[0])
    start_ts = window["decision_ts"].iloc[0]
    end_ts = window["decision_ts"].iloc[-1]
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(13.8, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.2]},
    )
    band_color = "#86efac"
    sns.lineplot(
        data=window,
        x="decision_ts",
        y="baseline_inventory_end_next_hour",
        ax=ax1,
        color="#9ca3af",
        linewidth=2.0,
        linestyle="--",
        label="Baseline end inventory",
    )
    sns.lineplot(
        data=window,
        x="decision_ts",
        y="inventory_after_rebalance",
        ax=ax1,
        color="#2563eb",
        linewidth=2.2,
        marker="o",
        markersize=4.2,
        label="After rebalance",
    )
    sns.lineplot(
        data=window,
        x="decision_ts",
        y="inventory_end_next_hour",
        ax=ax1,
        color="#111827",
        linewidth=2.0,
        marker="s",
        markersize=3.8,
        label="Realized end inventory",
    )
    lower = float(window["lower_target_inventory"].iloc[0])
    upper = float(window["upper_target_inventory"].iloc[0])
    ax1.fill_between(window["decision_ts"], lower, upper, color=band_color, alpha=0.18, label="Safety band")
    ax1.axhline(lower, color="#16a34a", linestyle="--", linewidth=1.0)
    ax1.axhline(upper, color="#16a34a", linestyle="--", linewidth=1.0)
    ax1.set_ylabel("库存（辆）")
    ax1.set_title(
        f"图7-6 单站库存回放与调度修正案例\n{station_name}（node {node_idx}）",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )
    ax1.grid(True, linestyle="--", alpha=0.22)
    ax1.legend(frameon=False, ncol=2, loc="upper right")
    transfer_colors = np.where(window["matched_transfer_delta"] >= 0, "#3b82f6", "#f97316")
    ax2.bar(window["decision_ts"], window["matched_transfer_delta"], color=transfer_colors, width=0.028)
    ax2.axhline(0, color="#64748b", linewidth=1.0)
    ax2.set_ylabel("调度\n增减")
    ax2.set_xlabel("决策时刻")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.22)
    ax2.text(
        0.01,
        0.9,
        f"窗口 {start_ts:%m-%d %H:%M} 至 {end_ts:%m-%d %H:%M}   "
        f"基线违规 {int(summary['base_violate'])} 次，调度后违规 {int(summary['reb_violate'])} 次",
        transform=ax2.transAxes,
        fontsize=9.2,
        color="#475569",
        va="top",
    )
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig7_6_inventory_case_study.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    # v7 keeps the GPT Image 2 architecture figures copied from v6 and only
    # regenerates result/mechanism charts with seaborn.
    fig_model_mae()
    fig_horizon_mae_curve()
    fig_quantile_interval()
    fig_attention_heatmap()
    fig_rebalancing_boundary()
    fig_inventory_case_study()


if __name__ == "__main__":
    main()
