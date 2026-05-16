from __future__ import annotations

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


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "figures"
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")


def configure_plotting() -> None:
    if FONT_PATH.exists():
        font_manager.fontManager.addfont(str(FONT_PATH))
        font_name = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
        plt.rcParams["font.family"] = font_name
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


def finish_diagram(fig, ax, title: str, out_name: str):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.965, title, fontsize=17, fontweight="bold", color="#111827", va="top")
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
    add_box(ax, (0.04, 0.58), (0.18, 0.15), "多源输入特征\n过去 12 小时\n883 站点", "#e0f2fe")
    add_box(ax, (0.04, 0.32), (0.18, 0.15), "未来已知特征\n目标小时/星期\n月份/周末", "#dcfce7")
    add_box(ax, (0.31, 0.60), (0.22, 0.14), "Graph WaveNet 分支\n扩张时序卷积\nadaptive + OD 图卷积", "#dbeafe")
    add_box(ax, (0.31, 0.30), (0.22, 0.16), "TFT-style 分支\nLSTM encoder\n时间注意力\n分位数输出", "#f5e8ff")
    add_box(ax, (0.62, 0.61), (0.16, 0.12), "dep/arr 点预测\nnet-flow", "#fee2e2")
    add_box(ax, (0.62, 0.31), (0.16, 0.13), "q10/q50/q90\n预测区间\nPinball Loss", "#ffedd5")
    add_box(ax, (0.84, 0.47), (0.12, 0.16), "调度输入\n未来净流量\n风险模式", "#dcfce7")
    for s, e in [
        ((0.22, 0.66), (0.31, 0.67)),
        ((0.22, 0.40), (0.31, 0.38)),
        ((0.22, 0.66), (0.31, 0.38)),
        ((0.22, 0.40), (0.31, 0.67)),
        ((0.53, 0.67), (0.62, 0.67)),
        ((0.53, 0.38), (0.62, 0.38)),
        ((0.78, 0.67), (0.84, 0.56)),
        ((0.78, 0.38), (0.84, 0.54)),
    ]:
        add_arrow(ax, s, e)
    ax.text(0.31, 0.20, "统一任务：past 12h -> future 12h，每站预测 dep_count 与 arr_count", fontsize=10.5, color="#374151")
    finish_diagram(fig, ax, "图4-1 预测模型总体结构", "fig4_1_forecast_architecture.png")


def fig_gwnet_time_netloss():
    fig, ax = plt.subplots(figsize=(13.5, 7.3))
    steps = [
        ((0.04, 0.58), (0.15, 0.13), "输入张量\n[B,12,N,F]\nlog1p 目标", "#e0f2fe"),
        ((0.25, 0.58), (0.17, 0.13), "Gated Dilated\nTemporal Conv\n1,2,4 × 2 blocks", "#dbeafe"),
        ((0.48, 0.58), (0.17, 0.13), "Graph Conv\nadaptive support\n+ weak OD top-k20", "#dcfce7"),
        ((0.70, 0.58), (0.16, 0.13), "Skip / End\nConv Readout\n共享时序表示", "#f5e8ff"),
        ((0.70, 0.31), (0.16, 0.13), "未来目标时间\nhour/dow/month\nweekend", "#ffedd5"),
        ((0.48, 0.31), (0.17, 0.13), "Time-conditioned\nHorizon Readout\n每步独立时间身份", "#fee2e2"),
        ((0.25, 0.31), (0.17, 0.13), "dep / arr 输出\n未来 12 小时\n反变换评价", "#dcfce7"),
        ((0.04, 0.31), (0.15, 0.13), "辅助损失\nnet-flow MAE\nλ = 0.10", "#e0f2fe"),
    ]
    for xy, wh, text, color in steps:
        add_box(ax, xy, wh, text, face=color, fontsize=9.7)
    for s, e in [
        ((0.19, 0.645), (0.25, 0.645)),
        ((0.42, 0.645), (0.48, 0.645)),
        ((0.65, 0.645), (0.70, 0.645)),
        ((0.78, 0.58), (0.78, 0.44)),
        ((0.70, 0.375), (0.65, 0.375)),
        ((0.48, 0.375), (0.42, 0.375)),
        ((0.25, 0.375), (0.19, 0.375)),
    ]:
        add_arrow(ax, s, e)
    ax.text(0.04, 0.20, "核心收益：Avg MAE 1.6238，net-flow MAE 1.5658；目标时间特征缓解 horizon 相位偏移。", fontsize=10.5, color="#374151")
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
    add_box(ax, (0.04, 0.58), (0.21, 0.17), "React + TypeScript 前端\nMapLibre 地图\nECharts 曲线\n决策时间/模型/算法选择", "#e0f2fe", fontsize=9.3)
    add_box(ax, (0.38, 0.61), (0.20, 0.12), "FastAPI Backend\n/api/meta\n/api/decision\n/api/station", "#dcfce7")
    add_box(ax, (0.72, 0.63), (0.19, 0.10), "DataRepository\n站点面板 / 静态属性\n测试窗口定位", "#ffedd5", fontsize=9.3)
    add_box(ax, (0.72, 0.48), (0.19, 0.10), "ForecastService\nTFT q10/q50/q90\nGraph WaveNet / oracle", "#dbeafe", fontsize=9.3)
    add_box(ax, (0.72, 0.33), (0.19, 0.10), "RebalancingService\ngreedy / min-cost\n库存回放", "#f5e8ff", fontsize=9.3)
    add_box(ax, (0.38, 0.30), (0.20, 0.12), "CacheService\n按决策时刻缓存\n避免重复计算", "#fee2e2")
    add_box(ax, (0.04, 0.25), (0.21, 0.13), "离线历史回测原型\n不接实时库存\n不接在线天气", "#f3f4f6", edge="#6b7280")
    for s, e in [
        ((0.25, 0.665), (0.38, 0.67)),
        ((0.58, 0.67), (0.72, 0.68)),
        ((0.58, 0.64), (0.72, 0.53)),
        ((0.58, 0.62), (0.72, 0.38)),
        ((0.48, 0.61), (0.48, 0.42)),
        ((0.38, 0.36), (0.25, 0.32)),
    ]:
        add_arrow(ax, s, e)
    finish_diagram(fig, ax, "图6-1 可视化平台系统架构", "fig6_1_platform_architecture.png")


def fig_dashboard_mock():
    fig, ax = plt.subplots(figsize=(13.5, 7.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.965, "图6-2 平台地图与调度路径展示（示意）", fontsize=17, fontweight="bold", va="top")
    add_box(ax, (0.04, 0.78), (0.92, 0.10), "决策时刻 2022-10-19 21:00 | 模型 TFT-style q50 | 算法 min-cost | cap 200", "#f3f4f6", edge="#6b7280")
    # Map panel
    add_box(ax, (0.04, 0.18), (0.56, 0.55), "", "#eef2ff", edge="#334155")
    ax.text(0.07, 0.69, "站点地图与调度路线", fontsize=12, fontweight="bold", color="#111827")
    rng = np.random.default_rng(8)
    pts = rng.uniform([0.09, 0.25], [0.55, 0.64], size=(55, 2))
    roles = rng.choice(["balanced", "donor", "receiver"], size=55, p=[0.62, 0.18, 0.20])
    color_map = {"balanced": "#64748b", "donor": "#2563eb", "receiver": "#dc2626"}
    for (x, y), r in zip(pts, roles):
        ax.scatter(x, y, s=28 if r == "balanced" else 46, c=color_map[r], alpha=0.88, edgecolors="white", linewidths=0.5)
    donors = pts[roles == "donor"][:6]
    recs = pts[roles == "receiver"][:6]
    for d, r in zip(donors, recs):
        add_arrow(ax, tuple(d), tuple(r), color="#f97316", lw=1.0, rad=0.08)
    ax.text(0.07, 0.21, "蓝色 donor  红色 receiver  橙色 transfer", fontsize=9, color="#374151")
    # Charts panel
    add_box(ax, (0.65, 0.47), (0.31, 0.26), "", "#ffffff", edge="#334155")
    ax.text(0.68, 0.69, "聚合净流量预测", fontsize=11, fontweight="bold")
    xs = np.linspace(0.69, 0.93, 12)
    actual = 0.58 + 0.07 * np.sin(np.linspace(0, 2.4 * np.pi, 12))
    pred = 0.57 + 0.06 * np.sin(np.linspace(0.2, 2.6 * np.pi, 12))
    ax.plot(xs, actual, color="#111827", lw=2.0, label="actual")
    ax.plot(xs, pred, color="#2563eb", lw=2.0, label="q50")
    ax.fill_between(xs, pred - 0.035, pred + 0.04, color="#93c5fd", alpha=0.35)
    ax.text(0.68, 0.50, "q10-q90 区间可在图例中开关", fontsize=8.5, color="#374151")
    add_box(ax, (0.65, 0.18), (0.31, 0.22), "", "#ffffff", edge="#334155")
    ax.text(0.68, 0.36, "当前决策指标", fontsize=11, fontweight="bold")
    metrics = [("搬运车辆", "32"), ("动作数", "9"), ("bike-km", "84.7"), ("越界改善", "显著")]
    for i, (k, v) in enumerate(metrics):
        ax.text(0.69, 0.315 - i * 0.045, k, fontsize=9, color="#374151")
        ax.text(0.87, 0.315 - i * 0.045, v, fontsize=9, color="#111827", fontweight="bold", ha="right")
    fig.savefig(OUT_DIR / "fig6_2_dashboard_map.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_model_mae():
    labels = [
        "AGCRN\nbaseline",
        "AGCRN\nlog1p+OD",
        "Graph\nWaveNet v1",
        "GWNet time\n+ net-loss",
        "GWNet\n+ POI",
        "TFT-style\nquantile",
    ]
    values = [2.2096, 2.1073, 1.7666, 1.6238, 1.7233, 1.5899]
    colors = ["#94a3b8", "#60a5fa", "#34d399", "#22c55e", "#f59e0b", "#ef4444"]
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, edgecolor="#1f2937", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("测试集 Avg MAE（原始订单计数尺度）")
    ax.set_title("图7-1 预测模型 Avg MAE 对比", fontsize=16, fontweight="bold", pad=16)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    for yi, v in enumerate(values):
        ax.text(v + 0.025, yi, f"{v:.4f}", va="center", fontsize=10)
    ax.set_xlim(1.45, 2.35)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig7_1_model_mae_comparison.png", bbox_inches="tight", facecolor="white")
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
    ax.fill_between(x, df["net_flow_q10"], df["net_flow_q90"], color="#93c5fd", alpha=0.35, label="q10-q90 区间")
    ax.plot(x, df["net_flow_q50"], color="#2563eb", lw=2.4, marker="o", label="TFT q50")
    ax.plot(x, df["actual_net_flow"], color="#111827", lw=2.2, marker="s", label="真实净流量")
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
    mat = df.drop(columns=["horizon"]).to_numpy()
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu")
    ax.set_title("图7-3 TFT-style 注意力滞后热力图", fontsize=16, fontweight="bold", pad=16)
    ax.set_xlabel("历史相对时间")
    ax.set_ylabel("预测步长 horizon")
    ax.set_xticks(np.arange(len(df.columns) - 1))
    ax.set_xticklabels(df.columns[1:], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["horizon"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("平均注意力权重")
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
    below = np.array([412308, 96889, 93842, 99860, 106077, 219667, 65415])
    above = np.array([631700, 13331, 16378, 16956, 18424, 75693, 349587])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    ax.bar(x, below, label="below lower", color="#60a5fa", edgecolor="#1f2937", linewidth=0.4)
    ax.bar(x, above, bottom=below, label="above upper", color="#f97316", edgecolor="#1f2937", linewidth=0.4)
    totals = below + above
    for xi, total in zip(x, totals):
        ax.text(xi, total + 18000, f"{total:,}", ha="center", va="bottom", fontsize=9)
    ax.set_title("图7-4 调度结果安全库存带违规小时对比", fontsize=16, fontweight="bold", pad=16)
    ax.set_ylabel("安全库存带违规站点-小时")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig7_4_rebalancing_boundary_comparison.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    fig_data_pipeline()
    fig_forecast_architecture()
    fig_gwnet_time_netloss()
    fig_tft_quantile()
    fig_rebalancing_pipeline()
    fig_platform_architecture()
    fig_dashboard_mock()
    fig_model_mae()
    fig_quantile_interval()
    fig_attention_heatmap()
    fig_rebalancing_boundary()


if __name__ == "__main__":
    main()
