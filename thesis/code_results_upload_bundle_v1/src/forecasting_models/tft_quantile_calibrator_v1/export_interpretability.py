"""Export post-hoc interpretability artifacts for the TFT-style quantile model."""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from forecasting_models.agcrn_nyc.train import resolve_device
from forecasting_models.agcrn_nyc_gwnet_time_netloss_v1.data import make_time_aware_dataloaders
from forecasting_models.tft_quantile_calibrator_v1.export_forecasts import load_model


DEFAULT_RUN_DIR = Path("forecasting_models/tft_quantile_calibrator_v1/runs/tft_quantile_top883_poi_v1_b16_e8")
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "best_model.pt"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "interpretability_v1"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Export TFT-style attention and saliency artifacts.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT.as_posix())
    parser.add_argument("--bundle", default="dataset/preprocessing/processed/nyc_top883_poi_v1/nyc_agcrn_bundle.npz")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lag", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--attention-batches", type=int, default=16)
    parser.add_argument("--saliency-batches", type=int, default=8)
    return parser.parse_args()


def feature_group(name: str) -> str:
    """Map a feature name to a thesis-friendly group."""
    if name.startswith("poi_"):
        return "poi"
    if name.startswith("wx_"):
        return "weather"
    if "lag_" in name or "rolling_" in name:
        return "history"
    if name in {"hour", "day_of_week", "day_of_month", "month", "is_weekend"} or name.endswith("_sin") or name.endswith("_cos"):
        return "time"
    if "holiday" in name:
        return "holiday"
    if "inventory" in name or name in {"dep_count", "arr_count", "net_flow"}:
        return "flow_inventory"
    if name in {"station_lat", "station_lng", "capacity_hat", "initial_inventory_hat"}:
        return "static_station"
    if "classic" in name or "electric" in name or "docked" in name or "member" in name or "casual" in name:
        return "ride_type"
    return "other"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    """Write rows to a UTF-8 CSV file."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def color_scale(value: float, min_value: float, max_value: float) -> str:
    """Return a muted blue heatmap color."""
    if max_value <= min_value:
        t = 0.0
    else:
        t = (value - min_value) / (max_value - min_value)
    t = float(np.clip(t, 0.0, 1.0))
    r = round(238 - 205 * t)
    g = round(246 - 104 * t)
    b = round(255 - 45 * t)
    return f"rgb({r},{g},{b})"


def render_attention_html(path: Path, horizon_lag: np.ndarray, lag_summary: np.ndarray) -> None:
    """Render a standalone SVG heatmap and lag bar chart."""
    lag = horizon_lag.shape[1]
    horizon = horizon_lag.shape[0]
    cell_w = 42
    cell_h = 28
    left = 94
    top = 46
    width = left + lag * cell_w + 28
    height = top + horizon * cell_h + 110
    min_value = float(horizon_lag.min())
    max_value = float(horizon_lag.max())
    rects = []
    labels = []
    for h in range(horizon):
        labels.append(f'<text x="62" y="{top + h * cell_h + 19}" class="axis">h+{h + 1}</text>')
        for lag_index in range(lag):
            x = left + lag_index * cell_w
            y = top + h * cell_h
            value = float(horizon_lag[h, lag_index])
            rects.append(
                f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" '
                f'fill="{color_scale(value, min_value, max_value)}"><title>h+{h + 1}, '
                f't-{lag - lag_index - 1}: {value:.4f}</title></rect>'
            )
    for lag_index in range(lag):
        x = left + lag_index * cell_w + cell_w / 2
        rel = lag_index - lag + 1
        labels.append(f'<text x="{x}" y="{top - 12}" class="axis" text-anchor="middle">t{rel:+d}</text>')

    bar_top = top + horizon * cell_h + 52
    bar_max = float(lag_summary.max()) if len(lag_summary) else 1.0
    bars = []
    for lag_index, value in enumerate(lag_summary):
        x = left + lag_index * cell_w
        bar_h = 48.0 * float(value) / max(bar_max, 1e-12)
        bars.append(
            f'<rect x="{x + 6}" y="{bar_top + 48 - bar_h}" width="{cell_w - 14}" height="{bar_h}" '
            f'fill="#2563eb"><title>t{lag_index - lag + 1:+d}: {float(value):.4f}</title></rect>'
        )
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>TFT Attention Heatmap</title>
  <style>
    body {{ margin: 0; padding: 24px; font-family: Inter, Arial, "Microsoft YaHei", sans-serif; color: #172033; background: #f7fafc; }}
    .wrap {{ max-width: {width}px; margin: 0 auto; background: white; border: 1px solid #d8e0e7; border-radius: 8px; padding: 18px; }}
    h1 {{ font-size: 20px; margin: 0 0 6px; }}
    p {{ color: #526173; margin: 0 0 14px; }}
    svg {{ width: 100%; height: auto; }}
    .axis {{ fill: #526173; font-size: 12px; font-weight: 700; }}
    .caption {{ fill: #203047; font-size: 13px; font-weight: 800; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>TFT-style 时间注意力热力图</h1>
    <p>颜色越深表示模型在对应预测步长上越关注该历史输入小时；下方柱形为跨预测步长平均注意力。</p>
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="TFT attention heatmap">
      <text x="{left}" y="22" class="caption">历史输入相对决策时刻</text>
      <text x="14" y="{top + horizon * cell_h / 2}" class="caption" transform="rotate(-90 14 {top + horizon * cell_h / 2})">预测步长</text>
      {''.join(labels)}
      {''.join(rects)}
      <text x="{left}" y="{bar_top - 14}" class="caption">平均注意力</text>
      {''.join(bars)}
    </svg>
  </div>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def render_saliency_html(path: Path, rows: list[dict[str, object]], *, title: str) -> None:
    """Render a simple horizontal bar chart for saliency rows."""
    top_rows = rows[:20]
    max_value = max(float(row["importance"]) for row in top_rows) if top_rows else 1.0
    items = []
    for row in top_rows:
        name = html.escape(str(row["feature"]))
        group = html.escape(str(row["group"]))
        value = float(row["importance"])
        pct = 100.0 * value / max(max_value, 1e-12)
        items.append(
            f'<div class="row"><div class="name">{name}<span>{group}</span></div>'
            f'<div class="bar"><i style="width:{pct:.2f}%"></i></div><div class="value">{value:.6f}</div></div>'
        )
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 0; padding: 24px; font-family: Inter, Arial, "Microsoft YaHei", sans-serif; color: #172033; background: #f7fafc; }}
    .wrap {{ max-width: 980px; margin: 0 auto; background: white; border: 1px solid #d8e0e7; border-radius: 8px; padding: 18px; }}
    h1 {{ font-size: 20px; margin: 0 0 6px; }}
    p {{ color: #526173; margin: 0 0 16px; }}
    .row {{ display: grid; grid-template-columns: minmax(220px, 1fr) 2fr 92px; gap: 10px; align-items: center; margin: 8px 0; }}
    .name {{ font-size: 13px; font-weight: 800; overflow-wrap: anywhere; }}
    .name span {{ display: block; color: #64748b; font-size: 11px; font-weight: 700; }}
    .bar {{ height: 12px; background: #e6ebf0; border-radius: 999px; overflow: hidden; }}
    .bar i {{ display: block; height: 100%; background: #0f8f68; }}
    .value {{ color: #526173; font-size: 12px; font-variant-numeric: tabular-nums; text-align: right; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{html.escape(title)}</h1>
    <p>基于测试窗口 q50 输出的输入梯度 saliency，数值为归一化输入空间中的相对敏感度。</p>
    {''.join(items)}
  </div>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


@torch.no_grad()
def collect_attention(model, loader, *, device: torch.device, limit_batches: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Collect attention summaries."""
    model.eval()
    lag_sum = None
    horizon_lag_sum = None
    head_lag_sum = None
    sample_count = 0
    batch_count = 0
    for x, future_time, _y, _raw_y, _baseline in loader:
        if batch_count >= limit_batches:
            break
        x = x.to(device, non_blocking=True)
        future_time = future_time.to(device, non_blocking=True)
        _pred, weights = model(x, future_time, return_attention=True)
        weights_cpu = weights.detach().cpu()
        if lag_sum is None:
            lag = weights_cpu.shape[-1]
            horizon = weights_cpu.shape[-2]
            heads = weights_cpu.shape[-3]
            lag_sum = torch.zeros(lag, dtype=torch.float64)
            horizon_lag_sum = torch.zeros(horizon, lag, dtype=torch.float64)
            head_lag_sum = torch.zeros(heads, lag, dtype=torch.float64)
        lag_sum += weights_cpu.sum(dim=(0, 1, 2, 3)).double()
        horizon_lag_sum += weights_cpu.sum(dim=(0, 1, 2)).double()
        head_lag_sum += weights_cpu.sum(dim=(0, 1, 3)).double()
        sample_count += int(weights_cpu.shape[0] * weights_cpu.shape[1])
        batch_count += 1
    if lag_sum is None or horizon_lag_sum is None or head_lag_sum is None or sample_count == 0:
        raise RuntimeError("No attention batches were processed")
    heads = int(head_lag_sum.shape[0])
    horizon = int(horizon_lag_sum.shape[0])
    lag = int(lag_sum.shape[0])
    return (
        (lag_sum / (sample_count * heads * horizon)).numpy(),
        (horizon_lag_sum / (sample_count * heads)).numpy(),
        (head_lag_sum / (sample_count * horizon)).numpy(),
        batch_count,
    )


def collect_saliency(model, loader, *, device: torch.device, limit_batches: int) -> tuple[np.ndarray, int]:
    """Collect gradient times input saliency by feature."""
    model.eval()
    saliency_sum = None
    count = 0
    batch_count = 0
    for x, future_time, _y, _raw_y, _baseline in loader:
        if batch_count >= limit_batches:
            break
        x = x.to(device, non_blocking=True).detach().requires_grad_(True)
        future_time = future_time.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        with torch.backends.cudnn.flags(enabled=False):
            pred = model(x, future_time)
            if isinstance(pred, tuple):
                pred = pred[0]
            q50 = pred[..., 1]
            score = q50.mean()
            score.backward()
        batch_saliency = (x.grad.detach().abs() * x.detach().abs()).sum(dim=(0, 1, 2)).cpu().double()
        if saliency_sum is None:
            saliency_sum = torch.zeros_like(batch_saliency)
        saliency_sum += batch_saliency
        count += int(x.shape[0] * x.shape[1] * x.shape[2])
        batch_count += 1
    if saliency_sum is None or count == 0:
        raise RuntimeError("No saliency batches were processed")
    return (saliency_sum / count).numpy(), batch_count


def main() -> int:
    """CLI entrypoint."""
    try:
        args = parse_args()
        device = resolve_device(args.device)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        data = make_time_aware_dataloaders(
            args.bundle,
            target_mode="log1p",
            lag=args.lag,
            horizon=args.horizon,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        model = load_model(args.checkpoint, device)

        lag_attention, horizon_lag_attention, head_lag_attention, attention_batches = collect_attention(
            model,
            data.test_loader,
            device=device,
            limit_batches=args.attention_batches,
        )
        saliency, saliency_batches = collect_saliency(
            model,
            data.test_loader,
            device=device,
            limit_batches=args.saliency_batches,
        )

        lag = len(lag_attention)
        lag_rows = [
            {
                "input_step": index + 1,
                "relative_hour_to_decision": index - lag + 1,
                "attention_weight": float(value),
            }
            for index, value in enumerate(lag_attention)
        ]
        write_csv(output_dir / "attention_lag_summary.csv", list(lag_rows[0].keys()), lag_rows)

        horizon_rows: list[dict[str, object]] = []
        for horizon_index in range(horizon_lag_attention.shape[0]):
            row: dict[str, object] = {"horizon": horizon_index + 1}
            for lag_index in range(horizon_lag_attention.shape[1]):
                row[f"t{lag_index - lag + 1:+d}"] = float(horizon_lag_attention[horizon_index, lag_index])
            horizon_rows.append(row)
        write_csv(output_dir / "attention_horizon_lag_matrix.csv", list(horizon_rows[0].keys()), horizon_rows)

        head_rows = []
        for head_index in range(head_lag_attention.shape[0]):
            for lag_index in range(head_lag_attention.shape[1]):
                head_rows.append(
                    {
                        "head": head_index + 1,
                        "input_step": lag_index + 1,
                        "relative_hour_to_decision": lag_index - lag + 1,
                        "attention_weight": float(head_lag_attention[head_index, lag_index]),
                    }
                )
        write_csv(output_dir / "attention_head_lag_summary.csv", list(head_rows[0].keys()), head_rows)

        feature_rows = []
        for index, value in enumerate(saliency):
            name = data.feature_names[index]
            feature_rows.append(
                {
                    "feature": name,
                    "group": feature_group(name),
                    "importance": float(value),
                }
            )
        feature_rows.sort(key=lambda row: float(row["importance"]), reverse=True)
        write_csv(output_dir / "feature_saliency.csv", ["feature", "group", "importance"], feature_rows)

        group_values: dict[str, float] = defaultdict(float)
        for row in feature_rows:
            group_values[str(row["group"])] += float(row["importance"])
        group_rows = [
            {"group": group, "importance": value}
            for group, value in sorted(group_values.items(), key=lambda item: item[1], reverse=True)
        ]
        write_csv(output_dir / "feature_group_saliency.csv", ["group", "importance"], group_rows)

        render_attention_html(output_dir / "attention_heatmap.html", horizon_lag_attention, lag_attention)
        render_saliency_html(output_dir / "feature_saliency.html", feature_rows, title="TFT-style 特征敏感度 Top 20")

        top_lag_index = int(np.argmax(lag_attention))
        summary = {
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "checkpoint": args.checkpoint,
            "bundle": args.bundle,
            "output_dir": str(output_dir),
            "attention_batches": attention_batches,
            "saliency_batches": saliency_batches,
            "sampled_test_windows_for_attention": attention_batches * args.batch_size,
            "sampled_test_windows_for_saliency": saliency_batches * args.batch_size,
            "top_attention_relative_hour": int(top_lag_index - lag + 1),
            "top_attention_weight": float(lag_attention[top_lag_index]),
            "top_feature_saliency": feature_rows[:12],
            "feature_group_saliency": group_rows,
            "artifacts": {
                "attention_lag_summary": "attention_lag_summary.csv",
                "attention_horizon_lag_matrix": "attention_horizon_lag_matrix.csv",
                "attention_head_lag_summary": "attention_head_lag_summary.csv",
                "feature_saliency": "feature_saliency.csv",
                "feature_group_saliency": "feature_group_saliency.csv",
                "attention_heatmap": "attention_heatmap.html",
                "feature_saliency_html": "feature_saliency.html",
            },
            "notes": [
                "Attention weights are averaged over sampled test windows, all stations, all heads, and forecast horizons.",
                "Feature saliency is post-hoc gradient times input sensitivity in normalized input space, not official PyTorch Forecasting variable selection.",
            ],
        }
        (output_dir / "interpretability_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
