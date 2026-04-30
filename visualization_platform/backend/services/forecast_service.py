"""Graph WaveNet forecasting service for visualization backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from forecasting_models.agcrn_nyc_gwnet_time_netloss_v1.data import build_timestamp_features
from forecasting_models.agcrn_nyc_gwnet_time_netloss_v1.model import (
    GraphWaveNet as TimeAwareGraphWaveNet,
    GraphWaveNetConfig as TimeAwareGraphWaveNetConfig,
)
from forecasting_models.agcrn_nyc_gwnet_time_netloss_v1.model import load_relation_graphs
from forecasting_models.agcrn_nyc_gwnet_v1.model import (
    GraphWaveNet as LegacyGraphWaveNet,
    GraphWaveNetConfig as LegacyGraphWaveNetConfig,
)

from .config import HORIZON, MODEL_SPECS, RELATION_GRAPHS_PATH
from .data_repository import DataRepository, DecisionContext


@dataclass(frozen=True)
class LoadedForecastModel:
    """Loaded model and scaler state for one forecast version."""

    model: torch.nn.Module
    kind: str
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: torch.Tensor
    target_std: torch.Tensor
    target_mode: str
    forecast_file: str | None = None


class ForecastService:
    """Load Graph WaveNet variants and predict future station dep/arr demand."""

    def __init__(self, repository: DataRepository, *, device: str = "auto") -> None:
        self.repository = repository
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self._models: dict[str, LoadedForecastModel] = {}
        self.future_time_features, self.future_time_feature_names = build_timestamp_features(
            [str(item) for item in repository.bundle_timestamps.tolist()]
        )

    def _load_model(self, model_id: str) -> LoadedForecastModel:
        """Load a forecast model lazily by id."""
        if model_id in self._models:
            return self._models[model_id]
        if model_id not in MODEL_SPECS:
            raise ValueError(f"Unsupported model: {model_id}")

        spec = MODEL_SPECS[model_id]
        checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
        checkpoint_station_ids = [str(item) for item in checkpoint["station_ids"]]
        if checkpoint_station_ids != self.repository.station_ids:
            raise ValueError(f"Checkpoint station_ids do not match dataset station order: {model_id}")

        if spec["kind"] == "tft_quantile_forecast_file":
            loaded = LoadedForecastModel(
                model=torch.nn.Identity(),
                kind=str(spec["kind"]),
                feature_mean=np.empty((1, 1, 0), dtype=np.float32),
                feature_std=np.empty((1, 1, 0), dtype=np.float32),
                target_mean=torch.empty((1, 1, 1, 0), dtype=torch.float32, device=self.device),
                target_std=torch.empty((1, 1, 1, 0), dtype=torch.float32, device=self.device),
                target_mode=str(checkpoint.get("target_mode", "log1p")),
                forecast_file=str(spec["forecast_file"]),
            )
            self._models[model_id] = loaded
            return loaded

        relation_supports, _metadata = load_relation_graphs(RELATION_GRAPHS_PATH, self.repository.station_ids)
        if spec["kind"] == "gwnet_time_netloss":
            config = TimeAwareGraphWaveNetConfig(**checkpoint["model_config"])
            model = TimeAwareGraphWaveNet(config, relation_supports)
        elif spec["kind"] == "gwnet":
            config = LegacyGraphWaveNetConfig(**checkpoint["model_config"])
            model = LegacyGraphWaveNet(config, relation_supports)
        else:
            raise ValueError(f"Unsupported model kind: {spec['kind']}")

        model.load_state_dict(checkpoint["model_state"])
        model.to(self.device)
        model.eval()

        loaded = LoadedForecastModel(
            model=model,
            kind=str(spec["kind"]),
            feature_mean=np.asarray(checkpoint["feature_scaler"]["mean"], dtype=np.float32).reshape(1, 1, -1),
            feature_std=np.asarray(checkpoint["feature_scaler"]["std"], dtype=np.float32).reshape(1, 1, -1),
            target_mean=torch.as_tensor(checkpoint["target_scaler"]["mean"], dtype=torch.float32, device=self.device).reshape(
                1, 1, 1, -1
            ),
            target_std=torch.as_tensor(checkpoint["target_scaler"]["std"], dtype=torch.float32, device=self.device).reshape(
                1, 1, 1, -1
            ),
            target_mode=str(checkpoint.get("target_mode", "log1p")),
        )
        self._models[model_id] = loaded
        return loaded

    def _future_time_window(self, context: DecisionContext) -> torch.Tensor:
        """Return future target-time features shaped [1, horizon, feature_dim]."""
        start = context.index + 1
        end = start + HORIZON
        values = self.future_time_features[start:end]
        if len(values) != HORIZON:
            raise ValueError(f"Timestamp is outside the valid forecast range: {context.ts}")
        return torch.from_numpy(values).unsqueeze(0).to(self.device)

    def _inverse_target(self, pred: torch.Tensor, loaded: LoadedForecastModel) -> torch.Tensor:
        """Convert model-space predictions to original count space."""
        unscaled = pred * loaded.target_std + loaded.target_mean
        if loaded.target_mode == "log1p":
            return torch.expm1(unscaled).clamp_min(0)
        if loaded.target_mode == "raw":
            return unscaled.clamp_min(0)
        raise ValueError(f"Unsupported backend target mode: {loaded.target_mode}")

    def _predict_from_quantile_file(self, context: DecisionContext, loaded: LoadedForecastModel) -> dict[str, np.ndarray]:
        """Load q10/q50/q90 forecasts for one test decision timestamp."""
        if loaded.forecast_file is None:
            raise ValueError("Quantile forecast file is not configured")
        columns = [
            "decision_ts",
            "target_ts",
            "node_idx",
            "dep_q10",
            "dep_q50",
            "dep_q90",
            "arr_q10",
            "arr_q50",
            "arr_q90",
            "net_flow_q10",
            "net_flow_q50",
            "net_flow_q90",
            "net_flow_pred",
        ]
        frame = pd.read_parquet(
            loaded.forecast_file,
            columns=columns,
            filters=[("decision_ts", "==", pd.Timestamp(context.ts))],
        )
        if frame.empty:
            raise ValueError(
                f"TFT quantile forecasts are available for the official test split only; no cached forecast for {context.ts}"
            )
        frame = frame.sort_values(["target_ts", "node_idx"], kind="stable")
        expected_rows = HORIZON * self.repository.node_count
        if len(frame) != expected_rows:
            raise ValueError(f"Unexpected TFT forecast row count for {context.ts}: {len(frame)} != {expected_rows}")
        node_idx = frame["node_idx"].to_numpy(dtype=np.int32, copy=False).reshape(HORIZON, self.repository.node_count)
        expected_node_idx = np.arange(self.repository.node_count, dtype=np.int32)[None, :]
        if not bool(np.all(node_idx == expected_node_idx)):
            raise ValueError(f"TFT forecast node order is invalid for {context.ts}")

        def column(name: str) -> np.ndarray:
            return frame[name].to_numpy(dtype=np.float32, copy=False).reshape(HORIZON, self.repository.node_count)

        dep = column("dep_q50")
        arr = column("arr_q50")
        return {
            "dep": dep,
            "arr": arr,
            "net_flow": column("net_flow_q50"),
            "dep_q10": column("dep_q10"),
            "dep_q50": dep,
            "dep_q90": column("dep_q90"),
            "arr_q10": column("arr_q10"),
            "arr_q50": arr,
            "arr_q90": column("arr_q90"),
            "net_flow_q10": column("net_flow_q10"),
            "net_flow_q50": column("net_flow_q50"),
            "net_flow_q90": column("net_flow_q90"),
        }

    @torch.no_grad()
    def predict(self, context: DecisionContext, *, model_id: str) -> dict[str, np.ndarray]:
        """Predict future dep, arr, and net flow for one decision."""
        loaded = self._load_model(model_id)
        if loaded.kind == "tft_quantile_forecast_file":
            return self._predict_from_quantile_file(context, loaded)

        raw_features = self.repository.feature_window(context)
        x = ((raw_features - loaded.feature_mean) / loaded.feature_std).astype(np.float32)
        source = torch.from_numpy(x).unsqueeze(0).to(self.device)
        if loaded.kind == "gwnet_time_netloss":
            pred = loaded.model(source, self._future_time_window(context))
        else:
            pred = loaded.model(source)
        counts = self._inverse_target(pred, loaded).squeeze(0).detach().cpu().numpy().astype(np.float32)
        dep = counts[..., 0]
        arr = counts[..., 1]
        return {"dep": dep, "arr": arr, "net_flow": arr - dep}
