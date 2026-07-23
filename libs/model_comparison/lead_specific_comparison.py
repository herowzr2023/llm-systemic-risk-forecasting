from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler


SUPPORTED_MODELS = ("Persistence", "AR(1)", "ETS", "GBR", "LSTM", "PatchTST")
OUTPUT_COLUMNS = {
    "Persistence": "Persistence_pred",
    "AR(1)": "AR1_pred",
    "ETS": "ETS_pred",
    "GBR": "GBR_pred",
    "LSTM": "LSTM_pred",
    "PatchTST": "PatchTST_pred",
}


@dataclass
class PreparedSamples:
    normalized: pd.DataFrame
    dates: pd.Series
    raw_target: np.ndarray
    target_scaler: Any
    sequences: np.ndarray
    targets: np.ndarray
    first_target_indices: np.ndarray
    train_count: int


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _read_forecasting_data(
    path: Path,
    date_col: str,
    target_col: str,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if date_col not in frame or target_col not in frame:
        raise ValueError(f"{path} must contain {date_col!r} and {target_col!r}")
    if feature_columns is not None:
        requested = list(dict.fromkeys([target_col, *feature_columns]))
        missing_requested = [column for column in requested if column not in frame.columns]
        if missing_requested:
            raise ValueError(f"Configured forecasting variables are absent from {path}: {missing_requested}")
        frame = frame[[date_col, *requested]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="raise")
    frame = frame.sort_values(date_col).reset_index(drop=True)
    if frame[date_col].duplicated().any():
        raise ValueError(f"Duplicate dates in {path}")

    numeric = [column for column in frame.columns if column != date_col]
    converted = frame[numeric].apply(pd.to_numeric, errors="coerce")
    nonnumeric = [column for column in numeric if converted[column].isna().all()]
    if nonnumeric:
        frame = frame.drop(columns=nonnumeric)
        numeric = [column for column in numeric if column not in nonnumeric]
        converted = converted[numeric]
    if converted.isna().any().any():
        missing = converted.columns[converted.isna().any()].tolist()
        raise ValueError(f"Missing or non-numeric values in forecasting variables: {missing}")
    frame[numeric] = converted
    return frame[[date_col, *numeric]]


def _column(frame: pd.DataFrame, candidates: Iterable[str], description: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"Could not find {description}; tried {list(candidates)}")


def _legacy_horizons(raw_dates: pd.Series, lead: int) -> np.ndarray:
    """Recover horizons from the legacy flattened-and-date-sorted Informer file."""
    dates = pd.to_datetime(raw_dates, errors="raise").to_numpy()
    if lead == 1:
        if pd.Series(dates).duplicated().any():
            raise ValueError("A one-step Informer archive must have one row per target date")
        return np.ones(len(dates), dtype=int)
    if len(dates) % lead:
        raise ValueError(f"Legacy Informer row count {len(dates)} is not divisible by lead {lead}")

    unique_dates = np.sort(np.unique(dates))
    window_count = len(dates) // lead
    if len(unique_dates) != window_count + lead - 1:
        raise ValueError(
            "Legacy Informer dates are inconsistent with consecutive rolling forecast windows: "
            f"{len(unique_dates)} unique dates for {window_count} windows and lead {lead}"
        )
    flattened_dates = np.concatenate(
        [unique_dates[start : start + lead] for start in range(window_count)]
    )
    flattened_horizons = np.tile(np.arange(1, lead + 1, dtype=int), window_count)
    legacy_sort_order = np.argsort(flattened_dates)
    reconstructed_dates = flattened_dates[legacy_sort_order]
    if not np.array_equal(reconstructed_dates, dates):
        raise ValueError(
            "The baseline file does not match the legacy Informer rolling-window export order; "
            "rerun Informer to obtain explicit horizon metadata."
        )
    return flattened_horizons[legacy_sort_order]


def load_informer_lead(
    baseline_csv: Path,
    forecasting_data: pd.DataFrame,
    lead: int,
    date_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Return exactly one Informer forecast for each origin at the requested lead."""
    raw = pd.read_csv(baseline_csv)
    baseline_date = _column(raw, (date_col, "target_date"), "Informer target-date column")
    actual_col = _column(raw, ("true", "TRUE", "actual", target_col), "Informer actual column")
    prediction_col = _column(raw, ("pred", "Informer_pred", "baseline_pred"), "Informer prediction column")
    raw[baseline_date] = pd.to_datetime(raw[baseline_date], errors="raise")
    raw[actual_col] = pd.to_numeric(raw[actual_col], errors="raise")
    raw[prediction_col] = pd.to_numeric(raw[prediction_col], errors="raise")

    horizon_col = next((column for column in ("lead", "horizon") if column in raw), None)
    if horizon_col:
        raw[horizon_col] = pd.to_numeric(raw[horizon_col], errors="raise").astype(int)
        selected = raw.loc[raw[horizon_col].eq(lead)].copy()
        if selected.empty:
            raise ValueError(f"No Informer forecasts found for lead {lead}")
    else:
        raw["_reconstructed_lead"] = _legacy_horizons(raw[baseline_date], lead)
        selected = raw.loc[raw["_reconstructed_lead"].eq(lead)].copy()

    data_dates = pd.to_datetime(forecasting_data[date_col]).reset_index(drop=True)
    date_to_index = pd.Series(np.arange(len(data_dates)), index=data_dates).to_dict()
    target_indices = selected[baseline_date].map(date_to_index)
    if target_indices.isna().any():
        examples = selected.loc[target_indices.isna(), baseline_date].head(3).dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Informer target dates absent from forecasting data: {examples}")
    target_indices = target_indices.astype(int)
    if (target_indices < lead).any():
        raise ValueError("At least one Informer forecast has no valid information-cutoff date")

    if "origin_date" in selected:
        supplied_origin = pd.to_datetime(selected["origin_date"], errors="raise").reset_index(drop=True)
        expected_origin = data_dates.iloc[(target_indices - lead).to_numpy()].reset_index(drop=True)
        if not supplied_origin.equals(expected_origin):
            raise ValueError("Informer origin_date is inconsistent with target date and requested lead")
        origin_dates = supplied_origin
    else:
        origin_dates = data_dates.iloc[(target_indices - lead).to_numpy()].reset_index(drop=True)

    data_actual = forecasting_data[target_col].to_numpy(dtype=float)[target_indices.to_numpy()]
    archive_actual = selected[actual_col].to_numpy(dtype=float)
    if not np.allclose(data_actual, archive_actual, rtol=1e-6, atol=1e-8):
        difference = float(np.max(np.abs(data_actual - archive_actual)))
        raise ValueError(f"Informer actual values do not align with forecasting data (max difference {difference})")

    result = pd.DataFrame(
        {
            "origin_date": origin_dates,
            date_col: selected[baseline_date].reset_index(drop=True),
            "lead": int(lead),
            "TRUE": archive_actual,
            "Informer_pred": selected[prediction_col].to_numpy(dtype=float),
        }
    ).sort_values("origin_date").reset_index(drop=True)
    if result.duplicated(["origin_date", date_col, "lead"]).any():
        raise ValueError("More than one Informer forecast remains for an origin-target-lead combination")
    if not result[date_col].is_monotonic_increasing:
        raise ValueError("Selected Informer target dates are not increasing")
    return result


def _prepare_samples(
    data: pd.DataFrame,
    lead: int,
    lags: int,
    train_ratio: float,
    normalization: str,
    date_col: str,
    target_col: str,
) -> PreparedSamples:
    if lead < 1 or lags < 1:
        raise ValueError("lead and lags must be positive")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must lie between zero and one")

    dates = pd.to_datetime(data[date_col]).reset_index(drop=True)
    value_columns = [column for column in data.columns if column != date_col]
    predictor_columns = [column for column in value_columns if column != target_col]
    fit_end = int(len(data) * train_ratio)
    if fit_end <= lags:
        raise ValueError("Training segment is too short for the requested lag length")

    scaler_type = MinMaxScaler if normalization == "minmax" else StandardScaler
    if normalization not in {"minmax", "standard"}:
        raise ValueError("normalization must be 'minmax' or 'standard'")
    target_scaler = scaler_type().fit(data[[target_col]].iloc[:fit_end])
    normalized = data.copy()
    normalized[target_col] = target_scaler.transform(data[[target_col]])
    if predictor_columns:
        predictor_scaler = scaler_type().fit(data[predictor_columns].iloc[:fit_end])
        normalized[predictor_columns] = predictor_scaler.transform(data[predictor_columns])

    matrix = normalized[value_columns].to_numpy(dtype=np.float32)
    target = normalized[target_col].to_numpy(dtype=np.float32)
    first_targets = np.arange(lags, len(data) - lead + 1, dtype=int)
    if len(first_targets) < 2:
        raise ValueError("Insufficient observations for the requested lead and lag length")
    sequences = np.stack([matrix[index - lags : index] for index in first_targets])
    targets = np.stack([target[index : index + lead] for index in first_targets])
    train_count = int(len(first_targets) * train_ratio)
    if train_count <= 0 or train_count >= len(first_targets):
        raise ValueError("The train/test split leaves an empty segment")
    return PreparedSamples(
        normalized=normalized,
        dates=dates,
        raw_target=data[target_col].to_numpy(dtype=float),
        target_scaler=target_scaler,
        sequences=sequences,
        targets=targets,
        first_target_indices=first_targets,
        train_count=train_count,
    )


def _prediction_frame(
    prepared: PreparedSamples,
    normalized_predictions: np.ndarray,
    lead: int,
    output_column: str,
    date_col: str,
) -> pd.DataFrame:
    predictions = np.asarray(normalized_predictions)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    expected_rows = len(prepared.first_target_indices) - prepared.train_count
    if predictions.shape != (expected_rows, lead):
        raise ValueError(f"{output_column}: expected shape {(expected_rows, lead)}, got {predictions.shape}")
    first_targets = prepared.first_target_indices[prepared.train_count :]
    target_indices = first_targets + lead - 1
    last_component = predictions[:, lead - 1].reshape(-1, 1)
    inverse = prepared.target_scaler.inverse_transform(last_component).reshape(-1)
    return pd.DataFrame(
        {
            "origin_date": prepared.dates.iloc[first_targets - 1].to_numpy(),
            date_col: prepared.dates.iloc[target_indices].to_numpy(),
            "lead": int(lead),
            output_column: inverse,
        }
    )


def _fit_ar1(values: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(values) - 1), values[:-1]])
    intercept, coefficient = np.linalg.lstsq(design, values[1:], rcond=None)[0]
    return float(intercept), float(coefficient)


def _ar_forecast(last_value: float, intercept: float, coefficient: float, lead: int) -> float:
    current = float(last_value)
    for _ in range(lead):
        current = intercept + coefficient * current
    return current


def _fit_ets(values: np.ndarray) -> tuple[float, float]:
    def objective(alpha: float) -> float:
        level = float(values[0])
        errors = []
        for observation in values[1:]:
            errors.append((float(observation) - level) ** 2)
            level = alpha * float(observation) + (1.0 - alpha) * level
        return float(np.sum(errors))

    fit = minimize_scalar(objective, bounds=(1e-6, 1.0 - 1e-6), method="bounded")
    if not fit.success:
        raise RuntimeError(f"ETS(A,N,N) optimization failed: {fit.message}")
    alpha = float(fit.x)
    level = float(values[0])
    for observation in values[1:]:
        level = alpha * float(observation) + (1.0 - alpha) * level
    return alpha, level


def _simple_predictions(
    prepared: PreparedSamples,
    models: set[str],
    lead: int,
    date_col: str,
) -> dict[str, pd.DataFrame]:
    first_targets = prepared.first_target_indices[prepared.train_count :]
    first_test_target = int(first_targets[0])
    training_values = prepared.raw_target[:first_test_target]
    outputs: dict[str, list[float]] = {model: [] for model in models & {"Persistence", "AR(1)", "ETS"}}
    if "AR(1)" in outputs:
        intercept, coefficient = _fit_ar1(training_values)
    if "ETS" in outputs:
        alpha, level = _fit_ets(training_values)

    for first_target in first_targets:
        last_observation = float(prepared.raw_target[first_target - 1])
        if "Persistence" in outputs:
            outputs["Persistence"].append(last_observation)
        if "AR(1)" in outputs:
            outputs["AR(1)"].append(_ar_forecast(last_observation, intercept, coefficient, lead))
        if "ETS" in outputs:
            outputs["ETS"].append(level)
            level = alpha * float(prepared.raw_target[first_target]) + (1.0 - alpha) * level

    target_indices = first_targets + lead - 1
    frames: dict[str, pd.DataFrame] = {}
    for model, predictions in outputs.items():
        frames[model] = pd.DataFrame(
            {
                "origin_date": prepared.dates.iloc[first_targets - 1].to_numpy(),
                date_col: prepared.dates.iloc[target_indices].to_numpy(),
                "lead": int(lead),
                OUTPUT_COLUMNS[model]: predictions,
            }
        )
    return frames


def _gbr_predictions(
    prepared: PreparedSamples,
    lead: int,
    date_col: str,
    settings: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    estimator = GradientBoostingRegressor(
        n_estimators=int(settings.get("n_estimators", 100)),
        learning_rate=float(settings.get("learning_rate", 0.1)),
        max_depth=int(settings.get("max_depth", 3)),
        random_state=seed,
    )
    model = estimator if lead == 1 else MultiOutputRegressor(estimator)
    x_train = prepared.sequences[: prepared.train_count].reshape(prepared.train_count, -1)
    x_test = prepared.sequences[prepared.train_count :].reshape(
        len(prepared.sequences) - prepared.train_count, -1
    )
    y_train = prepared.targets[: prepared.train_count]
    model.fit(x_train, y_train.reshape(-1) if lead == 1 else y_train)
    predicted = model.predict(x_test)
    return _prediction_frame(prepared, predicted, lead, "GBR_pred", date_col)


def _lstm_predictions(
    prepared: PreparedSamples,
    lead: int,
    date_col: str,
    settings: dict[str, Any],
) -> pd.DataFrame:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:
        raise RuntimeError("LSTM requires PyTorch; install requirements_full_training.txt") from error

    class LSTMRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=prepared.sequences.shape[2],
                hidden_size=int(settings.get("hidden_size", 50)),
                num_layers=int(settings.get("num_layers", 1)),
                batch_first=True,
            )
            self.output = nn.Linear(int(settings.get("hidden_size", 50)), lead)

        def forward(self, values):
            encoded, _ = self.lstm(values)
            return self.output(encoded[:, -1, :])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor().to(device)
    train_data = TensorDataset(
        torch.tensor(prepared.sequences[: prepared.train_count], dtype=torch.float32),
        torch.tensor(prepared.targets[: prepared.train_count], dtype=torch.float32),
    )
    loader = DataLoader(
        train_data,
        batch_size=int(settings.get("batch_size", 32)),
        shuffle=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings.get("learning_rate", 0.001)))
    criterion = nn.MSELoss()
    model.train()
    epochs = int(settings.get("epochs", 50))
    for epoch in range(epochs):
        losses = []
        for features, target in loader:
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if epoch == 0 or epoch + 1 == epochs or (epoch + 1) % 10 == 0:
            print(f"LSTM epoch {epoch + 1}/{epochs}: loss={np.mean(losses):.6f}")
    model.eval()
    with torch.no_grad():
        predicted = model(
            torch.tensor(prepared.sequences[prepared.train_count :], dtype=torch.float32, device=device)
        ).cpu().numpy()
    return _prediction_frame(prepared, predicted, lead, "LSTM_pred", date_col)


def _patchtst_predictions(
    prepared: PreparedSamples,
    lead: int,
    lags: int,
    date_col: str,
    target_col: str,
    settings: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:
        raise RuntimeError("PatchTST requires PyTorch; install requirements_full_training.txt") from error

    value_columns = [column for column in prepared.normalized.columns if column != date_col]
    predictor_columns = [column for column in value_columns if column != target_col]
    expected_covariates = int(settings.get("expected_historical_covariates", 16))
    if len(predictor_columns) != expected_covariates:
        raise ValueError(
            "PatchTST must receive the configured historical covariates together with SR: "
            f"expected {expected_covariates}, found {len(predictor_columns)}"
        )
    target_index = value_columns.index(target_col)
    channel_count = len(value_columns)
    if prepared.sequences.shape[2] != channel_count:
        raise ValueError(
            "PatchTST input-channel mismatch: "
            f"prepared {prepared.sequences.shape[2]} channels for {channel_count} variables"
        )

    patch_len = max(1, min(int(settings.get("patch_len", 3)), lags))
    stride = max(1, min(int(settings.get("stride", 1)), lags))
    patch_count = 1 + (lags - patch_len) // stride
    hidden_size = int(settings.get("hidden_size", 128))
    n_heads = int(settings.get("n_heads", 4))
    if hidden_size % n_heads:
        raise ValueError("PatchTST hidden_size must be divisible by n_heads")

    class CovariatePatchTST(nn.Module):
        """Channel-independent patch encoder with a joint SR forecasting head."""

        def __init__(self) -> None:
            super().__init__()
            self.target_index = target_index
            self.use_revin = bool(settings.get("revin", True))
            self.revin_affine = bool(settings.get("revin_affine", False))
            self.subtract_last = bool(settings.get("revin_subtract_last", True))
            if self.revin_affine:
                self.revin_weight = nn.Parameter(torch.ones(1, channel_count, 1))
                self.revin_bias = nn.Parameter(torch.zeros(1, channel_count, 1))

            self.patch_embedding = nn.Linear(patch_len, hidden_size)
            self.position_embedding = nn.Parameter(
                torch.zeros(1, 1, patch_count, hidden_size)
            )
            self.channel_embedding = nn.Parameter(
                torch.zeros(1, channel_count, 1, hidden_size)
            )
            self.input_dropout = nn.Dropout(float(settings.get("fc_dropout", 0.1)))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=int(settings.get("linear_hidden_size", 256)),
                dropout=max(
                    float(settings.get("dropout", 0.1)),
                    float(settings.get("attn_dropout", 0.0)),
                ),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=int(settings.get("encoder_layers", 2)),
            )
            self.head = nn.Sequential(
                nn.Dropout(float(settings.get("head_dropout", 0.0))),
                nn.Linear(channel_count * patch_count * hidden_size, lead),
            )
            nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
            nn.init.normal_(self.channel_embedding, mean=0.0, std=0.02)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            # inputs: [batch, look-back, SR + 16 historical covariates]
            channels = inputs.transpose(1, 2)
            if self.use_revin:
                center = (
                    channels[:, :, -1:]
                    if self.subtract_last
                    else channels.mean(dim=2, keepdim=True)
                )
                scale = torch.sqrt(
                    channels.var(dim=2, keepdim=True, unbiased=False) + 1e-5
                )
                channels = (channels - center) / scale
                if self.revin_affine:
                    channels = channels * self.revin_weight + self.revin_bias

            patches = channels.unfold(dimension=2, size=patch_len, step=stride)
            tokens = self.patch_embedding(patches)
            tokens = self.input_dropout(
                tokens + self.position_embedding + self.channel_embedding
            )
            batch_size = tokens.shape[0]
            encoded = self.encoder(
                tokens.reshape(batch_size * channel_count, patch_count, hidden_size)
            )
            forecast = self.head(encoded.reshape(batch_size, -1))

            if self.use_revin:
                if self.revin_affine:
                    forecast = (
                        forecast - self.revin_bias[:, self.target_index, :]
                    ) / (self.revin_weight[:, self.target_index, :] + 1e-8)
                forecast = (
                    forecast * scale[:, self.target_index, :]
                    + center[:, self.target_index, :]
                )
            return forecast

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CovariatePatchTST().to(device)
    training_data = TensorDataset(
        torch.tensor(
            prepared.sequences[: prepared.train_count],
            dtype=torch.float32,
        ),
        torch.tensor(
            prepared.targets[: prepared.train_count],
            dtype=torch.float32,
        ),
    )
    loader = DataLoader(
        training_data,
        batch_size=int(settings.get("windows_batch_size", 32)),
        shuffle=False,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(settings.get("learning_rate", 1e-4)),
    )
    loss_name = str(settings.get("loss", "mae")).lower()
    if loss_name not in {"mae", "mse"}:
        raise ValueError("PatchTST loss must be 'mae' or 'mse'")
    criterion: nn.Module = nn.L1Loss() if loss_name == "mae" else nn.MSELoss()
    max_steps = int(settings.get("max_steps", 500))
    if max_steps < 1:
        raise ValueError("PatchTST max_steps must be positive")

    model.train()
    step = 0
    while step < max_steps:
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            step += 1
            if bool(settings.get("enable_progress_bar", True)) and (
                step == 1 or step == max_steps or step % 100 == 0
            ):
                print(f"PatchTST step {step}/{max_steps}: loss={loss.item():.6f}")
            if step >= max_steps:
                break

    model.eval()
    with torch.no_grad():
        predicted = model(
            torch.tensor(
                prepared.sequences[prepared.train_count :],
                dtype=torch.float32,
                device=device,
            )
        ).cpu().numpy()
    return _prediction_frame(
        prepared,
        predicted,
        lead,
        "PatchTST_pred",
        date_col,
    )


def _merge_prediction(
    result: pd.DataFrame,
    addition: pd.DataFrame,
    model: str,
    date_col: str,
) -> pd.DataFrame:
    keys = ["origin_date", date_col, "lead"]
    if addition.duplicated(keys).any():
        raise ValueError(f"{model} produced duplicate origin-target-lead rows")
    merged = result.merge(addition, on=keys, how="left", validate="one_to_one")
    output_column = OUTPUT_COLUMNS[model]
    if merged[output_column].isna().any():
        missing = merged.loc[merged[output_column].isna(), keys].head(3).to_dict("records")
        raise ValueError(f"{model} is missing forecasts for requested Informer origins: {missing}")
    return merged


def run_lead_specific_comparison(
    baseline_csv: Path,
    data_csv: Path,
    output_csv: Path,
    lead: int,
    lags: int,
    selected_models: list[str],
    config: dict[str, Any],
) -> Path:
    """Fit selected comparators and save one forecast per origin at one forecast lead."""
    unknown = sorted(set(selected_models) - set(SUPPORTED_MODELS))
    if unknown:
        raise ValueError(f"Unsupported models {unknown}; choose from {SUPPORTED_MODELS}")
    if len(selected_models) != len(set(selected_models)):
        raise ValueError("selected_models contains duplicates")

    date_col = config.get("date_col", "date")
    target_col = config.get("target_col", "ALL_CoVaR")
    train_ratio = float(config.get("train_ratio", 0.7))
    normalization = config.get("normalization", "minmax")
    seed = int(config.get("random_seed", 2024))
    _set_seed(seed)

    feature_columns = config.get("feature_columns")
    data = _read_forecasting_data(Path(data_csv), date_col, target_col, feature_columns)
    result = load_informer_lead(Path(baseline_csv), data, lead, date_col, target_col)
    prepared = _prepare_samples(data, lead, lags, train_ratio, normalization, date_col, target_col)
    model_set = set(selected_models)

    generated = _simple_predictions(prepared, model_set, lead, date_col)
    settings = config.get("model_settings", {})
    if "GBR" in model_set:
        generated["GBR"] = _gbr_predictions(prepared, lead, date_col, settings.get("GBR", {}), seed)
    if "LSTM" in model_set:
        generated["LSTM"] = _lstm_predictions(prepared, lead, date_col, settings.get("LSTM", {}))
    if "PatchTST" in model_set:
        generated["PatchTST"] = _patchtst_predictions(
            prepared, lead, lags, date_col, target_col, settings.get("PatchTST", {}), seed
        )

    for model in selected_models:
        result = _merge_prediction(result, generated[model], model, date_col)
    ordered = ["origin_date", date_col, "lead", "TRUE", "Informer_pred"] + [
        OUTPUT_COLUMNS[model] for model in selected_models
    ]
    result = result[ordered]
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    return output_csv
