"""Model loading and calorie prediction.

Nothing in this file is hard-coded: every number shown to the user is
produced by the trained models stored in the models/ folder.
"""
import json

import joblib
import numpy as np
import pandas as pd

from src.config import (BEST_MODEL_PATH, DL_METRICS_PATH, DNN_NAME, DNN_PATH,
                        FEATURE_RANGES_PATH, FEATURES, ML_METRICS_PATH,
                        ML_MODEL_FILES, SCALER_PATH)
from src.preprocessing import encode_gender

AUTO_CHOICE = "Best model (auto)"


# --------------------------------------------------------------------------
# Helpers for the app's start-up checks
# --------------------------------------------------------------------------
def _read_json(path):
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except (json.JSONDecodeError, OSError):
        return None


def get_missing_artifacts() -> list:
    """Names of the files that the training scripts have not created yet."""
    needed = [SCALER_PATH, ML_MODEL_FILES["Random Forest"],
              ML_MODEL_FILES["Gradient Boosting"], DNN_PATH, BEST_MODEL_PATH]
    return [p.name for p in needed if not p.exists()]


def can_predict() -> bool:
    """True when at least the scaler and one ML model exist."""
    return SCALER_PATH.exists() and any(p.exists() for p in ML_MODEL_FILES.values())


def artifacts_signature() -> tuple:
    """Changes whenever a model file changes -> used as the cache key."""
    paths = [SCALER_PATH, DNN_PATH, BEST_MODEL_PATH, *ML_MODEL_FILES.values()]
    return tuple(p.stat().st_mtime if p.exists() else 0 for p in paths)


def load_artifacts() -> dict | None:
    """Load scaler + ML models + deep-learning model (+ metrics)."""
    if not can_predict():
        return None
    art = {"scaler": joblib.load(SCALER_PATH), "ml": {}, "dnn": None,
           "dl_meta": _read_json(DL_METRICS_PATH) or {}, "warnings": []}
    for name, path in ML_MODEL_FILES.items():
        if path.exists():
            art["ml"][name] = joblib.load(path)
    if DNN_PATH.exists():
        try:
            from tensorflow import keras  # imported lazily: TensorFlow is heavy
            art["dnn"] = keras.models.load_model(DNN_PATH, compile=False)
        except Exception:  # noqa: BLE001 - show a friendly message instead
            art["warnings"].append("The deep-learning model could not be loaded "
                                   "(is TensorFlow installed?). ML models are used instead.")
    art["metrics"] = load_all_metrics()
    best = _read_json(BEST_MODEL_PATH) or {}
    art["best"] = best.get("best_model")
    if art["best"] not in available_models(art):
        art["best"] = next(iter(art["ml"]))
    art["ranges"] = _read_json(FEATURE_RANGES_PATH) or {}
    return art


def available_models(art: dict) -> list:
    names = list(art["ml"].keys())
    if art.get("dnn") is not None:
        names.append(DNN_NAME)
    return names


def load_all_metrics() -> dict:
    """Merge ML and DL test-set metrics into one {model_name: metrics} dict."""
    out = {}
    ml = _read_json(ML_METRICS_PATH)
    if ml:
        out.update(ml.get("models", {}))
    dl = _read_json(DL_METRICS_PATH)
    if dl:
        out[DNN_NAME] = {k: dl[k] for k in ("mae", "mse", "rmse", "r2")}
    return out


def select_best_model() -> str | None:
    """Pick the model with the lowest test RMSE and remember it."""
    metrics = load_all_metrics()
    if not metrics:
        return None
    best = min(metrics, key=lambda n: metrics[n]["rmse"])
    BEST_MODEL_PATH.write_text(json.dumps(
        {"best_model": best, "selected_by": "lowest test RMSE",
         "rmse": metrics[best]["rmse"], "r2": metrics[best]["r2"]}, indent=2))
    return best


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------
def validate_inputs(age, height, weight, duration, heart_rate, body_temp) -> list:
    """Return a list of friendly error messages (empty list = all valid)."""
    errors = []
    values = [age, height, weight, duration, heart_rate, body_temp]
    if any(v is None or not np.isfinite(v) for v in values):
        return ["Please fill in every field with a valid number."]
    if not 10 <= age <= 100:
        errors.append("Age must be between 10 and 100.")
    if height <= 0:
        errors.append("Height must be positive.")
    elif not 100 <= height <= 250:
        errors.append("Height should be between 100 and 250 cm.")
    if weight <= 0:
        errors.append("Weight must be positive.")
    elif not 25 <= weight <= 300:
        errors.append("Weight should be between 25 and 300 kg.")
    if duration <= 0:
        errors.append("Workout duration must be greater than 0.")
    elif duration > 600:
        errors.append("Workout duration looks too long (maximum 600 minutes).")
    if not 40 <= heart_rate <= 220:
        errors.append("Heart rate must be within a realistic range (40-220 BPM).")
    if not 35 <= body_temp <= 42:
        errors.append("Body temperature must be between 35 and 42 °C.")
    return errors


def out_of_range_warnings(features: pd.DataFrame, ranges: dict) -> list:
    """Warn when an input lies outside the data the models were trained on."""
    labels = {"AGE": "Age", "HEIGHT": "Height", "WEIGHT": "Weight",
              "DURATION": "Duration", "HEART_RATE": "Heart rate", "BODY_TEMP": "Body temperature"}
    warnings = []
    for col, label in labels.items():
        r = ranges.get(col)
        if not r:
            continue
        v = float(features[col].iloc[0])
        if v < r["min"] or v > r["max"]:
            warnings.append(f"{label} ({v:g}) is outside the training-data range "
                            f"({r['min']:g}-{r['max']:g}); the estimate may be less accurate.")
    return warnings


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def make_feature_frame(gender, age, height, weight, duration, heart_rate, body_temp) -> pd.DataFrame:
    """One-row DataFrame with the exact column order used in training."""
    return pd.DataFrame([{
        "GENDER": encode_gender(gender), "AGE": age, "HEIGHT": height, "WEIGHT": weight,
        "DURATION": duration, "HEART_RATE": heart_rate, "BODY_TEMP": body_temp}])[FEATURES]


def predict_all(art: dict, features: pd.DataFrame) -> dict:
    """Predict with every available model -> {model_name: kcal}."""
    X = art["scaler"].transform(features[FEATURES])          # same scaling as training
    preds = {name: float(model.predict(X)[0]) for name, model in art["ml"].items()}
    if art.get("dnn") is not None:
        meta = art["dl_meta"]
        raw = art["dnn"](X.astype("float32"), training=False).numpy()[0, 0]
        # The network was trained on a standardised target -> convert back to kcal.
        preds[DNN_NAME] = float(raw * meta.get("target_std", 1.0) + meta.get("target_mean", 0.0))
    return {k: max(0.0, v) for k, v in preds.items() if np.isfinite(v)}


def predict_calories(art: dict, features: pd.DataFrame, choice: str = AUTO_CHOICE) -> dict:
    """Predict calories with the chosen (or best) model and describe its reliability."""
    preds = predict_all(art, features)
    if not preds:
        raise RuntimeError("No model produced a valid prediction.")
    model_used = art["best"] if choice == AUTO_CHOICE or choice not in preds else choice
    if model_used not in preds:
        model_used = next(iter(preds))
    kcal = preds[model_used]

    # Prediction-quality indicator: do the independent models agree?
    values = list(preds.values())
    spread = (max(values) - min(values)) / max(np.mean(values), 1.0) if len(values) > 1 else 0.0
    out_of_range = bool(out_of_range_warnings(features, art.get("ranges", {})))
    if out_of_range or spread > 0.25:
        quality = "Low"
    elif spread > 0.10:
        quality = "Medium"
    else:
        quality = "High"
    rmse = art["metrics"].get(model_used, {}).get("rmse")
    return {"calories": round(kcal), "model_used": model_used, "all_predictions": preds,
            "quality": quality, "model_spread_pct": round(spread * 100, 1),
            "typical_error": round(rmse, 1) if rmse else None,
            "warnings": out_of_range_warnings(features, art.get("ranges", {}))}


# --------------------------------------------------------------------------
# Intensity / MET
# --------------------------------------------------------------------------
def classify_intensity(heart_rate: float, age: float) -> str:
    """Intensity from % of estimated maximum heart rate (220 - age)."""
    pct = heart_rate / max(220 - age, 1) * 100
    if pct < 60:
        return "Light"
    if pct < 75:
        return "Moderate"
    if pct < 90:
        return "Vigorous"
    return "Max Effort"


def estimate_met(calories: float, weight_kg: float, duration_min: float) -> float:
    """MET = kcal / (weight in kg x hours).  1 MET ~ resting energy use."""
    hours = max(duration_min, 1e-6) / 60
    return calories / (max(weight_kg, 1e-6) * hours)


def met_category(met: float) -> str:
    if met < 3:
        return "Light activity (< 3 MET)"
    if met < 6:
        return "Moderate activity (3-6 MET)"
    return "Vigorous activity (≥ 6 MET)"
