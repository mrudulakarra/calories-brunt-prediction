"""Dataset loading, cleaning and preparation.

Pipeline implemented here (matches the project flow diagram):
    load  ->  standardise column names  ->  clean  ->  encode  ->
    train/test split  ->  feature scaling
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (DATA_FILE, FEATURE_RANGES_PATH, FEATURES, MODELS_DIR,
                        RANDOM_STATE, SCALER_PATH, TARGET, TEST_SIZE)

# Different Kaggle versions use different column names.  Every known alias
# is mapped to one standard name so the rest of the code never has to care.
COLUMN_ALIASES = {
    "USER_ID": ["user_id", "userid", "id"],
    "GENDER": ["gender", "sex"],
    "AGE": ["age"],
    "HEIGHT": ["height", "height_cm"],
    "WEIGHT": ["weight", "weight_kg"],
    "DURATION": ["duration", "duration_min", "workout_duration", "time"],
    "HEART_RATE": ["heart_rate", "heartrate", "hr"],
    "BODY_TEMP": ["body_temp", "body_temperature", "temperature", "bodytemp"],
    "CALORIES": ["calories", "calories_burned", "calories_burnt", "calorie"],
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename dataset columns to the standard upper-case names."""
    lookup = {alias: std for std, aliases in COLUMN_ALIASES.items() for alias in aliases}
    renamed = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in lookup:
            renamed[col] = lookup[key]
    return df.rename(columns=renamed)


def _missing_columns(df: pd.DataFrame) -> list:
    return [c for c in FEATURES + [TARGET] if c not in df.columns]


def load_dataset(path=DATA_FILE) -> pd.DataFrame:
    """Read the CSV and return it with standardised column names.

    The popular Kaggle "Calories Burnt Prediction" dataset ships as two files
    (exercise.csv + calories.csv).  If calories.csv only holds User_ID and
    Calories, we automatically merge it with exercise.csv.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download the Kaggle calories dataset and place it at data/calories.csv "
            "(see data/README.md)."
        )
    df = standardize_columns(pd.read_csv(path))

    if _missing_columns(df):
        exercise_file = path.with_name("exercise.csv")
        if exercise_file.exists() and "USER_ID" in df.columns:
            ex = standardize_columns(pd.read_csv(exercise_file))
            df = ex.merge(df, on="USER_ID", how="inner", suffixes=("", "_dup"))
        missing = _missing_columns(df)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}. "
                             f"Found: {list(df.columns)}")
    return df


def encode_gender(value) -> int:
    """male -> 1, female -> 0 (numbers pass through)."""
    if isinstance(value, str):
        return 1 if value.strip().lower() in ("male", "m", "1") else 0
    return int(value)


def clean_data(df: pd.DataFrame):
    """Clean the raw data. Returns (clean_df, report_dict)."""
    report = {"rows_raw": int(len(df))}
    df = df[FEATURES + [TARGET]].copy()
    df["GENDER"] = df["GENDER"].map(encode_gender)          # encoding step
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    report["missing_values_dropped"] = int(df.isna().any(axis=1).sum())
    df = df.dropna()
    report["duplicates_dropped"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # Physically impossible rows (zero/negative measurements) are removed.
    positive_cols = ["AGE", "HEIGHT", "WEIGHT", "DURATION", "HEART_RATE", "BODY_TEMP"]
    invalid = (df[positive_cols] <= 0).any(axis=1) | (df[TARGET] < 0)
    report["invalid_rows_dropped"] = int(invalid.sum())
    df = df[~invalid].reset_index(drop=True)
    report["rows_clean"] = int(len(df))
    return df, report


def split_data(df: pd.DataFrame):
    """80/20 train-test split.  The same random_state is used by the ML and DL
    scripts, so every model is evaluated on exactly the same test rows."""
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fit a StandardScaler on TRAINING data only (avoids data leakage)."""
    scaler = StandardScaler().fit(X_train)
    return scaler, scaler.transform(X_train), scaler.transform(X_test)


def prepare_data(save_artifacts: bool = True):
    """Run the full preprocessing pipeline and return everything needed."""
    df, report = clean_data(load_dataset())
    X_train, X_test, y_train, y_test = split_data(df)
    scaler, X_train_s, X_test_s = scale_features(X_train, X_test)
    if save_artifacts:
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        save_feature_ranges(X_train)
    return {"df": df, "report": report, "scaler": scaler,
            "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
            "X_train_s": X_train_s, "X_test_s": X_test_s}


def save_feature_ranges(X_train: pd.DataFrame) -> None:
    """Remember the min/max/median of each feature seen in training.
    The app warns users when they enter values outside this range, because
    models cannot be trusted far outside the data they were trained on."""
    ranges = {c: {"min": float(X_train[c].min()), "max": float(X_train[c].max()),
                  "median": float(X_train[c].median())} for c in FEATURES}
    FEATURE_RANGES_PATH.write_text(json.dumps(ranges, indent=2))
