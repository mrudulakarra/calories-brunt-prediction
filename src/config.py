"""Central configuration: file paths, feature names and option lists.

Every other module imports its constants from here so that there is a
single place to change things.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DB_DIR = ROOT / "database"
ASSETS_DIR = ROOT / "assets"
EDA_DIR = ASSETS_DIR / "eda"

DATA_FILE = DATA_DIR / "calories.csv"
DB_PATH = DB_DIR / "fitness.db"

# ---- Model artefacts -------------------------------------------------------
SCALER_PATH = MODELS_DIR / "scaler.pkl"
DNN_PATH = MODELS_DIR / "calories_dnn.keras"
ML_MODEL_FILES = {
    "Linear Regression": MODELS_DIR / "linear_regression.pkl",
    "Random Forest": MODELS_DIR / "random_forest.pkl",
    "Gradient Boosting": MODELS_DIR / "gradient_boosting.pkl",
    "XGBoost": MODELS_DIR / "xgboost.pkl",
}
DNN_NAME = "Deep Neural Network"
ML_METRICS_PATH = MODELS_DIR / "ml_metrics.json"
DL_METRICS_PATH = MODELS_DIR / "dl_metrics.json"
DL_HISTORY_PATH = MODELS_DIR / "dl_history.json"
BEST_MODEL_PATH = MODELS_DIR / "best_model.json"
FEATURE_RANGES_PATH = MODELS_DIR / "feature_ranges.json"

# ---- Standardised dataset columns -----------------------------------------
FEATURES = ["GENDER", "AGE", "HEIGHT", "WEIGHT", "DURATION", "HEART_RATE", "BODY_TEMP"]
TARGET = "CALORIES"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---- App option lists -------------------------------------------------------
EXERCISE_TYPES = ["Walking", "Running", "Cycling", "Swimming", "Gym", "HIIT",
                  "Yoga", "Strength Training", "Other"]
FITNESS_GOALS = ["Weight Loss", "Weight Maintenance", "Muscle Building",
                 "General Fitness", "Endurance"]
ACTIVITY_LEVELS = ["Sedentary", "Lightly Active", "Moderately Active",
                   "Very Active", "Athlete"]
INTENSITY_LEVELS = ["Light", "Moderate", "Vigorous", "Max Effort"]

DEFAULT_USER_ID = 1
