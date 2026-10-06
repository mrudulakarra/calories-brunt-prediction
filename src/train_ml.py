"""Train and compare the classic Machine-Learning regressors.

Run from the project root:   python src/train_ml.py

Steps: load -> clean -> EDA -> split -> scale -> train -> evaluate ->
compare -> choose best -> save.
"""
import json
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")  # draw charts to files, no window needed
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (EDA_DIR, FEATURES, ML_METRICS_PATH, ML_MODEL_FILES,
                        RANDOM_STATE, TARGET)
from src.prediction import select_best_model
from src.preprocessing import prepare_data


def run_eda(df) -> None:
    """Exploratory Data Analysis: save a few plots to assets/eda/."""
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    print("\n--- EDA: dataset summary ---")
    print(df.describe().round(2).to_string())

    # 1) Distribution of every column
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.ravel(), FEATURES + [TARGET]):
        ax.hist(df[col], bins=30, color="#FF5A3C", edgecolor="white")
        ax.set_title(col)
    fig.suptitle("Feature distributions")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "distributions.png", dpi=110)
    plt.close(fig)

    # 2) Correlation heat-map
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im)
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "correlation.png", dpi=110)
    plt.close(fig)

    # 3) The strongest relationships with calories
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["DURATION", "HEART_RATE", "BODY_TEMP"]):
        ax.scatter(df[col], df[TARGET], s=6, alpha=0.4, color="#7C3AED")
        ax.set_xlabel(col)
        ax.set_ylabel(TARGET)
    fig.suptitle("Calories vs key features")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "calories_vs_features.png", dpi=110)
    plt.close(fig)
    print(f"EDA plots saved to {EDA_DIR}")


def evaluate(model, X_test, y_test) -> dict:
    """MAE, MSE, RMSE and R² on the unseen test set."""
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    return {"mae": float(mean_absolute_error(y_test, pred)), "mse": float(mse),
            "rmse": float(np.sqrt(mse)), "r2": float(r2_score(y_test, pred))}


def build_models() -> dict:
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=150, min_samples_leaf=2,
                                               n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.1,
                                                       max_depth=3, random_state=RANDOM_STATE),
    }
    try:  # XGBoost is optional
        XGBRegressor = importlib.import_module("xgboost").XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5,
                                         random_state=RANDOM_STATE)
    except ImportError:
        print("(xgboost not installed - skipping XGBoost, this is optional)")
    return models


def plot_comparison(results: dict) -> None:
    names = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(names, [results[n]["r2"] for n in names], color="#FF5A3C")
    axes[0].set_title("R² score (higher is better)")
    axes[1].bar(names, [results[n]["rmse"] for n in names], color="#7C3AED")
    axes[1].set_title("RMSE in kcal (lower is better)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "ml_model_comparison.png", dpi=110)
    plt.close(fig)


def main() -> None:
    data = prepare_data(save_artifacts=True)
    print("Cleaning report:", data["report"])
    run_eda(data["df"])

    # Models are trained on the SCALED features (the same scaler is used in the app).
    results = {}
    for name, model in build_models().items():
        print(f"\nTraining {name} ...")
        model.fit(data["X_train_s"], data["y_train"])
        results[name] = evaluate(model, data["X_test_s"], data["y_test"])
        joblib.dump(model, ML_MODEL_FILES[name], compress=3)
        m = results[name]
        print(f"  MAE={m['mae']:.2f}  MSE={m['mse']:.2f}  RMSE={m['rmse']:.2f}  R²={m['r2']:.4f}")

    best_ml = min(results, key=lambda n: results[n]["rmse"])
    ML_METRICS_PATH.write_text(json.dumps(
        {"models": results, "best_ml_model": best_ml,
         "n_train": len(data["X_train"]), "n_test": len(data["X_test"])}, indent=2))
    plot_comparison(results)

    best_overall = select_best_model()  # also considers the DL model if it is already trained
    print(f"\nBest ML model: {best_ml}")
    print(f"Best model overall (so far): {best_overall}")
    print("Next step:  python src/train_dl.py")


if __name__ == "__main__":
    main()
