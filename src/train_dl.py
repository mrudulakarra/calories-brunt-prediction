"""Train the Deep-Learning model (Dense neural network / MLP) with Keras.

Run from the project root:   python src/train_dl.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide noisy TensorFlow logs

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers

from src.config import (DL_HISTORY_PATH, DL_METRICS_PATH, DNN_NAME, DNN_PATH,
                        EDA_DIR, RANDOM_STATE)
from src.prediction import select_best_model
from src.preprocessing import prepare_data


def build_model(n_features: int) -> keras.Model:
    """Input -> 128 ReLU -> Dropout -> 64 ReLU -> Dropout -> 32 ReLU -> 1 output."""
    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),          # randomly switches off 20% of neurons -> less over-fitting
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),              # one neuron = predicted calories (regression)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse", metrics=["mae"])
    return model


def plot_history(history: dict) -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["loss"], label="Training loss", color="#FF5A3C")
    ax.plot(history["val_loss"], label="Validation loss", color="#7C3AED")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE, standardised target)")
    ax.set_title("Training Loss vs Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(EDA_DIR / "dl_loss_curve.png", dpi=110)
    plt.close(fig)


def main() -> None:
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    data = prepare_data(save_artifacts=True)      # same cleaning/split/scaler as the ML models

    # Neural networks train better when the target is also standardised.
    # We remember mean/std so the app can convert predictions back to kcal.
    y_mean, y_std = float(data["y_train"].mean()), float(data["y_train"].std())
    y_train_n = ((data["y_train"] - y_mean) / y_std).to_numpy()

    model = build_model(data["X_train_s"].shape[1])
    model.summary()

    # Early stopping: stop when validation loss no longer improves and keep the best weights.
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                               restore_best_weights=True)
    history = model.fit(data["X_train_s"], y_train_n, validation_split=0.15,
                        epochs=200, batch_size=32, callbacks=[early_stop], verbose=2)

    # Evaluate on the untouched test set, in real kcal units.
    pred = model.predict(data["X_test_s"], verbose=0).ravel() * y_std + y_mean
    y_test = data["y_test"].to_numpy()
    mse = mean_squared_error(y_test, pred)
    metrics = {"model": DNN_NAME,
               "mae": float(mean_absolute_error(y_test, pred)), "mse": float(mse),
               "rmse": float(np.sqrt(mse)), "r2": float(r2_score(y_test, pred)),
               "train_loss_final": float(history.history["loss"][-1]),
               "val_loss_final": float(history.history["val_loss"][-1]),
               "epochs_run": len(history.history["loss"]),
               "target_mean": y_mean, "target_std": y_std}

    model.save(DNN_PATH)
    DL_METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    DL_HISTORY_PATH.write_text(json.dumps({k: [float(x) for x in v]
                                           for k, v in history.history.items()}, indent=2))
    plot_history(history.history)

    print(f"\nDNN  MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  R²={metrics['r2']:.4f}")
    print(f"Best model overall: {select_best_model()}")
    print("Next step:  streamlit run app.py")


if __name__ == "__main__":
    main()
