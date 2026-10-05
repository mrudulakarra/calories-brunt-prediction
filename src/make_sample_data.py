"""OPTIONAL - creates a SYNTHETIC dataset so you can smoke-test the pipeline.

!!! This is NOT real data. !!!
For your project/viva, download the real Kaggle dataset and place it at
data/calories.csv.  Models trained on this synthetic file only demonstrate
that the code works; do not report their scores as real results.

Usage:  python src/make_sample_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import DATA_FILE


def main(n: int = 15000, seed: int = 0) -> None:
    if DATA_FILE.exists():
        sys.exit(f"{DATA_FILE} already exists - refusing to overwrite your dataset.")
    rng = np.random.default_rng(seed)
    male = rng.integers(0, 2, n)
    age = rng.integers(20, 80, n)
    height = np.where(male == 1, rng.normal(176, 8, n), rng.normal(163, 7, n)).round()
    bmi = rng.normal(24.5, 3.5, n).clip(16, 38)
    weight = (bmi * (height / 100) ** 2).round()
    duration = rng.integers(1, 31, n)
    heart_rate = (rng.normal(72, 5, n) + duration * rng.uniform(1.0, 2.2, n)).clip(67, 128).round()
    body_temp = (37.0 + duration * rng.uniform(0.04, 0.14, n) + rng.normal(0, 0.1, n)).clip(37.1, 41.5).round(1)
    # Keytel et al. (2005) heart-rate based energy-expenditure equation + noise
    kcal_min = np.where(
        male == 1,
        (-55.0969 + 0.6309 * heart_rate + 0.1988 * weight + 0.2017 * age) / 4.184,
        (-20.4022 + 0.4472 * heart_rate - 0.1263 * weight + 0.074 * age) / 4.184)
    calories = (kcal_min.clip(1) * duration * rng.normal(1.0, 0.04, n)).round()
    pd.DataFrame({"User_ID": np.arange(10000000, 10000000 + n),
                  "Gender": np.where(male == 1, "male", "female"), "Age": age, "Height": height,
                  "Weight": weight, "Duration": duration, "Heart_Rate": heart_rate,
                  "Body_Temp": body_temp, "Calories": calories}).to_csv(DATA_FILE, index=False)
    print(f"SYNTHETIC sample dataset written to {DATA_FILE} ({n} rows). Replace it with the Kaggle data!")


if __name__ == "__main__":
    main()
