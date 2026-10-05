# 🔥 Calories Burnt Prediction AI
**AI-Powered Personal Fitness & Calorie Intelligence Platform**

A complete Machine Learning + Deep Learning + Data Visualisation project built with **Python, Streamlit, scikit-learn, TensorFlow/Keras, Plotly and SQLite**.

## Project overview
Users enter workout details (gender, age, height, weight, duration, heart rate, body temperature, exercise type).
Trained models predict the calories burned; the app then stores the workout, updates goals, unlocks achievements
and generates insights from the user's own history. All numbers on screen come from real models or real stored data.

## Problem statement
Knowing how many calories a workout burns is useful for fitness planning, but wearables are expensive and
formula-based estimates ignore physiological signals. Can a model learn the relationship between body
measurements, workout duration, heart rate and body temperature to predict calories burned accurately?

## Objectives
- Build a full ML pipeline (clean → EDA → encode → split → scale → train → evaluate → compare → save).
- Build a Keras neural network and compare it fairly with classic ML models on the same test set.
- Deliver a premium dashboard with history, goals, achievements, profile and analytics backed by SQLite.

## Features
Dashboard KPIs & charts · calorie prediction with intensity/MET/quality indicator · history with filters and delete ·
daily/weekly goals with status · 7 achievements with progress · profile with completion % · AI insights from your data ·
AI Model Performance page · light/dark mode · CSV export · input validation and friendly errors.

## ML methodology
Linear Regression, Random Forest, Gradient Boosting (and XGBoost if installed) are trained on standardised features.
Metrics on the held-out test set: **MAE, MSE, RMSE, R²**. The model with the lowest RMSE is saved as the "best model".

## DL methodology
`Input → Dense(128, ReLU) → Dropout → Dense(64, ReLU) → Dropout → Dense(32, ReLU) → Dense(1)`
Adam optimiser, MSE loss, MAE metric, early stopping (patience 15, best weights restored). The target is standardised
for training and converted back to kcal at prediction time. Training/validation loss curves are shown in the app.

## Dataset
Place the Kaggle *Calories Burnt Prediction* data in `data/calories.csv` (see `data/README.md`). Column names are
mapped automatically; the two-file Kaggle version (`exercise.csv` + `calories.csv`) is merged automatically.
No dataset is bundled. `python src/make_sample_data.py` can create a **synthetic** file for a smoke test only.

## Preprocessing
Standardise column names → encode gender (male=1, female=0) → drop missing/duplicate/impossible rows →
80/20 train-test split (`random_state=42`, same split for ML and DL) → `StandardScaler` fitted on training data only.

## Model comparison
After training, open **🤖 AI Insights → AI Model Performance** to see R²/RMSE cards (best model highlighted), comparison
charts, the DNN loss curves and Random Forest feature importance. Fill in your own numbers in the table below after training:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | | | |
| Random Forest | | | |
| Gradient Boosting | | | |
| Deep Neural Network | | | |

## Architecture
```
User → Streamlit dashboard → Profile / Goals / History
     → Prediction form → preprocessing (same scaler as training)
     → ML models (RF, GB, LR) + Deep Neural Network → best model result
     → AI insights → SQLite history + goals → achievements
```

## Screenshots
_Add screenshots here after running the app (Dashboard, Predict, History, Goals, Achievements, AI Insights)._

## Installation (Windows, Python 3.10–3.12)
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Put the dataset in `data/calories.csv`, then train and run:
```powershell
python src/train_ml.py
python src/train_dl.py
streamlit run app.py
```
Optional: `pip install xgboost` adds XGBoost to the comparison.

## Project structure
```
app.py                    Streamlit app (all pages)
src/config.py             paths, feature names, option lists
src/preprocessing.py      load / clean / encode / split / scale
src/train_ml.py           EDA + ML training + comparison
src/train_dl.py           Keras neural network training
src/prediction.py         model loading, validation, prediction, intensity, MET
src/database.py           SQLite tables and queries
src/analytics.py          statistics from stored workouts
src/achievements.py       achievement rules and unlocking
src/recommendations.py    rule-based insights
src/ui.py                 CSS theme + UI helpers
src/make_sample_data.py   OPTIONAL synthetic data for smoke tests
notebooks/model_training.ipynb   step-by-step training notebook
data/  models/  database/  assets/
```

## Important notes
- **Exercise type is not a model input** - the Kaggle dataset has no activity column. The model uses the 7 physical/physiological features; exercise type is stored in history and statistics only.
- The Kaggle data covers workouts of ≤ 30 minutes and heart rates ≤ 128 BPM. The app warns when your inputs fall outside the training range, because predictions there are less reliable (tree models cannot extrapolate).
- Intensity is derived from heart rate as a share of estimated maximum heart rate (220 − age). MET = kcal ÷ (kg × hours).
- Predictions are **estimates, not medical measurements or advice**.

## Future improvements
Multi-user login · wearable/CSV import · hyper-parameter tuning · exercise-aware models · cloud database · PDF reports.

## Technologies used
Python · Streamlit · Pandas · NumPy · scikit-learn · TensorFlow/Keras · Plotly · Matplotlib · Joblib · SQLite
