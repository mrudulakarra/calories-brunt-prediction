# Dataset

Place the Kaggle **Calories Burnt Prediction** data in this folder as:

    data/calories.csv

Expected columns (names are matched case-insensitively and common aliases are accepted):

    User_ID, Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp, Calories

**Kaggle ships it as two files** (`exercise.csv` + `calories.csv`). That works too:
put BOTH files in this folder and the loader merges them on `User_ID` automatically.

## No dataset yet?
`python src/make_sample_data.py` writes a **synthetic** `data/calories.csv` (generated from a
published heart-rate energy formula + noise) so you can check that everything runs.
It is NOT real data - replace it with the Kaggle file before reporting any results.
