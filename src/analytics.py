"""Statistics calculated from the stored workout history (no hard-coded numbers)."""
import pandas as pd


def today_ts() -> pd.Timestamp:
    return pd.Timestamp.today().normalize()


def week_start(day: pd.Timestamp) -> pd.Timestamp:
    """Monday of the week containing `day`."""
    return (day - pd.Timedelta(days=day.weekday())).normalize()


def between(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Rows with start <= date <= end (inclusive)."""
    return df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]


def day_summary(df: pd.DataFrame, day) -> dict:
    d = df[df["date"] == pd.Timestamp(day).normalize()]
    return {"calories": float(d["calories_burned"].sum()), "workouts": int(len(d)),
            "minutes": float(d["duration"].sum())}


def week_summary(df: pd.DataFrame, day=None) -> dict:
    day = today_ts() if day is None else pd.Timestamp(day).normalize()
    start = week_start(day)
    w = between(df, start, start + pd.Timedelta(days=6))
    return {"calories": float(w["calories_burned"].sum()), "workouts": int(len(w)),
            "minutes": float(w["duration"].sum()), "start": start}


def week_daily_series(df: pd.DataFrame, day=None) -> pd.DataFrame:
    """Calories + minutes for each day Monday..Sunday of the current week."""
    day = today_ts() if day is None else pd.Timestamp(day).normalize()
    start = week_start(day)
    days = pd.date_range(start, periods=7)
    w = between(df, start, start + pd.Timedelta(days=6))
    daily = w.groupby("date").agg(calories=("calories_burned", "sum"),
                                  minutes=("duration", "sum"), workouts=("id", "count"))
    out = daily.reindex(days, fill_value=0).rename_axis("date").reset_index()
    out["day"] = out["date"].dt.strftime("%a")
    out["is_future"] = out["date"] > day
    return out


def _streaks(dates) -> list:
    """Lengths of all runs of consecutive days in a sorted list of unique dates."""
    runs, run, prev = [], 0, None
    for d in dates:
        run = run + 1 if prev is not None and (d - prev).days == 1 else 1
        runs.append(run)
        prev = d
    return runs


def current_streak(df: pd.DataFrame, today=None) -> int:
    """Consecutive workout days ending today (or yesterday if you haven't trained yet today)."""
    if df.empty:
        return 0
    today = today_ts() if today is None else pd.Timestamp(today).normalize()
    days = set(df["date"].dt.normalize())
    cursor = today if today in days else today - pd.Timedelta(days=1)
    count = 0
    while cursor in days:
        count += 1
        cursor -= pd.Timedelta(days=1)
    return count


def longest_streak(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return max(_streaks(sorted(set(df["date"].dt.normalize()))))


def weekly_goal_streak(df: pd.DataFrame, weekly_goal: int) -> int:
    """Longest run of consecutive weeks in which the weekly workout goal was reached."""
    if df.empty or not weekly_goal:
        return 0
    starts = df["date"].apply(week_start)
    counts = df.groupby(starts)["id"].count()
    good = sorted(w for w, n in counts.items() if n >= weekly_goal)
    best, run, prev = 0, 0, None
    for w in good:
        run = run + 1 if prev is not None and (w - prev).days == 7 else 1
        best, prev = max(best, run), w
    return best


def best_day_calories(df: pd.DataFrame) -> float:
    return 0.0 if df.empty else float(df.groupby("date")["calories_burned"].sum().max())


def overall_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"total_calories": 0, "avg_calories": 0, "total_minutes": 0, "total_workouts": 0,
                "current_streak": 0, "best_workout": None}
    best = df.loc[df["calories_burned"].idxmax()]
    return {"total_calories": float(df["calories_burned"].sum()),
            "avg_calories": float(df["calories_burned"].mean()),
            "total_minutes": float(df["duration"].sum()), "total_workouts": int(len(df)),
            "current_streak": current_streak(df), "best_workout": best}


def period_trend(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Total calories / workouts / avg heart-rate per week ('W') or month ('M')."""
    if df.empty:
        return pd.DataFrame(columns=["period", "calories", "workouts", "avg_hr"])
    key = df["date"].dt.to_period(freq).dt.start_time
    g = df.groupby(key).agg(calories=("calories_burned", "sum"), workouts=("id", "count"),
                            avg_hr=("heart_rate", "mean"))
    return g.rename_axis("period").reset_index()
