"""SQLite storage: users, workouts, goals, achievements (+ a small settings table).

The app is a personal, single-user dashboard, so the profile always has id = 1.
"""
import sqlite3
from contextlib import contextmanager

import pandas as pd

from src.config import DB_DIR, DB_PATH, DEFAULT_USER_ID

WORKOUT_COLUMNS = ["id", "date", "exercise_type", "duration", "heart_rate",
                   "body_temperature", "calories_burned", "intensity"]
USER_FIELDS = {"name", "age", "gender", "height", "weight", "fitness_goal",
               "activity_level", "daily_calorie_goal", "weekly_workout_goal"}


@contextmanager
def get_conn():
    DB_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create all tables (if missing) plus the default user and goals rows."""
    with get_conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY, name TEXT, age INTEGER, gender TEXT,
            height REAL, weight REAL, fitness_goal TEXT, activity_level TEXT,
            daily_calorie_goal REAL, weekly_workout_goal INTEGER);
        CREATE TABLE IF NOT EXISTS workouts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, date TEXT NOT NULL,
            exercise_type TEXT, duration REAL, heart_rate REAL, body_temperature REAL,
            calories_burned REAL, intensity TEXT);
        CREATE TABLE IF NOT EXISTS goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
            daily_calorie_goal REAL, weekly_calorie_goal REAL,
            weekly_workout_goal INTEGER, daily_duration_goal REAL);
        CREATE TABLE IF NOT EXISTS achievements (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
            achievement_name TEXT NOT NULL, description TEXT, unlocked_date TEXT);
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
        """)
        if not c.execute("SELECT 1 FROM users WHERE id=?", (DEFAULT_USER_ID,)).fetchone():
            # Only the goals get defaults; personal details stay empty until the user fills them in.
            c.execute("INSERT INTO users (id, daily_calorie_goal, weekly_workout_goal) VALUES (?,?,?)",
                      (DEFAULT_USER_ID, 600, 4))
        if not c.execute("SELECT 1 FROM goals WHERE user_id=?", (DEFAULT_USER_ID,)).fetchone():
            c.execute("""INSERT INTO goals (user_id, daily_calorie_goal, weekly_calorie_goal,
                         weekly_workout_goal, daily_duration_goal) VALUES (?,?,?,?,?)""",
                      (DEFAULT_USER_ID, 600, 3000, 4, 45))


# ------------------------------------------------------------------ users
def get_user(user_id: int = DEFAULT_USER_ID) -> dict:
    with get_conn() as c:
        return dict(c.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone())


def update_user(user_id: int = DEFAULT_USER_ID, **fields) -> None:
    fields = {k: v for k, v in fields.items() if k in USER_FIELDS}
    if not fields:
        return
    with get_conn() as c:
        sets = ", ".join(f"{k}=?" for k in fields)
        c.execute(f"UPDATE users SET {sets} WHERE id=?", (*fields.values(), user_id))
        # Keep the goals table in sync with the two goals that also live on the profile.
        if "daily_calorie_goal" in fields:
            c.execute("UPDATE goals SET daily_calorie_goal=? WHERE user_id=?",
                      (fields["daily_calorie_goal"], user_id))
        if "weekly_workout_goal" in fields:
            c.execute("UPDATE goals SET weekly_workout_goal=? WHERE user_id=?",
                      (fields["weekly_workout_goal"], user_id))


def profile_completion(user: dict) -> int:
    """Percentage of the 9 profile fields that are filled in."""
    keys = ["name", "age", "gender", "height", "weight", "fitness_goal",
            "activity_level", "daily_calorie_goal", "weekly_workout_goal"]
    filled = sum(1 for k in keys if user.get(k) not in (None, "", 0))
    return round(filled / len(keys) * 100)


# ------------------------------------------------------------------ goals
def get_goals(user_id: int = DEFAULT_USER_ID) -> dict:
    with get_conn() as c:
        return dict(c.execute("SELECT * FROM goals WHERE user_id=?", (user_id,)).fetchone())


def save_goals(user_id, daily_calories, weekly_calories, weekly_workouts, daily_duration) -> None:
    with get_conn() as c:
        c.execute("""UPDATE goals SET daily_calorie_goal=?, weekly_calorie_goal=?,
                     weekly_workout_goal=?, daily_duration_goal=? WHERE user_id=?""",
                  (daily_calories, weekly_calories, weekly_workouts, daily_duration, user_id))
        c.execute("UPDATE users SET daily_calorie_goal=?, weekly_workout_goal=? WHERE id=?",
                  (daily_calories, weekly_workouts, user_id))


# ---------------------------------------------------------------- workouts
def add_workout(user_id, date, exercise_type, duration, heart_rate,
                body_temperature, calories_burned, intensity) -> int:
    with get_conn() as c:
        cur = c.execute("""INSERT INTO workouts (user_id, date, exercise_type, duration, heart_rate,
                           body_temperature, calories_burned, intensity) VALUES (?,?,?,?,?,?,?,?)""",
                        (user_id, str(date), exercise_type, duration, heart_rate,
                         body_temperature, calories_burned, intensity))
        return cur.lastrowid


def get_workouts(user_id: int = DEFAULT_USER_ID) -> pd.DataFrame:
    """All workouts, newest first.  `date` is a pandas datetime column."""
    with get_conn() as c:
        rows = c.execute("""SELECT id, date, exercise_type, duration, heart_rate, body_temperature,
                            calories_burned, intensity FROM workouts WHERE user_id=?
                            ORDER BY date DESC, id DESC""", (user_id,)).fetchall()
    df = pd.DataFrame([dict(r) for r in rows], columns=WORKOUT_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    return df


def delete_workout(workout_id: int, user_id: int = DEFAULT_USER_ID) -> None:
    with get_conn() as c:
        c.execute("DELETE FROM workouts WHERE id=? AND user_id=?", (workout_id, user_id))


def reset_history(user_id: int = DEFAULT_USER_ID) -> None:
    """Delete all workouts and achievements (profile and goals are kept)."""
    with get_conn() as c:
        c.execute("DELETE FROM workouts WHERE user_id=?", (user_id,))
        c.execute("DELETE FROM achievements WHERE user_id=?", (user_id,))


# ----------------------------------------------------------- achievements
def get_achievements(user_id: int = DEFAULT_USER_ID) -> pd.DataFrame:
    with get_conn() as c:
        rows = c.execute("""SELECT achievement_name, description, unlocked_date FROM achievements
                            WHERE user_id=? ORDER BY unlocked_date""", (user_id,)).fetchall()
    return pd.DataFrame([dict(r) for r in rows],
                        columns=["achievement_name", "description", "unlocked_date"])


def unlock_achievement(user_id, name, description, date) -> bool:
    """Insert the achievement once.  Returns True if it was newly unlocked."""
    with get_conn() as c:
        if c.execute("SELECT 1 FROM achievements WHERE user_id=? AND achievement_name=?",
                     (user_id, name)).fetchone():
            return False
        c.execute("""INSERT INTO achievements (user_id, achievement_name, description, unlocked_date)
                     VALUES (?,?,?,?)""", (user_id, name, description, str(date)))
        return True


# --------------------------------------------------------------- settings
def get_setting(key: str, default=None):
    with get_conn() as c:
        row = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    with get_conn() as c:
        c.execute("INSERT INTO settings (key, value) VALUES (?,?) "
                  "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
