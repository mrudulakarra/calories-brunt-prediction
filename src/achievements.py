"""Gamification: achievement definitions, progress and automatic unlocking.

Each achievement has a `progress` function returning (current, target); it is
unlocked when current >= target.  Everything is computed from the real workout
history stored in SQLite.
"""
import pandas as pd

from src import analytics, database as db
from src.config import DEFAULT_USER_ID


def _total_cal(df, goals):
    return float(df["calories_burned"].sum()), 1000


ACHIEVEMENTS = [
    {"name": "First Workout", "icon": "🥇", "description": "Complete your first workout.",
     "progress": lambda df, g: (len(df), 1)},
    {"name": "1K Club", "icon": "🔥", "description": "Burn 1,000 total calories.",
     "progress": lambda df, g: (float(df["calories_burned"].sum()), 1000)},
    {"name": "5K Club", "icon": "🚀", "description": "Burn 5,000 total calories.",
     "progress": lambda df, g: (float(df["calories_burned"].sum()), 5000)},
    {"name": "10 Workouts", "icon": "💪", "description": "Complete 10 workouts.",
     "progress": lambda df, g: (len(df), 10)},
    {"name": "7 Day Streak", "icon": "📅", "description": "Workout for 7 consecutive days.",
     "progress": lambda df, g: (analytics.longest_streak(df), 7)},
    {"name": "Goal Crusher", "icon": "🎯", "description": "Reach your daily calorie goal in a single day.",
     "progress": lambda df, g: (analytics.best_day_calories(df), float(g["daily_calorie_goal"] or 1))},
    {"name": "Consistency King", "icon": "👑",
     "description": "Hit your weekly workout goal two weeks in a row.",
     "progress": lambda df, g: (analytics.weekly_goal_streak(df, g["weekly_workout_goal"]), 2)},
]


def get_progress(df: pd.DataFrame, goals: dict) -> dict:
    """{achievement_name: (current, target)} for every achievement."""
    return {a["name"]: a["progress"](df, goals) for a in ACHIEVEMENTS}


def check_and_unlock(user_id: int = DEFAULT_USER_ID) -> list:
    """Unlock every achievement whose condition is now met. Returns the new ones."""
    df, goals = db.get_workouts(user_id), db.get_goals(user_id)
    if df.empty:
        return []
    today = pd.Timestamp.today().date()
    progress = get_progress(df, goals)
    new = []
    for a in ACHIEVEMENTS:
        current, target = progress[a["name"]]
        if current >= target and db.unlock_achievement(user_id, a["name"], a["description"], today):
            new.append(a["name"])
    return new
