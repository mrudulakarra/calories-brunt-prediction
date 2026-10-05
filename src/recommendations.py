"""Rule-based "AI" insights and recommendations.

These are transparent if/else rules computed from the model output and the
user's real history - easy to explain in a viva.  They are fitness tips, not
medical advice.
"""
import pandas as pd

from src import analytics


def prediction_insights(calories, intensity, duration, goal_progress_pct, previous_calories,
                        fitness_goal=None) -> list:
    """Messages shown right after a prediction."""
    msgs = [f"🔥 You burned approximately {calories:,.0f} calories.",
            f"💪 Your workout intensity is {intensity.lower()}.",
            (f"🎯 You are {goal_progress_pct:.0f}% of the way to your daily calorie-burning goal."
             if goal_progress_pct < 100 else
             f"🎯 You would reach {goal_progress_pct:.0f}% of your daily goal - goal completed!")]
    if previous_calories and previous_calories > 0:
        change = (calories - previous_calories) / previous_calories * 100
        arrow = "📈" if change >= 0 else "📉"
        word = "increased" if change >= 0 else "decreased"
        msgs.append(f"{arrow} Your calorie burn {word} by {abs(change):.0f}% compared with your previous workout.")
    msgs.append("💡 Recommendation: " + recommendation(intensity, duration, goal_progress_pct, fitness_goal))
    return msgs


def recommendation(intensity, duration, goal_progress_pct, fitness_goal=None) -> str:
    if goal_progress_pct >= 100:
        return "Great job reaching today's goal - prioritise recovery, hydration and sleep."
    if intensity == "Light":
        base = "Try raising the pace or adding short intervals to move into the moderate zone."
    elif intensity == "Moderate":
        base = "Maintain your current workout duration and gradually increase intensity."
    elif intensity == "Vigorous":
        base = "Strong effort! Balance hard sessions with easier days so your body can recover."
    else:
        base = "This was a very intense effort - make sure you rest and rehydrate before the next hard session."
    if duration < 20 and intensity in ("Light", "Moderate"):
        base += " Extending the session towards 30 minutes would also help."
    if fitness_goal == "Weight Loss" and intensity != "Max Effort":
        base += " For weight loss, consistency across the week matters more than any single session."
    elif fitness_goal == "Endurance":
        base += " For endurance, gradually extend your longer sessions."
    return base


def history_insights(df: pd.DataFrame) -> list:
    """Insights generated from the user's actual stored workouts."""
    if df.empty:
        return ["Log your first workout to unlock personalised insights."]
    msgs = []
    # 1) Trend: last 5 workouts vs the 5 before them
    ordered = df.sort_values(["date", "id"])
    if len(ordered) >= 4:
        n = min(5, len(ordered) // 2)
        recent = ordered["calories_burned"].tail(n).mean()
        earlier = ordered["calories_burned"].iloc[-2 * n:-n].mean()
        if earlier > 0:
            change = (recent - earlier) / earlier * 100
            if change > 3:
                msgs.append(f"📈 Your average calorie burn is increasing (+{change:.0f}% over your last {n} workouts).")
            elif change < -3:
                msgs.append(f"📉 Your average calorie burn has dipped ({change:.0f}% over your last {n} workouts).")
            else:
                msgs.append("➖ Your calorie burn per workout is steady.")
    else:
        msgs.append("📊 Log a few more workouts to see calorie-burn trends.")
    # 2) Most effective workout type = highest calories per minute
    typed = df.groupby("exercise_type").agg(c=("calories_burned", "sum"), m=("duration", "sum"))
    typed = typed[typed["m"] > 0]
    if not typed.empty:
        rate = (typed["c"] / typed["m"]).sort_values(ascending=False)
        msgs.append(f"🏅 Your most effective workout type is {rate.index[0]} "
                    f"(about {rate.iloc[0]:.1f} kcal per minute).")
    # 3) Duration
    msgs.append(f"⏱ Your average workout duration is {df['duration'].mean():.0f} minutes.")
    # 4) Consistency by weekday
    if len(df) >= 3:
        days = df["date"].dt.day_name().value_counts()
        weekday_share = df["date"].dt.weekday.lt(5).mean()
        label = "weekdays" if weekday_share >= 0.5 else "weekends"
        msgs.append(f"📅 You are most consistent on {label}; {days.index[0]} is your most frequent workout day.")
    # 5) Streak
    streak = analytics.current_streak(df)
    msgs.append(f"🔥 Your current streak is {streak} day{'s' if streak != 1 else ''}."
                if streak else "💤 You have no active streak - a workout today starts a new one!")
    # 6) Heart rate
    msgs.append(f"❤️ Your average workout heart rate is {df['heart_rate'].mean():.0f} BPM.")
    return msgs
