"""Calories Burnt Prediction AI - Streamlit dashboard.

Run with:  streamlit run app.py
"""
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
# pyrefly: ignore [missing-import]
import plotly.express as px
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go
import streamlit as st

from src import achievements as ach
from src import analytics, database as db, prediction, recommendations as rec, ui
from src.config import (ACTIVITY_LEVELS, ASSETS_DIR, DEFAULT_USER_ID, DL_HISTORY_PATH,
                        EXERCISE_TYPES, FEATURES, FITNESS_GOALS, ML_METRICS_PATH)

st.set_page_config(page_title="Calories Burnt Prediction AI", page_icon="🔥", layout="wide")
log = logging.getLogger("calories-ai")
UID = DEFAULT_USER_ID
PAGES = ["🏠 Dashboard", "🔥 Predict Calories", "📊 History", "🎯 Goals",
         "🏆 Achievements", "👤 Profile", "🤖 AI Insights", "⚙️ Settings"]


# ----------------------------------------------------------------- start-up
@st.cache_resource(show_spinner="Loading AI models...")
def get_artifacts(signature: tuple):
    """Load scaler + ML + DL models once.  `signature` changes when a model file
    changes, so re-training automatically refreshes the cache."""
    return prediction.load_artifacts()


def flash(kind: str, message: str) -> None:
    """Queue a message that is shown after the next rerun."""
    st.session_state.setdefault("flash", []).append((kind, message))


def show_flash() -> None:
    for kind, message in st.session_state.pop("flash", []):
        getattr(st, kind)(message)


def workouts_table(df: pd.DataFrame, with_id: bool = False) -> pd.DataFrame:
    out = pd.DataFrame({"Date": df["date"].dt.strftime("%Y-%m-%d"), "Workout": df["exercise_type"],
                        "Duration (min)": df["duration"].round(0).astype(int),
                        "Heart Rate (BPM)": df["heart_rate"].round(0).astype(int),
                        "Calories Burned (kcal)": df["calories_burned"].round(0).astype(int),
                        "Intensity": df["intensity"]})
    if with_id:
        out.insert(0, "ID", df["id"].astype(int))
    return out


def empty_state(text: str = "No workouts logged yet. Open **🔥 Predict Calories** to log your first one!") -> None:
    st.info(text)


# ---------------------------------------------------------------- Dashboard
def page_dashboard(ctx: dict) -> None:
    user, goals, wk, theme = ctx["user"], ctx["goals"], ctx["wk"], ctx["theme"]
    today = analytics.today_ts()
    hour = pd.Timestamp.now().hour
    greeting = "Good Morning" if hour < 12 else "Good Afternoon" if hour < 17 else "Good Evening"
    ui.hero(f"{greeting}, {user.get('name') or 'there'} 👋", "Your AI-powered fitness overview")
    if not user.get("name"):
        st.caption("Tip: add your name and details in the 👤 Profile page to personalise the dashboard.")

    t = analytics.day_summary(wk, today)
    y = analytics.day_summary(wk, today - pd.Timedelta(days=1))
    w = analytics.week_summary(wk, today)
    goal = float(goals["daily_calorie_goal"] or 0)
    pct = t["calories"] / goal * 100 if goal else 0.0

    if y["calories"] > 0:
        change = (t["calories"] - y["calories"]) / y["calories"] * 100
        delta, direction = f"{'↑' if change >= 0 else '↓'} {abs(change):.0f}% from yesterday", "up" if change >= 0 else "down"
    else:
        delta, direction = ("No workouts yesterday" if t["calories"] else "No workouts logged today"), ""

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ui.kpi_card("🔥", "Calories Burned Today", f"{t['calories']:,.0f}", "kcal", delta, direction)
    with c2:
        ui.kpi_card("🏃", "Workouts Completed", str(t["workouts"]), "today", f"{w['workouts']} this week")
    with c3:
        ui.kpi_card("🎯", "Goal Progress", f"{pct:.0f}%", "",
                    f"{max(goal - t['calories'], 0):,.0f} kcal to go" if goal else "Set a goal in 🎯 Goals")
    with c4:
        ui.kpi_card("⏱", "Total Workout Time", f"{t['minutes']:,.0f}", "min", f"{w['minutes']:,.0f} min this week")

    st.write("")
    left, right = st.columns([3, 2])
    with left:
        ui.section("Weekly Calories")
        series = analytics.week_daily_series(wk, today)
        series.loc[series["is_future"], "calories"] = None      # don't draw days that haven't happened
        fig = px.line(series, x="day", y="calories", markers=True)
        fig.update_traces(line={"width": 3, "color": "#FF5A3C"}, marker={"size": 9})
        fig.update_layout(xaxis_title=None, yaxis_title="kcal")
        ui.show_plot(ui.style_fig(fig, theme))
    with right:
        ui.section("Goal Progress")
        fig = go.Figure(go.Indicator(mode="gauge+number", value=round(pct), number={"suffix": "%"},
                                     gauge={"axis": {"range": [0, max(100, pct)]},
                                            "bar": {"color": "#FF5A3C"}}))
        ui.show_plot(ui.style_fig(fig, theme))

    left, right = st.columns([2, 3])
    with left:
        ui.section("Workout Activity (last 30 days)")
        recent = analytics.between(wk, today - pd.Timedelta(days=29), today)
        if recent.empty:
            st.caption("No workouts in the last 30 days.")
        else:
            counts = recent["exercise_type"].value_counts().rename_axis("Workout").reset_index(name="Workouts")
            fig = px.bar(counts, x="Workout", y="Workouts", color="Workout")
            fig.update_layout(showlegend=False, xaxis_title=None)
            ui.show_plot(ui.style_fig(fig, theme))
    with right:
        ui.section("Recent Activity")
        if wk.empty:
            empty_state()
        else:
            ui.show_df(workouts_table(wk.head(8)))


# ------------------------------------------------------------------ Predict
def page_predict(ctx: dict) -> None:
    user, goals, wk, theme, art = ctx["user"], ctx["goals"], ctx["wk"], ctx["theme"], ctx["art"]
    ui.page_header("🔥 Predict Calories", "Enter your workout details - a trained AI model estimates the calories you burned.")
    if art is None:
        st.warning("⚠️ Models are not trained yet.\n\nPlease run the training pipeline first: "
                   "`python src/train_ml.py` then `python src/train_dl.py`.")
        return
    for message in art["warnings"]:
        st.warning(message)
    ranges = art["ranges"]
    med = lambda col, fb: float(ranges.get(col, {}).get("median", fb))   # noqa: E731

    with st.form("predict_form"):
        c1, c2, c3, c4 = st.columns(4)
        gender = c1.selectbox("Gender", ["Male", "Female"],
                              index=1 if (user.get("gender") == "Female") else 0)
        age = c2.number_input("Age (years)", 0, 120, int(user.get("age") or 25), 1)
        height = c3.number_input("Height (cm)", 0.0, 300.0, float(user.get("height") or 170.0), 1.0)
        weight = c4.number_input("Weight (kg)", 0.0, 400.0, float(user.get("weight") or 70.0), 0.5)
        c5, c6, c7, c8 = st.columns(4)
        duration = c5.number_input("Workout Duration (min)", 0.0, 1000.0, float(round(med("DURATION", 20))), 1.0)
        heart_rate = c6.number_input("Heart Rate (BPM)", 0, 300, int(med("HEART_RATE", 100)), 1)
        body_temp = c7.number_input("Body Temperature (°C)", 30.0, 45.0, round(med("BODY_TEMP", 38.5), 1), 0.1,
                                    help="Body temperature at the end of the workout.")
        exercise = c8.selectbox("Exercise Type", EXERCISE_TYPES)
        workout_date = st.date_input("Workout date", value=date.today(), max_value=date.today())
        submitted = st.form_submit_button("🔥 Predict Calories")

    if submitted:
        errors = prediction.validate_inputs(age, height, weight, duration, heart_rate, body_temp)
        if errors:
            for e in errors:
                st.error(e)
        else:
            try:
                feats = prediction.make_feature_frame(gender, age, height, weight, duration, heart_rate, body_temp)
                res = prediction.predict_calories(art, feats, ctx["model_choice"])
                intensity = prediction.classify_intensity(heart_rate, age)
                met = prediction.estimate_met(res["calories"], weight, duration)
                daily_goal = float(goals["daily_calorie_goal"] or 0)
                day_total = analytics.day_summary(wk, workout_date)["calories"] + res["calories"]
                goal_pct = day_total / daily_goal * 100 if daily_goal else 0.0
                prev = float(wk["calories_burned"].iloc[0]) if not wk.empty else None
                st.session_state["last_prediction"] = {
                    "res": res, "intensity": intensity, "met": met, "goal_pct": goal_pct,
                    "inputs": {"date": workout_date, "exercise": exercise, "duration": duration,
                               "heart_rate": heart_rate, "body_temp": body_temp},
                    "insights": rec.prediction_insights(res["calories"], intensity, duration, goal_pct,
                                                        prev, user.get("fitness_goal"))}
            except Exception:  # noqa: BLE001
                log.exception("Prediction failed")
                st.error("Sorry, the prediction could not be completed. Please check your inputs and try again.")

    last = st.session_state.get("last_prediction")
    if not last:
        return
    res, inp = last["res"], last["inputs"]
    st.write("")
    left, right = st.columns([2, 3])
    with left:
        ui.result_card("Estimated Calories Burned", f"{res['calories']:,}", "kcal")
        st.write("")
        ui.pills([f"Workout Intensity: {last['intensity']}", f"Duration: {inp['duration']:g} min",
                  f"Heart Rate: {inp['heart_rate']:g} BPM"])
        extra = [f"Prediction quality: {res['quality']}", f"Model: {res['model_used']}",
                 f"≈ {last['met']:.1f} MET ({prediction.met_category(last['met'])})"]
        if res["typical_error"]:
            extra.append(f"Typical error: ±{res['typical_error']:.0f} kcal")
        ui.pills(extra)
    with right:
        g1, g2 = st.columns(2)
        with g1:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=round(last["goal_pct"]),
                                         number={"suffix": "%"}, title={"text": "Daily goal (incl. this workout)"},
                                         gauge={"axis": {"range": [0, max(100, last["goal_pct"])]},
                                                "bar": {"color": "#FF5A3C"}}))
            ui.show_plot(ui.style_fig(fig, theme, 260))
        with g2:
            preds = pd.DataFrame({"Model": list(res["all_predictions"]),
                                  "kcal": [round(v) for v in res["all_predictions"].values()]})
            fig = px.bar(preds, x="Model", y="kcal", text="kcal", color="Model", title="Estimate by model")
            fig.update_layout(showlegend=False, xaxis_title=None)
            ui.show_plot(ui.style_fig(fig, theme, 260))
    for w in res["warnings"]:
        st.warning(w)
    ui.section("AI Fitness Insights")
    for line in last["insights"]:
        ui.insight(line)
    st.caption("ℹ️ This is an **estimate** produced by a machine-learning model, not a medical measurement. "
               "It is not medical advice.")
    if st.button("💾 Save to workout history"):
        db.add_workout(UID, inp["date"], inp["exercise"], inp["duration"], inp["heart_rate"],
                       inp["body_temp"], res["calories"], last["intensity"])
        flash("success", f"Workout saved: {res['calories']} kcal.")
        for name in ach.check_and_unlock(UID):
            flash("success", f"🏆 Achievement unlocked: {name}!")
        st.session_state.pop("last_prediction", None)
        st.rerun()


# ------------------------------------------------------------------ History
def page_history(ctx: dict) -> None:
    wk, theme = ctx["wk"], ctx["theme"]
    ui.page_header("📊 Workout History", "Browse, filter and manage every workout you have saved.")
    if wk.empty:
        empty_state()
        return
    lo, hi = wk["date"].min().date(), wk["date"].max().date()
    c1, c2, c3 = st.columns(3)
    rng = c1.date_input("Date Range", value=(lo, hi), min_value=lo, max_value=hi)
    types = c2.multiselect("Workout Type", sorted(wk["exercise_type"].unique()),
                           default=sorted(wk["exercise_type"].unique()))
    levels = c3.multiselect("Intensity", ui_levels(wk), default=ui_levels(wk))
    start, end = (rng[0], rng[1]) if isinstance(rng, (tuple, list)) and len(rng) == 2 else (lo, hi)
    f = analytics.between(wk, start, end)
    f = f[f["exercise_type"].isin(types) & f["intensity"].isin(levels)]
    if f.empty:
        st.info("No workouts match these filters.")
        return

    m1, m2, m3 = st.columns(3)
    with m1:
        ui.kpi_card("📈", "Average Calories / Workout", f"{f['calories_burned'].mean():,.0f}", "kcal")
    with m2:
        ui.kpi_card("🏃", "Workouts", str(len(f)))
    with m3:
        ui.kpi_card("🔥", "Total Calories", f"{f['calories_burned'].sum():,.0f}", "kcal")
    st.write("")
    a, b = st.columns([3, 2])
    with a:
        ui.section("Calories Over Time")
        fig = px.line(f.sort_values("date"), x="date", y="calories_burned", markers=True,
                      hover_data=["exercise_type"])
        fig.update_traces(line={"color": "#FF5A3C"})
        fig.update_layout(xaxis_title=None, yaxis_title="kcal")
        ui.show_plot(ui.style_fig(fig, theme))
    with b:
        ui.section("Workout Distribution")
        dist = f["exercise_type"].value_counts().rename_axis("Workout").reset_index(name="Count")
        fig = px.pie(dist, names="Workout", values="Count", hole=0.55)
        ui.show_plot(ui.style_fig(fig, theme))
    ui.section("All Workouts")
    ui.show_df(workouts_table(f, with_id=True))

    with st.expander("🗑 Delete a record"):
        options = {int(r.id): f"#{int(r.id)} · {r.date:%Y-%m-%d} · {r.exercise_type} · {r.calories_burned:.0f} kcal"
                   for r in f.itertuples()}
        chosen = st.selectbox("Select a workout", list(options), format_func=options.get)
        confirm = st.checkbox("Yes, delete this workout permanently")
        if st.button("Delete workout"):
            if confirm:
                db.delete_workout(chosen, UID)
                ach_note = "Workout deleted."
                flash("success", ach_note)
                st.rerun()
            else:
                st.warning("Tick the confirmation box first.")


def ui_levels(df):
    from src.config import INTENSITY_LEVELS
    return [l for l in INTENSITY_LEVELS if l in set(df["intensity"])]


# -------------------------------------------------------------------- Goals
def page_goals(ctx: dict) -> None:
    goals, wk = ctx["goals"], ctx["wk"]
    ui.page_header("🎯 Goals", "Set targets and track how close you are.")
    today = analytics.today_ts()
    t, w = analytics.day_summary(wk, today), analytics.week_summary(wk, today)
    week_pace = (today.weekday() + 1) / 7 * 100      # share of the week that has passed
    rows = [("Daily Calorie Goal", t["calories"], goals["daily_calorie_goal"], "kcal", 40),
            ("Daily Workout Duration Goal", t["minutes"], goals["daily_duration_goal"], "min", 40),
            ("Weekly Workout Goal", w["workouts"], goals["weekly_workout_goal"], "workouts", week_pace),
            ("Weekly Calorie Goal", w["calories"], goals["weekly_calorie_goal"], "kcal", week_pace)]
    cols = st.columns(2)
    for i, (label, cur, target, unit, expected) in enumerate(rows):
        with cols[i % 2], st.container(border=True):
            pct = ui.progress_bar(label, cur, float(target or 0), unit)
            text, cls = ui.goal_status(pct, expected)
            ui.badge(text, cls)
    ui.section("Update Your Goals")
    with st.form("goals_form"):
        a, b = st.columns(2)
        daily = a.number_input("Daily Calorie Goal (kcal)", 0, 10000, int(goals["daily_calorie_goal"] or 600), 50)
        duration = b.number_input("Daily Workout Duration Goal (min)", 0, 600, int(goals["daily_duration_goal"] or 45), 5)
        weekly_w = a.number_input("Weekly Workout Goal (sessions)", 0, 21, int(goals["weekly_workout_goal"] or 4), 1)
        weekly_c = b.number_input("Weekly Calorie Goal (kcal)", 0, 50000, int(goals["weekly_calorie_goal"] or 3000), 100)
        if st.form_submit_button("💾 Save Goals"):
            if min(daily, duration, weekly_w, weekly_c) <= 0:
                st.error("Every goal must be greater than 0.")
            else:
                db.save_goals(UID, daily, weekly_c, weekly_w, duration)
                flash("success", "Goals updated.")
                st.rerun()


# ------------------------------------------------------------- Achievements
def page_achievements(ctx: dict) -> None:
    ui.page_header("🏆 Achievements", "Unlock badges as you build your fitness habit.")
    for name in ach.check_and_unlock(UID):
        st.success(f"🏆 Achievement unlocked: {name}!")
    unlocked = db.get_achievements(UID).set_index("achievement_name")
    progress = ach.get_progress(db.get_workouts(UID), db.get_goals(UID))
    total = len(ach.ACHIEVEMENTS)
    ui.progress_bar("Achievements unlocked", len(unlocked), total, "")
    cols = st.columns(3)
    for i, a in enumerate(ach.ACHIEVEMENTS):
        cur, target = progress[a["name"]]
        if a["name"] in unlocked.index:
            meta = f"Unlocked on {unlocked.loc[a['name'], 'unlocked_date']}"
            bar, cls = "", ""
        else:
            pct = min(cur / target * 100, 100) if target else 0
            meta = f"Locked · {cur:,.0f} / {target:,.0f}"
            bar = f"<div class='cb-bar' style='margin-top:8px'><div style='width:{pct:.0f}%'></div></div>"
            cls = "locked"
        icon = a["icon"] if not cls else "🔒"
        with cols[i % 3]:
            st.markdown(f"<div class='cb-ach {cls}'><div class='name'>{icon} {ui.esc(a['name'])}</div>"
                        f"<div class='desc'>{ui.esc(a['description'])}</div>"
                        f"<div class='meta'>{ui.esc(meta)}</div>{bar}</div>", unsafe_allow_html=True)
            st.write("")


# ------------------------------------------------------------------ Profile
def page_profile(ctx: dict) -> None:
    user = ctx["user"]
    ui.page_header("👤 Profile", "Your personal details help the app personalise predictions and goals.")
    ui.progress_bar("Profile Completion", db.profile_completion(user), 100, "%")
    with st.form("profile_form"):
        a, b = st.columns(2)
        name = a.text_input("Name", user.get("name") or "")
        age = b.number_input("Age", 0, 120, int(user.get("age") or 25), 1)
        gender = a.selectbox("Gender", ["Male", "Female"], index=1 if user.get("gender") == "Female" else 0)
        height = b.number_input("Height (cm)", 0.0, 300.0, float(user.get("height") or 170.0), 1.0)
        weight = a.number_input("Weight (kg)", 0.0, 400.0, float(user.get("weight") or 70.0), 0.5)
        goal = b.selectbox("Fitness Goal", FITNESS_GOALS,
                           index=FITNESS_GOALS.index(user["fitness_goal"]) if user.get("fitness_goal") in FITNESS_GOALS else 3)
        level = a.selectbox("Activity Level", ACTIVITY_LEVELS,
                            index=ACTIVITY_LEVELS.index(user["activity_level"]) if user.get("activity_level") in ACTIVITY_LEVELS else 1)
        daily = b.number_input("Daily Calorie Goal (kcal)", 0, 10000, int(user.get("daily_calorie_goal") or 600), 50)
        weekly = a.number_input("Weekly Workout Goal (sessions)", 0, 21, int(user.get("weekly_workout_goal") or 4), 1)
        if st.form_submit_button("💾 Save Profile"):
            errors = []
            if not name.strip():
                errors.append("Please enter your name.")
            errors += [e for e in prediction.validate_inputs(age, height, weight, 1, 100, 37)]
            if daily <= 0 or weekly <= 0:
                errors.append("Goals must be greater than 0.")
            if errors:
                for e in errors:
                    st.error(e)
            else:
                db.update_user(UID, name=name.strip(), age=int(age), gender=gender, height=height, weight=weight,
                               fitness_goal=goal, activity_level=level, daily_calorie_goal=daily,
                               weekly_workout_goal=int(weekly))
                flash("success", "Profile saved.")
                st.rerun()
    if user.get("height") and user.get("weight"):
        bmi = user["weight"] / (user["height"] / 100) ** 2
        st.caption(f"Your BMI is {bmi:.1f} (a general screening number only - not a diagnosis).")


# -------------------------------------------------------------- AI Insights
def page_insights(ctx: dict) -> None:
    wk, theme = ctx["wk"], ctx["theme"]
    ui.page_header("🤖 AI Insights", "Analytics generated from your own workout history and the trained models.")
    tab1, tab2 = st.tabs(["📈 My Insights", "🧠 AI Model Performance"])
    with tab1:
        if wk.empty:
            empty_state()
        else:
            s = analytics.overall_stats(wk)
            best = s["best_workout"]
            cards = [("🔥", "Total Calories Burned", f"{s['total_calories']:,.0f}", "kcal"),
                     ("📊", "Average Calories / Workout", f"{s['avg_calories']:,.0f}", "kcal"),
                     ("⏱", "Total Workout Time", f"{s['total_minutes']:,.0f}", "min"),
                     ("🏃", "Total Workouts", str(s["total_workouts"]), ""),
                     ("⚡", "Current Streak", str(s["current_streak"]), "days"),
                     ("🥇", "Best Workout", f"{best['calories_burned']:,.0f}", f"kcal · {best['exercise_type']}")]
            for row in (cards[:3], cards[3:]):
                for col, c in zip(st.columns(3), row):
                    with col:
                        ui.kpi_card(*c)
                st.write("")
            weekly, monthly = analytics.period_trend(wk, "W"), analytics.period_trend(wk, "M")
            a, b = st.columns(2)
            with a:
                ui.section("Weekly Trend")
                fig = px.line(weekly, x="period", y="calories", markers=True)
                fig.update_layout(xaxis_title=None, yaxis_title="kcal / week")
                ui.show_plot(ui.style_fig(fig, theme))
            with b:
                ui.section("Monthly Trend")
                fig = px.bar(monthly, x="period", y="calories")
                fig.update_layout(xaxis_title=None, yaxis_title="kcal / month")
                ui.show_plot(ui.style_fig(fig, theme))
            a, b = st.columns(2)
            with a:
                ui.section("Workout Frequency (by weekday)")
                order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                freq = wk["date"].dt.day_name().value_counts().reindex(order, fill_value=0)
                fig = px.bar(x=[d[:3] for d in order], y=freq.values, labels={"x": "", "y": "Workouts"})
                ui.show_plot(ui.style_fig(fig, theme))
            with b:
                ui.section("Calorie Trend (per workout)")
                ordered = wk.sort_values(["date", "id"]).reset_index(drop=True)
                ordered["Rolling avg (5)"] = ordered["calories_burned"].rolling(5, min_periods=1).mean()
                fig = px.line(ordered, y=["calories_burned", "Rolling avg (5)"], labels={"index": "Workout #", "value": "kcal"})
                ui.show_plot(ui.style_fig(fig, theme))
            ui.section("Average Heart Rate (per week)")
            fig = px.line(weekly, x="period", y="avg_hr", markers=True)
            fig.update_layout(xaxis_title=None, yaxis_title="BPM")
            ui.show_plot(ui.style_fig(fig, theme, 280))
            ui.section("AI-Generated Insights")
            for line in rec.history_insights(wk):
                ui.insight(line)
    with tab2:
        render_model_performance(ctx)


def render_model_performance(ctx: dict) -> None:
    theme, art = ctx["theme"], ctx["art"]
    metrics = prediction.load_all_metrics()
    if not metrics:
        st.warning("⚠️ Models are not trained yet.\n\nPlease run the training pipeline first.")
        return
    best = min(metrics, key=lambda n: metrics[n]["rmse"])
    order = [n for n in ["Random Forest", "Gradient Boosting", "Deep Neural Network", "Linear Regression", "XGBoost"] if n in metrics]
    for start in range(0, len(order), 3):
        for col, name in zip(st.columns(3), order[start:start + 3]):
            m = metrics[name]
            crown = " 🏆 Best" if name == best else ""
            with col:
                st.markdown(f"<div class='cb-model {'best' if name == best else ''}'><div class='mname'>{ui.esc(name)}{crown}</div>"
                            f"<div class='cb-kpi-value'>R²: {m['r2']:.4f}</div>"
                            f"<div class='cb-delta'>RMSE: {m['rmse']:.2f} kcal · MAE: {m['mae']:.2f} · MSE: {m['mse']:.1f}</div></div>",
                            unsafe_allow_html=True)
        st.write("")
    st.caption("Scores are measured on a held-out test set the models never saw during training. "
               "The best model (lowest RMSE) is used for predictions when 'Best model (auto)' is selected.")
    df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})
    a, b = st.columns(2)
    with a:
        fig = px.bar(df, x="Model", y="r2", color="Model", title="R² score (higher is better)")
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)
        ui.show_plot(ui.style_fig(fig, theme, 300))
    with b:
        fig = px.bar(df, x="Model", y="rmse", color="Model", title="RMSE in kcal (lower is better)")
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)
        ui.show_plot(ui.style_fig(fig, theme, 300))
    a, b = st.columns(2)
    with a:
        if DL_HISTORY_PATH.exists():
            hist = pd.read_json(DL_HISTORY_PATH)
            fig = px.line(hist, y=["loss", "val_loss"], title="Deep Neural Network: Training vs Validation Loss",
                          labels={"index": "Epoch", "value": "Loss (MSE)", "variable": ""})
            ui.show_plot(ui.style_fig(fig, theme, 300))
        else:
            st.info("Train the deep-learning model (`python src/train_dl.py`) to see its loss curves.")
    with b:
        rf = art["ml"].get("Random Forest") if art else None
        if rf is not None:
            imp = pd.DataFrame({"Feature": FEATURES, "Importance": rf.feature_importances_}).sort_values("Importance")
            fig = px.bar(imp, x="Importance", y="Feature", orientation="h", title="Random Forest: feature importance")
            ui.show_plot(ui.style_fig(fig, theme, 300))
    dl = ctx["art"]["dl_meta"] if art else {}
    if dl:
        st.caption(f"Deep Neural Network: trained for {dl.get('epochs_run', '?')} epochs (early stopping) · "
                   f"final training loss {dl.get('train_loss_final', float('nan')):.4f} · "
                   f"final validation loss {dl.get('val_loss_final', float('nan')):.4f}")


# ----------------------------------------------------------------- Settings
def page_settings(ctx: dict) -> None:
    wk, art = ctx["wk"], ctx["art"]
    ui.page_header("⚙️ Settings", "Appearance, model selection and data management.")
    st.markdown("**Appearance**")
    choice = st.radio("Theme", ["Light", "Dark"], index=1 if ctx["theme"] == "dark" else 0, horizontal=True)
    if choice.lower() != ctx["theme"]:
        db.set_setting("theme", choice.lower())
        st.rerun()
    st.markdown("**Prediction model**")
    options = [prediction.AUTO_CHOICE] + (prediction.available_models(art) if art else [])
    current = ctx["model_choice"] if ctx["model_choice"] in options else options[0]
    picked = st.selectbox("Model used for predictions", options, index=options.index(current))
    if picked != ctx["model_choice"]:
        db.set_setting("model_choice", picked)
        st.rerun()
    if art:
        st.caption(f"Best model right now: **{art['best']}**")
    st.markdown("**Your data**")
    st.download_button("⬇️ Export workout history (CSV)", workouts_table(wk).to_csv(index=False).encode("utf-8"),
                       file_name="workout_history.csv", mime="text/csv", disabled=wk.empty)
    with st.expander("Danger zone - delete all workouts and achievements"):
        typed = st.text_input("Type DELETE to confirm")
        if st.button("Delete all history"):
            if typed.strip() == "DELETE":
                db.reset_history(UID)
                flash("success", "All workout history and achievements were deleted.")
                st.rerun()
            else:
                st.warning("Type DELETE (in capitals) to confirm.")
    st.markdown("**System status**")
    missing = prediction.get_missing_artifacts()
    st.write("✅ All model files found." if not missing else f"⚠️ Missing model files: {', '.join(missing)}")


# --------------------------------------------------------------------- main
PAGE_FUNCS = {PAGES[0]: page_dashboard, PAGES[1]: page_predict, PAGES[2]: page_history, PAGES[3]: page_goals,
              PAGES[4]: page_achievements, PAGES[5]: page_profile, PAGES[6]: page_insights, PAGES[7]: page_settings}


def main() -> None:
    db.init_db()
    theme = db.get_setting("theme", "light")
    ui.inject_css(theme)

    logo = ASSETS_DIR / "logo.png"
    with st.sidebar:
        if logo.exists():
            st.image(str(logo), width=64)
        st.markdown("### Calories Burnt Prediction AI")
        st.caption("AI-Powered Personal Fitness & Calorie Intelligence Platform")
        page = st.radio("Navigation", PAGES, label_visibility="collapsed")
        st.caption("Predictions are estimates, not medical measurements.")

    art = get_artifacts(prediction.artifacts_signature())
    ctx = {"user": db.get_user(UID), "goals": db.get_goals(UID), "wk": db.get_workouts(UID), "theme": theme,
           "art": art, "model_choice": db.get_setting("model_choice", prediction.AUTO_CHOICE)}
    show_flash()
    if art is None and page != PAGES[1]:
        st.warning("⚠️ Models are not trained yet.\n\nPlease run the training pipeline first: "
                   "`python src/train_ml.py` then `python src/train_dl.py`.")
    try:
        PAGE_FUNCS[page](ctx)
    except Exception:  # noqa: BLE001 - never show a traceback to normal users
        log.exception("Page failed: %s", page)
        st.error("Something went wrong while loading this page. Please try again or choose another page.")


main()
