"""Look & feel: custom CSS (light/dark), HTML card helpers and Plotly styling."""
import html

import streamlit as st

THEMES = {
    "light": {"bg": "#F5F6FB", "card": "#FFFFFF", "text": "#1B1F3B", "muted": "#6B7280",
              "border": "#E6E8F0", "input": "#FFFFFF", "shadow": "0 4px 18px rgba(27,31,59,.07)",
              "plotly": "plotly_white"},
    "dark": {"bg": "#0E111C", "card": "#181C2E", "text": "#F2F3F8", "muted": "#9AA0B4",
             "border": "#2A2F47", "input": "#222740", "shadow": "0 4px 18px rgba(0,0,0,.35)",
             "plotly": "plotly_dark"},
}
BRAND = "linear-gradient(135deg,#FF7A3D 0%,#FF3D68 100%)"
COLORS = ["#FF5A3C", "#7C3AED", "#14B8A6", "#F59E0B", "#3B82F6", "#EC4899", "#84CC16", "#64748B", "#06B6D4"]


def inject_css(theme: str) -> None:
    t = THEMES.get(theme, THEMES["light"])
    st.markdown(f"""<style>
:root {{ --bg:{t['bg']}; --card:{t['card']}; --text:{t['text']}; --muted:{t['muted']};
  --border:{t['border']}; --input:{t['input']}; --shadow:{t['shadow']}; --brand:{BRAND}; }}
.stApp, [data-testid="stAppViewContainer"] {{ background: var(--bg); color: var(--text); }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: var(--card); border-right: 1px solid var(--border); }}
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px; }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp label, .stApp p, .stApp li,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stWidgetLabel"] p {{ color: var(--text); }}
.stApp [data-testid="stCaptionContainer"], .stApp small {{ color: var(--muted); }}
/* inputs */
div[data-baseweb="input"] > div, div[data-baseweb="base-input"], div[data-baseweb="select"] > div,
div[data-baseweb="textarea"] {{ background: var(--input) !important; border-color: var(--border) !important; border-radius: 10px !important; }}
.stApp input, .stApp textarea {{ color: var(--text) !important; -webkit-text-fill-color: var(--text) !important; background: transparent !important; }}
div[data-baseweb="select"] span, div[data-baseweb="select"] div {{ color: var(--text); }}
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li, div[data-baseweb="menu"] {{ background: var(--card) !important; color: var(--text) !important; }}
[data-testid="stNumberInput"] button {{ background: var(--input) !important; color: var(--text) !important; }}
/* buttons */
.stApp div.stButton > button, .stApp div.stFormSubmitButton > button, .stApp div.stDownloadButton > button {{
  background: var(--brand); color: #fff !important; border: 0; border-radius: 12px; font-weight: 600;
  padding: .55rem 1.3rem; box-shadow: 0 6px 16px rgba(255,61,104,.28); transition: transform .12s ease; }}
.stApp div.stButton > button:hover, .stApp div.stFormSubmitButton > button:hover,
.stApp div.stDownloadButton > button:hover {{ transform: translateY(-1px); color: #fff; }}
.stApp div.stButton > button p, .stApp div.stFormSubmitButton > button p, .stApp div.stDownloadButton > button p {{ color: #fff !important; }}
/* tabs / expanders / tables */
.stApp [data-baseweb="tab-list"] {{ gap: 6px; }}
.stApp [data-baseweb="tab"] p {{ color: var(--text); font-weight: 600; }}
.stApp [data-testid="stExpander"] {{ background: var(--card); border: 1px solid var(--border); border-radius: 14px; }}
.stApp [data-testid="stDataFrame"] {{ border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }}
/* sidebar navigation as pills */
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 4px; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{ padding: 9px 12px; border-radius: 12px; width: 100%; cursor: pointer; }}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: var(--bg); }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{ background: var(--brand); }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{ color: #fff !important; font-weight: 700; }}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{ display: none; }}
/* custom components */
.cb-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 18px; padding: 18px 20px;
  box-shadow: var(--shadow); height: 100%; }}
.cb-kpi-label {{ color: var(--muted); font-size: .85rem; font-weight: 600; }}
.cb-kpi-value {{ color: var(--text); font-size: 1.9rem; font-weight: 800; line-height: 1.2; margin-top: 4px; }}
.cb-kpi-value small {{ font-size: .95rem; color: var(--muted); font-weight: 600; }}
.cb-delta {{ font-size: .8rem; font-weight: 600; margin-top: 4px; color: var(--muted); }}
.cb-up {{ color: #16A34A; }} .cb-down {{ color: #EF4444; }}
.cb-hero {{ background: var(--brand); border-radius: 22px; padding: 26px 30px; color: #fff; margin-bottom: 18px; }}
.cb-hero h1 {{ color: #fff !important; margin: 0; font-size: 1.9rem; }}
.cb-hero p {{ color: rgba(255,255,255,.92) !important; margin: 4px 0 0 0; }}
.cb-title {{ font-size: 1.5rem; font-weight: 800; margin: 0 0 2px 0; color: var(--text); }}
.cb-sub {{ color: var(--muted); margin-bottom: 14px; }}
.cb-section {{ font-size: 1.1rem; font-weight: 700; margin: 18px 0 8px 0; color: var(--text); }}
.cb-result {{ background: var(--brand); border-radius: 22px; padding: 28px; color: #fff; text-align: center;
  box-shadow: 0 10px 28px rgba(255,61,104,.30); }}
.cb-result .lbl {{ font-size: 1rem; opacity: .95; font-weight: 600; }}
.cb-result .val {{ font-size: 3.6rem; font-weight: 900; line-height: 1.1; }}
.cb-result .unit {{ font-size: 1.3rem; font-weight: 600; opacity: .95; }}
.cb-pill {{ display: inline-block; background: var(--bg); border: 1px solid var(--border); border-radius: 999px;
  padding: 4px 12px; font-size: .82rem; font-weight: 600; margin: 2px 4px 2px 0; color: var(--text); }}
.cb-bar {{ background: var(--border); border-radius: 999px; height: 12px; overflow: hidden; }}
.cb-bar > div {{ height: 100%; border-radius: 999px; background: var(--brand); }}
.cb-row {{ display:flex; justify-content:space-between; font-weight:600; margin: 0 0 6px 0; color: var(--text); }}
.cb-badge {{ display:inline-block; padding: 3px 11px; border-radius: 999px; font-size: .8rem; font-weight: 700;
  background: rgba(255,90,60,.14); color: #FF5A3C; }}
.cb-badge.done {{ background: rgba(22,163,74,.15); color: #16A34A; }}
.cb-badge.almost {{ background: rgba(245,158,11,.17); color: #D97706; }}
.cb-ach {{ background: var(--card); border: 1px solid var(--border); border-radius: 18px; padding: 18px; height: 100%;
  box-shadow: var(--shadow); }}
.cb-ach.locked {{ opacity: .55; filter: grayscale(1); box-shadow: none; }}
.cb-ach .name {{ font-weight: 800; font-size: 1.05rem; color: var(--text); }}
.cb-ach .desc {{ color: var(--muted); font-size: .88rem; margin: 4px 0 8px 0; }}
.cb-ach .meta {{ font-size: .8rem; color: var(--muted); }}
.cb-insight {{ background: var(--card); border: 1px solid var(--border); border-left: 5px solid #FF5A3C;
  border-radius: 12px; padding: 12px 16px; margin: 8px 0; color: var(--text); }}
.cb-model {{ background: var(--card); border: 1px solid var(--border); border-radius: 18px; padding: 18px; height: 100%; }}
.cb-model.best {{ border: 2px solid #FF5A3C; box-shadow: 0 8px 24px rgba(255,90,60,.2); }}
.cb-model .mname {{ font-weight: 800; color: var(--text); }}
.cb-note {{ color: var(--muted); font-size: .82rem; }}
@media (max-width: 768px) {{
  .block-container {{ padding: 1rem .8rem 3rem; }}
  .cb-result .val {{ font-size: 2.6rem; }} .cb-kpi-value {{ font-size: 1.5rem; }}
  .cb-hero {{ padding: 18px; }} .cb-hero h1 {{ font-size: 1.4rem; }}
}}
</style>""", unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape(str(x))


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"<div class='cb-title'>{esc(title)}</div><div class='cb-sub'>{esc(subtitle)}</div>",
                unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f"<div class='cb-section'>{esc(title)}</div>", unsafe_allow_html=True)


def hero(greeting: str, subtitle: str) -> None:
    st.markdown(f"<div class='cb-hero'><h1>{esc(greeting)}</h1><p>{esc(subtitle)}</p></div>",
                unsafe_allow_html=True)


def kpi_card(icon: str, label: str, value: str, unit: str = "", delta: str = "", delta_dir: str = "") -> None:
    cls = {"up": "cb-up", "down": "cb-down"}.get(delta_dir, "")
    unit_html = f" <small>{esc(unit)}</small>" if unit else ""
    delta_html = f"<div class='cb-delta {cls}'>{esc(delta)}</div>" if delta else ""
    st.markdown(f"<div class='cb-card'><div class='cb-kpi-label'>{icon} {esc(label)}</div>"
                f"<div class='cb-kpi-value'>{esc(value)}{unit_html}</div>{delta_html}</div>",
                unsafe_allow_html=True)


def progress_bar(label: str, current: float, target: float, unit: str = "") -> float:
    """Draw a labelled progress bar; returns the completion percentage."""
    pct = 0.0 if not target else current / target * 100
    st.markdown(f"<div class='cb-row'><span>{esc(label)}</span><span>{current:,.0f} / {target:,.0f} {esc(unit)}"
                f"</span></div><div class='cb-bar'><div style='width:{min(pct, 100):.0f}%'></div></div>"
                f"<div class='cb-note' style='margin:4px 0 14px 0'>{pct:.0f}% complete</div>",
                unsafe_allow_html=True)
    return pct


def goal_status(pct: float, expected_pct: float = 50.0) -> tuple:
    """(badge_text, css_class).  expected_pct = progress you should have by now."""
    if pct >= 100:
        return "🎯 Goal Completed", "done"
    if pct >= 75:
        return "⚡ Almost There", "almost"
    if pct >= expected_pct:
        return "🔥 Goal On Track", ""
    return "💪 Keep Going", ""


def badge(text: str, cls: str = "") -> None:
    st.markdown(f"<span class='cb-badge {cls}'>{esc(text)}</span>", unsafe_allow_html=True)


def pills(items: list) -> None:
    st.markdown("".join(f"<span class='cb-pill'>{esc(i)}</span>" for i in items), unsafe_allow_html=True)


def insight(text: str) -> None:
    st.markdown(f"<div class='cb-insight'>{esc(text)}</div>", unsafe_allow_html=True)


def result_card(label: str, value: str, unit: str) -> None:
    st.markdown(f"<div class='cb-result'><div class='lbl'>{esc(label)}</div>"
                f"<div class='val'>{esc(value)}</div><div class='unit'>{esc(unit)}</div></div>",
                unsafe_allow_html=True)


def style_fig(fig, theme: str, height: int = 320):
    """Apply a consistent, theme-aware look to a Plotly figure."""
    t = THEMES.get(theme, THEMES["light"])
    fig.update_layout(template=t["plotly"], height=height, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font={"color": t["text"]},
                      margin={"l": 10, "r": 10, "t": 40, "b": 10}, colorway=COLORS,
                      legend={"orientation": "h", "y": -0.2})
    return fig


# Streamlit renamed `use_container_width` to `width="stretch"`; support both versions.
def show_plot(fig, key=None) -> None:
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_df(df, **kwargs) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)
