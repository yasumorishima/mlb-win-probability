"""
MLB Win Probability Dashboard

Dark theme + neon cyan RPG-style UI.
Standalone — no CSV data needed, pure math engine.
"""

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

from win_probability import (
    MLB_RPG,
    NPB_RPG,
    SCENARIOS,
    full_analysis,
    get_re24_table,
    calculate_wp,
    calculate_li,
    li_label,
)


st.set_page_config(
    page_title="MLB Win Probability",
    page_icon="&#9918;",
    layout="wide",
)


# ============================================================
# Custom CSS
# ============================================================

st.markdown("""
<style>
    .stApp { background-color: #0a0a1a; }
    h1, h2, h3 { color: #00e5ff !important; }
    .metric-card {
        background: linear-gradient(135deg, #141428 0%, #1a1a3e 100%);
        border: 1px solid #00e5ff33;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00e5ff;
        text-shadow: 0 0 10px rgba(0, 229, 255, 0.5);
    }
    .metric-label {
        font-size: 0.9rem;
        color: #a0a0b0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .li-badge {
        display: inline-block;
        padding: 4px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .li-low { background: #1a3a1a; color: #4caf50; border: 1px solid #4caf50; }
    .li-medium { background: #3a3a1a; color: #ff9800; border: 1px solid #ff9800; }
    .li-high { background: #3a1a1a; color: #f44336; border: 1px solid #f44336; }
    .li-veryhigh { background: #4a0a0a; color: #ff1744; border: 1px solid #ff1744;
                   box-shadow: 0 0 10px rgba(255, 23, 68, 0.4); }
    .tactic-card {
        background: #141428;
        border-left: 4px solid #00e5ff;
        padding: 12px 16px;
        margin: 6px 0;
        border-radius: 0 8px 8px 0;
    }
    .tactic-recommended { border-left-color: #4caf50; }
    .tactic-neutral { border-left-color: #ff9800; }
    .tactic-notrecommended { border-left-color: #f44336; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Title
# ============================================================

st.markdown("# MLB Win Probability Engine")
st.markdown("*Markov Chain + RE24 approach — real-time WP, LI, and tactical analysis*")
st.markdown("---")


# ============================================================
# Game State Input
# ============================================================

st.markdown("## Game State")

col_settings, col_preset = st.columns([3, 2])

with col_settings:
    c1, c2, c3 = st.columns(3)
    with c1:
        inning = st.selectbox("Inning", list(range(1, 13)), index=0)
    with c2:
        top_bottom = st.selectbox("Half", ["top", "bottom"], format_func=lambda x: "Top" if x == "top" else "Bottom")
    with c3:
        outs = st.selectbox("Outs", [0, 1, 2])

    c4, c5 = st.columns(2)
    with c4:
        score_diff = st.slider("Score (Home - Away)", -10, 10, 0)
    with c5:
        rpg = st.selectbox("Environment", [("MLB (4.5 R/G)", MLB_RPG), ("NPB (4.0 R/G)", NPB_RPG)],
                           format_func=lambda x: x[0])[1]

    r1, r2, r3_col = st.columns(3)
    with r1:
        runner1 = st.checkbox("Runner on 1B")
    with r2:
        runner2 = st.checkbox("Runner on 2B")
    with r3_col:
        runner3 = st.checkbox("Runner on 3B")

with col_preset:
    st.markdown("### Presets")
    preset_selected = None
    for key, scenario in SCENARIOS.items():
        if st.button(f"{scenario['name_ja']}", key=f"preset_{key}", use_container_width=True):
            preset_selected = key

# Apply preset if selected
if preset_selected:
    s = SCENARIOS[preset_selected]["state"]
    st.session_state["_preset"] = {
        "inning": s.inning,
        "top_bottom": s.top_bottom,
        "outs": s.outs,
        "runners": s.runners,
        "score_diff": s.score_diff,
    }
    st.rerun()

# Override from session state if preset was just applied
if "_preset" in st.session_state:
    p = st.session_state.pop("_preset")
    inning = p["inning"]
    top_bottom = p["top_bottom"]
    outs = p["outs"]
    runner1 = bool(p["runners"][0])
    runner2 = bool(p["runners"][1])
    runner3 = bool(p["runners"][2])
    score_diff = p["score_diff"]

runners = (int(runner1), int(runner2), int(runner3))


# ============================================================
# Optional: Matchup adjustment
# ============================================================

with st.expander("Matchup Adjustment (optional)"):
    mc1, mc2 = st.columns(2)
    with mc1:
        batter_ops_input = st.number_input("Batter OPS", min_value=0.0, max_value=2.0,
                                            value=0.0, step=0.01, format="%.3f",
                                            help="Leave 0 to skip adjustment")
    with mc2:
        pitcher_era_input = st.number_input("Pitcher ERA", min_value=0.0, max_value=15.0,
                                             value=0.0, step=0.1, format="%.2f",
                                             help="Leave 0 to skip adjustment")
    batter_ops = batter_ops_input if batter_ops_input > 0 else None
    pitcher_era = pitcher_era_input if pitcher_era_input > 0 else None


# ============================================================
# Analysis
# ============================================================

result = full_analysis(inning, top_bottom, outs, runners, score_diff,
                       rpg, batter_ops, pitcher_era)

wp = result["win_probability"]
li_val = result["leverage_index"]
li_lbl = result["leverage_label"]

st.markdown("---")


# ============================================================
# Diamond SVG + WP Gauge + LI Badge
# ============================================================

col_diamond, col_gauge, col_li = st.columns([1, 2, 1])

with col_diamond:
    st.markdown("### Diamond")
    r1_color = "#00e5ff" if runner1 else "#333"
    r2_color = "#00e5ff" if runner2 else "#333"
    r3_color = "#00e5ff" if runner3 else "#333"
    diamond_html = f"""
    <svg width="140" height="140" viewBox="0 0 140 140">
        <rect x="0" y="0" width="140" height="140" fill="#0a0a1a"/>
        <!-- Field lines -->
        <line x1="70" y1="120" x2="120" y2="70" stroke="#444" stroke-width="2"/>
        <line x1="120" y1="70" x2="70" y2="20" stroke="#444" stroke-width="2"/>
        <line x1="70" y1="20" x2="20" y2="70" stroke="#444" stroke-width="2"/>
        <line x1="20" y1="70" x2="70" y2="120" stroke="#444" stroke-width="2"/>
        <!-- Home plate -->
        <polygon points="70,115 65,120 70,125 75,120" fill="#fff"/>
        <!-- Bases -->
        <rect x="113" y="63" width="14" height="14" rx="2" fill="{r1_color}" transform="rotate(45 120 70)"/>
        <rect x="63" y="13" width="14" height="14" rx="2" fill="{r2_color}" transform="rotate(45 70 20)"/>
        <rect x="13" y="63" width="14" height="14" rx="2" fill="{r3_color}" transform="rotate(45 20 70)"/>
        <!-- Labels -->
        <text x="70" y="138" fill="#666" font-size="10" text-anchor="middle">{'Top' if top_bottom == 'top' else 'Bot'} {inning} | {outs} out</text>
    </svg>
    """
    components.html(diamond_html, height=145)

    # Score display
    if score_diff > 0:
        st.markdown(f"**Home leads by {score_diff}**")
    elif score_diff < 0:
        st.markdown(f"**Away leads by {abs(score_diff)}**")
    else:
        st.markdown("**Tie game**")

with col_gauge:
    st.markdown("### Win Probability (Home)")
    gauge_color = "#4caf50" if wp > 0.55 else "#ff9800" if wp > 0.45 else "#f44336"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=wp * 100,
        number={"suffix": "%", "font": {"size": 48, "color": "#00e5ff"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#666"},
            "bar": {"color": gauge_color},
            "bgcolor": "#1a1a3e",
            "bordercolor": "#00e5ff33",
            "steps": [
                {"range": [0, 30], "color": "#1a0a0a"},
                {"range": [30, 50], "color": "#1a1a0a"},
                {"range": [50, 70], "color": "#0a1a0a"},
                {"range": [70, 100], "color": "#0a2a0a"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="#0a0a1a",
        plot_bgcolor="#0a0a1a",
        font={"color": "#e0e0e0"},
        height=250,
        margin=dict(t=30, b=10, l=30, r=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    if result.get("adjusted_wp") is not None:
        adj = result["adjusted_wp"]
        st.markdown(f"**Matchup-adjusted: {adj * 100:.1f}%**")

with col_li:
    st.markdown("### Leverage Index")
    badge_class = {
        "Low": "li-low", "Medium": "li-medium",
        "High": "li-high", "Very High": "li-veryhigh",
    }.get(li_lbl, "li-medium")

    st.markdown(f"""
    <div class="metric-card" style="text-align: center;">
        <div class="metric-value">{li_val:.1f}</div>
        <div class="metric-label">Leverage Index</div>
        <div style="margin-top: 8px;">
            <span class="li-badge {badge_class}">{li_lbl}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size: 0.75rem; color: #666; margin-top: 12px;">
    LI 1.0 = average | 2.0+ = high pressure | 4.0+ = critical
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Tactical Recommendations
# ============================================================

st.markdown("---")
st.markdown("## Tactical Recommendations")

tactics = result.get("tactics", [])
if not tactics:
    st.info("No specific tactical recommendations for this situation.")
else:
    for t in tactics:
        rec = t["recommendation"]
        css_class = {
            "Recommended": "tactic-recommended",
            "Consider": "tactic-neutral",
            "Neutral": "tactic-neutral",
            "Not recommended": "tactic-notrecommended",
        }.get(rec, "")

        delta_str = f"RE24 delta: {t['re24_delta']:+.3f}" if t['re24_delta'] != 0 else ""
        sr_str = f"Success rate: {t.get('success_rate', 0):.0%}" if t.get('success_rate') else ""
        reason = t.get('reason', '')

        details = " | ".join(filter(None, [delta_str, sr_str, reason]))

        st.markdown(f"""
        <div class="tactic-card {css_class}">
            <strong>{t['tactic']}</strong> ({t['tactic_ja']})
            &nbsp; <span style="color: {'#4caf50' if rec == 'Recommended' else '#ff9800' if rec in ('Consider', 'Neutral') else '#f44336'}">{rec}</span>
            <br><span style="color: #888; font-size: 0.85rem;">{details}</span>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# What-If Section
# ============================================================

st.markdown("---")
st.markdown("## What-If: Next Play")
st.markdown("*What happens to Win Probability if the next play is...*")

whatif_outcomes = {
    "Strikeout": {"runners": runners, "outs": min(outs + 1, 3), "runs": 0},
    "Single": {"runners": (1, runners[0], runners[1]), "outs": outs, "runs": runners[2]},
    "Double": {"runners": (0, 1, runners[0]), "outs": outs, "runs": runners[1] + runners[2]},
    "Home Run": {"runners": (0, 0, 0), "outs": outs, "runs": 1 + sum(runners)},
    "Walk": {
        "runners": (
            1,
            1 if runners[0] else runners[1],
            1 if (runners[0] and runners[1]) else runners[2],
        ),
        "outs": outs,
        "runs": 1 if (runners[0] and runners[1] and runners[2]) else 0,
    },
    "Ground Out": {"runners": (0, runners[1], runners[2]), "outs": min(outs + 1, 3), "runs": 0},
}

cols = st.columns(len(whatif_outcomes))
for i, (label, outcome) in enumerate(whatif_outcomes.items()):
    with cols[i]:
        new_outs = outcome["outs"]
        new_runners = outcome["runners"]
        runs_scored = outcome["runs"]

        # Calculate new state WP
        if new_outs >= 3:
            if top_bottom == "top":
                new_wp = calculate_wp(inning, "bottom", 0, (0, 0, 0),
                                      score_diff - runs_scored, rpg)
            else:
                new_wp = calculate_wp(inning + 1, "top", 0, (0, 0, 0),
                                      score_diff + runs_scored, rpg)
        else:
            if top_bottom == "top":
                new_wp = calculate_wp(inning, top_bottom, new_outs, new_runners,
                                      score_diff - runs_scored, rpg)
            else:
                new_wp = calculate_wp(inning, top_bottom, new_outs, new_runners,
                                      score_diff + runs_scored, rpg)

        wpa = new_wp - wp
        wpa_color = "#4caf50" if wpa > 0 else "#f44336" if wpa < 0 else "#888"

        st.markdown(f"""
        <div class="metric-card" style="text-align: center; padding: 12px;">
            <div style="font-size: 0.85rem; color: #a0a0b0;">{label}</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #00e5ff;">{new_wp * 100:.1f}%</div>
            <div style="font-size: 0.9rem; color: {wpa_color};">WPA: {wpa:+.1%}</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# RE24 Reference Table
# ============================================================

st.markdown("---")
with st.expander("RE24 Reference Table"):
    st.markdown(f"*Run Expectancy by 24 base-out states ({rpg:.1f} R/G environment)*")
    re24_data = get_re24_table(rpg)

    # Format as table
    header = "| Runners | 0 out | 1 out | 2 out |"
    sep = "|---------|-------|-------|-------|"
    rows = {}
    for entry in re24_data:
        key = entry["runners"]
        if key not in rows:
            rows[key] = {}
        rows[key][entry["outs"]] = entry["expected_runs"]

    lines = [header, sep]
    for key in ["---", "1--", "-2-", "12-", "--3", "1-3", "-23", "123"]:
        if key in rows:
            lines.append(f"| {key} | {rows[key].get(0, 0):.3f} | {rows[key].get(1, 0):.3f} | {rows[key].get(2, 0):.3f} |")

    st.markdown("\n".join(lines))


# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #555; font-size: 0.8rem;">
    MLB Win Probability Engine | Markov Chain + RE24 Approach<br>
    <a href="https://github.com/yasumorishima/mlb-win-probability" style="color: #00e5ff;">GitHub</a>
</div>
""", unsafe_allow_html=True)
