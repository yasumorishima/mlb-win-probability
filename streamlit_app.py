"""
MLB Win Probability Dashboard

Dark theme UI — bilingual (EN / JA).
Standalone — no CSV data needed, pure math engine.
"""

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from datetime import datetime, timezone

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
from live_feed import get_todays_games, get_live_state


st.set_page_config(
    page_title="MLB Win Probability",
    page_icon="&#9918;",
    layout="wide",
)


# ============================================================
# Language selector
# ============================================================

lang = st.sidebar.radio("Language / 言語", ["EN", "JA"], horizontal=True)

T = {
    "EN": {
        "title": "MLB Win Probability Engine",
        "subtitle": "Markov Chain + RE24 approach — real-time WP, LI, and tactical analysis",
        "live_games": "Live Games",
        "fetch_games": "Fetch Today's Games",
        "clear_live": "Clear",
        "load_live_state": "Load Live State",
        "auto_refresh": "Auto-refresh (30s)",
        "next_refresh": "Next refresh in {s}s",
        "no_live_state": "Could not fetch live state. Game may not be active yet.",
        "no_games": "No games found for today.",
        "apply": "Apply to Analysis ↓",
        "game_state": "Game State",
        "inning": "Inning",
        "half": "Half",
        "half_top": "Top",
        "half_bottom": "Bottom",
        "outs_label": "Outs",
        "score_diff": "Score (Home − Away)",
        "environment": "Environment",
        "runner_1b": "Runner on 1B",
        "runner_2b": "Runner on 2B",
        "runner_3b": "Runner on 3B",
        "presets": "Presets",
        "matchup_adj": "Matchup Adjustment (optional)",
        "batter_ops": "Batter OPS",
        "pitcher_era": "Pitcher ERA",
        "batter_ops_help": "Leave 0 to skip adjustment",
        "pitcher_era_help": "Leave 0 to skip adjustment",
        "diamond": "Diamond",
        "home_leads": "Home leads by {n}",
        "away_leads": "Away leads by {n}",
        "tie": "Tie game",
        "wp_home": "Win Probability (Home)",
        "matchup_adj_label": "Matchup-adjusted: {pct:.1f}%",
        "leverage_index": "Leverage Index",
        "li_desc": "LI 1.0 = average | 2.0+ = high pressure | 4.0+ = critical",
        "li_labels": {"Low": "Low", "Medium": "Medium", "High": "High", "Very High": "Very High"},
        "tactical": "Tactical Recommendations",
        "no_tactics": "No specific tactical recommendations for this situation.",
        "rec_labels": {
            "Recommended": "Recommended",
            "Consider": "Consider",
            "Neutral": "Neutral",
            "Not recommended": "Not recommended",
        },
        "whatif": "What-If: Next Play",
        "whatif_sub": "What happens to Win Probability if the next play is...",
        "whatif_outcomes": {
            "Strikeout": "Strikeout",
            "Single": "Single",
            "Double": "Double",
            "Home Run": "Home Run",
            "Walk": "Walk",
            "Ground Out": "Ground Out",
        },
        "re24_title": "RE24 Reference Table",
        "re24_sub": "Run Expectancy by 24 base-out states ({rpg:.1f} R/G environment)",
        "footer": "MLB Win Probability Engine | Markov Chain + RE24 Approach",
        "top_inning": "Top",
        "bot_inning": "Bot",
        "out_unit": "out",
        "scenario_name_key": "name",
    },
    "JA": {
        "title": "MLB 勝利確率エンジン",
        "subtitle": "マルコフ連鎖 + RE24 方式 — リアルタイム WP・LI・作戦提案",
        "live_games": "ライブゲーム",
        "fetch_games": "今日の試合を取得",
        "clear_live": "クリア",
        "load_live_state": "ライブ状態を読み込む",
        "auto_refresh": "30秒ごとに自動更新",
        "next_refresh": "次の更新まで {s}秒",
        "no_live_state": "ライブ状態を取得できませんでした。試合がまだ開始されていない可能性があります。",
        "no_games": "今日の試合は見つかりませんでした。",
        "apply": "↓ 分析に反映",
        "game_state": "ゲーム状態",
        "inning": "イニング",
        "half": "表/裏",
        "half_top": "表",
        "half_bottom": "裏",
        "outs_label": "アウト数",
        "score_diff": "得点差（ホーム − アウェイ）",
        "environment": "環境",
        "runner_1b": "1塁にランナーあり",
        "runner_2b": "2塁にランナーあり",
        "runner_3b": "3塁にランナーあり",
        "presets": "プリセット",
        "matchup_adj": "マッチアップ補正（任意）",
        "batter_ops": "打者 OPS",
        "pitcher_era": "投手 ERA",
        "batter_ops_help": "0 のままにすると補正なし",
        "pitcher_era_help": "0 のままにすると補正なし",
        "diamond": "ダイヤモンド",
        "home_leads": "ホーム {n} 点リード",
        "away_leads": "アウェイ {n} 点リード",
        "tie": "同点",
        "wp_home": "勝利確率（ホーム）",
        "matchup_adj_label": "マッチアップ補正後: {pct:.1f}%",
        "leverage_index": "レバレッジ指数",
        "li_desc": "LI 1.0 = 平均 | 2.0+ = 高プレッシャー | 4.0+ = 超高",
        "li_labels": {"Low": "低", "Medium": "中", "High": "高", "Very High": "超高"},
        "tactical": "作戦提案",
        "no_tactics": "この状況では特定の作戦提案はありません。",
        "rec_labels": {
            "Recommended": "推奨",
            "Consider": "検討",
            "Neutral": "中立",
            "Not recommended": "非推奨",
        },
        "whatif": "仮説分析：次のプレイ",
        "whatif_sub": "次のプレイが以下の場合、勝利確率はどう変わるか",
        "whatif_outcomes": {
            "Strikeout": "三振",
            "Single": "シングル",
            "Double": "二塁打",
            "Home Run": "本塁打",
            "Walk": "四球",
            "Ground Out": "ゴロアウト",
        },
        "re24_title": "RE24 参照テーブル",
        "re24_sub": "24 塁上アウト状態別期待得点（{rpg:.1f} R/G 環境）",
        "footer": "MLB 勝利確率エンジン | マルコフ連鎖 + RE24 方式",
        "top_inning": "表",
        "bot_inning": "裏",
        "out_unit": "アウト",
        "scenario_name_key": "name_ja",
    },
}

_ = T[lang]  # shorthand


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

st.markdown(f"# {_['title']}")
st.markdown(f"*{_['subtitle']}*")
st.markdown("---")


# ============================================================
# Live Game Section
# ============================================================

st.markdown(f"## {_['live_games']}")

_live_state_loaded = None

with st.container():
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    live_col1, live_col2 = st.columns([3, 1])

    with live_col1:
        if st.button(_["fetch_games"], key="fetch_games", use_container_width=True):
            with st.spinner("Fetching schedule from MLB Stats API..."):
                games = get_todays_games()
            st.session_state["todays_games"] = games
            st.session_state["live_last_fetched"] = today_str

    with live_col2:
        if st.button(_["clear_live"], key="clear_live", use_container_width=True):
            for k in ("todays_games", "live_game_pk", "live_state_cache", "live_last_fetched"):
                st.session_state.pop(k, None)
            st.rerun()

    games_list = st.session_state.get("todays_games", [])
    if games_list:
        def _game_label(g: dict) -> str:
            status_icon = "🔴" if g["status"] == "In Progress" else "⚪" if g["status"] == "Final" else "🕒"
            return f"{status_icon} {g['away_team']} @ {g['home_team']}  [{g['status']}]"

        game_options = {g["gamePk"]: _game_label(g) for g in games_list}
        if lang == "JA":
            select_label = f"試合を選択（{today_str}、{len(games_list)}試合）"
        else:
            select_label = f"Select game ({len(games_list)} games on {today_str})"

        selected_pk = st.selectbox(
            select_label,
            options=list(game_options.keys()),
            format_func=lambda pk: game_options[pk],
            key="live_game_select",
        )

        load_col, refresh_col = st.columns([2, 1])
        with load_col:
            load_live = st.button(_["load_live_state"], key="load_live", use_container_width=True)
        with refresh_col:
            auto_refresh = st.checkbox(_["auto_refresh"], key="auto_refresh")

        if load_live:
            with st.spinner("Fetching live game state..."):
                live_state = get_live_state(selected_pk)
            if live_state:
                st.session_state["live_state_cache"] = live_state
                st.session_state["live_game_pk"] = selected_pk
            else:
                st.warning(_["no_live_state"])

        # Auto-refresh logic
        if auto_refresh and st.session_state.get("live_game_pk"):
            import time
            if "live_last_refresh" not in st.session_state:
                st.session_state["live_last_refresh"] = time.time()
            elapsed = time.time() - st.session_state.get("live_last_refresh", 0)
            if elapsed >= 30:
                live_state = get_live_state(st.session_state["live_game_pk"])
                if live_state:
                    st.session_state["live_state_cache"] = live_state
                st.session_state["live_last_refresh"] = time.time()
                st.rerun()
            else:
                remaining = int(30 - elapsed)
                st.caption(_["next_refresh"].format(s=remaining))

        # Display cached live state
        cached = st.session_state.get("live_state_cache")
        if cached:
            inning_half = _["top_inning"] if cached["top_bottom"] == "top" else _["bot_inning"]
            runners_str = (
                ("1B " if cached["runners"][0] else "")
                + ("2B " if cached["runners"][1] else "")
                + ("3B" if cached["runners"][2] else "")
                or "---"
            )
            st.markdown(f"""
            <div style="background: #141428; border: 1px solid #00e5ff44; border-radius: 10px; padding: 14px; margin: 8px 0;">
                <span style="color: #00e5ff; font-weight: bold; font-size: 1.1rem;">
                    {cached['away_team']} @ {cached['home_team']}
                </span>
                &nbsp;
                <span style="color: #aaa; font-size: 0.9rem;">{cached['status']}</span>
                <br>
                <span style="color: #e0e0e0; font-size: 1.2rem; font-weight: bold;">
                    Away {cached['score_away']} — Home {cached['score_home']}
                </span>
                &nbsp;&nbsp;
                <span style="color: #888; font-size: 0.9rem;">
                    {inning_half} {cached['inning']}
                    | {cached['outs']} {_['out_unit']}
                    | {runners_str}
                </span>
                <br>
                <span style="color: #ccc; font-size: 0.85rem;">
                    🥎 {cached['batter_name'] or '—'} &nbsp; ⚾ {cached['pitcher_name'] or '—'}
                </span>
            </div>
            """, unsafe_allow_html=True)

            if st.button(_["apply"], key="apply_live", use_container_width=True):
                st.session_state["_preset"] = {
                    "inning": cached["inning"],
                    "top_bottom": cached["top_bottom"],
                    "outs": cached["outs"],
                    "runners": cached["runners"],
                    "score_diff": cached["score_diff"],
                }
                st.rerun()

    elif not games_list and st.session_state.get("live_last_fetched"):
        st.info(_["no_games"])

st.markdown("---")


# ============================================================
# Game State Input
# ============================================================

st.markdown(f"## {_['game_state']}")

col_settings, col_preset = st.columns([3, 2])

with col_settings:
    c1, c2, c3 = st.columns(3)
    with c1:
        inning = st.selectbox(_["inning"], list(range(1, 13)), index=0)
    with c2:
        top_bottom = st.selectbox(
            _["half"],
            ["top", "bottom"],
            format_func=lambda x: _["half_top"] if x == "top" else _["half_bottom"],
        )
    with c3:
        outs = st.selectbox(_["outs_label"], [0, 1, 2])

    c4, c5 = st.columns(2)
    with c4:
        score_diff = st.slider(_["score_diff"], -10, 10, 0)
    with c5:
        rpg = st.selectbox(
            _["environment"],
            [("MLB (4.5 R/G)", MLB_RPG), ("NPB (4.0 R/G)", NPB_RPG)],
            format_func=lambda x: x[0],
        )[1]

    r1, r2, r3_col = st.columns(3)
    with r1:
        runner1 = st.checkbox(_["runner_1b"])
    with r2:
        runner2 = st.checkbox(_["runner_2b"])
    with r3_col:
        runner3 = st.checkbox(_["runner_3b"])

with col_preset:
    st.markdown(f"### {_['presets']}")
    scenario_name_key = _["scenario_name_key"]
    preset_selected = None
    for key, scenario in SCENARIOS.items():
        label = scenario[scenario_name_key]
        if st.button(label, key=f"preset_{key}", use_container_width=True):
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

with st.expander(_["matchup_adj"]):
    mc1, mc2 = st.columns(2)
    with mc1:
        batter_ops_input = st.number_input(
            _["batter_ops"], min_value=0.0, max_value=2.0,
            value=0.0, step=0.01, format="%.3f",
            help=_["batter_ops_help"],
        )
    with mc2:
        pitcher_era_input = st.number_input(
            _["pitcher_era"], min_value=0.0, max_value=15.0,
            value=0.0, step=0.1, format="%.2f",
            help=_["pitcher_era_help"],
        )
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
    st.markdown(f"### {_['diamond']}")
    r1_color = "#00e5ff" if runner1 else "#333"
    r2_color = "#00e5ff" if runner2 else "#333"
    r3_color = "#00e5ff" if runner3 else "#333"
    inning_label = f"{'Top' if top_bottom == 'top' else 'Bot'} {inning} | {outs} out"
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
        <text x="70" y="138" fill="#666" font-size="10" text-anchor="middle">{inning_label}</text>
    </svg>
    """
    components.html(diamond_html, height=145)

    if score_diff > 0:
        st.markdown(f"**{_['home_leads'].format(n=score_diff)}**")
    elif score_diff < 0:
        st.markdown(f"**{_['away_leads'].format(n=abs(score_diff))}**")
    else:
        st.markdown(f"**{_['tie']}**")

with col_gauge:
    st.markdown(f"### {_['wp_home']}")
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
        st.markdown(f"**{_['matchup_adj_label'].format(pct=adj * 100)}**")

with col_li:
    st.markdown(f"### {_['leverage_index']}")
    badge_class = {
        "Low": "li-low", "Medium": "li-medium",
        "High": "li-high", "Very High": "li-veryhigh",
    }.get(li_lbl, "li-medium")
    li_display = _["li_labels"].get(li_lbl, li_lbl)

    st.markdown(f"""
    <div class="metric-card" style="text-align: center;">
        <div class="metric-value">{li_val:.1f}</div>
        <div class="metric-label">{_['leverage_index']}</div>
        <div style="margin-top: 8px;">
            <span class="li-badge {badge_class}">{li_display}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-size: 0.75rem; color: #666; margin-top: 12px;">
    {_['li_desc']}
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Tactical Recommendations
# ============================================================

st.markdown("---")
st.markdown(f"## {_['tactical']}")

tactics = result.get("tactics", [])
if not tactics:
    st.info(_["no_tactics"])
else:
    for t in tactics:
        rec = t["recommendation"]
        css_class = {
            "Recommended": "tactic-recommended",
            "Consider": "tactic-neutral",
            "Neutral": "tactic-neutral",
            "Not recommended": "tactic-notrecommended",
        }.get(rec, "")

        rec_display = _["rec_labels"].get(rec, rec)
        tactic_name = t["tactic_ja"] if lang == "JA" else t["tactic"]
        rec_color = "#4caf50" if rec == "Recommended" else "#ff9800" if rec in ("Consider", "Neutral") else "#f44336"

        delta_str = f"RE24 delta: {t['re24_delta']:+.3f}" if t["re24_delta"] != 0 else ""
        sr_str = f"Success rate: {t.get('success_rate', 0):.0%}" if t.get("success_rate") else ""
        reason = t.get("reason", "")
        details = " | ".join(filter(None, [delta_str, sr_str, reason]))

        st.markdown(f"""
        <div class="tactic-card {css_class}">
            <strong>{tactic_name}</strong>
            &nbsp; <span style="color: {rec_color}">{rec_display}</span>
            <br><span style="color: #888; font-size: 0.85rem;">{details}</span>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# What-If Section
# ============================================================

st.markdown("---")
st.markdown(f"## {_['whatif']}")
st.markdown(f"*{_['whatif_sub']}*")

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
for i, (label_en, outcome) in enumerate(whatif_outcomes.items()):
    with cols[i]:
        new_outs = outcome["outs"]
        new_runners = outcome["runners"]
        runs_scored = outcome["runs"]

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
        label_display = _["whatif_outcomes"].get(label_en, label_en)

        st.markdown(f"""
        <div class="metric-card" style="text-align: center; padding: 12px;">
            <div style="font-size: 0.85rem; color: #a0a0b0;">{label_display}</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #00e5ff;">{new_wp * 100:.1f}%</div>
            <div style="font-size: 0.9rem; color: {wpa_color};">WPA: {wpa:+.1%}</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# RE24 Reference Table
# ============================================================

st.markdown("---")
with st.expander(_["re24_title"]):
    st.markdown(f"*{_['re24_sub'].format(rpg=rpg)}*")
    re24_data = get_re24_table(rpg)

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
st.markdown(f"""
<div style="text-align: center; color: #555; font-size: 0.8rem;">
    {_['footer']}<br>
    <a href="https://github.com/yasumorishima/mlb-win-probability" style="color: #00e5ff;">GitHub</a>
</div>
""", unsafe_allow_html=True)
