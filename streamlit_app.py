"""
MLB Win Probability Dashboard

Dark theme UI — bilingual (EN / JA).
Numbers come from the engine fitted to MLB plate-appearance transitions
(wp_engine.py + engine_tables.json, through analysis.py). The Claude bench
coach (strategist.py) runs only when ANTHROPIC_API_KEY is configured.
"""

import html
import json
import os
import threading
import time
from datetime import datetime, timezone

import plotly.graph_objects as go
import streamlit as st

from analysis import (
    SCENARIOS,
    card_note,
    engine_info,
    full_analysis,
    get_advice,
    has_api_key,
    re24_table,
    team_layer_wp,
    wp_after_play,
)
from live_feed import get_game_plays, get_live_state, get_todays_games, has_live_state

st.set_page_config(
    page_title="MLB Win Probability",
    page_icon="&#9918;",
    layout="wide",
)


# ============================================================
# Language selector
# ============================================================

lang = st.sidebar.radio("Language / 言語", ["EN", "JA"], horizontal=True)
MODES = ["⚾ WP Calculator", "🎬 Replay Mode"]
mode = st.sidebar.radio("Mode", MODES, horizontal=True, label_visibility="collapsed",
                        key="mode")

_info = engine_info()
_seasons = _info.get("train_seasons") or []
_season_span = f"{min(_seasons)}–{max(_seasons)}" if _seasons else "?"
_n_pa = _info.get("n_pa") or 0

T = {
    "EN": {
        "title": "MLB Win Probability Engine",
        "subtitle": (f"Fitted to {_n_pa:,} MLB plate appearances ({_season_span}) — "
                     "win probability, leverage, tactics, and a Claude bench coach"),
        "live_games": "Live Games",
        "fetch_games": "Fetch Today's Games",
        "clear_live": "Clear",
        "load_live_state": "Load Live State",
        "auto_refresh": "Auto-refresh (30s)",
        "next_refresh": "Refreshes every 30 s · last update {t} UTC",
        "refresh_failed": "The last refresh failed; showing the state from {t} UTC.",
        "no_live_state": "Could not fetch live state. Game may not be active yet.",
        "no_games": "No games found for today.",
        "apply": "Apply to Analysis ↓",
        "apply_none": "This game is over or not being played, so there is no state to analyse.",
        "inning_clipped": "The live inning ({i}) is beyond the selector; analysed as inning {c}.",
        "coach_limit": ("The bench coach has reached its limit ({n} questions per {per}). "
                        "Try again later."),
        "per_session": "session",
        "per_day": "day",
        "game_state": "Game State",
        "inning": "Inning",
        "half": "Half",
        "half_top": "Top",
        "half_bottom": "Bottom",
        "outs_label": "Outs",
        "score_diff": "Score (Home − Away)",
        "balls": "Balls",
        "strikes": "Strikes",
        "sched": "Scheduled innings",
        "runner_1b": "Runner on 1B",
        "runner_2b": "Runner on 2B",
        "runner_3b": "Runner on 3B",
        "extra_note": "Extra innings: regular-season half-innings start with a runner on 2nd.",
        "xr": "Extra innings start with a runner on 2nd (off for postseason)",
        "diff_clipped": "The live score difference ({d:+d}) is outside the slider; analysed as {c:+d}.",
        "presets": "Presets",
        "diamond": "Diamond",
        "home_leads": "Home leads by {n}",
        "away_leads": "Away leads by {n}",
        "tie": "Tie game",
        "wp_home": "Win Probability (Home)",
        "batting_wp": "{team} (batting): {pct:.1f}%",
        "home": "Home",
        "away": "Away",
        "game_over": "Game over: the home team leads in the bottom of the {n}th or later.",
        "leverage_index": "Leverage Index",
        "li_desc": "LI 1.0 = average | 2.0+ = high pressure | 4.0+ = critical",
        "li_labels": {"Low": "Low", "Medium": "Medium", "High": "High", "Very High": "Very High"},
        "re_now": "Expected runs (rest of half): {er:.2f} · P(score): {p:.0%}",
        "tactical": "Tactics (engine)",
        "tactical_sub": "Every player is league-average to the engine.",
        "no_tactics": "No tactic applies in this situation.",
        "breakeven": "Break-even success rate: {be:.1%} ({team} batting)",
        "no_breakeven": "No success rate makes this steal pay off",
        "delta_wp": "{team} WP: {now:.1f}% → {after:.1f}% (ΔWP {d:+.1f} pts)",
        "n_obs": "{n:,} observed bunts",
        "whatif": "What-If: Next Plate Appearance",
        "whatif_sub": ("Win probability of the batting team ({team}) if the plate appearance ends in... "
                       "(average over what happened on such plays in this base-out state)"),
        "coach": "Claude Bench Coach",
        "coach_sub": ("Claude reads the situation and the engine's numbers, recommends what to do "
                      "before the next pitch, and cites the numbers it used."),
        "coach_no_key": ("The public demo does not call the Claude API. To enable the bench coach, run "
                         "the app yourself with `ANTHROPIC_API_KEY` set as an environment variable or in "
                         "Streamlit secrets."),
        "coach_ask": "Ask the bench coach",
        "coach_asking": "Asking Claude...",
        "coach_failed": "No valid response from the model.",
        "coach_rec": "Recommendation",
        "coach_side": "Side",
        "coach_conf": "Confidence",
        "coach_follows": "Follows the engine",
        "coach_departs": "Departs from the engine",
        "coach_cited": "Numbers cited",
        "coach_problems": "Cited numbers that do not match the input",
        "coach_ok": "Every cited number matches the input.",
        "coach_game_over": "The game is over in this state.",
        "yes": "yes",
        "no": "no",
        "re24_title": "Run Expectancy (24 base-out states)",
        "re24_sub": ("Expected runs in the rest of the half / P(at least one run), "
                     "{half} half ({team} batting)"),
        "footer": "MLB Win Probability Engine | fitted to Stats API play-by-play",
        "top_inning": "Top",
        "bot_inning": "Bot",
        "out_unit": "out",
        "scenario_name_key": "name",
        "replay_caption": "Select any past MLB game and replay it play-by-play with WP updates.",
        "wp_change": "WP (home): {b:.1f}% → {a:.1f}% (WPA {d:+.1f} pts)",
        "team_wp": ("Team-adjusted home WP for these bases and outs (count left out): **{t:.1f}%** "
                    "(engine {e:.1f}%). Run differential per game, home − away: "
                    "last season {dp:+.2f}, this season so far {dc:+.2f}."),
        "team_wp_na_type": "The team-strength adjustment is used only for regular-season games.",
        "team_wp_na_data": ("The team-strength adjustment is not available "
                            "(the teams' results could not be read)."),
    },
    "JA": {
        "title": "MLB 勝利確率エンジン",
        "subtitle": (f"MLB {_season_span} の {_n_pa:,} 打席の推移に当てはめたエンジン — "
                     "勝利確率・レバレッジ・作戦・Claude ベンチコーチ"),
        "live_games": "ライブゲーム",
        "fetch_games": "今日の試合を取得",
        "clear_live": "クリア",
        "load_live_state": "ライブ状態を読み込む",
        "auto_refresh": "30秒ごとに自動更新",
        "next_refresh": "30 秒ごとに更新 · 最終更新 {t} UTC",
        "refresh_failed": "直近の更新に失敗しました。{t} UTC の状態を表示しています。",
        "no_live_state": "ライブ状態を取得できませんでした。試合がまだ開始されていない可能性があります。",
        "no_games": "今日の試合は見つかりませんでした。",
        "apply": "↓ 分析に反映",
        "apply_none": "この試合は終了しているか行われていないため、分析する局面はありません。",
        "inning_clipped": "ライブのイニング（{i}回）は選択肢の範囲外のため {c}回として分析しています。",
        "coach_limit": "ベンチコーチの利用上限（1 {per}あたり {n} 回）に達しました。時間をおいて試してください。",
        "per_session": "セッション",
        "per_day": "日",
        "game_state": "ゲーム状態",
        "inning": "イニング",
        "half": "表/裏",
        "half_top": "表",
        "half_bottom": "裏",
        "outs_label": "アウト数",
        "score_diff": "得点差（ホーム − アウェイ）",
        "balls": "ボール",
        "strikes": "ストライク",
        "sched": "規定イニング",
        "runner_1b": "1塁にランナーあり",
        "runner_2b": "2塁にランナーあり",
        "runner_3b": "3塁にランナーあり",
        "extra_note": "延長戦：レギュラーシーズンは各回とも走者二塁から始まります。",
        "xr": "延長は走者二塁から（ポストシーズンはオフ）",
        "diff_clipped": "ライブの得点差（{d:+d}）はスライダーの範囲外のため {c:+d} として分析しています。",
        "presets": "プリセット",
        "diamond": "ダイヤモンド",
        "home_leads": "ホーム {n} 点リード",
        "away_leads": "アウェイ {n} 点リード",
        "tie": "同点",
        "wp_home": "勝利確率（ホーム）",
        "batting_wp": "{team}（攻撃中）: {pct:.1f}%",
        "home": "ホーム",
        "away": "アウェイ",
        "game_over": "試合終了：{n}回以降の裏でホームがリードしています。",
        "leverage_index": "レバレッジ指数",
        "li_desc": "LI 1.0 = 平均 | 2.0+ = 高プレッシャー | 4.0+ = 超高",
        "li_labels": {"Low": "低", "Medium": "中", "High": "高", "Very High": "超高"},
        "re_now": "この回の残り期待得点: {er:.2f} · 1点以上の確率: {p:.0%}",
        "tactical": "作戦（エンジン）",
        "tactical_sub": "エンジンは全選手をリーグ平均として扱います。",
        "no_tactics": "この状況で使える作戦はありません。",
        "breakeven": "損益分岐の成功率: {be:.1%}（{team} の攻撃）",
        "no_breakeven": "どの成功率でもこの盗塁は得になりません",
        "delta_wp": "{team} の勝利確率: {now:.1f}% → {after:.1f}%（ΔWP {d:+.1f} pt）",
        "n_obs": "観測されたバント {n:,} 件",
        "whatif": "仮説分析：次の打席",
        "whatif_sub": ("打席が次の結果で終わったときの攻撃側（{team}）の勝利確率"
                       "（この塁上アウト状況で実際に起きた推移の平均）"),
        "coach": "Claude ベンチコーチ",
        "coach_sub": "Claude が状況とエンジンの数字を読み、次の投球の前に取るべき作戦を、使った数字を挙げて提案します。",
        "coach_no_key": ("公開デモでは Claude API を呼びません。ベンチコーチを使うには、環境変数または "
                         "Streamlit secrets に `ANTHROPIC_API_KEY` を設定して自分でアプリを起動してください。"),
        "coach_ask": "ベンチコーチに聞く",
        "coach_asking": "Claude に問い合わせ中...",
        "coach_failed": "モデルから有効な応答がありませんでした。",
        "coach_rec": "推奨",
        "coach_side": "側",
        "coach_conf": "確信度",
        "coach_follows": "エンジンに従う",
        "coach_departs": "エンジンと異なる理由",
        "coach_cited": "引用した数字",
        "coach_problems": "入力と一致しない引用",
        "coach_ok": "引用した数字はすべて入力と一致しました。",
        "coach_game_over": "この状況では試合が終わっています。",
        "yes": "はい",
        "no": "いいえ",
        "re24_title": "期待得点（24 塁上アウト状況）",
        "re24_sub": "この回の残り期待得点 / 1点以上の確率、{half}（{team} の攻撃）",
        "footer": "MLB 勝利確率エンジン | Stats API の一球速報に当てはめ",
        "top_inning": "表",
        "bot_inning": "裏",
        "out_unit": "アウト",
        "scenario_name_key": "name_ja",
        "replay_caption": "過去の試合を選んで1プレーずつ勝利確率をリプレイ",
        "wp_change": "勝利確率（ホーム）: {b:.1f}% → {a:.1f}%（WPA {d:+.1f} pt）",
        "team_wp": ("チーム力込みのホーム勝率（この塁とアウト・カウントは除く）：**{t:.1f}%**（エンジン {e:.1f}%）。"
                    "1 試合あたりの得失点差（ホーム − ビジター）：前年 {dp:+.2f}・今季ここまで {dc:+.2f}"),
        "team_wp_na_type": "チーム力の補正はレギュラーシーズンの試合だけに使います。",
        "team_wp_na_data": "チーム力の補正は使えません（両チームの成績を読めませんでした）。",
    },
}

_ = T[lang]  # shorthand

RECOMMENDATION_NAMES = {
    "none": ("Play it straight", "通常どおり"),
    "steal_2b": ("Steal 2nd base", "二盗"),
    "steal_3b": ("Steal 3rd base", "三盗"),
    "sacrifice_bunt": ("Sacrifice bunt", "送りバント"),
    "intentional_walk": ("Intentional walk", "敬遠"),
    "pitching_change": ("Pitching change", "継投"),
    "pinch_hitter": ("Pinch hitter", "代打"),
    "other": ("Other", "その他"),
}


def esc(x) -> str:
    return html.escape(str(x))


def team_label(team: str) -> str:
    return _["home"] if team == "home" else _["away"]


def li_badge(li_lbl: str) -> str:
    return {"Low": "li-low", "Medium": "li-medium", "High": "li-high",
            "Very High": "li-veryhigh"}.get(li_lbl, "li-medium")


def diamond_svg(runners, label: str) -> str:
    c = ["#00e5ff" if r else "#333" for r in runners]
    return (
        '<svg width="140" height="140" viewBox="0 0 140 140">'
        '<rect x="0" y="0" width="140" height="140" fill="#0a0a1a"/>'
        '<line x1="70" y1="120" x2="120" y2="70" stroke="#444" stroke-width="2"/>'
        '<line x1="120" y1="70" x2="70" y2="20" stroke="#444" stroke-width="2"/>'
        '<line x1="70" y1="20" x2="20" y2="70" stroke="#444" stroke-width="2"/>'
        '<line x1="20" y1="70" x2="70" y2="120" stroke="#444" stroke-width="2"/>'
        '<polygon points="70,115 65,120 70,125 75,120" fill="#fff"/>'
        f'<rect x="113" y="63" width="14" height="14" rx="2" fill="{c[0]}" transform="rotate(45 120 70)"/>'
        f'<rect x="63" y="13" width="14" height="14" rx="2" fill="{c[1]}" transform="rotate(45 70 20)"/>'
        f'<rect x="13" y="63" width="14" height="14" rx="2" fill="{c[2]}" transform="rotate(45 20 70)"/>'
        f'<text x="70" y="138" fill="#666" font-size="10" text-anchor="middle">{esc(label)}</text>'
        '</svg>'
    )


def team_line(s: dict) -> str:
    """The team layer's line (research/wp_layer) for a live or replayed state."""
    t = team_layer_wp(s["inning"], s["top_bottom"], s["outs"], s["runners"], s["score_diff"],
                      s.get("scheduled_innings", 9), s.get("extra_runner", True),
                      s.get("home_id"), s.get("away_id"), s.get("game_date"), s.get("game_type"))
    if t["applied"]:
        return _["team_wp"].format(t=t["team_wp"] * 100, e=t["engine_wp"] * 100,
                                   dp=t["d_prev"], dc=t["d_cur"])
    return _["team_wp_na_type"] if t["reason"] == "not_regular_season" else _["team_wp_na_data"]


def wp_gauge(wp: float, title: str | None = None) -> go.Figure:
    gauge_color = "#4caf50" if wp > 0.55 else "#ff9800" if wp > 0.45 else "#f44336"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=wp * 100,
        title={"text": title, "font": {"color": "#e0e0e0", "size": 13}} if title else None,
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
    fig.update_layout(paper_bgcolor="#0a0a1a", plot_bgcolor="#0a0a1a",
                      font={"color": "#e0e0e0"}, height=250,
                      margin=dict(t=30, b=10, l=30, r=30))
    return fig


def li_card(li_val: float, li_lbl: str) -> str:
    return (
        '<div class="metric-card" style="text-align:center;">'
        f'<div class="metric-value">{li_val:.1f}</div>'
        f'<div class="metric-label">{esc(_["leverage_index"])}</div>'
        f'<div style="margin-top:8px;"><span class="li-badge {li_badge(li_lbl)}">'
        f'{esc(_["li_labels"].get(li_lbl, li_lbl))}</span></div></div>'
    )


COACH_SESSION_LIMIT = 10


def _setting(name: str, default: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    try:
        return str(st.secrets.get(name, default))
    except Exception:  # no secrets file
        return default


@st.cache_resource
def _coach_calls_today() -> dict:
    """Shared by every session (thread) of this server process."""
    return {"day": None, "n": 0, "lock": threading.Lock()}


def coach_limit_reached():
    """None and count the call if it is allowed; otherwise (limit, period key).
    At most COACH_DAILY_LIMIT (default 30) calls per UTC day for the whole
    server, which is the real bound; COACH_SESSION_LIMIT per browser session
    only spreads that budget (reloading the page starts a new session).
    A call is counted before it is made, so failed calls count too."""
    try:
        daily = max(0, int(_setting("COACH_DAILY_LIMIT", "30")))
    except ValueError:
        daily = 30
    used = st.session_state.get("coach_calls", 0)
    if used >= COACH_SESSION_LIMIT:
        return COACH_SESSION_LIMIT, "per_session"
    day = datetime.now(timezone.utc).date().isoformat()
    box = _coach_calls_today()
    with box["lock"]:
        if box["day"] != day:
            box["day"], box["n"] = day, 0
        if box["n"] >= daily:
            return daily, "per_day"
        box["n"] += 1
    st.session_state["coach_calls"] = used + 1
    return None


def api_key_available() -> bool:
    """ANTHROPIC_API_KEY from the environment, or from Streamlit secrets
    (copied into this process's environment for strategist.py)."""
    if has_api_key():
        return True
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:  # no secrets file
        key = None
    if key:
        os.environ["ANTHROPIC_API_KEY"] = str(key)
        return True
    return False


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
    .tactic-gain { border-left-color: #4caf50; }
    .tactic-loss { border-left-color: #f44336; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Title
# ============================================================

st.markdown(f"# {_['title']}")
st.markdown(f"*{_['subtitle']}*")
st.markdown("---")


# ============================================================
# Replay Mode
# ============================================================

if mode == "🎬 Replay Mode":
    st.markdown("## 🎬 Replay Mode")
    st.caption(_["replay_caption"])
    st.markdown("---")

    col_date, col_load = st.columns([2, 1])
    with col_date:
        replay_date = st.date_input("Game Date", value=datetime.now(timezone.utc).date())
    with col_load:
        st.write("")
        load_games_btn = st.button("Load Games", width="stretch", type="primary")

    if load_games_btn:
        with st.spinner("Fetching schedule..."):
            _games = get_todays_games(replay_date.strftime("%Y-%m-%d"))
        st.session_state["replay_games"] = _games
        st.session_state.pop("replay_plays", None)
        st.session_state.pop("replay_idx", None)

    replay_games = st.session_state.get("replay_games", [])

    if replay_games:
        game_opts = {f"{g['away_team']} @ {g['home_team']}  [{g['status']}]": g["gamePk"]
                     for g in replay_games}
        sel_label = st.selectbox("Select Game", list(game_opts.keys()))
        sel_pk = game_opts[sel_label]

        if st.button("▶ Load Replay", width="stretch", type="primary"):
            with st.spinner("Loading play-by-play data..."):
                _plays = get_game_plays(sel_pk)
            if _plays:
                st.session_state["replay_plays"] = _plays
                st.session_state["replay_idx"] = 0
                st.session_state["replay_playing"] = False
            else:
                st.error("No play data found. The game may not have started or data is unavailable.")

    plays = st.session_state.get("replay_plays", [])

    if plays:
        n = len(plays)
        idx = max(0, min(st.session_state.get("replay_idx", 0), n - 1))

        st.progress(idx / max(n - 1, 1), text=f"Play {idx + 1} / {n}")

        ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([1, 1, 2, 1, 2])
        with ctrl1:
            if st.button("⏮", help="First"):
                st.session_state["replay_idx"] = 0
                st.session_state["replay_playing"] = False
                st.rerun()
        with ctrl2:
            if st.button("◀", help="Prev"):
                st.session_state["replay_idx"] = max(0, idx - 1)
                st.session_state["replay_playing"] = False
                st.rerun()
        with ctrl3:
            is_playing = st.session_state.get("replay_playing", False)
            if st.button("⏸ Pause" if is_playing else "▶ Play", width="stretch"):
                st.session_state["replay_playing"] = not is_playing
                st.rerun()
        with ctrl4:
            if st.button("▶", help="Next"):
                st.session_state["replay_idx"] = min(n - 1, idx + 1)
                st.session_state["replay_playing"] = False
                st.rerun()
        with ctrl5:
            speed = st.select_slider("Speed", options=[0.5, 1.0, 2.0, 3.0],
                                     value=st.session_state.get("replay_speed", 1.0),
                                     format_func=lambda x: f"{x}s/play")
            st.session_state["replay_speed"] = speed

        # State before the play, and after it
        play = plays[idx]
        res = full_analysis(play["inning"], play["top_bottom"], play["outs"], play["runners"],
                            play["score_diff"], play["scheduled_innings"],
                            extra_runner=play["extra_runner"])
        wp = res["win_probability"]
        wp_after = wp_after_play(play["inning"], play["top_bottom"], play["outs_after"],
                                 play["runners_after"], play["runs_on_play"], play["score_diff"],
                                 play["scheduled_innings"], extra_runner=play["extra_runner"])

        half_lbl = _["top_inning"] if play["top_bottom"] == "top" else _["bot_inning"]
        st.markdown(
            f"### {esc(play['away_team'])} **{play['away_score']}** — "
            f"**{play['home_score']}** {esc(play['home_team'])}  "
            f"| {half_lbl} {play['inning']} · {play['outs']} {_['out_unit']}"
        )
        if play["batter"] or play["pitcher"]:
            st.caption(f"🏏 {play['batter']}  ⚔  ⚾ {play['pitcher']}")

        if play["description"]:
            event_color = {
                "Home Run": "#ffd700", "Single": "#4caf50", "Double": "#00e5ff",
                "Triple": "#ff9800", "Walk": "#9c27b0", "Strikeout": "#f44336",
            }.get(play["event"], "#aaaaaa")
            st.markdown(
                f'<div style="background:#141428;border-left:4px solid {event_color};'
                f'padding:12px 16px;border-radius:0 8px 8px 0;margin:8px 0;">'
                f'<strong style="color:{event_color};">{esc(play["event"])}</strong> — '
                f'<span style="color:#ccc;">{esc(play["description"])}</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown(_["wp_change"].format(b=wp * 100, a=wp_after * 100, d=(wp_after - wp) * 100))
        st.caption(team_line(play))

        col_diamond, col_gauge, col_li = st.columns([1, 2, 1])
        with col_diamond:
            st.markdown(diamond_svg(play["runners"],
                                    f"{half_lbl} {play['inning']} | {play['outs']} {_['out_unit']}"),
                        unsafe_allow_html=True)
            sd = play["score_diff"]
            if sd > 0:
                st.markdown(f"**{play['home_team']} +{sd}**")
            elif sd < 0:
                st.markdown(f"**{play['away_team']} +{abs(sd)}**")
            else:
                st.markdown(f"**{_['tie']}**")
        with col_gauge:
            st.plotly_chart(wp_gauge(wp, f"Win Prob ({play['home_team']})"), width="stretch")
        with col_li:
            st.markdown(li_card(res["leverage_index"], res["leverage_label"]), unsafe_allow_html=True)

        if st.session_state.get("replay_playing", False):
            if idx < n - 1:
                time.sleep(speed)
                st.session_state["replay_idx"] = idx + 1
                st.rerun()
            else:
                st.session_state["replay_playing"] = False
                st.success("🏁 Replay complete!")

    st.stop()


# ============================================================
# Live Game Section
# ============================================================

st.markdown(f"## {_['live_games']}")

with st.container():
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    live_col1, live_col2 = st.columns([3, 1])

    with live_col1:
        if st.button(_["fetch_games"], key="fetch_games", width="stretch"):
            with st.spinner("Fetching schedule from MLB Stats API..."):
                games = get_todays_games()
            st.session_state["todays_games"] = games
            st.session_state["live_last_fetched"] = today_str

    with live_col2:
        if st.button(_["clear_live"], key="clear_live", width="stretch"):
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
            load_live = st.button(_["load_live_state"], key="load_live", width="stretch")
        with refresh_col:
            auto_refresh = st.checkbox(_["auto_refresh"], key="auto_refresh")

        if load_live:
            with st.spinner("Fetching live game state..."):
                live_state = get_live_state(selected_pk)
            if live_state:
                st.session_state["live_state_cache"] = live_state
                st.session_state["live_game_pk"] = selected_pk
                st.session_state["live_last_refresh"] = time.time()
                st.session_state["live_last_ok"] = time.time()
                st.session_state["live_refresh_failed"] = False
            else:
                st.warning(_["no_live_state"])

        def _live_panel():
            """The live state card and its Apply button. With auto-refresh on, this
            runs as a fragment every 30 s, so the state is fetched again without
            any click."""
            if st.session_state.get("mode", MODES[0]) != MODES[0]:
                return  # a timer tick that arrives after switching to Replay
            # The timer ticks every 30 s; the attempt is stamped when it starts, so a
            # slow fetch does not make the next tick skip. A click on Apply reruns
            # this fragment too: it applies the state on screen and fetches nothing.
            if (st.session_state.get("auto_refresh") and st.session_state.get("live_game_pk")
                    and not st.session_state.get("apply_live")):
                if time.time() - st.session_state.get("live_last_refresh", 0.0) >= 20:
                    st.session_state["live_last_refresh"] = time.time()
                    fresh = get_live_state(st.session_state["live_game_pk"])
                    if fresh:
                        st.session_state["live_state_cache"] = fresh
                        st.session_state["live_last_ok"] = time.time()
                    st.session_state["live_refresh_failed"] = not fresh
                ok = datetime.fromtimestamp(st.session_state.get("live_last_ok", 0.0),
                                            timezone.utc).strftime("%H:%M:%S")
                key = "refresh_failed" if st.session_state.get("live_refresh_failed") else "next_refresh"
                st.caption(_[key].format(t=ok))
            cached = st.session_state.get("live_state_cache")
            if cached:
                inning_half = _["top_inning"] if cached["top_bottom"] == "top" else _["bot_inning"]
                runners_str = (
                    ("1B " if cached["runners"][0] else "")
                    + ("2B " if cached["runners"][1] else "")
                    + ("3B" if cached["runners"][2] else "")
                    or "---"
                )
                count_str = ("" if cached.get("balls") is None
                             else f" | {cached['balls']}-{cached['strikes']}")
                st.markdown(
                    '<div style="background:#141428;border:1px solid #00e5ff44;border-radius:10px;'
                    'padding:14px;margin:8px 0;">'
                    f'<span style="color:#00e5ff;font-weight:bold;font-size:1.1rem;">'
                    f'{esc(cached["away_team"])} @ {esc(cached["home_team"])}</span>&nbsp;'
                    f'<span style="color:#aaa;font-size:0.9rem;">{esc(cached["status"])}</span><br>'
                    f'<span style="color:#e0e0e0;font-size:1.2rem;font-weight:bold;">'
                    f'Away {cached["score_away"]} — Home {cached["score_home"]}</span>&nbsp;&nbsp;'
                    f'<span style="color:#888;font-size:0.9rem;">{inning_half} {cached["inning"]} | '
                    f'{cached["outs"]} {_["out_unit"]} | {runners_str}{count_str}</span><br>'
                    f'<span style="color:#ccc;font-size:0.85rem;">🥎 {esc(cached["batter_name"] or "—")}'
                    f' &nbsp; ⚾ {esc(cached["pitcher_name"] or "—")}</span></div>',
                    unsafe_allow_html=True,
                )

                if has_live_state(cached["status"]):
                    st.caption(team_line(cached))
                if not has_live_state(cached["status"]):
                    st.caption(_["apply_none"])
                elif st.button(_["apply"], key="apply_live", width="stretch"):
                    st.session_state["_preset"] = {
                        "inning": cached["inning"],
                        "top_bottom": cached["top_bottom"],
                        "outs": cached["outs"],
                        "runners": cached["runners"],
                        "score_diff": cached["score_diff"],
                        "balls": cached.get("balls"),
                        "strikes": cached.get("strikes"),
                        "sched": cached.get("scheduled_innings", 9),
                        "extra_runner": cached.get("extra_runner", True),
                    }
                    st.rerun()

        if auto_refresh and st.session_state.get("live_game_pk"):
            st.fragment(run_every=30)(_live_panel)()
        else:
            _live_panel()

    elif st.session_state.get("live_last_fetched"):
        st.info(_["no_games"])

st.markdown("---")


# ============================================================
# Game State Input
# ============================================================

INNINGS = list(range(1, 21))
SCHED_OPTIONS = [9, 7]

# A preset or a live state sets the widgets' values before they are drawn.
if "_preset" in st.session_state:
    p = st.session_state.pop("_preset")
    st.session_state["w_inning"] = min(max(int(p["inning"]), 1), INNINGS[-1])
    st.session_state["inning_clipped"] = (int(p["inning"]) if int(p["inning"]) > INNINGS[-1]
                                         else None)
    st.session_state["half"] = p["top_bottom"]
    for k in ("w_half_EN", "w_half_JA"):
        st.session_state.pop(k, None)
    st.session_state["w_outs"] = int(p["outs"])
    st.session_state["w_diff"] = int(max(-10, min(10, p["score_diff"])))
    st.session_state["diff_clipped"] = (int(p["score_diff"]) if abs(int(p["score_diff"])) > 10
                                        else None)
    st.session_state["w_xr"] = bool(p.get("extra_runner", True))
    st.session_state["w_r1"], st.session_state["w_r2"], st.session_state["w_r3"] = (
        bool(x) for x in p["runners"])
    st.session_state["w_balls"] = int(p.get("balls") or 0)
    st.session_state["w_strikes"] = int(p.get("strikes") or 0)
    st.session_state["w_sched"] = p.get("sched", 9) if p.get("sched", 9) in SCHED_OPTIONS else 9
st.session_state.setdefault("w_diff", 0)
st.session_state.setdefault("w_xr", True)
# The half selector shows translated labels; the frontend keeps the labels of a
# keyed widget, so each language gets its own widget and "half" carries the value.
st.session_state.setdefault("half", "top")
HALF_KEY = f"w_half_{lang}"
st.session_state.pop("w_half_JA" if lang == "EN" else "w_half_EN", None)
if HALF_KEY not in st.session_state:
    st.session_state[HALF_KEY] = st.session_state["half"]

st.markdown(f"## {_['game_state']}")

col_settings, col_preset = st.columns([3, 2])

with col_settings:
    c1, c2, c3 = st.columns(3)
    with c1:
        inning = st.selectbox(_["inning"], INNINGS, key="w_inning")
    with c2:
        top_bottom = st.selectbox(
            _["half"], ["top", "bottom"], key=HALF_KEY,
            format_func=lambda x: _["half_top"] if x == "top" else _["half_bottom"],
        )
        st.session_state["half"] = top_bottom
    with c3:
        outs = st.selectbox(_["outs_label"], [0, 1, 2], key="w_outs")

    c4, c5, c6, c7 = st.columns([3, 1, 1, 1])
    with c4:
        score_diff = st.slider(_["score_diff"], -10, 10, key="w_diff")
    with c5:
        balls = st.selectbox(_["balls"], [0, 1, 2, 3], key="w_balls")
    with c6:
        strikes = st.selectbox(_["strikes"], [0, 1, 2], key="w_strikes")
    with c7:
        sched = st.selectbox(_["sched"], SCHED_OPTIONS, key="w_sched")

    r1, r2, r3_col = st.columns(3)
    with r1:
        runner1 = st.checkbox(_["runner_1b"], key="w_r1")
    with r2:
        runner2 = st.checkbox(_["runner_2b"], key="w_r2")
    with r3_col:
        runner3 = st.checkbox(_["runner_3b"], key="w_r3")
    extra_runner = st.checkbox(_["xr"], key="w_xr")
    if inning > sched and extra_runner:
        st.caption(_["extra_note"])
    if inning != INNINGS[-1]:
        st.session_state.pop("inning_clipped", None)  # the user moved away from it
    inning_clipped = st.session_state.get("inning_clipped")
    if inning_clipped is not None:
        st.caption(_["inning_clipped"].format(i=inning_clipped, c=inning))
    clipped = st.session_state.get("diff_clipped")
    if clipped is not None and score_diff == max(-10, min(10, clipped)):
        st.caption(_["diff_clipped"].format(d=clipped, c=score_diff))

with col_preset:
    st.markdown(f"### {_['presets']}")
    for key, scenario in SCENARIOS.items():
        if st.button(scenario[_["scenario_name_key"]], key=f"preset_{key}", width="stretch"):
            s = scenario["state"]
            st.session_state["_preset"] = {
                "inning": s.inning, "top_bottom": s.top_bottom, "outs": s.outs,
                "runners": s.runners, "score_diff": s.score_diff,
                "balls": 0, "strikes": 0, "sched": s.scheduled_innings, "extra_runner": True,
            }
            st.rerun()

runners = (int(runner1), int(runner2), int(runner3))


# ============================================================
# Analysis
# ============================================================

result = full_analysis(inning, top_bottom, outs, runners, score_diff, sched, balls, strikes,
                       extra_runner=extra_runner)

wp = result["win_probability"]
li_val = result["leverage_index"]
li_lbl = result["leverage_label"]
batting_team = result["batting_team"]

st.markdown("---")


# ============================================================
# Diamond SVG + WP Gauge + LI Badge
# ============================================================

col_diamond, col_gauge, col_li = st.columns([1, 2, 1])

with col_diamond:
    st.markdown(f"### {_['diamond']}")
    half_lbl = _["top_inning"] if top_bottom == "top" else _["bot_inning"]
    st.markdown(diamond_svg(runners, f"{half_lbl} {inning} | {outs} {_['out_unit']} | {balls}-{strikes}"),
                unsafe_allow_html=True)
    if score_diff > 0:
        st.markdown(f"**{_['home_leads'].format(n=score_diff)}**")
    elif score_diff < 0:
        st.markdown(f"**{_['away_leads'].format(n=abs(score_diff))}**")
    else:
        st.markdown(f"**{_['tie']}**")

with col_gauge:
    st.markdown(f"### {_['wp_home']}")
    st.plotly_chart(wp_gauge(wp), width="stretch")
    st.markdown(_["batting_wp"].format(team=team_label(batting_team),
                                       pct=result["batting_team_wp"] * 100))

with col_li:
    st.markdown(f"### {_['leverage_index']}")
    st.markdown(li_card(li_val, li_lbl), unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.75rem;color:#666;margin-top:12px;">{esc(_["li_desc"])}</div>',
                unsafe_allow_html=True)
    st.caption(_["re_now"].format(er=result["run_expectancy"], p=result["p_score"]))

if result["game_over"]:
    st.info(_["game_over"].format(n=sched))


# ============================================================
# Tactics
# ============================================================

st.markdown("---")
st.markdown(f"## {_['tactical']}")
st.caption(_["tactical_sub"])

tactics = result["tactics"]
if not tactics:
    st.info(_["no_tactics"])
else:
    for t in tactics:
        name = t["name_ja"] if lang == "JA" else t["name"]
        team = team_label(t["team"])
        if t["breakeven"] is not None:
            css = ""
            if t["breakeven"] >= 1.0:
                detail = _["no_breakeven"]
            else:
                detail = _["breakeven"].format(be=t["breakeven"], team=team)
        else:
            css = "tactic-gain" if t["delta"] > 0 else "tactic-loss"
            detail = _["delta_wp"].format(team=team, now=t["wp_now"] * 100,
                                          after=t["wp_if"] * 100, d=t["delta"] * 100)
            if t["n"]:
                detail += " | " + _["n_obs"].format(n=t["n"])
        st.markdown(
            f'<div class="tactic-card {css}"><strong>{esc(name)}</strong><br>'
            f'<span style="color:#ddd;font-size:0.95rem;">{esc(detail)}</span><br>'
            f'<span style="color:#888;font-size:0.8rem;">'
            f'{esc(card_note(t["note"], lang == "JA", (t["breakeven"] or 0) >= 1.0))}</span></div>',
            unsafe_allow_html=True,
        )


# ============================================================
# Claude Bench Coach
# ============================================================

st.markdown("---")
st.markdown(f"## {_['coach']}")
st.markdown(f"*{_['coach_sub']}*")

if not api_key_available():
    st.info(_["coach_no_key"])
elif result["game_over"]:
    st.info(_["coach_game_over"])
else:
    state_key = json.dumps([inning, top_bottom, outs, runners, score_diff, sched, balls, strikes,
                            extra_runner, lang])
    if st.button(_["coach_ask"], key="ask_coach"):
        limit = coach_limit_reached()
        if limit:
            st.warning(_["coach_limit"].format(n=limit[0], per=_[limit[1]]))
        else:
            with st.spinner(_["coach_asking"]):
                adv, problems, _payload = get_advice(result, lang)
            st.session_state["coach"] = {"key": state_key, "advice": adv,
                                          "problems": problems}

    last = st.session_state.get("coach")
    if last and last["key"] == state_key:
        adv = last["advice"]
        if adv is None:
            st.warning(_["coach_failed"] + " " + "; ".join(last["problems"]))
        else:
            rec_en, rec_ja = RECOMMENDATION_NAMES.get(adv.recommendation,
                                                      (adv.recommendation, adv.recommendation))
            st.markdown(
                '<div class="metric-card" style="padding:20px;">'
                f'<div class="metric-label">{esc(_["coach_rec"])}</div>'
                f'<div style="font-size:1.6rem;font-weight:bold;color:#00e5ff;">'
                f'{esc(rec_ja if lang == "JA" else rec_en)}</div>'
                f'<div style="color:#aaa;font-size:0.9rem;margin:6px 0;">{esc(_["coach_side"])}: '
                f'{esc(adv.side)} · {esc(_["coach_conf"])}: {adv.confidence:.0%} · '
                f'{esc(_["coach_follows"])}: {esc(_["yes"] if adv.follows_engine else _["no"])}</div>'
                f'<div style="font-size:1.05rem;line-height:1.7;">{esc(adv.reason)}</div></div>',
                unsafe_allow_html=True,
            )
            if not adv.follows_engine and adv.departure_reason:
                st.markdown(f'<div><strong>{esc(_["coach_departs"])}:</strong> '
                            f'{esc(adv.departure_reason)}</div>', unsafe_allow_html=True)
            if adv.cited:
                st.markdown(f"**{_['coach_cited']}**")
                st.table([{"key": c.name, "value": c.value} for c in adv.cited])
            if last["problems"]:
                st.warning(f"{_['coach_problems']}: " + "; ".join(last["problems"]))
            else:
                st.caption(_["coach_ok"])


# ============================================================
# What-If Section
# ============================================================

st.markdown("---")
st.markdown(f"## {_['whatif']}")
st.markdown(f"*{_['whatif_sub'].format(team=team_label(batting_team))}*")

outcomes = result["what_if"]
if outcomes:
    cols = st.columns(len(outcomes))
    for col, o in zip(cols, outcomes):
        with col:
            d = o["delta"]
            d_color = "#4caf50" if d > 0 else "#f44336" if d < 0 else "#888"
            st.markdown(
                '<div class="metric-card" style="text-align:center;padding:12px;">'
                f'<div style="font-size:0.85rem;color:#a0a0b0;">'
                f'{esc(o["name_ja"] if lang == "JA" else o["name"])}</div>'
                f'<div style="font-size:1.5rem;font-weight:bold;color:#00e5ff;">'
                f'{o["batting_wp"] * 100:.1f}%</div>'
                f'<div style="font-size:0.9rem;color:{d_color};">ΔWP: {d * 100:+.1f}</div></div>',
                unsafe_allow_html=True,
            )


# ============================================================
# Run Expectancy Table
# ============================================================

st.markdown("---")
with st.expander(_["re24_title"]):
    half_name = _["half_top"] if top_bottom == "top" else _["half_bottom"]
    st.markdown(f"*{_['re24_sub'].format(half=half_name, team=team_label(batting_team))}*")
    rows = {}
    for entry in re24_table(top_bottom):
        rows.setdefault(entry["runners"], {})[entry["outs"]] = entry
    lines = ["| Runners | 0 out | 1 out | 2 out |", "|---------|-------|-------|-------|"]
    for key in ["---", "1--", "-2-", "12-", "--3", "1-3", "-23", "123"]:
        cells = [f"{rows[key][o]['expected_runs']:.3f} / {rows[key][o]['p_score']:.0%}" for o in range(3)]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    st.markdown("\n".join(lines))


# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#555;font-size:0.8rem;">'
    f'{esc(_["footer"])}<br>'
    '<a href="https://github.com/yasumorishima/mlb-win-probability" style="color:#00e5ff;">GitHub</a>'
    '</div>',
    unsafe_allow_html=True,
)
