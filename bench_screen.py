"""
Bench screen: one page that follows a live game by itself.

Pick the mode, pick a game (an in-progress game is picked by default); the
page refreshes every 30 s and shows, on one landscape screen:
  score / inning / outs / bases, the batting team's win probability
  (engine + team layer as the headline, the player layer next to it) and
  the leverage index;
  the current batter and the next three against the current pitcher
  (on-base, strikeout, extra-base hit, home run chances);
  the bench options (bench_live.py): bunt, pinch hitters, intentional
  walk, relievers, with the change in win probability.
"""

from __future__ import annotations

import html
import time
from datetime import datetime, timezone

import streamlit as st

import bench_live
import bench_viz
from analysis import leverage_index, li_label, state_wp, team_layer_wp
from live_feed import get_todays_games, has_live_state, is_game_over_status, is_not_started, parse_live_state

T = {
    "JA": {"title": "📋 ベンチ", "pick": "試合", "none": "今日の試合はありません。", "reload": "日程を読み直す",
           "follow": "自動で追う（30 秒ごと）", "updated": "更新 {t} UTC", "failed": "取得できませんでした（最後の成功 {t} UTC）",
           "top": "表", "bot": "裏", "outs": "アウト", "wp": "{team} の勝率", "player": "選手込み", "li": "重要度 LI",
           "matchup": "対戦（投手 {p}）", "batter": "打者", "on_base": "出塁", "k": "三振", "xbh": "長打", "hr": "本塁打",
           "now": "今", "next": "次 {n}", "not_live": "試合中ではありません（{s}）。", "cannot": "打順を読み取れませんでした。",
           "options": "作戦", "off": "攻撃側", "def": "守備側", "opt": "選択肢", "runs": "この回の得点期待値",
           "score": "1点以上", "owp": "攻撃側の勝率", "delta": "差（ポイント）",
           "note": "勝率の見出しはエンジン＋チーム補正、「選手込み」と作戦の値は選手を入れた層（事前登録して 2026 年で確認）。"
                   "選手の成績は {d} まで。守備固めは考えていません。7 回以降の救援は、2022〜2026 年の試合ではそのチームの直近 30 日の起用から、それ以外はブルペン全体の平均で計算しています。"},
    "EN": {"title": "📋 Bench", "pick": "Game", "none": "No games today.", "reload": "Reload schedule",
           "follow": "Follow (every 30 s)", "updated": "Updated {t} UTC", "failed": "Fetch failed (last success {t} UTC)",
           "top": "Top", "bot": "Bot", "outs": "out", "wp": "{team} win prob.", "player": "with players", "li": "Leverage",
           "matchup": "Matchups (pitcher {p})", "batter": "Batter", "on_base": "On base", "k": "K", "xbh": "XBH", "hr": "HR",
           "now": "now", "next": "next {n}", "not_live": "Not in progress ({s}).", "cannot": "Could not read the lineups.",
           "options": "Options", "off": "Offense", "def": "Defense", "opt": "Option", "runs": "Runs this half",
           "score": "P(score)", "owp": "Offense WP", "delta": "Change (pt)",
           "note": "Headline WP: engine + team layer; 'with players' and the options: the player layer (pre-registered, "
                   "confirmed on 2026). Player records through {d}. Defensive subs are not modelled. Relief from the 7th: the team's "
                   "usage over the last 30 days for 2022-2026 games, otherwise the pooled bullpen."},
}


def _li_word(label: str, lang: str) -> str:
    if lang != "JA":
        return label
    return {"Low": "低い", "Medium": "普通", "High": "高い", "Very High": "勝負所"}.get(label, label)


def _games() -> list[dict]:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if st.session_state.get("bench_games_day") != today or "bench_games" not in st.session_state:
        st.session_state["bench_games"] = get_todays_games()
        st.session_state["bench_games_day"] = today
    return st.session_state["bench_games"]


def _fetch(pk: int) -> None:
    feed = bench_live.get_feed(pk)
    st.session_state["bench_last_try"] = time.time()
    if feed:
        st.session_state["bench_feed"] = feed
        st.session_state["bench_feed_pk"] = pk
        st.session_state["bench_last_ok"] = time.time()
        st.session_state["bench_failed"] = False
        st.session_state.pop("bench_cache", None)
    else:
        st.session_state["bench_failed"] = True


def _computed(feed: dict) -> dict:
    """Everything the screen shows for one fetched feed (cached until the next fetch)."""
    if "bench_cache" not in st.session_state:
        out = {"state": parse_live_state(feed)}
        try:
            out["options"] = bench_live.bench_options(feed)
            out["matchups"] = bench_live.matchups(feed)
        except Exception as e:          # never break the screen
            out["options"] = out["matchups"] = None
            out["error"] = repr(e)[:200]
        try:
            out["series"] = bench_viz.wp_series(feed)
        except Exception:
            out["series"] = []
        st.session_state["bench_cache"] = out
    return st.session_state["bench_cache"]


def _panel(lang: str) -> None:
    t = T[lang]
    pk = st.session_state.get("bench_pk")
    if pk is None:
        return
    if (st.session_state.get("bench_feed_pk") != pk
            or (st.session_state.get("bench_follow", True) and time.time() - st.session_state.get("bench_last_try", 0) >= 20)):
        _fetch(pk)
    ok = datetime.fromtimestamp(st.session_state.get("bench_last_ok", 0.0), timezone.utc).strftime("%H:%M:%S")
    st.caption(t["failed" if st.session_state.get("bench_failed") else "updated"].format(t=ok))
    feed = st.session_state.get("bench_feed")
    if not feed or st.session_state.get("bench_feed_pk") != pk:
        return
    c = _computed(feed)
    s = c["state"]
    if not has_live_state(s["status"]) or is_not_started(s["status"]) or is_game_over_status(s["status"]):
        st.info(t["not_live"].format(s=s["status"]))
        return
    _draw(lang, s, c.get("options"), c.get("matchups"), c.get("series") or [])


def _plan_html(plan: list, lang: str, s: dict) -> str:
    """Who the model expects to pitch the late innings for the fielding team."""
    ja = lang == "JA"
    fld = s["home_team"] if s["top_bottom"] == "top" else s["away_team"]
    head = (f"終盤の継投の見込み（守備側 {fld}：直近 30 日の起用と今の点差から。今の投手を除く）" if ja else
            f"Expected late-inning arms for {fld} (usage over the last 30 days at the current score; current pitcher excluded)")
    lines = []
    for r in plan:
        parts = [f"{html.escape(n)} <b>{x:.0%}</b>" for n, x in r["rows"]]
        parts.append(("その他 " if ja else "others ") + f"{r['other']:.0%}")
        inn = f"{r['inning']}回" if ja else f"{r['inning']}th"
        lines.append(f'<div style="margin:2px 0"><span style="display:inline-block;width:3.2em;color:#c3c2b7">{inn}</span>'
                     + " · ".join(parts) + "</div>")
    return (f'<div style="background:#141428;border-radius:10px;padding:10px 14px;margin:6px 0;color:#f0efec;font-size:14px">'
            f'<div style="color:#c3c2b7;font-size:12px;margin-bottom:4px">{head}</div>{"".join(lines)}</div>')


def _draw(lang: str, s: dict, opts: dict | None, mu: dict | None, series: list, mark_i: int | None = None,
          chart_key: str | None = None, home_wp: float | None = None, wp_note: str | None = None):
    """The bench screen's cards and charts for one game moment (live or replayed).
    Returns the WP chart's selection event when `chart_key` is given (review mode)."""
    t = T[lang]
    event = None
    bat_home = s["top_bottom"] == "bottom"
    bat_team = s["home_team"] if bat_home else s["away_team"]
    sched, xr = s.get("scheduled_innings", 9), s.get("extra_runner", True)
    tl = team_layer_wp(s["inning"], s["top_bottom"], s["outs"], s["runners"], s["score_diff"], sched, xr,
                       s.get("home_id"), s.get("away_id"), s.get("game_date"), s.get("game_type"))
    home = tl["team_wp"] if tl.get("applied") else state_wp(s["inning"], s["top_bottom"], s["outs"], s["runners"],
                                                               s["score_diff"], sched, xr)
    player_wp = opts["offense"][0]["wp_offense"] if opts else None
    if home_wp is not None:          # look-back: PEN_RECAL (players + late bullpen usage, recalibrated)
        home, player_wp = home_wp, None
    bat_wp = home if bat_home else 1 - home
    li = leverage_index(s["inning"], s["top_bottom"], s["outs"], s["runners"], s["score_diff"], sched, xr) if s["outs"] < 3 else 0.0
    hot = li >= 1.5
    st.markdown(
        "<style>@keyframes lipulse{0%{box-shadow:0 0 0 0 rgba(230,103,103,.55)}70%{box-shadow:0 0 0 14px rgba(230,103,103,0)}"
        "100%{box-shadow:0 0 0 0 rgba(230,103,103,0)}}"
        ".li-hot{animation:lipulse 1.6s infinite;border-radius:10px}"
        "@keyframes wpin{from{opacity:.25;transform:translateY(6px)}to{opacity:1;transform:none}}"
        ".wp-big{animation:wpin .6s ease-out}</style>", unsafe_allow_html=True)
    left, mid, right = st.columns([2, 2, 4])
    with left:
        st.markdown(
            f'<div style="background:#141428;border-radius:12px;padding:14px;">'
            f'<div style="color:#c3c2b7">{html.escape(s["away_team"])} @ {html.escape(s["home_team"])}</div>'
            f'<div style="font-size:2.4rem;font-weight:bold;color:#f0efec">{s["score_away"]} – {s["score_home"]}</div>'
            f'<div style="font-size:1.2rem;color:#c3c2b7">{s["inning"]}{t["top"] if s["top_bottom"] == "top" else t["bot"]}'
            + (f' &nbsp; {s["balls"]}-{s["strikes"]}' if s.get("balls") is not None else "") + '</div>'
            + bench_viz.diamond(s["runners"], min(s["outs"], 3)) + '</div>', unsafe_allow_html=True)
    with mid:
        st.markdown(
            f'<div style="background:#141428;border-radius:12px;padding:14px;">'
            f'<div style="color:#c3c2b7">{html.escape(t["wp"].format(team=bat_team))}</div>'
            f'<div class="wp-big" style="font-size:3.4rem;font-weight:bold;color:#f0efec">{bat_wp:.0%}</div>'
            + (f'<div style="color:#c3c2b7">{t["player"]} <b style="color:#f0efec">{player_wp:.0%}</b></div>'
               if player_wp is not None else "")
            + (f'<div style="color:#898781;font-size:.85rem">{html.escape(wp_note)}</div>' if wp_note else "")
            + f'<div class="{"li-hot" if hot else ""}" style="margin-top:12px;padding:8px 10px;background:#1f1f3a">'
            f'<span style="color:#c3c2b7">{t["li"]}</span> '
            f'<b style="font-size:1.8rem;color:{"#e66767" if hot else "#f0efec"}">{li:.1f}</b>'
            f' <span style="color:#898781">{_li_word(li_label(li), lang)}</span></div></div>', unsafe_allow_html=True)
    with right:
        if mu:
            st.markdown(f"**{html.escape(t['matchup'].format(p=mu['pitcher']))}**")
            st.plotly_chart(bench_viz.fig_matchups(mu, lang), width="stretch", config={"displayModeBar": False})
        elif not opts:
            st.caption(t["cannot"])
    if len(series) > 1:
        fig = bench_viz.fig_wp(series, s["home_team"], s["away_team"], lang, mark_i=mark_i)
        if chart_key:
            event = st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=chart_key,
                                    on_select="rerun", selection_mode="points")
        else:
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    if opts:
        for o in opts["offense"]:
            if o["kind"] == "steal":
                st.markdown(bench_viz.steal_badge(o, lang), unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            st.plotly_chart(bench_viz.fig_options(opts["offense"], lang, t["off"] + f"（{opts['batter']}）"),
                            width="stretch", config={"displayModeBar": False})
        with b:
            st.plotly_chart(bench_viz.fig_options(opts["defense"], lang, t["def"] + f"（{opts['pitcher']}）"),
                            width="stretch", config={"displayModeBar": False})

        plan = opts.get("pen_plan")
        if plan:
            st.markdown(_plan_html(plan, lang, s), unsafe_allow_html=True)

        def rows(o):
            return [{t["opt"]: x["label_ja"] if lang == "JA" else x.get("label_en", x["option"]), t["runs"]: round(x["runs_mean"], 2),
                     t["score"]: f"{x['p_score']:.0%}", t["owp"]: f"{x['wp_offense']:.1%}",
                     t["delta"]: f"{100 * x['delta_wp_offense']:+.1f}"} for x in o]
        with st.expander("表で見る" if lang == "JA" else "Show as tables"):
            if mu:
                st.dataframe([{t["batter"]: (t["now"] if r["order"] == 0 else t["next"].format(n=r["order"])) + "  " + r["name"],
                               t["on_base"]: f"{r['p_on_base']:.0%}", t["k"]: f"{r['p_k']:.0%}",
                               t["xbh"]: f"{r['p_xbh']:.0%}", t["hr"]: f"{r['p_hr']:.1%}"} for r in mu["rows"]],
                             hide_index=True, width="stretch")
            st.dataframe(rows(opts["offense"]), hide_index=True, width="stretch")
            st.dataframe(rows(opts["defense"]), hide_index=True, width="stretch")
        st.caption(t["note"].format(d=opts["records_through"]))


    return event


def render(lang: str) -> None:
    t = T[lang]
    st.markdown(f"## {t['title']}")
    view = st.radio("", ["ライブ", "振り返り"] if lang == "JA" else ["Live", "Look back"], horizontal=True,
                    key=f"bench_view_{lang}", label_visibility="collapsed")
    if view in ("振り返り", "Look back"):
        import bench_review
        bench_review.render(lang)
        return
    games = _games()
    if not games:
        st.info(t["none"])
        if st.button(t["reload"], key="bench_reload"):
            st.session_state.pop("bench_games", None)
            st.rerun()
        return
    live = [g for g in games if g.get("status") == "In Progress"]
    pks = [g["gamePk"] for g in games]
    default = pks.index(live[0]["gamePk"]) if live else 0
    label = {g["gamePk"]: f"{'🔴' if g.get('status') == 'In Progress' else '⚪'} {g['away_team']} @ {g['home_team']}"
             for g in games}
    c1, c2 = st.columns([3, 1])
    with c1:
        st.session_state["bench_pk"] = st.selectbox(t["pick"], pks, index=default, format_func=label.get, key="bench_select")
    with c2:
        st.checkbox(t["follow"], value=True, key="bench_follow")
    if st.session_state.get("bench_follow", True):
        st.fragment(run_every=30)(_panel)(lang)
    else:
        _panel(lang)
