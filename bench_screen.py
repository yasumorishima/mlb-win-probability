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
                   "選手の成績は {d} まで。代走・守備固め・この回より先の継投は考えていません。"},
    "EN": {"title": "📋 Bench", "pick": "Game", "none": "No games today.", "reload": "Reload schedule",
           "follow": "Follow (every 30 s)", "updated": "Updated {t} UTC", "failed": "Fetch failed (last success {t} UTC)",
           "top": "Top", "bot": "Bot", "outs": "out", "wp": "{team} win prob.", "player": "with players", "li": "Leverage",
           "matchup": "Matchups (pitcher {p})", "batter": "Batter", "on_base": "On base", "k": "K", "xbh": "XBH", "hr": "HR",
           "now": "now", "next": "next {n}", "not_live": "Not in progress ({s}).", "cannot": "Could not read the lineups.",
           "options": "Options", "off": "Offense", "def": "Defense", "opt": "Option", "runs": "Runs this half",
           "score": "P(score)", "owp": "Offense WP", "delta": "Change (pt)",
           "note": "Headline WP: engine + team layer; 'with players' and the options: the player layer (pre-registered, "
                   "confirmed on 2026). Player records through {d}. Pinch runners, defensive subs and relief beyond this half "
                   "are not modelled."},
}


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
    bat_home = s["top_bottom"] == "bottom"
    bat_team = s["home_team"] if bat_home else s["away_team"]
    sched, xr = s.get("scheduled_innings", 9), s.get("extra_runner", True)
    tl = team_layer_wp(s["inning"], s["top_bottom"], s["outs"], s["runners"], s["score_diff"], sched, xr,
                       s.get("home_id"), s.get("away_id"), s.get("game_date"), s.get("game_type"))
    home = tl["team_wp"] if tl.get("applied") else state_wp(s["inning"], s["top_bottom"], s["outs"], s["runners"],
                                                               s["score_diff"], sched, xr)
    bat_wp = home if bat_home else 1 - home
    opts = c.get("options")
    player_wp = opts["offense"][0]["wp_offense"] if opts else None
    li = leverage_index(s["inning"], s["top_bottom"], s["outs"], s["runners"], s["score_diff"], sched, xr) if s["outs"] < 3 else 0.0
    bases = "".join(b if r else "·" for b, r in zip(("一", "二", "三") if lang == "JA" else ("1", "2", "3"), s["runners"]))
    left, right = st.columns([2, 3])
    with left:
        st.markdown(
            f'<div style="background:#141428;border-radius:12px;padding:14px;">'
            f'<div style="color:#aaa">{html.escape(s["away_team"])} @ {html.escape(s["home_team"])}</div>'
            f'<div style="font-size:2.2rem;font-weight:bold;color:#fff">{s["score_away"]} – {s["score_home"]}</div>'
            f'<div style="font-size:1.2rem;color:#ccc">{s["inning"]}{t["top"] if s["top_bottom"] == "top" else t["bot"]}'
            f' &nbsp; {s["outs"]} {t["outs"]} &nbsp; [{bases}]'
            + (f' &nbsp; {s["balls"]}-{s["strikes"]}' if s.get("balls") is not None else "") + '</div>'
            f'<div style="margin-top:10px;color:#aaa">{html.escape(t["wp"].format(team=bat_team))}</div>'
            f'<div style="font-size:3rem;font-weight:bold;color:#00e5ff">{bat_wp:.0%}</div>'
            + (f'<div style="color:#ccc">{t["player"]} {player_wp:.0%}</div>' if player_wp is not None else "")
            + f'<div style="margin-top:8px;color:#ccc">{t["li"]} <b style="font-size:1.4rem;color:#ff9800">{li:.1f}</b>'
            f' ({li_label(li)})</div></div>', unsafe_allow_html=True)
    with right:
        mu = c.get("matchups")
        if mu:
            st.markdown(f"**{html.escape(t['matchup'].format(p=mu['pitcher']))}**")
            st.dataframe([{t["batter"]: (t["now"] if r["order"] == 0 else t["next"].format(n=r["order"])) + "  " + r["name"],
                           t["on_base"]: f"{r['p_on_base']:.0%}", t["k"]: f"{r['p_k']:.0%}",
                           t["xbh"]: f"{r['p_xbh']:.0%}", t["hr"]: f"{r['p_hr']:.1%}"} for r in mu["rows"]],
                         hide_index=True, width="stretch")
        elif not opts:
            st.caption(t["cannot"])
    if opts:
        def rows(o):
            return [{t["opt"]: x["label_ja"] if lang == "JA" else x.get("label_en", x["option"]), t["runs"]: round(x["runs_mean"], 2),
                     t["score"]: f"{x['p_score']:.0%}", t["owp"]: f"{x['wp_offense']:.1%}",
                     t["delta"]: f"{100 * x['delta_wp_offense']:+.1f}"} for x in o]
        a, b = st.columns(2)
        with a:
            st.markdown(f"**{t['off']}**")
            st.dataframe(rows(opts["offense"]), hide_index=True, width="stretch")
        with b:
            st.markdown(f"**{t['def']}**")
            st.dataframe(rows(opts["defense"]), hide_index=True, width="stretch")
        st.caption(t["note"].format(d=opts["records_through"]))


def render(lang: str) -> None:
    t = T[lang]
    st.markdown(f"## {t['title']}")
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
