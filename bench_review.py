"""
Look back at a finished game, one plate appearance at a time.

Pick a date and a game; move through the game with the slider, the
previous / next buttons, the list of the bench's decisions, or a tap on the
win-probability chart. For the moment before each plate appearance the
screen shows what the bench screen would have shown then (player records
up to the previous season: bench_replay.py), what the manager actually did
and how the model rated it among the options, and what happened next.
"""

from __future__ import annotations

import html
from datetime import date, timedelta

import streamlit as st

import bench_live
import bench_replay
import bench_screen
import bench_viz
from analysis import leverage_index
from live_feed import get_todays_games

TXT = {
    "JA": {"date": "日付", "game": "試合", "none": "この日の終わった試合はありません。", "moment": "局面",
           "prev": "◀ 前", "next": "次 ▶", "hot": "勝負所へ", "decisions": "采配の場面", "pick": "（選ぶ）",
           "actual": "実際には", "result": "結果", "swing": "{team} の勝率 {a:.0%} → {b:.0%}（{d:+.0f}）",
           "rated": "モデルの評価 {d:+.1f} ポイント（候補 {n} 人中 {r} 位）", "rated1": "モデルの評価 {d:+.1f} ポイント",
           "not_rated": "（候補に含まれていません）", "steal_call": "モデルの判定は「{c}」",
           "records": "選手の成績は {d} まで（この試合の前年まで。後から分かった成績は使っていません）。",
           "loading": "試合を読み込み中…", "cannot": "この局面は再現できませんでした（打順が読み取れません）。",
           "tap": "グラフの点をタップしても局面を移れます。",
           "limits": "継投の値は、交代した投手がこの回を投げ、その後はブルペン全体の平均が投げるとして計算しています（序盤の継投は大きく出やすい）。"},
    "EN": {"date": "Date", "game": "Game", "none": "No finished games on this date.", "moment": "Moment",
           "prev": "◀ Prev", "next": "Next ▶", "hot": "Biggest moment", "decisions": "Bench decisions", "pick": "(pick)",
           "actual": "What happened", "result": "Result", "swing": "{team} WP {a:.0%} → {b:.0%} ({d:+.0f})",
           "rated": "Model: {d:+.1f} pt ({r} of {n} candidates)", "rated1": "Model: {d:+.1f} pt",
           "not_rated": "(not among the candidates)", "steal_call": "the model said '{c}'",
           "records": "Player records through {d} (the season before this game: no hindsight).",
           "loading": "Loading the game…", "cannot": "This moment could not be rebuilt (lineups unreadable).",
           "tap": "Tap a point on the chart to jump there.",
           "limits": "A pitching change assumes the new pitcher finishes this half and the pooled bullpen pitches after it (early changes look large)."},
}
DEC_JA = {"pinch": "代打・代走", "pitch_change": "継投", "steal": "盗塁", "bunt": "バント", "intentional_walk": "敬遠",
          "defense": "守備交代"}


def _label(t: dict, lang: str) -> str:
    half = ("表" if t["top_bottom"] == "top" else "裏") if lang == "JA" else ("T" if t["top_bottom"] == "top" else "B")
    bases = "".join(b if r else "" for b, r in zip(("一", "二", "三") if lang == "JA" else ("1", "2", "3"), t["runners"])) or ("走者なし" if lang == "JA" else "empty")
    inn = f"{t['inning']}{half}" if lang == "JA" else f"{half}{t['inning']}"
    outs = f"{t['outs']}死" if lang == "JA" else f"{t['outs']} out"
    return f"{inn} {outs} {bases} — {t['batter']}"


@st.cache_data(show_spinner=False, max_entries=8)
def _feed(pk: int) -> dict | None:
    return bench_live.get_feed(pk)


@st.cache_data(show_spinner=False, max_entries=16)
def _games_on(day: str) -> list[dict]:
    return [g for g in get_todays_games(day) if g.get("status") in ("Final", "Game Over", "Completed Early")]


def _moment(pk: int, k: int):
    key = ("review", pk, k)
    if st.session_state.get("review_cache_key") != key:
        feed = _feed(pk)
        got = bench_replay.moment_at(feed, k)
        out = None
        if got is not None:
            out = {"options": bench_live.bench_options(feed, got=got), "matchups": bench_live.matchups(feed, got=got)}
        st.session_state["review_cache"] = out
        st.session_state["review_cache_key"] = key
    return st.session_state["review_cache"]


def _rate(dec: dict, opts: dict, lang: str, x: dict) -> str:
    """How the model rated what the manager did, among the candidates of that kind."""
    allo = opts["all"]
    side_off = dec["type"] in ("pinch", "steal", "bunt")
    if dec["type"] == "steal":
        st_ = [o for o in allo if o["option"].startswith("steal")]
        if st_:
            c = ("走れ" if st_[0]["go"] else "走るな") if lang == "JA" else ("go" if st_[0]["go"] else "hold")
            return x["steal_call"].format(c=c) + f"（{st_[0]['p_success']:.0%} / {st_[0]['breakeven']:.0%}）"
        return ""
    if dec["type"] in ("bunt", "intentional_walk"):
        o = [o for o in allo if o["option"] == ("bunt" if dec["type"] == "bunt" else "intentional_walk")]
        if not o:
            return ""
        d = 100 * o[0]["delta_wp_offense"] * (1 if side_off else -1)
        return x["rated1"].format(d=d)
    pid = dec.get("player")
    kinds = ("pinch_hit", "pinch_run") if dec["type"] == "pinch" else ("pitch_change",)
    cands = [o for o in allo if o["option"].split(":")[0] in kinds]
    me = [o for o in cands if o["option"].endswith(f":{pid}")]
    if not me:
        return x["not_rated"]
    sign = 1 if dec["type"] == "pinch" else -1          # a pitching change is good when it lowers the offense WP
    ranked = sorted(cands, key=lambda o: -sign * o["wp_offense"])
    return x["rated"].format(d=sign * 100 * me[0]["delta_wp_offense"], n=len(ranked), r=ranked.index(me[0]) + 1)


def render(lang: str) -> None:
    x = TXT[lang]
    c1, c2 = st.columns([1, 3])
    with c1:
        day = st.date_input(x["date"], value=st.session_state.get("review_day", date(2026, 9, 27)),
                            max_value=date.today() - timedelta(days=1), key="review_day")
    games = _games_on(day.isoformat())
    if not games:
        st.info(x["none"])
        return
    with c2:
        pk = st.selectbox(x["game"], [g["gamePk"] for g in games], key="review_game",
                          format_func=lambda p: next(f"{g['away_team']} @ {g['home_team']}" for g in games if g["gamePk"] == p))
    with st.spinner(x["loading"]):
        feed = _feed(pk)
    if not feed:
        st.info(x["none"])
        return
    tl = bench_replay.timeline(feed)
    if not tl:
        st.info(x["none"])
        return
    n = len(tl)
    if st.session_state.get("review_pk") != pk:
        st.session_state["review_pk"] = pk
        st.session_state["review_k"] = 0
    k = st.session_state.get("review_k", 0)
    series = bench_viz.wp_series(feed)
    # navigation
    b1, b2, b3, b4 = st.columns([1, 1, 1, 3])
    if b1.button(x["prev"], key="review_prev", width="stretch") and k > 0:
        k -= 1
    if b2.button(x["next"], key="review_next", width="stretch") and k < n - 1:
        k += 1
    if b3.button(x["hot"], key="review_hot", width="stretch"):
        def li(t):
            try:
                return leverage_index(t["inning"], t["top_bottom"], t["outs"], t["runners"], t["score_diff"],
                                      t["scheduled_innings"], t["extra_runner"])
            except Exception:
                return 0.0
        k = max(range(n), key=lambda j: li(tl[j]))
    dec_ks = [t["k"] for t in tl if any(d["type"] != "defense" for d in t["decisions"])]
    with b4:
        pick = st.selectbox(x["decisions"], [None] + dec_ks, key=f"review_dec_{pk}",
                            format_func=lambda j: x["pick"] if j is None else
                            _label(tl[j], lang) + "  ・" + "／".join(sorted({DEC_JA.get(d["type"], d["type"]) if lang == "JA" else d["type"]
                                                                           for d in tl[j]["decisions"] if d["type"] != "defense"})))
    if pick is not None and st.session_state.get("review_last_pick") != pick:
        st.session_state["review_last_pick"] = pick
        k = pick
    k = st.select_slider(x["moment"], options=list(range(n)), value=k, format_func=lambda j: _label(tl[j], lang),
                         key=f"review_slider_{pk}_{k}")
    st.session_state["review_k"] = k

    t = tl[k]
    prev = tl[k - 1] if k else None
    home_s = prev["home_score"] if prev else 0
    away_s = prev["away_score"] if prev else 0
    s = {**t, "score_home": home_s, "score_away": away_s, "score_diff": home_s - away_s, "balls": None,
         "status": "Final"}
    got = _moment(pk, k)
    opts = got["options"] if got else None
    mu = got["matchups"] if got else None
    if got is None:
        st.caption(x["cannot"])
    # what happened
    bat_team = s["home_team"] if s["top_bottom"] == "bottom" else s["away_team"]
    a = series[k]["wp"] if k < len(series) else None
    b_ = series[k + 1]["wp"] if k + 1 < len(series) else None
    lines = []
    for d in t["decisions"]:
        if d["type"] == "defense":
            continue
        rate = _rate(d, opts, lang, x) if opts else ""
        lines.append(f"<li><b>{html.escape(DEC_JA.get(d['type'], d['type']) if lang == 'JA' else d['type'])}</b> — "
                     f"{html.escape(d.get('text', ''))}<br><span style='color:#c3c2b7'>{html.escape(rate)}</span></li>")
    swing = ""
    if a is not None and b_ is not None:
        aw, bw = (a, b_) if s["top_bottom"] == "bottom" else (1 - a, 1 - b_)
        swing = x["swing"].format(team=bat_team, a=aw, b=bw, d=100 * (bw - aw))
    st.markdown(
        f'<div style="background:#141428;border-left:4px solid #3987e5;border-radius:0 10px 10px 0;padding:12px 16px;margin-top:6px">'
        f'<div style="color:#f0efec;font-weight:bold">{x["actual"]}</div>'
        + (f'<ul style="color:#f0efec;margin:6px 0">{"".join(lines)}</ul>' if lines else "")
        + f'<div style="color:#f0efec">{x["result"]}: {html.escape(t["description"])}</div>'
        f'<div style="color:#c3c2b7">{html.escape(swing)}</div></div>', unsafe_allow_html=True)
    event = bench_screen._draw(lang, s, opts, mu, series, mark_i=k, chart_key=f"review_wp_{pk}")
    if event is not None:
        pts = (event.get("selection") or {}).get("points") if isinstance(event, dict) else getattr(getattr(event, "selection", None), "points", None)
        if pts:
            j = int(round(pts[0].get("x", k)))
            if 0 <= j < n and j != k:
                st.session_state["review_k"] = j
                st.rerun()
    st.caption(x["tap"])
    if opts:
        st.caption(x["records"].format(d=opts["records_through"]) + " " + x["limits"])
