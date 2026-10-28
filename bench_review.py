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
import json
from datetime import date, timedelta
from pathlib import Path

import streamlit as st

import bench_live
import bench_replay
import bench_screen
import bench_viz
import pen_wp
from analysis import leverage_index
import live_feed
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
           "limits": "継投の値は、交代した投手がこの回を投げ、7 回以降はそのチームの直近 60 日の起用（今の点差での使い分け。その投手と、この試合で投げた投手、2 日連続登板の投手を除く）、6 回まではブルペン全体の平均が投げるとして計算しています（序盤の継投は大きく出やすい）。",
           "views": ["⚾ 局面", "📝 試合のまとめ"], "view": "表示",
           "not_rated2": "対象外",
           "sum_head": "{team}：采配 {n} 回（評価できたもの {r} 回）　モデルの最善と一致 {a} 回　プラス評価 {p} 回",
           "sum_cols": ["回", "チーム", "采配", "モデル評価", "最善候補", "最善との差", "起用パターン", "結果", "勝率の変化", "記録"],
           "sum_none": "この試合には代打・代走・継投・盗塁・バント・敬遠の記録がありません。",
           "sum_pick": "表の行を選ぶと、その局面に移ります。",
           "sum_note": "モデル評価＝打席の前に、その采配で采配したチームの勝率が「何もしない」よりどれだけ上がると見ていたか（ポイント）。"
                       "結果と勝率の変化は、実際に起きたこと（運を含む）。両者は別物です。"
                       "「対象外」＝前年の記録がない選手か、モデルが扱わない場面（満塁の代走など）。守備位置・休養・故障などはモデルに入っていません。",
           "miss_btn": "🔎 モデルならここで動いた、という場面を探す（約 30 秒）",
           "miss_head": "モデルが 1 ポイント以上の上積みを見ていたのに、その種類の采配がなかった場面（大きい順）",
           "miss_none": "そのような場面はありませんでした（基準：1 ポイント以上）。",
           "miss_row": "{inn}　{team}：{opt}　{g:+.1f} ポイント",
           "miss_note": "継投は 7 回以降のみ（それより前は大きく出すぎるため）。盗塁は「走れ」の判定のときだけ。",
           "pen_note": "選手・終盤の継投の使い方込みの勝率（2026 年で事前登録して検証済み）",
           "note_head": "📝 試合の解説", "note_cap": "Claude（Sonnet 5.5）が、この画面と同じ勝率と試合記録だけを渡されて事前に書いた文章です。数字とスコアは渡した値と一致するものだけを残しています。",
           "after_sub": "この打席で先にあった交代のあとの状況で評価",
           "usage": "普段の起用パターンなら {p}（モデルの順位 {r}/{n}）", "usage_same": "普段の起用パターンどおり（モデルの順位 {r}/{n}）",
           "usage_note": "起用パターン＝そのチームが直近 30 日、この回・この点差でいちばん多く使った投手（使える投手のうち）。2026 年の検証では、監督の選択との一致はモデルの最善候補より高めでした（28.7% と 20.1%）。",
           "find": "探し方", "finds": ["📅 日付から", "⚾ チームから"], "team": "チーム", "season": "年",
           "no_team_games": "この年の終わった試合が見つかりません。", "win": "勝", "loss": "負",
           "keys": "山場（勝率が大きく動いた打席。押すとその局面へ）"},
    "EN": {"date": "Date", "game": "Game", "none": "No finished games on this date.", "moment": "Moment",
           "prev": "◀ Prev", "next": "Next ▶", "hot": "Biggest moment", "decisions": "Bench decisions", "pick": "(pick)",
           "actual": "What happened", "result": "Result", "swing": "{team} WP {a:.0%} → {b:.0%} ({d:+.0f})",
           "rated": "Model: {d:+.1f} pt ({r} of {n} candidates)", "rated1": "Model: {d:+.1f} pt",
           "not_rated": "(not among the candidates)", "steal_call": "the model said '{c}'",
           "records": "Player records through {d} (the season before this game: no hindsight).",
           "loading": "Loading the game…", "cannot": "This moment could not be rebuilt (lineups unreadable).",
           "tap": "Tap a point on the chart to jump there.",
           "limits": "A pitching change assumes the new pitcher finishes this half; from the 7th the team's usage over the last 60 days at the current score pitches (without him, pitchers already used and anyone who pitched on both previous days), before that the pooled bullpen (early changes look large).",
           "views": ["⚾ Moment", "📝 Game summary"], "view": "View",
           "not_rated2": "not rated",
           "sum_head": "{team}: {n} calls ({r} rated), matched the model's best {a}, positive {p}",
           "sum_cols": ["Inning", "Team", "Call", "Model", "Best candidate", "Gap to best", "Usage pick", "Result", "WP change", "Play-by-play"],
           "sum_none": "No pinch hitters, pinch runners, pitching changes, steals, bunts or intentional walks in this game.",
           "sum_pick": "Pick a row to jump to that moment.",
           "sum_note": "Model = how many WP points the model expected the call to add for the team that made it, before the play. "
                       "Result and WP change are what happened (luck included); they are different things. 'not rated' = no previous-season record, or a situation the model does not cover (e.g. a pinch runner with the bases loaded). Positions, rest and injuries are not in the model.",
           "miss_btn": "🔎 Find moments where the model would have moved (about 30 s)",
           "miss_head": "Moments where the model saw a gain of 1 point or more and no call of that kind was made (largest first)",
           "miss_none": "None (threshold: 1 point).",
           "miss_row": "{inn}  {team}: {opt}  {g:+.1f} pt",
           "miss_note": "Pitching changes from the 7th inning only (earlier ones are overstated). Steals only when the call is 'go'.",
           "pen_note": "WP with players and late-inning bullpen usage (pre-registered, checked on 2026)",
           "note_head": "📝 Game note (Japanese)", "note_cap": "Written in advance by Claude (Sonnet 5.5) from the same win probabilities and play log as this screen; only notes whose numbers and scores match those inputs are kept.",
           "after_sub": "rated after the earlier substitution in this plate appearance",
           "usage": "Usage pattern: {p} (model rank {r}/{n})", "usage_same": "Matches the usage pattern (model rank {r}/{n})",
           "usage_note": "Usage pick = the available reliever the team used most over the last 30 days in this inning and score state. In the 2026 check it matched managers more often than the model's best (28.7% vs 20.1%).",
           "find": "Find by", "finds": ["📅 Date", "⚾ Team"], "team": "Team", "season": "Season",
           "no_team_games": "No finished games found for this season.", "win": "W", "loss": "L",
           "keys": "Key moments (biggest win-probability swings; tap to jump)"},
}
DEC_JA = {"pinch": "代打・代走", "pitch_change": "継投", "steal": "盗塁", "bunt": "バント", "intentional_walk": "敬遠",
          "defense": "守備交代"}


NOTES_DIR = Path(__file__).resolve().parent / "research/game_notes/published"


@st.cache_data(show_spinner=False)
def _notes(season: int) -> dict:
    p = NOTES_DIR / f"notes_{season}.json"
    try:
        return json.loads(p.read_text()).get("notes", {}) if p.exists() else {}
    except (OSError, ValueError):
        return {}


def game_note(pk: int, game_date: str | None) -> str | None:
    """The pre-generated note for a finished game, or None."""
    try:
        season = int(str(game_date)[:4])
    except (TypeError, ValueError):
        return None
    return _notes(season).get(str(int(pk)))


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


@st.cache_data(show_spinner=False, ttl=86400)
def _teams() -> list[dict]:
    return live_feed.get_teams()


@st.cache_data(show_spinner=False, max_entries=32, ttl=3600)
def _team_games(team_id: int, season: int) -> list[dict]:
    games = live_feed.get_team_games(team_id, f"{season}-02-15", f"{season}-11-15")
    return [g for g in games if str(g.get("status", "")).startswith(("Final", "Game Over", "Completed Early"))]


def _team_label(g: dict, team_id: int, abbr: dict, x: dict) -> str:
    home = g["home_id"] == team_id
    opp = abbr.get(g["away_id"] if home else g["home_id"], g["away"] if home else g["home"])
    own, oth = (g["home_score"], g["away_score"]) if home else (g["away_score"], g["home_score"])
    res = ""
    if own is not None and oth is not None:
        res = f"  {own}-{oth} {x['win'] if own > oth else x['loss']}"
    return f"{g['date'][5:].replace('-', '/')} {'vs' if home else '@'} {opp}{res}"


def key_moments(tl: list[dict], series: list[dict], n: int = 5) -> list[tuple[int, float]]:
    """The n plate appearances with the largest win-probability swing, in game
    order: [(k, swing in points for the batting team)]."""
    sw = []
    for k, t in enumerate(tl):
        if k + 1 < len(series):
            d = 100 * (series[k + 1]["wp"] - series[k]["wp"])
            sw.append((k, d if t["top_bottom"] == "bottom" else -d))
    top = sorted(sw, key=lambda z: -abs(z[1]))[:n]
    return sorted(top)


def _moment(pk: int, k: int, upto: int | None = None):
    """Options (and matchups) before plate appearance k, or at event `upto` of it
    (a call made after an earlier substitution in the same plate appearance)."""
    key = ("review", pk, k)
    if st.session_state.get("review_cache_key") != key:
        st.session_state["review_cache"] = {}
        st.session_state["review_cache_key"] = key
    cache = st.session_state["review_cache"]
    if upto not in cache:
        feed = _feed(pk)
        got = bench_replay.moment_at(feed, k, upto_event=upto)
        out = None
        if got is not None:
            out = {"options": bench_live.bench_options(feed, got=got), "matchups": bench_live.matchups(feed, got=got)}
        cache[upto] = out
    return cache[upto]


def _series(pk: int, feed: dict, tl: list[dict]) -> list[dict]:
    """Home WP before each plate appearance: PEN_RECAL where available (regular
    season 2022-2026), the team layer elsewhere. The card, the chart, the key
    moments and the game summary all use this one series."""
    series = bench_viz.wp_series(feed)
    pw = pen_wp.game(pk, tl[0].get("game_date")) if tl and tl[0].get("game_type") == "R" else {}
    return pen_wp.apply_to_series(series, [t.get("at_bat_index") for t in tl], pw)[0]


def _rate(dec: dict, opts: dict, lang: str, x: dict) -> str:
    """How the model rated what the manager did, among the candidates of that kind."""
    r = bench_replay.rate(dec, opts)
    if r is None:
        return x["not_rated"]
    if dec["type"] == "steal":
        c = ("走れ" if r["call"] == "go" else "走るな") if lang == "JA" else r["call"]
        return x["steal_call"].format(c=c) + f"（{r['p_success']:.0%} / {r['breakeven']:.0%}）"
    if r["rank"] is None:
        return x["rated1"].format(d=r["gain"])
    return x["rated"].format(d=r["gain"], n=r["n"], r=r["rank"])


def _short(o: dict, ja: bool) -> str:
    """A candidate's name without the option word (継投 / new P)."""
    lab = o.get("label_ja" if ja else "label_en", "")
    for pre in ("継投 ", "代打 ", "代走 ", "new P ", "PH ", "PR "):
        if lab.startswith(pre):
            return lab[len(pre):]
    return lab


def _inn(t: dict, lang: str) -> str:
    if lang == "JA":
        return f"{t['inning']}回{'表' if t['top_bottom'] == 'top' else '裏'}"
    return f"{'T' if t['top_bottom'] == 'top' else 'B'}{t['inning']}"


@st.cache_data(show_spinner=False, max_entries=8)
def _review(pk: int) -> list[dict]:
    feed = _feed(pk)
    rows = bench_replay.game_review(feed, _series(pk, feed, bench_replay.timeline(feed)))
    for r in rows:                                     # keep only what the screen needs (cacheable, small)
        rt = r["rating"]
        if rt:
            r["rating"] = {k: v for k, v in rt.items() if k not in ("me", "best")}
            r["rating"]["me_ja"], r["rating"]["me_en"] = rt["me"]["label_ja"], rt["me"].get("label_en", "")
            r["rating"]["best_ja"] = rt["best"]["label_ja"] if rt.get("best") else None
            r["rating"]["best_en"] = rt["best"].get("label_en") if rt.get("best") else None
    return rows


@st.cache_data(show_spinner=False, max_entries=8)
def _missed(pk: int) -> list[dict]:
    rows = bench_replay.missed_chances(_feed(pk))
    for r in rows:
        o = r.pop("option")
        r["label_ja"], r["label_en"] = o["label_ja"], o.get("label_en", o["option"])
    return rows


def _summary(lang: str, pk: int, feed: dict, x: dict) -> None:
    ja = lang == "JA"
    gd = feed.get("gameData", {}).get("teams", {})
    team = {"home": gd.get("home", {}).get("name", "home"), "away": gd.get("away", {}).get("name", "away")}
    with st.spinner(x["loading"]):
        rows = _review(pk)
    if not rows:
        st.info(x["sum_none"])
        return
    for side in ("away", "home"):
        mine = [r for r in rows if r["side"] == side]
        rated = [r for r in mine if r["rating"]]
        st.markdown(x["sum_head"].format(team=team[side], n=len(mine), r=len(rated),
                                         a=sum(1 for r in rated if r["rating"]["agree"]),
                                         p=sum(1 for r in rated if r["rating"]["gain"] > 0)))
    kind_name = (lambda d: DEC_JA.get(d, d)) if ja else (lambda d: d.replace("_", " "))
    pts = []
    for r in rows:
        rt = r["rating"]
        if not rt:
            continue
        me = rt["me_ja"] if ja else rt["me_en"]
        if r["type"] in ("steal", "bunt", "intentional_walk"):
            me = kind_name(r["type"])
        pts.append({"label": f"{_inn(r, lang)} {me}", "side": r["side"], "gain": rt["gain"],
                    "best_gain": rt["best_gain"], "best_label": rt["best_ja"] if ja else rt["best_en"]})
    if pts:
        st.plotly_chart(bench_viz.fig_calls(pts, team["home"], team["away"], lang), width="stretch",
                        config={"displayModeBar": False}, key=f"review_calls_{pk}")
    c = x["sum_cols"]
    table = []
    for r in rows:
        rt = r["rating"]
        if rt is None:
            model, best, gap = x["not_rated2"], "", ""
        elif r["type"] == "steal":
            call = ("走れ" if rt["call"] == "go" else "走るな") if ja else rt["call"]
            model = f"{rt['gain']:+.2f}（{call} {rt['p_success']:.0%}/{rt['breakeven']:.0%}）"
            best, gap = "", ""
        else:
            model = f"{rt['gain']:+.2f}" + (f"（{rt['rank']}/{rt['n']}）" if rt["rank"] else "")
            best = (rt["best_ja"] if ja else rt["best_en"]) or ""
            gap = f"{rt['gain'] - rt['best_gain']:+.2f}" if rt["best_gain"] is not None else ""
        rt_me = (rt["me_ja"] if ja else rt["me_en"]) if rt and r["type"] in ("pinch", "pitch_change") else kind_name(r["type"])
        if not rt and r["type"] in ("pinch", "pitch_change"):
            rt_me = kind_name(r["type"])
        up = r.get("usage")
        upl = ""
        if up:
            upl = ("采配と同じ" if ja else "same") if up["player"] == r.get("player") else _short(up["option"], ja)
        table.append({c[0]: _inn(r, lang), c[1]: team[r["side"]], c[2]: rt_me,
                      c[3]: model, c[4]: best, c[5]: gap, c[6]: upl, c[7]: r["event"],
                      c[8]: "" if r["swing"] is None else f"{r['swing']:+.1f}", c[9]: r["text"]})
    ev = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                      key=f"review_table_{pk}")
    st.caption(x["sum_pick"] + " " + x["sum_note"])
    sel = (ev.get("selection") or {}).get("rows") if isinstance(ev, dict) else getattr(getattr(ev, "selection", None), "rows", None)
    if sel:
        st.session_state["review_k"] = rows[sel[0]]["k"]
        st.session_state["review_view_next"] = 0
        st.rerun()
    # moments the model would have moved
    if st.button(x["miss_btn"], key=f"review_miss_{pk}") or st.session_state.get("review_miss_pk") == pk:
        st.session_state["review_miss_pk"] = pk
        with st.spinner(x["loading"]):
            miss = _missed(pk)
        st.markdown(f"**{x['miss_head']}**")
        if not miss:
            st.caption(x["miss_none"])
        for i, m in enumerate(miss):
            label = x["miss_row"].format(inn=_label(m, lang), team=team[m["side"]],
                                         opt=m["label_ja"] if ja else m["label_en"], g=m["gain"])
            if st.button(label, key=f"review_miss_go_{pk}_{i}", width="stretch"):
                st.session_state["review_k"] = m["k"]
                st.session_state["review_view_next"] = 0
                st.rerun()
        st.caption(x["miss_note"])


def render(lang: str) -> None:
    x = TXT[lang]
    how = st.radio(x["find"], x["finds"], horizontal=True, key=f"review_find_{lang}")
    if how == x["finds"][0]:
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
    else:
        teams = _teams()
        if not teams:
            st.info(x["none"])
            return
        abbr = {t["id"]: t["abbr"] for t in teams}
        c1, c2, c3 = st.columns([2, 1, 3])
        with c1:
            tid = st.selectbox(x["team"], [t["id"] for t in teams], key="review_team",
                               format_func=lambda i: next(t["name"] for t in teams if t["id"] == i))
        with c2:
            season = st.selectbox(x["season"], [2026, 2025, 2024, 2023, 2022], key="review_season")
        tg = _team_games(tid, season)
        if not tg:
            st.info(x["no_team_games"])
            return
        with c3:
            pk = st.selectbox(x["game"], [g["gamePk"] for g in tg], key=f"review_tgame_{tid}_{season}",
                              format_func=lambda p: next(_team_label(g, tid, abbr, x) for g in tg if g["gamePk"] == p))
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
    vkey = f"review_view_{lang}"
    if "review_view_next" in st.session_state:          # a jump from the summary back to the moment view
        st.session_state[vkey] = x["views"][st.session_state.pop("review_view_next")]
    note = game_note(pk, tl[0].get("game_date"))
    if note:
        with st.expander(x["note_head"], expanded=True):
            st.markdown(html.escape(note))
            st.caption(x["note_cap"])
    view = st.radio(x["view"], x["views"], horizontal=True, key=vkey, label_visibility="collapsed")
    if view == x["views"][1]:
        _summary(lang, pk, feed, x)
        return
    k = st.session_state.get("review_k", 0)
    series = _series(pk, feed, tl)
    pw = pen_wp.game(pk, tl[0].get("game_date")) if tl[0].get("game_type") == "R" else {}
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
    km = key_moments(tl, series)
    if km:
        st.caption(x["keys"])
        cols = st.columns(len(km))
        for (kk, dd), col in zip(km, cols):
            t_ = tl[kk]
            inn = f"{t_['inning']}{'表' if t_['top_bottom'] == 'top' else '裏'}" if lang == "JA" else \
                f"{'T' if t_['top_bottom'] == 'top' else 'B'}{t_['inning']}"
            if col.button(f"{inn} {dd:+.0f}", key=f"review_km_{pk}_{kk}", help=t_.get("description", ""), width="stretch"):
                k = kk
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
        u = bench_replay.upto_for(t, d)
        o_d = opts if u is None else (_moment(pk, k, u) or {}).get("options")
        rate = _rate(d, o_d, lang, x) if o_d else ""
        if u is not None and o_d:
            rate += "（" + x["after_sub"] + "）" if lang == "JA" else f" ({x['after_sub']})"
        up = bench_replay.usage_pick(t, tl, d, o_d) if d["type"] == "pitch_change" else None
        if up:
            rate += "　／　" + (x["usage_same"].format(r=up["rank"], n=up["n"]) if up["player"] == d.get("player")
                               else x["usage"].format(p=_short(up["option"], lang == "JA"), r=up["rank"], n=up["n"]))
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
    ab = t.get("at_bat_index")
    home_wp = pw.get(ab) if (pw and ab is not None) else None
    event = bench_screen._draw(lang, s, opts, mu, series, mark_i=k, chart_key=f"review_wp_{pk}",
                               home_wp=home_wp, wp_note=x["pen_note"] if home_wp is not None else None)
    if event is not None:
        pts = (event.get("selection") or {}).get("points") if isinstance(event, dict) else getattr(getattr(event, "selection", None), "points", None)
        if pts:
            j = int(round(pts[0].get("x", k)))
            if 0 <= j < n and j != k:
                st.session_state["review_k"] = j
                st.rerun()
    st.caption(x["tap"])
    if opts:
        st.caption(x["records"].format(d=opts["records_through"]) + " " + x["limits"]
                   + (" " + x["usage_note"] if any(d["type"] == "pitch_change" for d in t["decisions"]) else ""))
