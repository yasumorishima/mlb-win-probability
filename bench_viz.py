"""
Charts for the bench screen (plotly, dark surface of the app).

Colors (validated with the dataviz skill's validate_palette.js against the
chart surface #141428, dark mode, all checks pass):
  home / away (diverging poles):       #3987e5 / #e66767
  outcome classes (categorical, fixed order: K, BB, 1B, XBH, HR, OUT):
      #3987e5 #d95926 #199e70 #c98500 #d55181 #008300
Ink: primary #f0efec, secondary #c3c2b7, muted #898781; grid #2c2c3a.
"""

from __future__ import annotations

import plotly.graph_objects as go

import team_layer
from analysis import state_wp, wp_after_play
from live_feed import parse_game_plays

SURFACE, INK, INK2, MUTED, GRID = "#141428", "#f0efec", "#c3c2b7", "#898781", "#2c2c3a"
HOME, AWAY = "#3987e5", "#e66767"
OUTCOME_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]
OUTCOME_JA = ["三振", "四死球", "単打", "長打", "本塁打", "凡打"]
OUTCOME_EN = ["K", "BB/HBP", "1B", "2B/3B", "HR", "Out"]
FONT = dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK2, size=13)


def _base_layout(height: int) -> dict:
    return dict(height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=FONT,
                margin=dict(l=48, r=16, t=36, b=36), hoverlabel=dict(bgcolor="#0a0a1a", font_color=INK))


def wp_series(feed: dict) -> list[dict]:
    """Home WP after every completed play, as the headline computes it (team
    layer when it applies, else the engine)."""
    plays = parse_game_plays(feed)
    if not plays:
        return []
    p0 = plays[0]
    strengths = None
    if p0.get("game_type") == "R" and p0.get("home_id") and p0.get("away_id") and p0.get("game_date"):
        try:
            if str(p0["game_date"])[:4] >= str(team_layer.FIRST_SEASON):
                strengths = team_layer.strengths(int(p0["home_id"]), int(p0["away_id"]), str(p0["game_date"]))
        except Exception:
            strengths = None
    sched, xr = p0["scheduled_innings"], p0["extra_runner"]

    def home_wp(inning, half, outs, runners, diff):
        if strengths is None:
            return state_wp(inning, half, outs, runners, diff, sched, xr)
        p = state_wp(inning, half, outs, runners, max(-15, min(15, diff)), sched, xr)
        if outs == 3:
            inning, half, outs = (inning, "bottom", 0) if half == "top" else (inning + 1, "top", 0)
        return team_layer.team_wp(p, inning, half, outs, sched, strengths["d_prev"], strengths["d_cur"])

    out = [{"i": 0, "wp": home_wp(1, "top", 0, (0, 0, 0), 0), "label": "", "text": "", "delta": 0.0}]
    for k, p in enumerate(plays, start=1):
        diff_after = p["home_score"] - p["away_score"]
        game_over = (p["top_bottom"] == "bottom" and p["inning"] >= sched and diff_after > 0) or (
            p["outs_after"] >= 3 and p["inning"] >= sched and
            ((p["top_bottom"] == "bottom" and diff_after != 0) or (p["top_bottom"] == "top" and diff_after < 0)))
        if game_over:
            w = 1.0 if diff_after > 0 else 0.0
        else:
            w = home_wp(p["inning"], p["top_bottom"], min(p["outs_after"], 3), p["runners_after"], diff_after)
        out.append({"i": k, "wp": float(w), "label": f"{p['inning']}{'表' if p['top_bottom'] == 'top' else '裏'}",
                    "label_en": f"{'T' if p['top_bottom'] == 'top' else 'B'}{p['inning']}",
                    "text": p.get("description", ""), "event": p.get("event", ""), "delta": float(w - out[-1]["wp"])})
    return out


def fig_wp(series: list[dict], home: str, away: str, lang: str, animate: bool = True, mark_i: int | None = None) -> go.Figure:
    """Home WP over the game: a line with the area above / below 50% in the
    home / away color, the three biggest swings marked and labelled, and a
    play button that redraws the game from the first pitch."""
    ja = lang == "JA"
    x = [s["i"] for s in series]
    y = [100 * s["wp"] for s in series]
    hover = [(f"{s['label' if ja else 'label_en']} {s['event']}<br>{s['text'][:90]}<br>"
              f"{home}: {100 * s['wp']:.0f}% ({100 * s['delta']:+.0f})") if s["i"] else f"{home}: {100 * s['wp']:.0f}%"
             for s in series]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, mode="lines", y=[max(v, 50) for v in y], line=dict(width=0), fill=None, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, mode="lines", y=[50] * len(x), line=dict(width=0), fill="tonexty", fillcolor="rgba(57,135,229,0.22)",
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, mode="lines", y=[min(v, 50) for v in y], line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(230,103,103,0.22)", hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", line=dict(color=INK, width=2), hovertext=hover,
                             marker=dict(size=9, color="rgba(0,0,0,0)"),       # invisible, but tappable (review)
                             hoverinfo="text", name=home, showlegend=False))
    swings = []
    gap = max(4, len(series) // 12)                 # keep the marked swings apart so their labels do not collide
    for s_ in sorted(series[1:], key=lambda s: -abs(s["delta"])):
        if all(abs(s_["i"] - t["i"]) >= gap for t in swings):
            swings.append(s_)
        if len(swings) == 3:
            break
    fig.add_trace(go.Scatter(x=[s["i"] for s in swings], y=[100 * s["wp"] for s in swings], mode="markers+text",
                             marker=dict(size=11, color=[HOME if s["delta"] > 0 else AWAY for s in swings],
                                         line=dict(color=SURFACE, width=2)),
                             text=[f"{s['label' if ja else 'label_en']} {100 * s['delta']:+.0f}" for s in swings],
                             textposition="top center", textfont=dict(color=INK, size=12), hoverinfo="skip",
                             showlegend=False))
    if mark_i is not None and 0 <= mark_i < len(x):       # the moment being looked at (review)
        fig.add_vline(x=mark_i, line=dict(color="#f0efec", width=1, dash="dot"))
        fig.add_trace(go.Scatter(x=[mark_i], y=[y[mark_i]], mode="markers", marker=dict(size=16, color="#f0efec",
                                 line=dict(color=SURFACE, width=3)), hoverinfo="skip", showlegend=False))
    elif x:
        fig.add_trace(go.Scatter(x=[x[-1]], y=[y[-1]], mode="markers", marker=dict(size=14, color=INK,
                                 line=dict(color=HOME if y[-1] >= 50 else AWAY, width=3)), hoverinfo="skip", showlegend=False))
    lay = _base_layout(300)
    lay.update(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, max(len(x) - 1, 1) + 1]),
               yaxis=dict(range=[0, 100], ticksuffix="%", gridcolor=GRID, zeroline=False, tickvals=[0, 25, 50, 75, 100]),
               hovermode="x unified", transition=dict(duration=500, easing="cubic-in-out"))
    lay["annotations"] = [
        dict(xref="paper", yref="y", x=0.005, y=96, text=f"▲ {home}", showarrow=False, xanchor="left", font=dict(color=HOME, size=12)),
        dict(xref="paper", yref="y", x=0.005, y=4, text=f"▼ {away}", showarrow=False, xanchor="left", font=dict(color=AWAY, size=12)),
    ]
    if animate and len(x) > 2:
        steps = sorted(set([max(2, round(len(x) * k / 25)) for k in range(1, 26)] + [len(x)]))
        # only the WP line is redrawn; the shaded areas stay (animating filled traces is unstable in plotly.js)
        frames = [go.Frame(data=[go.Scatter(x=x[:n], y=y[:n], mode="lines")], traces=[3], name=str(n)) for n in steps]
        fig.frames = frames
        lay["updatemenus"] = [dict(type="buttons", showactive=False, x=1.0, y=1.14, xanchor="right", yanchor="top",
                                   bgcolor="#1f1f3a", bordercolor=GRID, font=dict(color=INK, size=12),
                                   buttons=[dict(label="▶ 試合を再生" if ja else "▶ Replay game", method="animate",
                                                 args=[None, dict(frame=dict(duration=70, redraw=False),
                                                                  transition=dict(duration=0), fromcurrent=False,
                                                                  mode="immediate")])])]
    fig.update_layout(**lay)
    return fig


def fig_options(rows: list[dict], lang: str, title: str) -> go.Figure:
    """Change in the batting team's win probability for each option (points):
    blue = helps the batting team, red = helps the fielding team."""
    ja = lang == "JA"
    rows = [r for r in rows if r["kind"] != "none"]

    def short(r):
        if r["kind"] == "steal":            # the full call is shown as a badge above the charts
            return (("二盗" if "steal_2b" in r["option"] else "三盗") if ja else r["option"].split(":")[0].replace("_", " "))
        return r["label_ja"] if ja else r.get("label_en", r["option"])
    labels = [short(r) for r in rows][::-1]
    vals = [100 * r["delta_wp_offense"] for r in rows][::-1]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker=dict(color=[HOME if v >= 0 else AWAY for v in vals],
                                                                           line=dict(color=SURFACE, width=2)),
                           text=[f"{v:+.1f}" if abs(v) >= 0.95 else f"{v:+.2f}" for v in vals], textposition="outside", textfont=dict(color=INK, size=12),
                           hovertemplate="%{y}: %{x:+.2f} pt<extra></extra>", width=0.6))
    lay = _base_layout(max(140, 34 * len(rows) + 80))
    lay["margin"] = dict(l=48, r=16, t=56, b=36)
    span = max([abs(v) for v in vals] + [0.5]) * 1.35
    key = ("<span style='font-size:11px;color:#c3c2b7'>青＝攻撃側に有利　赤＝守備側に有利（勝率の増減、ポイント）</span>" if ja else
           "<span style='font-size:11px;color:#c3c2b7'>blue = helps the batting team, red = helps the fielding team (WP points)</span>")
    lay.update(title=dict(text=f"{title}<br>{key}", font=dict(color=INK, size=14), x=0, xanchor="left"),
               xaxis=dict(range=[-span, span], gridcolor=GRID, zeroline=True, zerolinecolor=MUTED, zerolinewidth=1,
                          ticksuffix=" pt"),
               yaxis=dict(automargin=True, tickfont=dict(color=INK, size=12)), bargap=0.35,
               transition=dict(duration=500, easing="cubic-in-out"))
    fig.update_layout(**lay)
    return fig


def fig_matchups(mu: dict, lang: str) -> go.Figure:
    """Outcome mix of the current batter and the next three against the
    current pitcher (stacked bars, fixed category order, values on the bars)."""
    ja = lang == "JA"
    names = [(("今 " if ja else "now ") if r["order"] == 0 else (f"次{r['order']} " if ja else f"+{r['order']} ")) + r["name"]
             for r in mu["rows"]][::-1]
    groups = [("p_k",), ("p_bb",), ("p_1b",), ("p_2b", "p_3b"), ("p_hr",), ("p_out",)]
    fig = go.Figure()
    for k, (cols, color) in enumerate(zip(groups, OUTCOME_COLORS)):
        v = [100 * sum(r[c] for c in cols) for r in mu["rows"]][::-1]
        fig.add_trace(go.Bar(x=v, y=names, orientation="h", name=(OUTCOME_JA if ja else OUTCOME_EN)[k],
                             marker=dict(color=color, line=dict(color=SURFACE, width=2)),
                             text=[f"{x:.0f}" if x >= 6 else "" for x in v], textposition="inside", insidetextanchor="middle",
                             textfont=dict(color="#ffffff", size=11),
                             hovertemplate="%{y}<br>" + (OUTCOME_JA if ja else OUTCOME_EN)[k] + ": %{x:.1f}%<extra></extra>"))
    lay = _base_layout(110 + 46 * len(names))
    lay.update(barmode="stack", bargap=0.3, xaxis=dict(range=[0, 100], ticksuffix="%", gridcolor=GRID, zeroline=False),
               yaxis=dict(automargin=True, tickfont=dict(color=INK, size=12)),
               legend=dict(orientation="h", y=-0.12, yanchor="top", x=0, font=dict(color=INK2, size=12), traceorder="normal"),
               margin=dict(l=48, r=16, t=10, b=70), transition=dict(duration=500, easing="cubic-in-out"))
    fig.update_layout(**lay)
    return fig


def diamond(runners, outs: int, size: int = 150) -> str:
    """Bases (lit when occupied) and out dots, as inline SVG."""
    c = [HOME if r else "#2c2c3a" for r in runners]
    dots = "".join(f'<circle cx="{46 + 22 * k}" cy="140" r="7" fill="{"#e66767" if k < outs else "#2c2c3a"}"/>'
                   for k in range(3))
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 150 150" role="img" aria-label="bases">'
            '<polygon points="75,22 128,75 75,128 22,75" fill="none" stroke="#3a3a55" stroke-width="2"/>'
            f'<rect x="116" y="63" width="22" height="22" rx="3" fill="{c[0]}" transform="rotate(45 127 74)"/>'
            f'<rect x="64" y="11" width="22" height="22" rx="3" fill="{c[1]}" transform="rotate(45 75 22)"/>'
            f'<rect x="12" y="63" width="22" height="22" rx="3" fill="{c[2]}" transform="rotate(45 23 74)"/>'
            '<polygon points="75,120 68,127 75,134 82,127" fill="#f0efec"/>' + dots + '</svg>')


def steal_badge(opt: dict, lang: str) -> str:
    """The steal call as a badge: icon + words + numbers (never color alone)."""
    ja = lang == "JA"
    go_ = opt["go"]
    color = HOME if go_ else AWAY
    word = ("走れ" if go_ else "走るな") if ja else ("GO" if go_ else "HOLD")
    icon = "▶" if go_ else "■"
    what = ("二盗" if "steal_2b" in opt["option"] else "三盗") if ja else ("steal 2B" if "steal_2b" in opt["option"] else "steal 3B")
    nums = (f"成功見込み {opt['p_success']:.0%} ／ 必要 {opt['breakeven']:.0%}" if ja
            else f"est. {opt['p_success']:.0%} / needed {opt['breakeven']:.0%}")
    return (f'<div style="display:inline-block;margin:4px 0 8px 0;padding:8px 14px;border-radius:10px;'
            f'background:#1f1f3a;border:2px solid {color}">'
            f'<span style="color:{color};font-weight:bold;font-size:1.25rem">{icon} {what}：{word}</span>'
            f'<span style="color:#c3c2b7;margin-left:12px">{nums}</span></div>')


def fig_calls(rows: list[dict], home: str, away: str, lang: str) -> go.Figure:
    """Each rated call of a game: the model's gain for the team that made it
    (filled dot, team color) and the best candidate of the same kind (hollow
    ring), joined by a line = how far the call was from the model's best.
    rows: dicts with label, side ('home'/'away'), gain, best_gain (or None)."""
    ja = lang == "JA"
    rows = rows[::-1]                                   # first call at the top
    y = list(range(len(rows)))
    fig = go.Figure()
    for r, yy in zip(rows, y):
        if r.get("best_gain") is not None and abs(r["best_gain"] - r["gain"]) > 1e-9:
            fig.add_trace(go.Scatter(x=[r["gain"], r["best_gain"]], y=[yy, yy], mode="lines",
                                     line=dict(color=MUTED, width=2), hoverinfo="skip", showlegend=False))
    best = [(r["best_gain"], yy, r.get("best_label", "")) for r, yy in zip(rows, y) if r.get("best_gain") is not None]
    if best:
        fig.add_trace(go.Scatter(x=[b[0] for b in best], y=[b[1] for b in best], mode="markers",
                                 name="モデルの最善候補" if ja else "model's best candidate",
                                 marker=dict(size=12, color=SURFACE, line=dict(color=INK2, width=2)),
                                 customdata=[b[2] for b in best],
                                 hovertemplate="%{customdata}: %{x:+.2f} pt<extra></extra>"))
    for side, name, color in (("away", away, AWAY), ("home", home, HOME)):
        pts = [(r, yy) for r, yy in zip(rows, y) if r["side"] == side]
        if not pts:
            continue
        fig.add_trace(go.Scatter(x=[r["gain"] for r, _ in pts], y=[yy for _, yy in pts], mode="markers", name=name,
                                 marker=dict(size=13, color=color, line=dict(color=SURFACE, width=2)),
                                 customdata=[r["label"] for r, _ in pts],
                                 hovertemplate="%{customdata}: %{x:+.2f} pt<extra></extra>"))
    vals = [r["gain"] for r in rows] + [r["best_gain"] for r in rows if r.get("best_gain") is not None]
    span = max([abs(v) for v in vals] + [0.5]) * 1.25
    lay = _base_layout(max(180, 30 * len(rows) + 110))
    lay["margin"] = dict(l=48, r=16, t=64, b=36)
    key = ("<span style='font-size:11px;color:#c3c2b7'>点＝実際の采配　白丸＝同じ種類で最善の候補　右ほど采配したチームに有利（勝率、ポイント）</span>" if ja else
           "<span style='font-size:11px;color:#c3c2b7'>dot = the call made, ring = best candidate of the same kind; right = better for the team that made it (WP points)</span>")
    lay.update(title=dict(text=("モデルから見た各采配" if ja else "Each call, as the model saw it") + f"<br>{key}",
                          font=dict(color=INK, size=14), x=0, xanchor="left"),
               xaxis=dict(range=[-span, span], gridcolor=GRID, zeroline=True, zerolinecolor=MUTED, zerolinewidth=1, ticksuffix=" pt"),
               yaxis=dict(tickvals=y, ticktext=[r["label"] for r in rows], automargin=True, tickfont=dict(color=INK, size=12),
                          showgrid=False),
               legend=dict(orientation="h", y=-0.12, x=0, font=dict(color=INK2)))
    fig.update_layout(**lay)
    return fig
