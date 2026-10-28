"""Animated README figures for the pre-registered results (numbers are read from the
published JSON files, never typed by hand).

  pip install matplotlib pillow
  python docs/make_gifs.py          ->  docs/images/{engine_2026,team_layer,player_layer,blend,steal_claude,pen_recal,penw,penr}.gif

The Claude bench-call figure is made by research/bench_claude/make_gif.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs/images"
J = lambda p: json.loads((ROOT / p).read_text())  # noqa: E731

for f in ("WenQuanYi Zen Hei", "Noto Sans CJK JP", "IPAexGothic"):
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        plt.rcParams["axes.unicode_minus"] = False
        break

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


def ease(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def new_fig(title: str, sub: str):
    fig = plt.figure(figsize=(10, 5.4), dpi=90, facecolor=SURF)
    fig.text(0.03, 0.94, title, fontsize=15, color=INK, weight="bold")
    fig.text(0.03, 0.885, sub, fontsize=10.5, color=INK2)
    return fig


def style(ax, xlabel: str = "", ylabel: str = "", grid: str = "x") -> None:
    ax.set_facecolor(SURF)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9.5)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(GRID)


def to_image(fig) -> Image.Image:
    fig.canvas.draw()
    im = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig)
    return im


def bars(ax, rows, g: float, fmt: str, xmax: float, xmin: float = 0.0) -> None:
    """Horizontal bars, top to bottom: rows = [(label, value, color, bold)]."""
    ys = list(range(len(rows)))[::-1]
    for y, (lab, v, col, bold) in zip(ys, rows):
        ax.barh(y, v * ease(g), height=0.56, color=col, edgecolor=SURF, linewidth=2)
        ax.text(xmin - (xmax - xmin) * 0.02, y, lab, ha="right", va="center", fontsize=10, color=INK,
                weight="bold" if bold else "normal")
        if g >= 1:
            ax.text(v + (xmax - xmin) * 0.015, y, fmt.format(v), va="center", fontsize=10.5, color=INK,
                    weight="bold" if bold else "normal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks([])
    style(ax)
    ax.spines["left"].set_visible(False)


def intervals(ax, rows, g: float, scale: float, unit: str, xlim) -> None:
    """Point + 95% interval rows (top to bottom): rows = [(label, mean, (lo, hi), verdict or None)]."""
    ax.axvline(0, color=INK2, lw=1)
    n = len(rows)
    for i, (lab, d, ci, verdict) in enumerate(rows):
        y = n - 1 - i
        d, lo, hi = d * scale, ci[0] * scale, ci[1] * scale
        e = ease(g)
        if g > 0:
            ax.plot([d - (d - lo) * e, d + (hi - d) * e], [y, y], color=INK, lw=2.2, solid_capstyle="round")
            ax.plot(d, y, "o", ms=9, color=INK, mec=SURF, mew=2)
        ax.text(xlim[0], y + 0.3, lab, fontsize=10, color=INK)
        if g >= 1:
            txt = f"{d:+.2f}{unit}（95% {lo:+.2f}〜{hi:+.2f}）"
            ok = verdict == "支持"
            if verdict:
                txt += f" → {verdict}"
            ax.text(xlim[0], y - 0.33, txt, fontsize=9.5, color=INK if ok else INK2, weight="bold" if ok else "normal")
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks([])
    style(ax)
    ax.spines["left"].set_visible(False)


def save(name: str, draw, n_grow: int = 15, n_open: int = 12) -> None:
    frames, durs = [], []
    for i in range(n_grow + 1):
        frames.append(draw(i / n_grow, 0.0)); durs.append(60)
    durs[-1] = 800
    for i in range(1, n_open + 1):
        frames.append(draw(1.0, i / n_open)); durs.append(60)
    durs[-1] = 4500
    pal = frames[-1].quantize(colors=128)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    out = IMG / f"{name}.gif"
    q[0].save(out, save_all=True, append_images=q[1:], duration=durs, loop=0, optimize=True)
    print(out.name, out.stat().st_size, "bytes", len(frames), "frames")


# ---------------------------------------------------------------- figures

def engine_2026() -> None:
    r = J("reports/engine_check_2026.json")
    e, o = r["engine"], r["old_engine"]
    gap = lambda arm: [100 * (b["mean_pred"] - b["observed"]) for b in arm["calibration"]]  # noqa: E731
    ge, go = gap(e), gap(o)
    labels = [b["bin"].replace("-", "〜") for b in e["calibration"]]

    def draw(g, h):
        fig = new_fig("2026年（学習に使っていない年）：新しいエンジンは旧エンジンより当たる",
                      f"打席前の勝率を試合の勝敗で採点（{e['n']:,}打席）。棒＝予測 - 実際（ポイント、上なら高めに出ている）")
        ax = fig.add_axes([0.08, 0.15, 0.62, 0.64])
        xs = range(len(labels))
        w = 0.38
        ax.bar([x - w / 2 for x in xs], [v * ease(g) for v in go], w, color=YELLOW, edgecolor=SURF, lw=1.5,
               label="旧エンジン（手で合わせた正規近似）")
        ax.bar([x + w / 2 for x in xs], [v * ease(g) for v in ge], w, color=BLUE, edgecolor=SURF, lw=1.5,
               label="このエンジン")
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylim(-5, 5)
        if h > 0:
            ax.axvspan(2.5, 4.5, color=ORANGE, alpha=0.10 * ease(h), lw=0)
            if h >= 1:
                ax.text(3.5, 4.4, "弱点：0.3〜0.5 帯は高め", ha="center", fontsize=9.5, color=INK)
        style(ax, "予測した勝率の帯", "予測 - 実際（ポイント）", grid="y")
        ax.legend(loc="upper right", fontsize=9, frameon=False, labelcolor=INK)
        bx = fig.add_axes([0.76, 0.15, 0.22, 0.64])
        bx.axis("off")
        rows = [("このエンジン", e["brier"], BLUE), ("旧エンジン", o["brier"], YELLOW),
                ("ホーム勝率の定数", r["constant_home_rate"]["brier"], INK2)]
        bx.text(0, 0.95, "Brier（小さいほど良い）", fontsize=10, color=INK2)
        for i, (lab, v, col) in enumerate(rows):
            y = 0.78 - i * 0.2
            bx.text(0, y, lab, fontsize=10, color=INK)
            if g >= 1:
                bx.text(0, y - 0.08, f"{v:.5f}", fontsize=13, color=col if col != INK2 else INK2, weight="bold")
        return to_image(fig)

    save("engine_2026", draw)


def team_layer() -> None:
    r = J("research/wp_layer/published/results_2026.json")
    a, t = r["arms"], r["tests"]
    rows_b = [("エンジン", a["ENGINE"]["brier"] * 1000, YELLOW, False),
              ("較正し直しただけ（対照）", a["RECAL"]["brier"] * 1000, AQUA, False),
              ("チームの強さを足した補正層", a["TEAM"]["brier"] * 1000, BLUE, True)]
    v = lambda k: "支持" if t[k]["SUPPORTED"] else "支持されず"  # noqa: E731
    rows_i = [("仮説1  補正層 - 対照", t["H1_TEAM_minus_RECAL_brier"]["point"],
               t["H1_TEAM_minus_RECAL_brier"]["ci95"], v("H1_TEAM_minus_RECAL_brier")),
              ("仮説2  補正層 - エンジン", t["H2_TEAM_minus_ENGINE_brier"]["point"],
               t["H2_TEAM_minus_ENGINE_brier"]["ci95"], v("H2_TEAM_minus_ENGINE_brier"))]

    def draw(g, h):
        fig = new_fig("チームの強さを足すと、2026年の勝率予測が良くなった（仮説2つとも支持）",
                      f"事前登録して1回だけ採点（{r['n_test']:,}打席・{r['n_games']:,}試合）。左：Brier×1000（小さいほど良い）、右：差と95%区間")
        ax = fig.add_axes([0.27, 0.15, 0.3, 0.62])
        bars(ax, rows_b, g, "{:.2f}", 159.0, 156.0)
        ax.set_xlabel("Brier × 1000（156 から表示）", color=INK2, fontsize=9.5)
        bx = fig.add_axes([0.64, 0.15, 0.34, 0.62])
        intervals(bx, rows_i, h, 1000, "", (-3.2, 1.2))
        bx.set_xlabel("Brier の差 × 1000（負＝良くなった）", color=INK2, fontsize=9.5)
        return to_image(fig)

    save("team_layer", draw)


def player_layer() -> None:
    r = J("research/player_layer/published/results_2026.json")
    a, t, s = r["arms"], r["tests"], r["supported"]
    rows_b = [("エンジン", a["ENGINE"]["band_0.3_0.5_gap_pt"], YELLOW, False),
              ("チーム補正層", a["TEAM"]["band_0.3_0.5_gap_pt"], AQUA, False),
              ("選手を入れた層", a["PLAYER"]["band_0.3_0.5_gap_pt"], BLUE, True)]
    v = lambda k: "支持" if s[k] else "区別できず"  # noqa: E731
    rows_i = [("仮説1  選手層 - エンジン（勝率 Brier）", t["H1_PLAYER_minus_ENGINE_brier"]["mean"],
               t["H1_PLAYER_minus_ENGINE_brier"]["ci95"], v("H1_PLAYER_minus_ENGINE_brier")),
              ("仮説2  選手層 - チーム補正層（勝率 Brier）", t["H2_PLAYER_minus_TEAM_brier"]["mean"],
               t["H2_PLAYER_minus_TEAM_brier"]["ci95"], v("H2_PLAYER_minus_TEAM_brier")),
              ("仮説3  選手層 - エンジン（この回1点以上 log loss）", t["H3_PLAYER_minus_ENGINE_p1_logloss"]["mean"],
               t["H3_PLAYER_minus_ENGINE_p1_logloss"]["ci95"], v("H3_PLAYER_minus_ENGINE_p1_logloss"))]

    def draw(g, h):
        fig = new_fig("選手を入れると、エンジンより当たる（チーム補正層とは区別できない）",
                      f"2026年に事前登録して1回だけ採点（{r['n_rows']:,}打席・{r['n_games']:,}試合）。左：0.3〜0.5帯の偏り、右：差×1000と95%区間")
        ax = fig.add_axes([0.2, 0.15, 0.26, 0.62])
        bars(ax, rows_b, g, "+{:.2f}", 4.0)
        ax.set_xlabel("0.3〜0.5 帯で高めに出る分（ポイント）", color=INK2, fontsize=9.5)
        bx = fig.add_axes([0.53, 0.15, 0.45, 0.62])
        intervals(bx, rows_i, h, 1000, "", (-3.6, 1.4))
        bx.set_xlabel("差 × 1000（負＝選手層のほうが当たる）", color=INK2, fontsize=9.5)
        return to_image(fig)

    save("player_layer", draw)


def blend() -> None:
    r = J("research/player_layer/published/results_blend_2026.json")
    a, t, s = r["arms"], r["tests"], r["supported"]
    rows_b = [("チーム補正層", a["TEAM"]["brier"] * 1000, AQUA, False),
              ("選手を入れた層", a["PLAYER"]["brier"] * 1000, YELLOW, False),
              ("選手層を較正し直したもの", a["P_RECAL"]["brier"] * 1000, ORANGE, False),
              ("BLEND（半々に混ぜたもの）", a["BLEND"]["brier"] * 1000, BLUE, True)]
    v = lambda k: "支持" if s[k] else "支持されず"  # noqa: E731
    rows_i = [("仮説1  BLEND - チーム補正層", t["H1_BLEND_minus_TEAM"]["mean"], t["H1_BLEND_minus_TEAM"]["ci95"],
               v("H1_BLEND_minus_TEAM")),
              ("仮説2  BLEND - 選手層", t["H2_BLEND_minus_PLAYER"]["mean"], t["H2_BLEND_minus_PLAYER"]["ci95"],
               v("H2_BLEND_minus_PLAYER"))]

    def draw(g, h):
        fig = new_fig("混ぜたもの（BLEND）は数字の上では最小、でも区間が0をまたぐ（支持されず）",
                      "2026年に事前登録して1回だけ採点。左：Brier×1000（小さいほど良い）、右：差×1000と95%区間")
        ax = fig.add_axes([0.27, 0.15, 0.3, 0.62])
        bars(ax, rows_b, g, "{:.2f}", 157.2, 156.0)
        ax.set_xlabel("Brier × 1000（156 から表示）", color=INK2, fontsize=9.5)
        bx = fig.add_axes([0.64, 0.15, 0.34, 0.62])
        intervals(bx, rows_i, h, 1000, "", (-1.4, 0.6))
        bx.set_xlabel("Brier の差 × 1000（負＝BLEND のほうが当たる）", color=INK2, fontsize=9.5)
        return to_image(fig)

    save("blend", draw)


def steal_claude() -> None:
    r = J("research/steal_claude/published/results.json")
    te = r["by_split"]["test"]
    s = r["supported"]
    rows_b = [("統計モデル", te["A"]["log_loss"], YELLOW, False),
              ("Claude 単独", te["B"]["log_loss"], AQUA, False),
              ("Claude ＋ モデルの予測値", te["C"]["log_loss"], BLUE, True)]
    rows_i = [("Claude 単独 - モデル", te["B_minus_A"]["mean"], te["B_minus_A"]["ci95"],
               "支持" if s["H2_B"] else "支持されず"),
              ("Claude＋モデル - モデル", te["C_minus_A"]["mean"], te["C_minus_A"]["ci95"],
               "支持" if s["H1_C"] else "支持されず")]

    def draw(g, h):
        fig = new_fig("盗塁の成功を、Claude は統計モデルより当てられなかった（仮説2つとも支持されず）",
                      f"2026年7〜9月の盗塁 {te['n']:,}件（{te['games']}試合）。左：log loss（小さいほど良い）、右：差×1000と95%区間")
        ax = fig.add_axes([0.25, 0.15, 0.3, 0.62])
        bars(ax, rows_b, g, "{:.4f}", 0.470, 0.455)
        ax.set_xlabel("log loss（0.455 から表示）", color=INK2, fontsize=9.5)
        bx = fig.add_axes([0.62, 0.15, 0.36, 0.62])
        intervals(bx, rows_i, h, 1000, "", (-5, 11))
        bx.set_xlabel("log loss の差 × 1000（負＝モデルより当たる）", color=INK2, fontsize=9.5)
        return to_image(fig)

    save("steal_claude", draw)


def pen_recal() -> None:
    r = J("research/player_layer/published/results_pen_2026.json")
    a, t, s = r["arms"], r["tests"], r["supported"]
    rows_b = [("エンジン", a["ENGINE"]["brier"] * 1000, INK2, False),
              ("チーム補正層（今の見出し）", a["TEAM"]["brier"] * 1000, AQUA, False),
              ("選手を入れた層（v1）", a["PLAYER"]["brier"] * 1000, YELLOW, False),
              ("新しい層（選手＋継投＋較正）", a["PEN_RECAL"]["brier"] * 1000, BLUE, True)]
    v = lambda k: "支持" if s[k] else "支持されず"  # noqa: E731
    rows_i = [("仮説1  新しい層 - チーム補正層", t["H1_PEN_RECAL_minus_TEAM"]["mean"], t["H1_PEN_RECAL_minus_TEAM"]["ci95"],
               v("H1_PEN_RECAL_minus_TEAM")),
              ("仮説2  新しい層 - 選手層 v1", t["H2_PEN_RECAL_minus_PLAYER"]["mean"], t["H2_PEN_RECAL_minus_PLAYER"]["ci95"],
               v("H2_PEN_RECAL_minus_PLAYER"))]

    def draw(g, h):
        fig = new_fig("終盤の継投の使い方を入れて較正し直すと、2026年の勝率予測がはっきり良くなった",
                      f"事前登録して1回だけ採点（{r['n_rows']:,}打席・{r['n_games']:,}試合）。左：Brier×1000（小さいほど良い）、右：差×1000と95%区間")
        ax = fig.add_axes([0.27, 0.15, 0.3, 0.62])
        bars(ax, rows_b, g, "{:.2f}", 159.0, 154.5)
        ax.set_xlabel("Brier × 1000（154.5 から表示）", color=INK2, fontsize=9.5)
        bx = fig.add_axes([0.64, 0.15, 0.34, 0.62])
        intervals(bx, rows_i, h, 1000, "", (-3.2, 0.8))
        bx.set_xlabel("Brier の差 × 1000（負＝新しい層のほうが当たる）", color=INK2, fontsize=9.5)
        return to_image(fig)

    save("pen_recal", draw)


def penw() -> None:
    r = J("research/player_layer/published/results_penw_2026.json")
    a, t, s = r["arms"], r["tests"], r["supported"]
    rows_b = [("チーム補正層", a["TEAM"]["brier"] * 1000, AQUA, False),
              ("継投の起用を直近30日で数える層", a["PEN_RECAL"]["brier"] * 1000, YELLOW, False),
              ("直近60日で数える層（新）", a["PENW_RECAL"]["brier"] * 1000, BLUE, True)]
    v = lambda k: "支持" if s[k] else "支持されず"  # noqa: E731
    k1 = "H1_PENW_RECAL_minus_PEN_RECAL"
    rows_i = [("仮説1  60日 - 30日", t[k1]["mean"], t[k1]["ci95"], v(k1))]

    def draw(g, h):
        fig = new_fig("救援投手の起用を直近60日で数えると、2026年の勝率予測がわずかに良くなった",
                      f"事前登録して1回だけ採点（{r['n_rows']:,}打席・{r['n_games']:,}試合）。左：Brier×1000（小さいほど良い）、右：差×1000と95%区間")
        ax = fig.add_axes([0.30, 0.15, 0.28, 0.62])
        bars(ax, rows_b, g, "{:.3f}", 157.2, 154.5)
        ax.set_xlabel("Brier × 1000（154.5 から表示）", color=INK2, fontsize=9.5)
        bx = fig.add_axes([0.65, 0.15, 0.33, 0.62])
        intervals(bx, rows_i, h, 1000, "", (-0.2, 0.05))
        bx.set_xlabel("Brier の差 × 1000（負＝60日のほうが当たる）", color=INK2, fontsize=9.5)
        return to_image(fig)

    save("penw", draw)


def penr() -> None:
    r = J("research/player_layer/published/results_penr_2026.json")
    a, t, s = r["arms"], r["tests"], r["supported"]
    rows_b = [("チーム補正層", a["TEAM"]["brier"] * 1000, AQUA, False),
              ("起用60日の層", a["PENW_RECAL"]["brier"] * 1000, YELLOW, False),
              ("役割を細かく見た層（新）", a["PENR_RECAL"]["brier"] * 1000, BLUE, True)]
    v = lambda k: "支持" if s[k] else "支持されず"  # noqa: E731
    k1 = "H1_PENR_RECAL_minus_PENW_RECAL"
    rows_i = [("仮説1  新 - 60日", t[k1]["mean"], t[k1]["ci95"], v(k1))]

    def draw(g, h):
        fig = new_fig("救援の役割を細かく見ると、2026年の勝率予測がさらに良くなった",
                      f"事前登録して1回だけ採点（{r['n_rows']:,}打席・{r['n_games']:,}試合）。左：Brier×1000（小さいほど良い）、右：差×1000と95%区間")
        ax = fig.add_axes([0.30, 0.15, 0.28, 0.62])
        bars(ax, rows_b, g, "{:.3f}", 157.2, 154.5)
        ax.set_xlabel("Brier × 1000（154.5 から表示）", color=INK2, fontsize=9.5)
        bx = fig.add_axes([0.65, 0.15, 0.33, 0.62])
        intervals(bx, rows_i, h, 1000, "", (-0.35, 0.05))
        bx.set_xlabel("Brier の差 × 1000（負＝新しい層のほうが当たる）", color=INK2, fontsize=9.5)
        return to_image(fig)

    save("penr", draw)


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    for fn in (engine_2026, team_layer, player_layer, blend, steal_claude, pen_recal, penw, penr):
        fn()
