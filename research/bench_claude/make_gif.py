"""Animated README figure from published/results.json (numbers are read, not typed).

  pip install matplotlib pillow
  python research/bench_claude/make_gif.py   ->  docs/images/bench_claude.gif
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "docs/images/bench_claude.gif"
r = json.loads((HERE / "published/results.json").read_text())
pc = r["pitching_change"]
H1 = r["hypotheses"]["H1 C > MODEL (pitching changes)"]
H2 = r["hypotheses"]["H2 C > USAGE (pitching changes)"]

for f in ("WenQuanYi Zen Hei", "Noto Sans CJK JP", "IPAexGothic"):
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        plt.rcParams["axes.unicode_minus"] = False
        break

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ARMS = [("B", "Claude（数字なし）", "#2a78d6"), ("C", "Claude（モデルの数字あり）", "#eb6834"),
        ("USAGE", "直近の起用パターン", "#1baf7a"), ("MODEL", "このアプリのモデル", "#eda100")]
HYP = [("仮説1  Claude（数字あり） - モデル", H1), ("仮説2  Claude（数字あり） - 起用パターン", H2)]


def ease(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def frame(gb: float, gc: float) -> Image.Image:
    fig = plt.figure(figsize=(10, 5.4), dpi=90, facecolor=SURF)
    fig.text(0.03, 0.94, f"継投で、監督と同じ投手を選んだ割合（2026年7〜9月の{pc['n']}件）",
             fontsize=15, color=INK, weight="bold")
    fig.text(0.03, 0.885, "Claude はモデルより監督に近い。ただし普段の起用パターンとの差は言えない"
             "（一致率で、采配の正しさではありません）", fontsize=10.5, color=INK2)

    ax = fig.add_axes([0.25, 0.13, 0.33, 0.66], facecolor=SURF)
    ys = list(range(len(ARMS)))[::-1]
    for y, (k, lab, col) in zip(ys, ARMS):
        v = pc["agreement"][k] * 100
        ax.barh(y, v * ease(gb), height=0.56, color=col, edgecolor=SURF, linewidth=2)
        ax.text(-1, y, lab, ha="right", va="center", fontsize=10.5, color=INK)
        if gb >= 1:
            ax.text(v + 0.8, y, f"{v:.1f}%", va="center", fontsize=11, color=INK, weight="bold")
    ch = pc["chance_top1"] * 100
    ax.axvline(ch, color=INK2, ls=(0, (3, 3)), lw=1.2)
    ax.text(ch, len(ARMS) - 0.35, f"偶然 {ch:.1f}%", ha="center", fontsize=9, color=INK2)
    ax.set_xlim(0, 40)
    ax.set_ylim(-0.6, len(ARMS) - 0.1)
    ax.set_yticks([])
    ax.set_xlabel("一致率（%）", color=INK2, fontsize=9.5)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    bx = fig.add_axes([0.66, 0.13, 0.31, 0.66], facecolor=SURF)
    bx.axvline(0, color=INK2, lw=1)
    for i, (lab, h) in enumerate(HYP):
        y = 1 - i
        d, lo, hi = h["diff"] * 100, h["ci95"][0] * 100, h["ci95"][1] * 100
        e = ease(gc)
        if gc > 0:
            bx.plot([d - (d - lo) * e, d + (hi - d) * e], [y, y], color=INK, lw=2.2, solid_capstyle="round")
            bx.plot(d, y, "o", ms=9, color=INK, mec=SURF, mew=2)
        bx.text(-5, y + 0.3, lab, fontsize=10, color=INK)
        if gc >= 1:
            ok = h["supported"]
            bx.text(-5, y - 0.33, f"{d:+.1f} pt（95% {lo:+.1f}〜{hi:+.1f}）→ {'支持' if ok else '支持されず'}",
                    fontsize=10, color=INK if ok else INK2, weight="bold" if ok else "normal")
    bx.set_xlim(-5, 16)
    bx.set_ylim(-0.7, 1.7)
    bx.set_yticks([])
    bx.set_xlabel("一致率の差（ポイント、試合単位のブートストラップ）", color=INK2, fontsize=9.5)
    bx.tick_params(colors=INK2, labelsize=9)
    bx.grid(axis="x", color=GRID, lw=0.8)
    bx.set_axisbelow(True)
    for s in ("top", "right", "left"):
        bx.spines[s].set_visible(False)
    bx.spines["bottom"].set_color(GRID)

    fig.canvas.draw()
    im = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig)
    return im


def main() -> None:
    frames, durs = [], []
    for i in range(16):                       # bars grow
        frames.append(frame(i / 15, 0)); durs.append(60)
    durs[-1] = 900
    for i in range(1, 13):                    # intervals open
        frames.append(frame(1, i / 12)); durs.append(60)
    durs[-1] = 4500
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pal = frames[-1].quantize(colors=128)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(OUT, save_all=True, append_images=q[1:], duration=durs, loop=0, optimize=True)
    print(OUT, OUT.stat().st_size, "bytes,", len(frames), "frames")


if __name__ == "__main__":
    main()
