"""bench_viz: the charts of the bench screen (offline)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bench_viz as V  # noqa: E402


def series(n=60):
    import math
    out = [{"i": 0, "wp": 0.5, "label": "", "label_en": "", "text": "", "event": "", "delta": 0.0}]
    for k in range(1, n):
        w = 0.5 + 0.4 * math.sin(k / 9)
        out.append({"i": k, "wp": w, "label": "5表", "label_en": "T5", "text": "single", "event": "Single",
                    "delta": w - out[-1]["wp"]})
    out[30]["delta"], out[31]["delta"], out[50]["delta"] = 0.3, 0.29, -0.2   # two big swings side by side
    return out


def test_wp_chart_marks_three_spaced_swings_and_replays():
    fig = V.fig_wp(series(), "Home", "Away", "JA")
    marks = fig.data[4]
    xs = sorted(marks.x)
    assert len(xs) == 3 and all(b - a >= 4 for a, b in zip(xs, xs[1:])), xs
    assert fig.frames and all(len(f.data) == 1 and f.traces == (3,) for f in fig.frames)   # only the line moves
    assert len(fig.frames[-1].data[0].x) == 60                                             # the last frame is the whole game
    assert fig.layout.updatemenus[0].buttons[0].label.startswith("▶")


def test_options_chart_signs_and_short_steal_label():
    rows = [{"kind": "none", "option": "none", "label_ja": "そのまま", "delta_wp_offense": 0.0},
            {"kind": "steal", "option": "steal_2b:1", "label_ja": "二盗（長い説明）", "delta_wp_offense": -0.002},
            {"kind": "pinch_hit", "option": "pinch_hit:2", "label_ja": "代打 X", "delta_wp_offense": 0.012}]
    fig = V.fig_options(rows, "JA", "攻撃側")
    bar = fig.data[0]
    assert list(bar.y) == ["代打 X", "二盗"]
    assert list(bar.marker.color) == [V.HOME, V.AWAY]
    html = V.steal_badge({"option": "steal_2b:1", "go": False, "p_success": 0.78, "breakeven": 0.85}, "JA")
    assert "走るな" in html and "78%" in html and "85%" in html and "■" in html


def test_matchup_chart_stacks_to_100():
    r = {"order": 0, "name": "A", "p_k": .2, "p_bb": .1, "p_1b": .15, "p_2b": .05, "p_3b": .01, "p_hr": .04, "p_out": .45}
    fig = V.fig_matchups({"rows": [r, {**r, "order": 1}]}, "EN")
    assert len(fig.data) == 6
    assert abs(sum(t.x[0] for t in fig.data) - 100) < 1e-9
