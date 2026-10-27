"""RESULTS.md from published/results.json (numbers are generated, not copied by hand)."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
r = json.loads((HERE / "published/results.json").read_text())
pct = lambda x: f"{100 * x:.1f}%"
pt = lambda x: f"{100 * x:+.1f}"
ci = lambda d: f"{pt(d['diff'])} pt (95% {pt(d['ci95'][0])} to {pt(d['ci95'][1])})"
L = ["# Results — Claude's bench calls vs the managers'", "",
     "Pre-registration: `PREREG.md` (frozen `78a0a97`, merged `7a98f7a`). Run once, in Claude Code cloud,",
     f"model(s) {', '.join(r['answering_models'])}, live Messages API calls (Amendment 1: the batches stalled "
     f"and were cancelled unread). Cost from the recorded tokens at full price: B ${2 * r['cost_usd_batch']['B']:.2f}, "
     f"C ${2 * r['cost_usd_batch']['C']:.2f} (analyze.py's `cost_usd_batch` is the batch rate, half of this).", "",
     "Agreement = the arm picked the player the manager actually sent in. It measures how much the picks look "
     "like real MLB managers' choices, not whether they were right.", "", "## Hypotheses (pitching changes)", ""]
for name, h in r["hypotheses"].items():
    L.append(f"- **{name}**: {ci(h)}, Holm p = {h['p_holm']:.3f} → **{'SUPPORTED' if h['supported'] else 'NOT SUPPORTED'}**")
for task, title in (("pitching_change", "Pitching changes (6th inning on)"), ("pinch_hitter", "Pinch hitters")):
    t = r[task]
    L += ["", f"## {title}", "", f"{t['n']} items in {t['games']} games; mean {t['mean_candidates']:.1f} candidates "
          f"(chance {pct(t['chance_top1'])}). Invalid answers: B {t['invalid_answers']['B']}, C {t['invalid_answers']['C']}.", "",
          "| arm | agreement | top-3 agreement |", "|---|---|---|"]
    for k, v in t["agreement"].items():
        L.append(f"| {k} | {pct(v)} | {pct(t['top3_agreement'][k]) if k in t['top3_agreement'] else '—'} |")
    L += ["", "Paired differences (bootstrap over games):", ""]
    for k, d in t["differences"].items():
        L.append(f"- {k}: {ci(d)}")
    if "by_inning" in t:
        L += ["", "| innings | n | " + " | ".join(k for k in t["agreement"]) + " |", "|---|---|" + "---|" * len(t["agreement"])]
        for lab, v in t["by_inning"].items():
            L.append(f"| {lab} | {v['n']} | " + " | ".join(pct(v[k]) for k in t["agreement"]) + " |")
L += ["", f"Excluded items: {r['excluded']}."]
(HERE / "RESULTS.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
