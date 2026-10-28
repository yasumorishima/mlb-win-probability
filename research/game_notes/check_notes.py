"""
Check the generated notes and write the file the app reads.

  python research/game_notes/check_notes.py --inputs IN.jsonl --raw RAW.jsonl --out research/game_notes/published/notes_2026.json

A note is kept only if
- stop_reason is end_turn and the text is 2-6 sentences, 60-450 characters;
- every number in it (digits, full-width digits too) appears among the numbers of that game's input;
- every percentage is one of the input's win-probability values, every score (X-Y, X対Y) one of its scores;
- both team names appear in the input (always) and at least one team or player name from the input
  appears in the note.
Rejected notes are counted by reason; nothing is regenerated.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


def numbers(x) -> set[str]:
    s = json.dumps(x, ensure_ascii=False)
    return set(re.findall(r"\d+", unicodedata.normalize("NFKC", s)))


def names(obj: dict) -> set[str]:
    out = {obj["home"], obj["away"]}
    for p in obj["big_plays"]:
        out |= {p["batter"], p["pitcher"]}
    for c in obj["late_pitching_changes"]:
        out.add(c["new_pitcher"])
    return out


def pct_values(obj: dict) -> list[int]:
    out = [obj["home_wp_start_pct"], obj["home_wp_min_pct"], obj["home_wp_max_pct"]]
    for p in obj["big_plays"]:
        out += [p["batting_team_wp_before_pct"], p["batting_team_wp_after_pct"]]
    out += [c["team_wp_pct"] for c in obj["late_pitching_changes"]]
    return out


def score_pairs(obj: dict) -> set[tuple[int, int]]:
    """Scores in the input (both orders): final, before / after each big play, at each change."""
    txt = [obj["final"]] + [p["score_before"] for p in obj["big_plays"]] + [p.get("score_after", "") for p in obj["big_plays"]]
    txt += [c["score"] for c in obj["late_pitching_changes"]]
    out = set()
    for s in txt:
        m = re.search(r"(\d+) - (\d+)", s)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            out |= {(a, b), (b, a)}
    return out


def verdict(obj: dict, rec: dict) -> str:
    if rec.get("stop_reason") != "end_turn":
        return "stop_reason"
    t = unicodedata.normalize("NFKC", rec.get("text", "")).strip()
    n_sent = len([s for s in re.split(r"[。！？]", t) if s.strip()])
    if not (60 <= len(t) <= 450) or not (2 <= n_sent <= 6):
        return "length"
    extra = set(re.findall(r"\d+", t)) - numbers(obj)
    if extra:
        return "number_not_in_input"
    pcts = {int(v) for v in re.findall(r"(\d+)\s*%", json.dumps(pct_values(obj)))} | set(pct_values(obj))
    if any(int(x) not in pcts for x in re.findall(r"(\d+)\s*%", t)):
        return "percent_not_in_input"
    scores = score_pairs(obj)
    for x, y in re.findall(r"(\d+)\s*(?:-|対|－|–)\s*(\d+)", t):
        if (int(x), int(y)) not in scores:
            return "score_not_in_input"
    if not any(n in t for n in names(obj)):
        return "no_name"
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    inp = {json.loads(l)["game_pk"]: json.loads(l) for l in a.inputs.read_text().splitlines() if l.strip()}
    recs = [json.loads(l) for l in a.raw.read_text().splitlines() if l.strip()]
    c = Counter()
    keep = {}
    examples = {}
    for r in recs:
        v = verdict(inp[r["game_pk"]], r)
        c[v] += 1
        if v == "ok":
            keep[str(r["game_pk"])] = r["text"].strip()
        else:
            examples.setdefault(v, (r["game_pk"], r.get("text", "")[:300]))
    print(dict(c), "of", len(recs))
    for v, (g, t) in examples.items():
        print("--", v, g, t)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        model = recs[0]["model"] if recs else ""
        a.out.write_text(json.dumps({"model": model, "season": 2026, "checked": dict(c), "notes": keep},
                                    ensure_ascii=False, indent=0))
        print("wrote", a.out, len(keep))


if __name__ == "__main__":
    main()
