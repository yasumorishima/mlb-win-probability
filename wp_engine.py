"""
Win-probability engine fitted to real plate-appearance transitions.

Replaces the hand-tuned normal approximation in win_probability.py.

Model
-----
1. Plate-appearance transitions, estimated separately for the top half
   (visitors batting) and the bottom half (home batting):
       P(next base-out state, runs on the play | base-out state)
   from scripts/build_states.py output. A state is (outs, on1, on2, on3);
   three outs is the absorbing END state.
2. Runs in the rest of a half-inning from any state, by dynamic
   programming over those transitions.
3. Game win probability by recursion over half-innings: walk-off rules
   in the bottom of the last scheduled inning and later, and the automatic
   runner on second in extra innings.
4. Count layer: the plate appearance's outcome class depends on the count
   (balls, strikes) it has reached; the state transition given the class
   does not. WP at a count is the expectation over the outcome class.

Tactics are scored in win probability, not run expectancy:
    stolen base  -> break-even success rate
    sacrifice bunt -> WP after the observed outcomes of bunts in that state
    intentional walk -> WP after putting the batter on (defense's view)

Everything the engine needs is a small JSON table (`engine_tables.json`)
built by `fit_tables()` and loaded by `Engine.load()`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

MAX_RUNS = 20          # runs-in-half distribution support 0..MAX_RUNS
MAX_DIFF = 30          # score differences are clipped to +-MAX_DIFF
MAX_INNING = 25        # recursion depth for extra innings

States = list[tuple[int, int, int, int]]
STATES: States = [(o, a, b, c) for o in range(3) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
END = (3, 0, 0, 0)
S_INDEX = {s: i for i, s in enumerate(STATES)}

# Outcome classes for the count layer (from Stats API eventType).
CLASS_OF = {
    "strikeout": "K", "strikeout_double_play": "K", "strikeout_triple_play": "K",
    "walk": "BB", "intent_walk": "BB", "hit_by_pitch": "BB",
    "single": "1B", "double": "2B", "triple": "3B", "home_run": "HR",
}
CLASSES = ["K", "BB", "1B", "2B", "3B", "HR", "OUT", "OTHER"]
# everything else that ends a PA with the ball in play and no hit is OUT
IN_PLAY_OUTS = {"field_out", "force_out", "grounded_into_double_play", "double_play",
                "triple_play", "fielders_choice_out", "sac_fly", "sac_bunt",
                "sac_fly_double_play", "sac_bunt_double_play", "fielders_choice",
                "field_error"}


def outcome_class(event_type: str | None) -> str:
    if event_type in CLASS_OF:
        return CLASS_OF[event_type]
    if event_type in IN_PLAY_OUTS:
        return "OUT"
    return "OTHER"


def _state(outs, b1, b2, b3) -> tuple:
    if outs >= 3:
        return END
    return (int(outs), int(b1), int(b2), int(b3))


def _occ(x) -> int:
    """Base occupied? Player ids are ints, empty is None/NaN; -1 = unknown runner."""
    if x is None:
        return 0
    try:
        return 0 if np.isnan(x) else 1
    except TypeError:
        return 1


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_tables(pa, pitch=None, min_bunts: int = 30) -> dict:
    """Estimate transition tables from build_states.py plate-appearance rows.

    pa: DataFrame with outs, on_1b/2b/3b, outs_after, post_1b/2b/3b,
        runs_on_play, half, event_type, is_bunt, at_bat_index, game_pk
    pitch: DataFrame with game_pk, at_bat_index, balls, strikes (count
        before each pitch); needed for the count layer.
    """
    trans = {"top": defaultdict(lambda: defaultdict(float)),
             "bottom": defaultdict(lambda: defaultdict(float))}
    by_class = defaultdict(lambda: defaultdict(float))     # (state, class) -> (next, runs)
    bunts = defaultdict(lambda: defaultdict(float))        # state -> (next, runs)
    n_state = defaultdict(float)

    cols = ["half", "outs", "on_1b", "on_2b", "on_3b", "outs_after",
            "post_1b", "post_2b", "post_3b", "runs_on_play", "event_type", "is_bunt"]
    for r in pa[cols].itertuples(index=False):
        s = _state(r.outs, _occ(r.on_1b), _occ(r.on_2b), _occ(r.on_3b))
        nxt = _state(r.outs_after, _occ(r.post_1b), _occ(r.post_2b), _occ(r.post_3b))
        runs = int(min(max(r.runs_on_play, 0), 4))
        key = (nxt, runs)
        trans[r.half][s][key] += 1
        by_class[(s, outcome_class(r.event_type))][key] += 1
        n_state[s] += 1
        if r.is_bunt:
            bunts[s][key] += 1

    def norm(d):
        tot = sum(d.values())
        return [[list(k[0]), k[1], v / tot] for k, v in sorted(d.items())]

    tables = {
        "transitions": {h: {_skey(s): norm(d) for s, d in t.items()} for h, t in trans.items()},
        "by_class": {f"{_skey(s)}|{c}": norm(d) for (s, c), d in by_class.items()},
        "bunt": {_skey(s): {"n": int(sum(d.values())), "dist": norm(d)}
                 for s, d in bunts.items() if sum(d.values()) >= min_bunts},
        "n_pa": int(len(pa)),
    }
    if pitch is not None:
        tables["class_given_count"] = _class_given_count(pa, pitch)
    return tables


def _class_given_count(pa, pitch) -> dict:
    """P(outcome class of the PA | the PA reached count (b, s))."""
    cls = {(g, a): outcome_class(e) for g, a, e in
           pa[["game_pk", "at_bat_index", "event_type"]].itertuples(index=False)}
    counts = defaultdict(lambda: defaultdict(float))
    seen = set()
    for g, a, b, s in pitch[["game_pk", "at_bat_index", "balls", "strikes"]].itertuples(index=False):
        k = (g, a, int(b), int(s))
        if k in seen or (g, a) not in cls:
            continue
        seen.add(k)
        counts[(int(b), int(s))][cls[(g, a)]] += 1
    out = {}
    for (b, s), d in counts.items():
        tot = sum(d.values())
        out[f"{b}-{s}"] = {"n": int(tot), "p": {c: d.get(c, 0.0) / tot for c in CLASSES}}
    return out


def _skey(s) -> str:
    return "".join(str(x) for x in s)


def _parse(k: str) -> tuple:
    return tuple(int(c) for c in k)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

@dataclass
class TacticResult:
    tactic: str
    side: str            # "offense" or "defense": whose WP is reported
    wp_now: float        # WP of `side` if nothing is done
    wp_if: float | None  # WP of `side` if the tactic is used (None for steals)
    delta: float | None
    breakeven: float | None = None   # steals: success rate needed
    n: int | None = None             # sample size behind the estimate
    note: str = ""


class Engine:
    def __init__(self, tables: dict, extra_runner: bool = True):
        self.t = tables
        self.extra_runner = extra_runner
        self.trans = {h: {_parse(k): [(tuple(n), r, p) for n, r, p in v]
                          for k, v in d.items()} for h, d in tables["transitions"].items()}
        self.by_class = {}
        for k, v in tables.get("by_class", {}).items():
            sk, c = k.split("|")
            self.by_class[(_parse(sk), c)] = [(tuple(n), r, p) for n, r, p in v]
        self.cgc = tables.get("class_given_count", {})
        self.runs = {h: self._runs_dist(h) for h in ("top", "bottom")}
        self._H.cache_clear()

    @classmethod
    def load(cls, path: str | Path, **kw) -> "Engine":
        return cls(json.loads(Path(path).read_text()), **kw)

    # -- half-inning run distributions ------------------------------------
    def _runs_dist(self, half: str) -> dict:
        D = {s: np.zeros(MAX_RUNS + 1) for s in STATES}
        D[END] = np.zeros(MAX_RUNS + 1)
        D[END][0] = 1.0
        for _ in range(400):
            delta = 0.0
            for s in STATES:
                new = np.zeros(MAX_RUNS + 1)
                for nxt, r, p in self.trans[half].get(s, []):
                    shifted = np.roll(D[nxt], r)
                    if r:
                        shifted[:r] = 0.0
                        shifted[-1] += D[nxt][-r:].sum()  # keep mass at the cap
                    new += p * shifted
                delta = max(delta, float(np.abs(new - D[s]).max()))
                D[s] = new
            if delta < 1e-12:
                break
        return D

    # -- game win probability ---------------------------------------------
    def _start_state(self, inning: int, sched: int) -> tuple:
        if self.extra_runner and inning > sched:
            return (0, 0, 1, 0)
        return (0, 0, 0, 0)

    @lru_cache(maxsize=None)
    def _H(self, inning: int, half: str, diff: int, sched: int) -> float:
        """P(home wins) at the start of a half-inning, diff = home - away."""
        if inning > MAX_INNING:
            return 0.5 + 0.04 * np.sign(diff)  # far tail; effectively unreachable
        return self._wp_in_half(inning, half, self._start_state(inning, sched), diff, sched)

    def _wp_in_half(self, inning, half, state, diff, sched) -> float:
        dist = self.runs[half][state]
        total = 0.0
        for r, p in enumerate(dist):
            if p == 0.0:
                continue
            if half == "top":
                d = diff - r
                if inning >= sched and d > 0:
                    v = 1.0                      # bottom half not needed
                else:
                    v = self._H(inning, "bottom", _clip(d), sched)
            else:
                d = diff + r
                if inning >= sched:
                    if d > 0:
                        v = 1.0                  # walk-off
                    elif d == 0:
                        v = self._H(inning + 1, "top", 0, sched)
                    else:
                        v = 0.0
                else:
                    v = self._H(inning + 1, "top", _clip(d), sched)
            total += p * v
        return total

    def wp(self, inning: int, half: str, outs: int, bases: tuple, diff: int,
           sched: int = 9) -> float:
        """Home win probability before a plate appearance."""
        if half == "bottom" and inning >= sched and diff > 0:
            return 1.0
        s = _state(outs, *bases)
        return float(self._wp_in_half(inning, half, s, _clip(diff), sched))

    def _wp_after(self, inning, half, nxt, runs, diff, sched) -> float:
        """Home WP right after a PA that ended in `nxt` with `runs` scored."""
        d = diff + runs if half == "bottom" else diff - runs
        if half == "bottom" and inning >= sched and d > 0:
            return 1.0
        if nxt == END:
            if half == "top":
                if inning >= sched and d > 0:
                    return 1.0
                return self._H(inning, "bottom", _clip(d), sched)
            if inning >= sched:
                return 1.0 if d > 0 else (self._H(inning + 1, "top", 0, sched) if d == 0 else 0.0)
            return self._H(inning + 1, "top", _clip(d), sched)
        return float(self._wp_in_half(inning, half, nxt, _clip(d), sched))

    def wp_at_count(self, inning, half, outs, bases, diff, balls, strikes, sched=9):
        """Home WP with the count (balls, strikes) reached in this PA."""
        key = f"{balls}-{strikes}"
        cg = self.cgc.get(key)
        s = _state(outs, *bases)
        if not cg:
            return self.wp(inning, half, outs, bases, diff, sched)
        total, mass = 0.0, 0.0
        for c, pc in cg["p"].items():
            rows = self.by_class.get((s, c))
            if not rows or pc == 0:
                continue
            v = sum(p * self._wp_after(inning, half, nxt, r, diff, sched) for nxt, r, p in rows)
            total += pc * v
            mass += pc
        return total / mass if mass else self.wp(inning, half, outs, bases, diff, sched)

    def leverage(self, inning, half, outs, bases, diff, sched=9, league_swing=None) -> float:
        """Expected |WP change| of the next PA (divide by league average for LI)."""
        s = _state(outs, *bases)
        now = self.wp(inning, half, outs, bases, diff, sched)
        swing = sum(p * abs(self._wp_after(inning, half, nxt, r, diff, sched) - now)
                    for nxt, r, p in self.trans[half].get(s, []))
        return swing / league_swing if league_swing else swing

    # -- tactics ----------------------------------------------------------
    def tactics(self, inning, half, outs, bases, diff, sched=9,
                balls=None, strikes=None) -> list[TacticResult]:
        b1, b2, b3 = bases
        home_bat = half == "bottom"

        def wp_off(inn, o, bs, d_home):
            """Offense's WP for a pre-PA state (count-aware when given)."""
            if balls is not None and strikes is not None:
                w = self.wp_at_count(inn, half, o, bs, d_home, balls, strikes, sched)
            else:
                w = self.wp(inn, half, o, bs, d_home, sched)
            return w if home_bat else 1.0 - w

        def after_out(o):
            return o + 1

        out = []
        now = wp_off(inning, outs, bases, diff)

        # stolen bases: break-even success rate
        steals = []
        if b1 and not b2:
            steals.append(("steal_2b", (0, 1, b3), (0, 0, b3)))
        if b2 and not b3:
            steals.append(("steal_3b", (b1, 0, 1), (b1, 0, 0)))
        for name, succ, fail in steals:
            w_s = wp_off(inning, outs, succ, diff)
            if after_out(outs) >= 3:
                w_f = self._wp_after(inning, half, END, 0, diff, sched)
                w_f = w_f if home_bat else 1.0 - w_f
            else:
                w_f = wp_off(inning, after_out(outs), fail, diff)
            be = (now - w_f) / (w_s - w_f) if w_s > w_f else None
            out.append(TacticResult(name, "offense", now, None, None, breakeven=be,
                                    note="count held fixed; pickoff and wild-pitch paths ignored"))

        # sacrifice bunt (observed outcomes of bunts from this state)
        bt = self.t.get("bunt", {}).get(_skey(_state(outs, *bases)))
        if bt and outs < 2 and (b1 or b2):
            v = 0.0
            for nxt, r, p in bt["dist"]:
                w = self._wp_after(inning, half, tuple(nxt), r, diff, sched)
                v += p * (w if home_bat else 1.0 - w)
            out.append(TacticResult("sacrifice_bunt", "offense", now, v, v - now, n=bt["n"],
                                    note="outcomes of PAs that ended on a bunt; bunts fouled off are not in the sample"))

        # intentional walk (defense's view)
        if not (b1 and b2 and b3):
            nb, runs = _force(bases)
            d2 = diff + runs if home_bat else diff - runs
            w_after = self.wp(inning, half, outs, nb, d2, sched) if not (
                home_bat and inning >= sched and d2 > 0) else 1.0
            def_now = 1.0 - now
            def_if = 1.0 - (w_after if home_bat else 1.0 - w_after)
            out.append(TacticResult("intentional_walk", "defense", def_now, def_if,
                                    def_if - def_now,
                                    note="next batter assumed league-average"))
        return out


def _force(bases):
    """Put the batter on first; forced runners advance. Returns (bases, runs)."""
    b1, b2, b3 = bases
    if not b1:
        return (1, b2, b3), 0
    if not b2:
        return (1, 1, b3), 0
    if not b3:
        return (1, 1, 1), 0
    return (1, 1, 1), 1


def _clip(d: int) -> int:
    return int(max(-MAX_DIFF, min(MAX_DIFF, d)))
