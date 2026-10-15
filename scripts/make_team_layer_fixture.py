"""
Sample of the published 2026 TEAM predictions with their inputs, for
tests/test_team_layer.py (tests/fixtures/team_layer_rows.json). Run from the
repository root after research/wp_layer has been run:

    python scripts/make_team_layer_fixture.py [dir with pa_2026.parquet]
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, ".")
sys.path.insert(0, "research/wp_layer")
import layer_features as lf
from wp_engine import Engine
STATES = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/states")
d = lf.build(STATES, [2026], Engine.load("engine_tables.json"), league_exclude=[2026])
pub = pd.read_parquet("research/wp_layer/published/pred_2026.parquet")
assert (d.game_pk.to_numpy() == pub.game_pk.to_numpy()).all()
d = d.assign(p_team=pub.p_TEAM.to_numpy())
rng = np.random.default_rng(20261016)
idx = list(rng.choice(len(d), 40, replace=False))
idx += list(np.flatnonzero((d.inning > d.scheduled_innings).to_numpy())[:3])
idx += list(np.flatnonzero((d.scheduled_innings == 7).to_numpy())[:2])
rows = [{"p_engine": float(r.p_eng), "inning": int(r.inning), "half": r.half, "outs": int(r.outs),
         "sched": int(r.scheduled_innings), "d_prev": float(r.d_prev), "d_cur": float(r.d_cur),
         "p_team": float(r.p_team)} for r in d.iloc[idx].itertuples()]
Path("tests/fixtures/team_layer_rows.json").write_text(json.dumps(rows, indent=0))
print(len(rows), sum(r["inning"] > r["sched"] for r in rows), sum(r["sched"] == 7 for r in rows))
