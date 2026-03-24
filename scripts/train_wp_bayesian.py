"""
Bayesian Hierarchical Win Probability Model (NumPyro SVI).

Architecture:
  logit(WP_i) = logit(StatcastLGBM_pred_i)     # Statcast 58-feature LightGBM base
              + α_home[i] - α_away[i]           # team strength (hierarchical)
              + β_park[i]                        # park effect (hierarchical)
              + γ_season[i]                      # era effect (random walk)
              + κ × LI_i × (α_home - α_away)    # leverage × team interaction

Bayesian stacking: LightGBM handles 58 Statcast features (pitch/hit/bat tracking),
Bayesian layer adds team/park/season effects + posterior credible intervals.

Key innovations:
  1. Same game state, different teams → different WP
  2. Park effects with Bayesian shrinkage (no overfitting)
  3. Posterior credible intervals on every WP estimate
  4. Temporal random walk captures era changes (juiced ball etc.)
  5. Leverage-team interaction: good teams gain more in clutch

Training: SVI (Stochastic Variational Inference) for scalability to 1.7M obs.
Output: Posterior parameter JSON + team/park/season effect summaries.

Usage:
  python scripts/train_wp_bayesian.py --data-dir data/ --output-dir data/
  python scripts/train_wp_bayesian.py --n-steps 30000 --lr 0.005
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import ndtr as _ndtr

# Lazy imports for JAX/NumPyro (fail fast with clear message)
try:
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from jax import random as jrandom
    from numpyro.infer import SVI, Predictive, Trace_ELBO
    from numpyro.infer.autoguide import AutoNormal
except ImportError as e:
    print(f"ERROR: {e}")
    print("Install: pip install numpyro jax jaxlib")
    sys.exit(1)


# ============================================================
# Data Loading
# ============================================================

# Full MLB team name -> abbreviation mapping
TEAM_ABBREV: dict[str, str] = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Cleveland Indians": "CLE",  # pre-2022
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP", "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSN",
}

# Park = home team abbreviation (simplification; works for 2015-2024)
ALL_TEAMS = sorted(set(TEAM_ABBREV.values()))
TEAM_TO_IDX = {t: i for i, t in enumerate(ALL_TEAMS)}
N_TEAMS = len(ALL_TEAMS)


def load_play_states(data_dir: Path,
                     exclude_years: set[int] | None = None,
                     ) -> list[dict]:
    """Load all play_states CSVs, return list of dicts."""
    exclude_years = exclude_years or set()
    all_states = []

    for csv_path in sorted(data_dir.glob("play_states_*.csv")):
        year = int(csv_path.stem.split("_")[-1])
        if year in exclude_years:
            print(f"  Skipping {csv_path.name} (excluded)")
            continue

        n = 0
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    home = TEAM_ABBREV.get(row["home_team"], row["home_team"])
                    away = TEAM_ABBREV.get(row["away_team"], row["away_team"])
                    if home not in TEAM_TO_IDX or away not in TEAM_TO_IDX:
                        continue

                    all_states.append({
                        "game_pk": int(row.get("game_pk", 0)),
                        "year": year,
                        "inning": int(row["inning"]),
                        "half_inning": row["half_inning"],
                        "outs": int(row["outs"]),
                        "r1": int(row["runner_1b"]),
                        "r2": int(row["runner_2b"]),
                        "r3": int(row["runner_3b"]),
                        "score_diff": int(row["score_diff"]),
                        "home_team": home,
                        "away_team": away,
                        "home_won": int(row["home_won"]),
                    })
                    n += 1
                except (ValueError, KeyError):
                    continue
        print(f"  {csv_path.name}: {n:,} plays")

    return all_states


# ============================================================
# Feature Engineering
# ============================================================

def compute_markov_wp(inning: int, half: str, outs: int,
                      r1: int, r2: int, r3: int,
                      score_diff: int) -> float:
    """Simplified Markov-like WP using Normal approximation.

    This is fast enough for all 1.7M rows. We use the v1 formula:
    remaining_innings -> expected runs remaining -> Normal CDF.
    """
    runs_per_game = 4.5
    rphi = runs_per_game / 18.0  # runs per half-inning

    # Remaining half-innings
    if half == "top":
        home_hi = max(0, 18 - 2 * inning + 1)
        away_hi = max(0, 18 - 2 * inning + 1)
    else:
        home_hi = max(0, 18 - 2 * inning)
        away_hi = max(0, 18 - 2 * inning)

    # Fractional half-inning for outs
    outs_frac = 1.0 - outs / 3.0

    # Runner adjustment
    runner_re = 0.0
    if half == "top":
        # Away batting: runners subtract from home advantage
        runner_re = -(r1 * 0.37 + r2 * 0.70 + r3 * 1.00)
    else:
        runner_re = r1 * 0.37 + r2 * 0.70 + r3 * 1.00

    mean_diff = score_diff + runner_re
    var = rphi * (home_hi + away_hi + outs_frac) * 1.3

    if var <= 0:
        return 0.99 if mean_diff > 0 else (0.01 if mean_diff < 0 else 0.52)

    wp = float(_ndtr(mean_diff / np.sqrt(var)))
    return np.clip(wp, 0.005, 0.995)


def compute_leverage_index(inning: int, half: str, outs: int,
                           r1: int, r2: int, r3: int,
                           score_diff: int) -> float:
    """Approximate leverage index based on game situation.

    LI ≈ how much a single event can swing WP at this moment.
    Higher in close games, late innings, runners on base.
    """
    # Inning factor: increases as game progresses
    inn_factor = min(inning / 9.0, 1.5)

    # Score closeness: highest when tied, drops with lead
    abs_diff = abs(score_diff)
    close_factor = max(0, 1.0 - abs_diff / 5.0)

    # Runner/out state: more runners + fewer outs = higher leverage
    runner_factor = 1.0 + 0.3 * (r1 + r2 + r3)
    out_factor = 1.0 - 0.2 * outs

    li = inn_factor * close_factor * runner_factor * out_factor

    # Late-and-close bonus
    if inning >= 7 and abs_diff <= 2:
        li *= 1.5
    if inning >= 9 and half == "bottom" and abs_diff <= 1:
        li *= 2.0

    return max(0.1, min(li, 5.0))


def _compute_statcast_base_wp(states: list[dict],
                               statcast_model_path: Path | None = None,
                               data_dir: Path | None = None,
                               ) -> np.ndarray | None:
    """Compute base WP using Statcast LightGBM model with real features.

    If statcast_pitches CSVs are in data_dir, uses real pitch/hit features
    for at-bat plays (joined by game state). Falls back to game-state-only
    prediction for non-matched plays and to Markov WP if model unavailable.

    Returns predicted WP array, or None if model not available.
    """
    if statcast_model_path is None or not statcast_model_path.exists():
        return None

    try:
        import lightgbm as lgb
        import pandas as pd
        from scripts.train_wp_statcast import engineer_features

        model = lgb.Booster(model_file=str(statcast_model_path))

        # Try to load real statcast features
        lookup = None
        if data_dir is not None:
            csvs = sorted(data_dir.glob("statcast_pitches_*.csv"))
            if csvs:
                dfs = [pd.read_csv(c) for c in csvs]
                sc_df = pd.concat(dfs, ignore_index=True)
                print(f"  Loaded {len(sc_df):,} at-bat outcomes from statcast CSVs")

                X, _ = engineer_features(sc_df)
                sc_preds = np.clip(model.predict(X), 0.005, 0.995)

                # Build lookup with runners for near-unique keys
                # Key: (game_pk, inning, is_bottom, outs, score_diff, r1, r2, r3)
                is_bottom = (
                    sc_df["is_bottom"].values if "is_bottom" in sc_df.columns
                    else (sc_df["inning_topbot"] == "Bot").astype(int).values
                )
                score_diff = (
                    sc_df["score_diff"].values if "score_diff" in sc_df.columns
                    else (sc_df["home_score"] - sc_df["away_score"]).values
                )
                sc_r1 = sc_df["on_1b"].notna().astype(int).values if "on_1b" in sc_df.columns else np.zeros(len(sc_df), dtype=int)
                sc_r2 = sc_df["on_2b"].notna().astype(int).values if "on_2b" in sc_df.columns else np.zeros(len(sc_df), dtype=int)
                sc_r3 = sc_df["on_3b"].notna().astype(int).values if "on_3b" in sc_df.columns else np.zeros(len(sc_df), dtype=int)

                lookup = {}
                for i in range(len(sc_df)):
                    key = (
                        int(sc_df["game_pk"].iloc[i]),
                        int(sc_df["inning"].iloc[i]),
                        int(is_bottom[i]),
                        int(sc_df["outs_when_up"].iloc[i]),
                        int(score_diff[i]),
                        int(sc_r1[i]), int(sc_r2[i]), int(sc_r3[i]),
                    )
                    lookup[key] = float(sc_preds[i])

                n_collisions = len(sc_df) - len(lookup)
                print(f"  Statcast lookup: {len(lookup):,} unique states "
                      f"({n_collisions:,} collisions)")

        # Game-state-only fallback engine
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from win_probability_statcast import WPEngineStatcast
        engine = WPEngineStatcast.__new__(WPEngineStatcast)
        engine._model = model
        engine._loaded = True

        # Predict for each play state
        preds = np.zeros(len(states), dtype=np.float32)
        n_matched = 0
        for i, s in enumerate(states):
            is_bot = 1 if s["half_inning"] == "bottom" else 0
            game_pk = s.get("game_pk", 0)
            key = (game_pk, s["inning"], is_bot, s["outs"], s["score_diff"],
                   s["r1"], s["r2"], s["r3"])

            if lookup and key in lookup:
                preds[i] = lookup[key]
                n_matched += 1
            else:
                # Fallback: game-state-only prediction
                gs = {
                    "inning": s["inning"],
                    "top_bottom": "bottom" if is_bot else "top",
                    "outs": s["outs"],
                    "runners": (s["r1"], s["r2"], s["r3"]),
                    "score_diff": s["score_diff"],
                    "balls": 0, "strikes": 0,
                }
                feat = engine._build_features(gs, None, None)
                preds[i] = model.predict(
                    np.array([feat], dtype=np.float32))[0]

        preds = np.clip(preds, 0.005, 0.995).astype(np.float32)

        if lookup:
            pct = n_matched / len(states) * 100 if states else 0
            print(f"  Statcast base WP: {n_matched:,}/{len(states):,} matched "
                  f"({pct:.1f}%) with real features")
        print(f"  Statcast LightGBM base: mean={preds.mean():.4f}, "
              f"std={preds.std():.4f}")
        return preds

    except ImportError as e:
        print(f"  LightGBM/pandas not installed, falling back to Markov WP: {e}")
        return None
    except Exception as e:
        print(f"  Statcast model load failed: {e}, falling back to Markov WP")
        import traceback
        traceback.print_exc()
        return None


def _compute_markov_wp(inning, is_bottom, outs, r1, r2, r3, score_diff):
    """Vectorized Markov WP computation (fallback when Statcast model unavailable)."""
    runs_per_game = 4.5
    rphi = runs_per_game / 18.0

    home_hi = np.where(is_bottom,
                       np.maximum(0, 18 - 2 * inning),
                       np.maximum(0, 18 - 2 * inning + 1))
    away_hi = home_hi.copy()
    outs_frac = 1.0 - outs / 3.0
    runner_re = r1 * 0.37 + r2 * 0.70 + r3 * 1.00
    runner_re = np.where(is_bottom, runner_re, -runner_re)
    mean_diff = score_diff + runner_re
    var = rphi * (home_hi + away_hi + outs_frac) * 1.3

    return np.where(
        var <= 0,
        np.where(mean_diff > 0, 0.99, np.where(mean_diff < 0, 0.01, 0.52)),
        np.clip(_ndtr(mean_diff / np.sqrt(np.maximum(var, 1e-10))),
                0.005, 0.995)
    ).astype(np.float32)


def prepare_arrays(states: list[dict],
                    year_to_idx: dict[int, int] | None = None,
                    statcast_model_path: Path | None = None,
                    data_dir: Path | None = None,
                    ) -> dict[str, np.ndarray]:
    """Convert play states to numpy arrays for NumPyro (vectorized).

    Args:
        states: List of play state dicts.
        year_to_idx: If provided, use this mapping for season indices.
            For test sets, pass the train's mapping so season effects align.
            Years not in the mapping use the last index (extrapolation).
        statcast_model_path: Path to Statcast LightGBM model file.
            If provided and valid, uses Statcast predictions as base WP.
            Falls back to Markov WP if unavailable.
        data_dir: Directory containing statcast_pitches CSVs for real features.
    """
    n = len(states)
    if year_to_idx is None:
        years = sorted(set(s["year"] for s in states))
        year_to_idx = {y: i for i, y in enumerate(years)}
    else:
        years = sorted(year_to_idx.keys())

    # Extract columns as arrays first (avoid per-row dict access)
    inning = np.array([s["inning"] for s in states], dtype=np.float32)
    is_bottom = np.array([s["half_inning"] == "bottom" for s in states])
    outs = np.array([s["outs"] for s in states], dtype=np.float32)
    r1 = np.array([s["r1"] for s in states], dtype=np.float32)
    r2 = np.array([s["r2"] for s in states], dtype=np.float32)
    r3 = np.array([s["r3"] for s in states], dtype=np.float32)
    score_diff = np.array([s["score_diff"] for s in states], dtype=np.float32)

    home_idx = np.array([TEAM_TO_IDX[s["home_team"]] for s in states],
                        dtype=np.int32)
    away_idx = np.array([TEAM_TO_IDX[s["away_team"]] for s in states],
                        dtype=np.int32)
    max_season_idx = max(year_to_idx.values())
    season_idx = np.array(
        [year_to_idx.get(s["year"], max_season_idx) for s in states],
        dtype=np.int32)
    y = np.array([s["home_won"] for s in states], dtype=np.float32)

    # --- Base WP: Statcast LightGBM (preferred) or Markov (fallback) ---
    base_wp = _compute_statcast_base_wp(states, statcast_model_path,
                                         data_dir=data_dir)
    use_statcast = base_wp is not None

    if base_wp is None:
        print("  Using Markov WP as base (Statcast model not available)")
        base_wp = _compute_markov_wp(inning, is_bottom, outs,
                                      r1, r2, r3, score_diff)

    # --- Vectorized Leverage Index ---
    inn_factor = np.minimum(inning / 9.0, 1.5)
    abs_diff = np.abs(score_diff)
    close_factor = np.maximum(0, 1.0 - abs_diff / 5.0)
    runner_factor = 1.0 + (r1 + r2 + r3) * 0.15
    outs_factor = 1.0 + outs * 0.1

    leverage = inn_factor * close_factor * runner_factor * outs_factor
    # High-leverage boosts
    leverage = np.where(
        (inning >= 7) & (abs_diff <= 2), leverage * 1.5, leverage)
    leverage = np.where(
        (inning >= 9) & is_bottom & (abs_diff <= 1), leverage * 2.0, leverage)
    leverage = np.clip(leverage, 0.1, 5.0).astype(np.float32)

    # Logit transform of base WP
    wp_clipped = np.clip(base_wp, 0.005, 0.995)
    logit_wp = np.log(wp_clipped / (1 - wp_clipped))

    # Normalize leverage
    li_mean = leverage.mean()
    li_std = leverage.std() + 1e-8
    leverage_norm = (leverage - li_mean) / li_std

    return {
        "logit_base_wp": logit_wp.astype(np.float32),
        "leverage": leverage.astype(np.float32),
        "leverage_norm": leverage_norm.astype(np.float32),
        "home_idx": home_idx,
        "away_idx": away_idx,
        "season_idx": season_idx,
        "y": y,
        "n_seasons": len(years),
        "years": years,
        "base_wp": base_wp,
        "use_statcast": use_statcast,
        "leverage_mean": float(li_mean),
        "leverage_std": float(li_std),
    }


# ============================================================
# NumPyro Model
# ============================================================

def bayesian_wp_model(logit_base_wp, leverage_norm,
                      home_idx, away_idx, season_idx,
                      n_teams, n_seasons, y=None):
    """Bayesian hierarchical WP model.

    Generative process:
      1. Each team has a latent strength α ~ Normal(0, σ_team)
      2. Each park (=home team) has an effect β ~ Normal(0, σ_park)
      3. Season effects follow a random walk γ_t = γ_{t-1} + ε, ε ~ Normal(0, σ_season)
      4. Leverage-team interaction: strong teams capitalize more in clutch
      5. WP = sigmoid(logit_base + home_strength - away_strength + park + season + interaction)

    logit_base_wp: logit of Statcast LightGBM predictions (or Markov WP fallback)
    """
    n_obs = logit_base_wp.shape[0]

    # --- Hyperpriors ---
    # Priors must be wide enough that posteriors don't hit the boundary.
    # Previous run: posteriors at 0.31/0.23/0.13 against 0.3/0.15/0.1 priors.
    sigma_team = numpyro.sample("sigma_team", dist.HalfNormal(1.0))
    sigma_park = numpyro.sample("sigma_park", dist.HalfNormal(0.5))
    sigma_season = numpyro.sample("sigma_season", dist.HalfNormal(0.5))

    # --- Team strengths (non-centered parameterization) ---
    # Zero-sum: the league average is already in the base WP
    team_raw = numpyro.sample("team_raw",
                               dist.Normal(0, 1).expand([n_teams]))
    team_strength = numpyro.deterministic(
        "team_strength", sigma_team * (team_raw - team_raw.mean()))

    # --- Park effects (non-centered) ---
    # park[i] = team's home field advantage beyond the league-average HFA
    park_raw = numpyro.sample("park_raw",
                               dist.Normal(0, 1).expand([n_teams]))
    park_effect = numpyro.deterministic(
        "park_effect", sigma_park * (park_raw - park_raw.mean()))

    # --- Season effects (random walk prior) ---
    # Captures era-level changes: juiced ball, mound height, etc.
    if n_seasons > 1:
        season_innov = numpyro.sample(
            "season_innov",
            dist.Normal(0, 1).expand([n_seasons - 1]))
        season_cumsum = jnp.concatenate(
            [jnp.zeros(1), jnp.cumsum(season_innov)])
        season_effect = numpyro.deterministic(
            "season_effect", sigma_season * season_cumsum)
    else:
        season_effect = numpyro.deterministic(
            "season_effect", jnp.zeros(1))

    # --- Leverage × team interaction ---
    # kappa > 0 means better teams gain more WP in high-leverage situations
    kappa = numpyro.sample("kappa", dist.Normal(0, 0.2))

    # --- Home field advantage correction ---
    # Base WP assumes average HFA; this learns the residual
    hfa_correction = numpyro.sample("hfa_correction", dist.Normal(0, 0.1))

    # --- Linear predictor ---
    team_diff = team_strength[home_idx] - team_strength[away_idx]

    logit_wp = (
        logit_base_wp
        + team_diff
        + park_effect[home_idx]
        + season_effect[season_idx]
        + kappa * leverage_norm * team_diff
        + hfa_correction
    )

    # --- Observation model ---
    # Subsampling is standard for SVI; plate auto-scales the likelihood.
    with numpyro.plate("obs", n_obs, subsample_size=min(n_obs, 65536)) as idx:
        numpyro.sample("y",
                        dist.Bernoulli(logits=logit_wp[idx]),
                        obs=y[idx] if y is not None else None)


# ============================================================
# Training
# ============================================================

def train_svi(arrays: dict, n_steps: int = 20000, lr: float = 0.005,
              seed: int = 42) -> dict:
    """Train with Stochastic Variational Inference."""
    rng_key = jrandom.PRNGKey(seed)

    n_teams = N_TEAMS
    n_seasons = arrays["n_seasons"]

    # SVI setup
    guide = AutoNormal(bayesian_wp_model)
    optimizer = numpyro.optim.ClippedAdam(step_size=lr)
    svi = SVI(bayesian_wp_model, guide, optimizer, loss=Trace_ELBO())

    # Convert to JAX arrays once (avoid re-conversion every step)
    jax_arrays = {
        "logit_base_wp": jnp.array(arrays["logit_base_wp"]),
        "leverage_norm": jnp.array(arrays["leverage_norm"]),
        "home_idx": jnp.array(arrays["home_idx"]),
        "away_idx": jnp.array(arrays["away_idx"]),
        "season_idx": jnp.array(arrays["season_idx"]),
        "y": jnp.array(arrays["y"]),
    }

    # Initialize
    svi_state = svi.init(
        rng_key,
        logit_base_wp=jax_arrays["logit_base_wp"],
        leverage_norm=jax_arrays["leverage_norm"],
        home_idx=jax_arrays["home_idx"],
        away_idx=jax_arrays["away_idx"],
        season_idx=jax_arrays["season_idx"],
        n_teams=n_teams,
        n_seasons=n_seasons,
        y=jax_arrays["y"],
    )

    # Training loop
    print(f"\n  SVI training: {n_steps} steps, lr={lr}")
    losses = []
    t0 = time.time()

    for step in range(n_steps):
        svi_state, loss = svi.update(
            svi_state,
            logit_base_wp=jax_arrays["logit_base_wp"],
            leverage_norm=jax_arrays["leverage_norm"],
            home_idx=jax_arrays["home_idx"],
            away_idx=jax_arrays["away_idx"],
            season_idx=jax_arrays["season_idx"],
            n_teams=n_teams,
            n_seasons=n_seasons,
            y=jax_arrays["y"],
        )
        losses.append(float(loss))

        if (step + 1) % 500 == 0 or step == 0:
            elapsed = time.time() - t0
            avg_loss = np.mean(losses[-500:]) if len(losses) >= 500 else np.mean(losses)

            # Always show convergence info when available
            rel_info = ""
            if len(losses) >= 1000:
                recent = np.mean(losses[-500:])
                earlier = np.mean(losses[-1000:-500])
                rel_change = abs(recent - earlier) / (abs(earlier) + 1e-8)
                rel_info = f" | rel_change={rel_change:.2e}"

            print(f"    Step {step + 1:>6}/{n_steps}: ELBO loss = {avg_loss:.2f} "
                  f"({elapsed:.0f}s){rel_info}")

            # Early stopping: check if ELBO has plateaued
            if len(losses) >= 2000:
                recent_es = np.mean(losses[-500:])
                earlier_es = np.mean(losses[-1500:-1000])
                rel_change_es = abs(recent_es - earlier_es) / (abs(earlier_es) + 1e-8)
                if rel_change_es < 5e-5:
                    print(f"  Early stopping at step {step + 1} "
                          f"(rel_change={rel_change_es:.2e} < 5e-5)")
                    break

    actual_steps = len(losses)
    elapsed = time.time() - t0
    final_loss = np.mean(losses[-500:]) if len(losses) >= 500 else np.mean(losses)
    print(f"  Training complete: {actual_steps} steps in {elapsed:.0f}s "
          f"({elapsed/actual_steps:.2f}s/step)")
    print(f"  Final ELBO loss: {final_loss:.2f}")
    if actual_steps < n_steps:
        print(f"  (Early stopped at {actual_steps}/{n_steps})")
    else:
        print(f"  (Ran full {n_steps} steps — consider increasing if still improving)")

    # Extract posterior samples
    params = svi.get_params(svi_state)
    rng_key, pred_key = jrandom.split(rng_key)

    predictive = Predictive(
        bayesian_wp_model, guide=guide, params=params,
        num_samples=500, return_sites=[
            "team_strength", "park_effect", "season_effect",
            "sigma_team", "sigma_park", "sigma_season",
            "kappa", "hfa_correction",
        ])

    posterior_samples = predictive(
        pred_key,
        logit_base_wp=jax_arrays["logit_base_wp"][:1],
        leverage_norm=jax_arrays["leverage_norm"][:1],
        home_idx=jax_arrays["home_idx"][:1],
        away_idx=jax_arrays["away_idx"][:1],
        season_idx=jax_arrays["season_idx"][:1],
        n_teams=n_teams,
        n_seasons=n_seasons,
    )

    return {
        k: np.array(v).tolist()
        for k, v in posterior_samples.items()
    }, losses


# ============================================================
# Evaluation
# ============================================================

def predict_with_uncertainty(arrays: dict, posterior: dict,
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate predictions with credible intervals from posterior samples.

    Returns: (mean_wp, lower_5, upper_95)
    """
    n_samples = len(posterior["team_strength"])
    n_obs = len(arrays["logit_base_wp"])

    team_strength = np.array(posterior["team_strength"])  # (n_samples, n_teams)
    park_effect = np.array(posterior["park_effect"])       # (n_samples, n_teams)
    season_effect = np.array(posterior["season_effect"])   # (n_samples, n_seasons)
    kappa = np.array(posterior["kappa"])                   # (n_samples,)
    hfa_corr = np.array(posterior["hfa_correction"])       # (n_samples,)

    logit_base = arrays["logit_base_wp"]
    lev_norm = arrays["leverage_norm"]
    h_idx = arrays["home_idx"]
    a_idx = arrays["away_idx"]
    s_idx = arrays["season_idx"]

    # Vectorized over samples
    all_wp = np.zeros((n_samples, n_obs), dtype=np.float32)

    for s in range(n_samples):
        team_diff = team_strength[s][h_idx] - team_strength[s][a_idx]
        logit_wp = (
            logit_base
            + team_diff
            + park_effect[s][h_idx]
            + season_effect[s][s_idx]
            + kappa[s] * lev_norm * team_diff
            + hfa_corr[s]
        )
        all_wp[s] = 1.0 / (1.0 + np.exp(-logit_wp))

    mean_wp = np.mean(all_wp, axis=0)
    lower_5 = np.percentile(all_wp, 5, axis=0)
    upper_95 = np.percentile(all_wp, 95, axis=0)

    return mean_wp, lower_5, upper_95


def evaluate(preds: np.ndarray, actuals: np.ndarray, name: str,
             lower: np.ndarray | None = None,
             upper: np.ndarray | None = None) -> dict:
    """Compute Brier score, BSS, LogLoss, ECE, and CI coverage."""
    brier = float(np.mean((preds - actuals) ** 2))
    brier_base = float(np.mean((0.5 - actuals) ** 2))
    brier_skill = 1 - brier / brier_base

    eps = 1e-7
    p = np.clip(preds, eps, 1 - eps)
    log_loss = float(-np.mean(actuals * np.log(p) + (1 - actuals) * np.log(1 - p)))

    # ECE (10 bins)
    ece = 0.0
    n = len(actuals)
    for low_bin in np.arange(0, 1.0, 0.1):
        mask = (preds >= low_bin) & (preds < low_bin + 0.1)
        if mask.sum() > 0:
            ece += abs(preds[mask].mean() - actuals[mask].mean()) * mask.sum() / n

    print(f"  {name}: Brier={brier:.6f} BSS={brier_skill:.4f} "
          f"LogLoss={log_loss:.6f} ECE={ece:.4f}")

    result = {
        "model": name,
        "brier": round(brier, 6),
        "brier_skill": round(brier_skill, 4),
        "log_loss": round(log_loss, 6),
        "ece": round(ece, 4),
    }

    # CI coverage (90% credible interval should contain ~90% of outcomes)
    if lower is not None and upper is not None:
        # For binary outcomes, "coverage" = fraction of times the CI
        # includes the empirical win rate in its bin
        ci_width = float(np.mean(upper - lower))
        print(f"    Mean 90% CI width: {ci_width:.4f}")
        result["ci_width_mean"] = round(ci_width, 4)

    return result


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Bayesian Hierarchical WP model (NumPyro SVI)")
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--output-dir", default="data/")
    parser.add_argument("--test-year", type=int, default=2024)
    parser.add_argument("--n-steps", type=int, default=5000,
                        help="SVI optimization steps")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--statcast-model", default=None,
                        help="Path to Statcast LightGBM model (wp_statcast_lgbm.txt)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # Load data
    # -------------------------------------------------------
    print("=" * 60)
    print("Loading play-by-play data")
    print("=" * 60)

    train_states = load_play_states(data_dir, exclude_years={args.test_year})
    test_states = load_play_states(data_dir,
                                   exclude_years=set(
                                       range(2000, 2030)) - {args.test_year})

    print(f"\n  Train: {len(train_states):,} plays (excl. {args.test_year})")
    print(f"  Test:  {len(test_states):,} plays ({args.test_year} only)")

    if not train_states or not test_states:
        print("ERROR: Insufficient data")
        sys.exit(1)

    # -------------------------------------------------------
    # Prepare arrays
    # -------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Preparing features")
    print("=" * 60)

    # Resolve Statcast model path
    statcast_path = None
    if args.statcast_model:
        statcast_path = Path(args.statcast_model)
    else:
        # Auto-detect: check data/ directory
        for candidate in [data_dir / "wp_statcast_lgbm.txt",
                          Path(__file__).parent.parent / "data" / "wp_statcast_lgbm.txt"]:
            if candidate.exists():
                statcast_path = candidate
                break
    if statcast_path and statcast_path.exists():
        print(f"  Statcast model: {statcast_path}")
    else:
        print("  Statcast model not found, will use Markov WP fallback")

    train_arrays = prepare_arrays(train_states,
                                   statcast_model_path=statcast_path,
                                   data_dir=data_dir)

    # Test uses train's year_to_idx so season effects align correctly.
    # 2024 (test year) maps to last season index (extrapolation from 2023).
    train_year_to_idx = {y: i for i, y in enumerate(train_arrays["years"])}
    test_arrays = prepare_arrays(test_states, year_to_idx=train_year_to_idx,
                                  statcast_model_path=statcast_path,
                                  data_dir=data_dir)

    # Use train normalization for test leverage
    test_arrays["leverage_norm"] = (
        (test_arrays["leverage"] - train_arrays["leverage_mean"])
        / train_arrays["leverage_std"]
    ).astype(np.float32)

    # Override test n_seasons to match train (for posterior indexing)
    test_arrays["n_seasons"] = train_arrays["n_seasons"]

    print(f"  Train features: {len(train_arrays['y']):,} obs, "
          f"{train_arrays['n_seasons']} seasons")
    print(f"  Test features:  {len(test_arrays['y']):,} obs")
    print(f"  Teams: {N_TEAMS}")
    print(f"  Leverage (train): mean={train_arrays['leverage'].mean():.2f}, "
          f"std={train_arrays['leverage'].std():.2f}")

    # Diagnostic: season index mapping
    print(f"\n  Season index mapping (train):")
    for yr, idx in sorted(train_year_to_idx.items()):
        print(f"    {yr} -> idx {idx}")
    test_sidx = test_arrays["season_idx"]
    print(f"  Test season_idx: min={test_sidx.min()}, max={test_sidx.max()}, "
          f"unique={np.unique(test_sidx).tolist()}")
    print(f"  (Test year {args.test_year} -> idx {test_sidx[0]} "
          f"= extrapolate from {train_arrays['years'][-1]})")

    # -------------------------------------------------------
    # Baseline (on test set)
    # -------------------------------------------------------
    print(f"\n{'=' * 60}")
    base_label = "Statcast_LightGBM" if test_arrays["use_statcast"] else "Markov_v1"
    print(f"BASELINE: {base_label}")
    print("=" * 60)
    base_test_wp = test_arrays["base_wp"]
    base_metrics = evaluate(base_test_wp, test_arrays["y"], base_label)

    # -------------------------------------------------------
    # Train Bayesian model
    # -------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Training Bayesian Hierarchical WP Model")
    print(f"  SVI steps: {args.n_steps}, lr: {args.lr}")
    print("=" * 60)

    posterior, losses = train_svi(
        train_arrays, n_steps=args.n_steps, lr=args.lr, seed=args.seed)

    # -------------------------------------------------------
    # Posterior summary
    # -------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("POSTERIOR SUMMARY")
    print("=" * 60)

    # Hyperparameters with prior boundary check
    prior_scales = {
        "sigma_team": 1.0, "sigma_park": 0.5, "sigma_season": 0.5,
        "kappa": 0.2, "hfa_correction": 0.1,
    }
    for param in ["sigma_team", "sigma_park", "sigma_season", "kappa",
                   "hfa_correction"]:
        vals = np.array(posterior[param])
        mean = vals.mean()
        ci_lo = np.percentile(vals, 5)
        ci_hi = np.percentile(vals, 95)
        prior_scale = prior_scales[param]
        # For HalfNormal, effective boundary ≈ 2*scale; for Normal, ≈ 2*scale
        ratio = abs(mean) / prior_scale
        boundary_warn = " *** NEAR PRIOR BOUNDARY ***" if ratio > 0.8 else ""
        print(f"  {param}: mean={mean:.4f}, std={vals.std():.4f}, "
              f"90% CI=[{ci_lo:.4f}, {ci_hi:.4f}] "
              f"(prior_scale={prior_scale}, ratio={ratio:.2f}){boundary_warn}")

    # Team strengths
    team_str = np.array(posterior["team_strength"])  # (500, n_teams)
    team_means = team_str.mean(axis=0)
    team_stds = team_str.std(axis=0)
    ranked = sorted(zip(ALL_TEAMS, team_means, team_stds),
                    key=lambda x: -x[1])

    print(f"\n  Team Strength Rankings (logit scale):")
    print(f"  {'Team':<6} {'Mean':>8} {'Std':>8} {'WP vs avg':>10}")
    for team, mean, std in ranked:
        # Convert logit-scale team strength to WP impact
        wp_impact = 1 / (1 + np.exp(-mean)) - 0.5
        print(f"  {team:<6} {mean:>8.4f} {std:>8.4f} {wp_impact:>+10.3f}")

    # Park effects
    park_eff = np.array(posterior["park_effect"])
    park_means = park_eff.mean(axis=0)
    park_ranked = sorted(zip(ALL_TEAMS, park_means), key=lambda x: -x[1])
    print(f"\n  Park Effects (top 5 / bottom 5):")
    for team, mean in park_ranked[:5]:
        print(f"    {team}: {mean:+.4f}")
    print(f"    ...")
    for team, mean in park_ranked[-5:]:
        print(f"    {team}: {mean:+.4f}")

    # Season effects
    season_eff = np.array(posterior["season_effect"])
    season_means = season_eff.mean(axis=0)
    print(f"\n  Season Effects:")
    for yr, eff in zip(train_arrays["years"], season_means):
        print(f"    {yr}: {eff:+.4f}")

    # -------------------------------------------------------
    # Evaluate on test set
    # -------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"EVALUATION (holdout: {args.test_year})")
    print("=" * 60)

    # Need to remap test season indices to train's year mapping
    # Since test is a single year, map it to the closest train year
    # Actually for prediction, we use the last season effect as proxy
    # (the test year is excluded from training, so use the trend)

    mean_wp, lower_5, upper_95 = predict_with_uncertainty(
        test_arrays, posterior)

    bayes_metrics = evaluate(mean_wp, test_arrays["y"], "Bayesian_hierarchical",
                             lower_5, upper_95)

    # Comparison
    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print("=" * 60)

    base_brier = base_metrics["brier"]
    bayes_brier = bayes_metrics["brier"]
    improvement = (base_brier - bayes_brier) / base_brier * 100
    print(f"  {base_label}:  Brier = {base_brier:.6f}")
    print(f"  Bayesian:        Brier = {bayes_brier:.6f} "
          f"({improvement:+.2f}% vs {base_label})")

    # -------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Saving outputs")
    print("=" * 60)

    # 1. Posterior parameters (for runtime prediction)
    posterior_path = output_dir / "bayesian_posterior.json"
    posterior_data = {
        "model_version": "bayesian_hierarchical_v1",
        "train_years": train_arrays["years"],
        "test_year": args.test_year,
        "n_steps": args.n_steps,
        "teams": ALL_TEAMS,
        "n_teams": N_TEAMS,
        "n_seasons": train_arrays["n_seasons"],
        "leverage_mean": train_arrays["leverage_mean"],
        "leverage_std": train_arrays["leverage_std"],
        # Posterior summaries (not full samples, for compact storage)
        "team_strength_mean": team_means.tolist(),
        "team_strength_std": team_stds.tolist(),
        "park_effect_mean": park_means.tolist(),
        "park_effect_std": park_eff.std(axis=0).tolist(),
        "season_effect_mean": season_means.tolist(),
        "season_effect_std": season_eff.std(axis=0).tolist(),
        "kappa_mean": float(np.array(posterior["kappa"]).mean()),
        "kappa_std": float(np.array(posterior["kappa"]).std()),
        "hfa_correction_mean": float(
            np.array(posterior["hfa_correction"]).mean()),
        "hfa_correction_std": float(
            np.array(posterior["hfa_correction"]).std()),
        "sigma_team_mean": float(np.array(posterior["sigma_team"]).mean()),
        "sigma_park_mean": float(np.array(posterior["sigma_park"]).mean()),
        "sigma_season_mean": float(np.array(posterior["sigma_season"]).mean()),
    }
    with open(posterior_path, "w") as f:
        json.dump(posterior_data, f, indent=2)
    print(f"  {posterior_path}")

    # 2. Full posterior samples (for CI computation at runtime)
    samples_path = output_dir / "bayesian_posterior_samples.json"
    with open(samples_path, "w") as f:
        json.dump(posterior, f)
    print(f"  {samples_path}")

    # 3. Results summary
    results_path = output_dir / "bayesian_results.json"
    results = {
        "test_year": args.test_year,
        "n_steps": args.n_steps,
        "baseline": base_metrics,
        "baseline_type": base_label,
        "bayesian": bayes_metrics,
        "improvement_vs_baseline_pct": round(improvement, 2),
        "final_elbo_loss": round(float(np.mean(losses[-500:])), 2),
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  {results_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
