# MLB Win Probability Engine

Real-time Win Probability (WP), Leverage Index (LI), and tactical recommendations from game state inputs.

**Approach**: Markov Chain + RE24 (Run Expectancy by 24 base-out states) + Normal distribution approximation.

No external data dependencies — pure mathematical model that works for both MLB and NPB scoring environments.

**[Live Demo](https://mlb-wp-engine.streamlit.app/)** | English / 日本語

## Features

- **Win Probability**: Home team win probability given inning, outs, runners, score
- **Leverage Index**: How critical the current plate appearance is (1.0 = average, 4.0+ = critical)
- **WPA (Win Probability Added)**: WP change from a single play
- **Tactical Recommendations**: RE24-based evaluation of 8 tactical options (bunt, steal, squeeze, etc.)
- **Matchup Adjustment**: Fine-tune WP with batter OPS and pitcher ERA
- **Live Feed**: Real-time game state via MLB Stats API (auto-refresh every 30s)
- **Preset Scenarios**: 6 pre-built game situations for quick analysis
- **Bilingual UI**: English / Japanese toggle

## Quick Start

### FastAPI

```bash
pip install -r requirements.txt
uvicorn api:app --reload
# → http://localhost:8000/docs
```

### Docker

```bash
docker compose up --build
# → http://localhost:8001/docs
```

### Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

## API Endpoints

### `GET /wp` — Win Probability + LI + Tactics

```bash
# 9th inning, bottom, 2 outs, bases loaded, tie game
curl "http://localhost:8000/wp?inning=9&top_bottom=bottom&outs=2&runner1=1&runner2=1&runner3=1&score_diff=0"
```

Response:
```json
{
  "win_probability": 0.93,
  "leverage_index": 8.6,
  "leverage_label": "Very High",
  "tactics": [...]
}
```

### `GET /wp/play` — WPA for a Single Play

```bash
# Before: Bot 9, 1 out, runner on 2nd, tie → After: Bot 9, 1 out, runners 1st & 3rd, tie
curl "http://localhost:8000/wp/play?before_inning=9&before_top_bottom=bottom&before_outs=1&before_runner2=1&before_score_diff=0&after_inning=9&after_top_bottom=bottom&after_outs=1&after_runner1=1&after_runner3=1&after_score_diff=0"
```

### `GET /re24` — RE24 Table

```bash
# MLB environment (default)
curl "http://localhost:8000/re24"

# NPB environment
curl "http://localhost:8000/re24?runs_per_game=4.0"
```

### `GET /wp/scenario` — Preset Scenarios

```bash
curl "http://localhost:8000/wp/scenario?name=ninth_inning_drama"
curl "http://localhost:8000/wp/scenario?name=walkoff_chance"
```

Available scenarios: `ninth_inning_drama`, `game_start`, `rally_7th`, `tied_8th`, `walkoff_chance`, `comfortable_lead`

## Technical Background

### RE24 (Run Expectancy Matrix)

24 base-out states (8 runner configurations × 3 out states), each with an expected number of runs to score in the remainder of the inning. Based on MLB 2010-2019 average data. Scaled linearly for different scoring environments.

### Win Probability

Uses remaining innings and scoring rate to estimate run distributions via normal approximation. Late-inning states (9th inning) use specialized calculations to handle walk-off and save situations.

### Leverage Index

For each of 9 representative plate appearance outcomes (strikeout, single, double, HR, walk, etc.), calculates the WP change. LI = expected |WP change| / league average |WP change per PA| (≈0.035).

| LI Range | Label | Meaning |
|----------|-------|---------|
| < 0.5 | Low | Routine situation |
| 0.5–1.5 | Medium | Average importance |
| 1.5–3.0 | High | Key moment |
| 3.0+ | Very High | Game-defining |

### Tactical Recommendations

Evaluates 8 tactics by comparing expected RE24 values:
- Sacrifice Bunt / Steal 2B / Steal 3B / Intentional Walk
- Pitching Change / Pinch Hitter / Hit and Run / Squeeze Play

Each tactic has preconditions (e.g., steal requires a runner) and a success probability. The expected RE24 delta determines the recommendation.

## Scoring Environment

| League | Runs/Game | Parameter |
|--------|-----------|-----------|
| MLB | 4.5 | `runs_per_game=4.5` (default) |
| NPB | 4.0 | `runs_per_game=4.0` |

## License

MIT
