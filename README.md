# MLB Win Probability Engine

イニング・アウト・走者・点差を入力すると、**ホーム勝率・局面の重要度・戦術提案**をリアルタイムで返すエンジン。

**[Live Demo](https://mlb-wp-engine.streamlit.app/)** | English / 日本語

---

公開して動いているのは **Streamlit デモ**だけで、FastAPI は手元で起動して使うもの（公開サーバーは無い）。新規モデル学習ワークフローは本リポに含まれない。

---

## Features

- **Win Probability**: Home team win probability given inning, outs, runners, score
- **Leverage Index**: How critical the current plate appearance is (1.0 = average, 4.0+ = critical)
- **WPA (Win Probability Added)**: WP change from a single play
- **Live Feed**: Real-time game state + pitch/hit data via MLB Stats API (auto-refresh every 30s)
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

### `GET /wp/play` — WPA for a Single Play

```bash
# Before: Bot 9, 1 out, runner on 2nd, tie → After: Bot 9, 1 out, runners 1st & 3rd, tie
curl "http://localhost:8000/wp/play?before_inning=9&before_top_bottom=bottom&before_outs=1&before_runner2=1&before_score_diff=0&after_inning=9&after_top_bottom=bottom&after_outs=1&after_runner1=1&after_runner3=1&after_score_diff=0"
```

### `GET /re24` — RE24 Table

```bash
curl "http://localhost:8000/re24"

```

### `GET /wp/scenario` — Preset Scenarios

```bash
curl "http://localhost:8000/wp/scenario?name=ninth_inning_drama"
curl "http://localhost:8000/wp/scenario?name=walkoff_chance"
```

Available scenarios: `ninth_inning_drama`, `game_start`, `rally_7th`, `tied_8th`, `walkoff_chance`, `comfortable_lead`

## Technical Background

### Leverage Index

| LI Range | Label | Meaning |
|----------|-------|---------|
| < 0.5 | Low | Routine situation |
| 0.5–1.5 | Medium | Average importance |
| 1.5–3.0 | High | Key moment |
| 3.0+ | Very High | Game-defining |

### FastAPI（ローカル / RPi5）

| 項目 | 値 |
|---|---|
| Swagger UI | ローカル起動: `http://localhost:8001/docs` |
| メモリ | 256Mi |

## Roadmap

### Phase 1: 基盤構築 ✅
- [x] WP エンジン v1（Markov Chain + Normal 近似 + Optuna 5 パラメータ最適化）
- [x] FastAPI + Streamlit ダッシュボード（バイリンガル、ライブフィード、What-If）

## License

MIT
