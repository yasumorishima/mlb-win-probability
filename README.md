# MLB Win Probability Engine

イニング・アウト・走者・点差を入力すると、**ホーム勝率・局面の重要度・戦術提案**をリアルタイムで返すエンジン。

**[Live Demo](https://mlb-wp-engine.streamlit.app/)** | English / 日本語 | MLB / NPB 対応

---

**計算の仕組み（3層構造）：**
1. **RE24** — 24通りの走者×アウト状況ごとに「この回に何点取れるか」の期待値テーブル
2. **Markov Chain** — 打席ごとの状態遷移（三振/単打/本塁打…）を確率モデルで繋ぎ、得点分布を推定
3. **Normal approximation** — 残りイニングの得点分布から最終的な勝率を計算

外部データ不要・純粋な数理モデルのため MLB / NPB どちらでも動作します。

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

## Model Validation（精度検証）

GitHub Actions で 2024 シーズン全試合（~2,430 試合・~170,000 打席）の play-by-play データを取得し、自作 WP モデルの精度を定量検証しています。

### 検証パイプライン

```
MLB Stats API → play-by-play CSV → WP算出 → 実際の勝敗と比較
                                  ↓
                          RE24再計算（実データ vs ハードコード値）
                                  ↓
                          Optuna 500trial パラメータ最適化
```

### メトリクス

| Metric | Description |
|--------|-------------|
| Brier Score | 確率予測の代表的評価指標。mean((predicted - actual)^2)、低いほど良い |
| Brier Skill Score | 常に 50% と予測するベースラインとの比較。正なら改善 |
| ECE (Expected Calibration Error) | 予測値と実際の勝率のズレ。0 に近いほど校正が正確 |
| Log Loss | 確信度の高い誤予測を重く罰するスコアリングルール |
| MAE by Inning | イニング別の平均絶対誤差。終盤の精度を個別に評価 |

### 検証ワークフロー

```bash
gh workflow run "Validate WP Model" \
  --repo yasumorishima/mlb-win-probability \
  -f memo="2024 full season validation" \
  -f season=2024 \
  -f optimize=true \
  -f n_trials=500
```

### パラメータ最適化

現在のモデルには以下のパラメータがあり、Optuna（TPE sampler）で最適値を探索：

| Parameter | Description | Default |
|-----------|-------------|---------|
| `variance_factor` | 残りイニングの得点分布の分散係数 | 1.3 |
| `scoring_factor` | 9回裏同点時のサヨナラ確率スケーリング | 1.8 |
| `behind_lambda_mult` | 9回裏ビハインド時のPoisson λ倍率 | 1.5 |
| `top9_lambda_mult` | 9回表ホームリード時のPoisson λ倍率 | 1.3 |
| `extras_win_prob` | 延長戦突入時のホーム勝率 | 0.50 |

## Scoring Environment（得点環境の調整）

得点の多い/少ないリーグでは勝率の変動幅も変わります。`runs_per_game` パラメータで切り替え可能。

| League | Runs/Game | Parameter |
|--------|-----------|-----------|
| MLB | 4.5 | `runs_per_game=4.5` (default) |
| NPB | 4.0 | `runs_per_game=4.0` |

## License

MIT
