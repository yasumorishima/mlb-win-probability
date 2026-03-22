# MLB Win Probability Engine

イニング・アウト・走者・点差を入力すると、**ホーム勝率・局面の重要度・戦術提案**をリアルタイムで返すエンジン。

**[Live Demo](https://mlb-wp-engine.streamlit.app/)** | English / 日本語

---

**3 エンジン + アンサンブル構成：**

| Engine | Approach |
|--------|----------|
| **v1** | RE24 + Markov Chain + Normal 近似（Optuna 最適化済み） |
| **v2** | 10 年分 MLB 実データの経験的 WP テーブル + Markov Chain フォールバック |
| **LightGBM** | 勾配ブースティング（11 特徴量、367K play states で学習） |

3 エンジンを **inverse-Brier 加重アンサンブル** + **Isotonic Regression キャリブレーション**で統合。
学習・検証データは全て BigQuery（`data-platform-490901.mlb_wp.play_states`）から取得。

## Features

- **Win Probability**: Home team win probability given inning, outs, runners, score
- **Leverage Index**: How critical the current plate appearance is (1.0 = average, 4.0+ = critical)
- **WPA (Win Probability Added)**: WP change from a single play
- **Tactical Recommendations**: RE24-based evaluation of 8 tactical options (bunt, steal, squeeze, etc.)
- **Matchup Adjustment**: Fine-tune WP with batter OPS and pitcher ERA
- **AI Commentary (Gemini 2.5 Flash)**: Natural language game situation analysis with automated quality evaluation
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

## Grafana Dashboard

[MLB Win Probability](https://yasumorishima.grafana.net/public-dashboards/8cf85d216d6e47068c3dcc7e807ac337) — Situation-based win expectancy analysis from 367K+ play states (2015–2024). Connected to BigQuery `data-platform-490901.mlb_wp`.

![MLB Win Probability — Grafana Dashboard](docs/images/grafana-preview.png)

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

### `GET /wp/commentary` — AI Commentary (Gemini 2.5 Flash)

```bash
# 9th inning drama with AI analysis
curl "http://localhost:8000/wp/commentary?inning=9&top_bottom=bottom&outs=2&runner1=1&runner2=1&runner3=1&score_diff=0&lang=JA"
```

Response:
```json
{
  "win_probability": 0.93,
  "leverage_index": 8.6,
  "ai_commentary": "9回裏2アウト満塁、同点という...",
  "quality_evaluation": {
    "score": 95,
    "pass": true,
    "checks": {"mentions_wp": true, "mentions_li": true, ...}
  },
  "model": "gemini-2.5-flash",
  "lang": "JA"
}
```

### `GET /commentary/info` — Commentary System Metadata

```bash
curl "http://localhost:8000/commentary/info"
```

Returns prompt version, quality criteria, and available versions (similar to `/model/info` in [baseball-mlops](https://github.com/yasumorishima/baseball-mlops)).

### `GET /wp/play` — WPA for a Single Play

```bash
# Before: Bot 9, 1 out, runner on 2nd, tie → After: Bot 9, 1 out, runners 1st & 3rd, tie
curl "http://localhost:8000/wp/play?before_inning=9&before_top_bottom=bottom&before_outs=1&before_runner2=1&before_score_diff=0&after_inning=9&after_top_bottom=bottom&after_outs=1&after_runner1=1&after_runner3=1&after_score_diff=0"
```

### `GET /re24` — RE24 Table

```bash
# MLB environment (default)
curl "http://localhost:8000/re24"

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

3 エンジン構成（詳細は [Model Validation](#model-validation精度検証) セクション参照）:
- **v1**: Normal 近似 + Optuna 最適化（5 パラメータ）
- **v2**: 実データ WP テーブル（10 年分）+ Markov Chain フォールバック
- **LightGBM**: 勾配ブースティング（11 特徴量）

アンサンブルで統合し、Isotonic Regression でキャリブレーション補正。

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

## AI Commentary Architecture

Gemini 2.5 Flash を使ったリアルタイム状況解説機能。WP エンジンの構造化データ（勝利確率・レバレッジ指数・RE24・作戦提案・What-If 分析）を LLM に渡し、データに基づいた自然言語解説を生成。

### データフロー

```
Game State (inning, outs, runners, score)
    ↓
WP Engine (Markov Chain + RE24)
    ↓ full_analysis()
┌─────────────────────────────────────────┐
│ WP: 87.1%  LI: 11.19  RE24: 1.541     │
│ Tactics: [Hit and Run: Recommended]     │
│ What-If: HR→98.2%, K→83.4%, 1B→95.1%  │
└─────────────────────────────────────────┘
    ↓ Structured prompt (v2)
Gemini 2.5 Flash API
    ↓
Natural language commentary
    ↓
Quality Evaluation (6 criteria, 100pt)
    ↓
W&B Logging (latency, tokens, quality score)
```

### MLOps Integration

[baseball-mlops](https://github.com/yasumorishima/baseball-mlops) と同じ MLOps 設計思想を LLM 解説生成に適用：

| baseball-mlops (ML model) | mlb-win-probability (LLM commentary) |
|---|---|
| Model versioning (W&B Artifact + production alias) | Prompt versioning (`PROMPT_REGISTRY` + `PROMPT_VERSION`) |
| MAE evaluation (prediction vs actual) | Quality evaluation (6-criteria rule-based scoring, 100pt) |
| W&B experiment tracking (MAE, hyperparams) | W&B tracking (quality score, latency, token count, prompt version) |
| `/model/info` endpoint | `/commentary/info` endpoint |
| APScheduler model cache (6h refresh) | Session-based commentary cache |
| `continue-on-error` (non-blocking GCP/Discord) | Non-blocking W&B logging |

### Quality Evaluation

解説が分析データを適切に活用しているかを自動スコアリング：

| Criterion | Weight | Check |
|---|---|---|
| Win Probability 言及 | 25pt | 勝利確率の数値またはキーワード |
| Leverage Index 言及 | 20pt | プレッシャー/重要度への言及 |
| Run Expectancy 言及 | 15pt | 期待得点/得点可能性への言及 |
| What-If 言及 | 20pt | 次のプレイ別の WP 変動への言及 |
| Tactics 言及 | 10pt | 推奨作戦がある場合の言及 |
| 適切な長さ | 10pt | JA: 100-600文字 / EN: 150-800文字 |

**60点以上で PASS**。品質スコアは W&B に時系列で記録され、プロンプト改善のフィードバックループを形成。

### Setup

```bash
# Google AI Studio で無料 API キーを取得
# https://aistudio.google.com/apikey

# 環境変数に設定
export GEMINI_API_KEY="your-api-key"

# Streamlit Cloud の場合は secrets.toml に設定
# GEMINI_API_KEY = "your-api-key"

# W&B トラッキングを有効にする場合（任意）
export WANDB_API_KEY="your-wandb-key"
```

## Model Validation（精度検証）

367K+ play states（2015–2024、10シーズン）を使って 3 つの WP エンジンを定量検証し、最良のアンサンブルを構築しています。

### データ基盤

全データは **BigQuery**（`data-platform-490901.mlb_wp.play_states`）に格納。学習・検証パイプラインは BQ から直接エクスポートして実行（MLB Stats API はフォールバック用）。

| Item | Value |
|------|-------|
| Project | `data-platform-490901` |
| Dataset / Table | `mlb_wp.play_states` |
| Rows | 367,564（2015–2024 全試合） |
| Cost | Free tier |

### 3 エンジン構成

| Engine | Approach | 特徴 |
|--------|----------|------|
| **v1 (Normal)** | Markov Chain + Normal 近似 + Optuna 最適化 | 5 パラメータ、数式ベース |
| **v2 (Empirical)** | 実データ WP テーブル + Markov Chain フォールバック | 10 年分の経験的確率、未知状態は Markov で補完 |
| **LightGBM** | 勾配ブースティング（11 特徴量） | ML ベース、非線形パターンを捕捉 |

### アンサンブル（inverse-Brier 加重）

[baseball-mlops](https://github.com/yasumorishima/baseball-mlops) の 5 モデルアンサンブルと同じ設計思想：

```
weight_i = 1 / brier_score_i
ensemble_pred = Σ(w_i × pred_i) / Σ(w_i)
```

さらに **Isotonic Regression** でキャリブレーション補正し、ECE（Expected Calibration Error）を削減。

### 検証パイプライン

```
BigQuery (play_states)
    ↓ export_from_bq.py（秒単位で完了）
data/play_states_{year}.csv
    ↓
┌─────────────────────────────────────────────────┐
│  v1 Normal    → Brier Score                     │
│  v2 Empirical → Brier Score                     │
│  LightGBM     → Brier Score                     │
│       ↓                                          │
│  Ensemble (inverse-Brier weighted)               │
│       ↓                                          │
│  Isotonic Regression Calibration                 │
│       ↓                                          │
│  Leave-one-year-out CV (2015–2024)              │
└─────────────────────────────────────────────────┘
    ↓
results/ (JSON + calibrator.pkl)
    ↓
W&B + Discord notification
```

### メトリクス

| Metric | Description |
|--------|-------------|
| Brier Score | 確率予測の代表的評価指標。mean((predicted - actual)²)、低いほど良い |
| Brier Skill Score | 常に 50% と予測するベースラインとの比較。正なら改善 |
| ECE | 予測値と実際の勝率のズレ。0 に近いほど校正が正確 |
| Log Loss | 確信度の高い誤予測を重く罰するスコアリングルール |

### v1 Optuna 最適化

Optuna（TPE sampler, 500 trial）で Brier Score **+3.85%** 改善（0.1651 → 0.1587）。

| Parameter | Description | Value |
|-----------|-------------|-------|
| `variance_factor` | 得点分布の分散係数 | 3.66 |
| `scoring_factor` | 9 回裏同点時のサヨナラ確率 | 0.87 |
| `behind_lambda_mult` | 9 回裏ビハインド時 Poisson λ倍率 | 0.45 |
| `top9_lambda_mult` | 9 回表ホームリード時 Poisson λ倍率 | 0.67 |
| `extras_win_prob` | 延長戦ホーム勝率 | 0.41 |

### ワークフロー

```bash
# 3エンジン比較 + アンサンブル + 年次CV（BQから自動データ取得）
gh workflow run "Build WP v2" \
  --repo yasumorishima/mlb-win-probability \
  -f memo="ensemble + calibration + CV" \
  -f step=wp_v2_full

# v1 単体の Optuna 最適化
gh workflow run "Validate WP Model" \
  --repo yasumorishima/mlb-win-probability \
  -f memo="2024 full season" \
  -f season=2024 -f optimize=true -f n_trials=500
```

### Cloud Run API

| 項目 | 値 |
|---|---|
| URL | デプロイ済み（認証付き） |
| Swagger UI | ローカル起動: `http://localhost:8001/docs` |
| Artifact Registry | `us-central1-docker.pkg.dev/data-platform-490901/apis/mlb-win-probability-api` |
| メモリ | 256Mi |

## Roadmap

### Phase 1: 基盤構築 ✅
- [x] WP エンジン v1（Markov Chain + Normal 近似 + Optuna 5 パラメータ最適化）
- [x] FastAPI + Streamlit ダッシュボード（バイリンガル、ライブフィード、What-If）
- [x] BigQuery データ基盤（367K+ play states、BQ エクスポートで秒単位データ取得）
- [x] Cloud Run API デプロイ（認証付き、Artifact Registry）
- [x] Grafana ダッシュボード（BQ 接続、公開）

### Phase 2: 精度追い込み 🔄（現在）
- [x] v2 エンジン構築（10 年分実データ WP テーブル + Markov Chain フォールバック）
- [x] LightGBM エンジン構築（11 特徴量、時系列 holdout 分割）
- [x] 3 エンジン比較パイプライン（GitHub Actions、BQ → CSV → 自動比較）
- [x] アンサンブル実装（inverse-Brier 加重、baseball-mlops 同設計）
- [x] Isotonic Regression キャリブレーション補正
- [x] Leave-one-year-out CV（2015–2024、年次安定性検証）
- [ ] **比較結果確認 → 最良構成を本番 `calculate_wp()` に反映**
- [ ] **BQML モデル — BigQuery SQL だけで WP モデルを構築（4 番目のエンジン候補）**

### Phase 3: AI Commentary 🔄（現在）
- [x] Gemini 2.5 Flash 解説生成（`/wp/commentary` エンドポイント）
- [x] プロンプトバージョニング（`PROMPT_REGISTRY`）+ 品質自動評価（100pt）
- [x] W&B 実験追跡（レイテンシ・トークン数・品質スコア）
- [ ] **Gemini API キー設定 + Streamlit Cloud 実動作確認**
- [ ] プロンプト v3 改善（v2 の品質スコア分析結果ベース）

### Phase 4: 統合デプロイ
- [ ] 本番エンジン切り替え（アンサンブル or 最良エンジン）
- [ ] Cloud Run 再デプロイ（アンサンブル + AI Commentary + BQML 統合）
- [ ] W&B Dashboard 構築（品質スコア・Brier Score の時系列可視化）

## License

MIT
