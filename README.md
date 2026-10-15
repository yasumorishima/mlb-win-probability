# MLB Win Probability Engine

イニング・アウト・走者・点差（とカウント）から、**ホームの勝利確率・局面の重要度（Leverage Index）・作戦ごとの勝率の増減**を返すエンジンと、その数字を読んで次の 1 球の前に作戦を提案する **Claude ベンチコーチ**。

**[Live Demo](https://mlb-wp-engine.streamlit.app/)**（English / 日本語）

公開して動いているのは Streamlit デモだけです。FastAPI は手元で起動して使います。デモには API キーを置いていないので、デモからは Claude を呼びません。

---

## エンジン

MLB Stats API の play-by-play から作った打席ごとの遷移（2021〜2025 年・915,418 打席）に当てはめた、リーグ平均のマルコフモデルです（`wp_engine.py`、表は `engine_tables.json`）。

1. 塁・アウトの状態ごとに「次の状態とその打席の得点」の分布を、表（ビジター攻撃）と裏（ホーム攻撃）で別に数える
2. そこから、半イニングの残りで何点入るかの分布を動的計画法で出す
3. 半イニングを再帰でたどって試合の勝率を出す（サヨナラ、延長の二塁走者〈レギュラーシーズン 2020 年〜。ポストシーズンは無し〉）
4. カウントの層：打席の結果の種類はカウントで変わり、種類ごとの遷移は変わらないとして、カウント別の勝率を出す（0-0 ではカウント無しの勝率と一致）

作戦は得点期待値でなく**勝率**で測ります。

| 作戦 | 出す数字 |
|---|---|
| 盗塁（二塁・三塁） | 損益分岐の成功率 |
| 送りバント | その状態で実際に起きたバントの結果の平均をとった勝率と、その増減 |
| 敬遠 | 打者を一塁に置いた後の勝率と、その増減（守備側から見る） |

### 2026 年（学習に使っていない年）での確認

| | Brier | log loss |
|---|---|---|
| このエンジン | 0.15845 | 0.47138 |
| 旧エンジン（手で合わせた正規近似） | 0.15970 | 0.47733 |
| ホーム勝率の定数 | 0.24976 | 0.69266 |

184,327 打席の結果で採点しました。7 回以降で 2 点差以内に限ると、旧エンジンとの Brier の差は −0.0045（試合単位のブートストラップで 95% 区間 −0.0073〜−0.0017）です。勝率帯ごとの数字は `reports/engine_check_2026.json` にあります。

**分かっている弱点**：勝率 0.3〜0.5 の帯で、どの季節も約 3 ポイント高めに出ます。全員をリーグ平均として扱うので、同じ試合の中でチームの強さが続くこと（序盤に点を取るチームは終盤も取る）を織り込んでいないためです。

## データ

`scripts/build_states.py` が MLB Stats API から打席・走者・投球の表を作ります（GitHub Actions の `build-states.yml` を手動で実行し、Release `states-v1` に出力）。各試合で次を確かめ、どれかが合わない試合は捨てます。

- スコアを打席ごとに積み上げると最終スコアに一致する
- アウト数が 0→3 で途切れずにつながる
- 延長の二塁走者がルールどおり置かれている
- 盗塁と盗塁死の数がボックススコアと一致する

2021〜2026 年の全試合で、捨てた試合は 0 です。表を当てはめ直すときは `scripts/fit_engine.py` を使います。

## Claude ベンチコーチ

`strategist.py` が、局面とエンジンの数字（作戦ごとの損益分岐や勝率の増減）を Claude（`claude-opus-5-5`）に渡し、推奨する作戦・理由・使った数字を構造化出力で受け取ります。**理由に出てくる数字は、渡した数字と 1 つずつ突き合わせ**、合わないものは画面と API の応答に出します。

- キー：環境変数 `ANTHROPIC_API_KEY`（または Streamlit secrets）。無ければ何も送りません
- 1 回の問い合わせは 1 リクエストです（SDK の再試行なし・60 秒で打ち切り。2026-10-15 の実測は 1 回 4〜7 秒）
- アプリでの上限は、サーバー全体で 1 日 `COACH_DAILY_LIMIT` 回（既定 30）と、1 セッション 10 回です
- API の `/wp/advice` は `ADVICE_TOKEN` も必要です。リクエストに同じ値の `X-Advice-Token` ヘッダーが無ければ断ります（回数の上限は無いので、トークンは人に渡さないでください）

### 盗塁の成功を、統計モデルより当てられるか

Claude 自身の予測が役に立つかを、事前登録して確かめました（`research/steal_claude/`）。2026 年 7〜9 月の盗塁 1,843 件で比べた結果、**Claude 単独でも、モデルの予測値を渡しても、統計モデル（log loss 0.4636）を上回りませんでした**（Claude 単独 0.4668、モデルの値つき 0.4624。事前登録した 2 つの仮説はどちらも支持されず）。詳細と Claude の全回答は [`research/steal_claude/RESULTS.md`](research/steal_claude/RESULTS.md) にあります。ベンチコーチは、エンジンの数字を言葉にして作戦をまとめる役として使っています。

## 使い方

### Streamlit

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

- 局面を入力するか、プリセット（9 回裏の満塁など 6 つ）を選ぶと、勝率・LI・作戦・次の打席の結果ごとの勝率が出ます
- **Live Games**：その日の試合を MLB Stats API から読み、進行中の局面をそのまま入力に反映できます（30 秒ごとの自動更新あり。終わった試合は反映しません）
- **Replay Mode**：過去の試合を 1 プレーずつ（自動再生もできます）たどり、各プレーの前後の勝率と WPA を見られます

### FastAPI

```bash
pip install -r requirements.txt
uvicorn api:app --reload
# → http://localhost:8000/docs
```

Docker なら `docker compose up --build` で `http://localhost:8001/docs` です（ホストの `ANTHROPIC_API_KEY` と `ADVICE_TOKEN` を渡します）。

| エンドポイント | 内容 |
|---|---|
| `GET /wp` | 勝率・打撃側の勝率・LI・残りの得点期待値・次の打席の結果ごとの勝率・作戦 |
| `GET /wp/play` | 1 プレーの前後の状態から WPA（`after_outs=3` は攻守交代） |
| `GET /wp/advice` | 上と同じ分析に Claude の推奨を付けたもの（キーとトークンが必要） |
| `GET /re24` | 24 の塁・アウト状態の得点期待値と、1 点以上入る確率 |
| `GET /wp/scenario` | プリセットの局面 |
| `GET /games/today` | その日の試合一覧（`date=YYYY-MM-DD`） |
| `GET /wp/live/{gamePk}` | 試合の現在の局面と勝率 |

`/wp`・`/wp/play`・`/wp/advice` は `extra_runner=false` でポストシーズン（延長の二塁走者なし）として計算します。カウントは `balls` と `strikes` を両方渡したときだけ使います。

```bash
# 9 回裏 2 死満塁 同点
curl "http://localhost:8000/wp?inning=9&top_bottom=bottom&outs=2&runner1=1&runner2=1&runner3=1&score_diff=0"

# Claude の推奨（8 回裏 無死一塁 1 点ビハインド）
curl -H "X-Advice-Token: $ADVICE_TOKEN" \
  "http://localhost:8000/wp/advice?inning=8&top_bottom=bottom&outs=0&runner1=1&score_diff=-1&lang=JA"
```

### Leverage Index の目安

| LI | 表示 |
|---|---|
| 0.5 未満 | Low |
| 0.5〜1.5 | Medium |
| 1.5〜3.0 | High |
| 3.0 以上 | Very High |

## 旧版からの変更（2026-10）

- 以前の学習データ（Hugging Face `mlb_wp/play_states`）は、アウト数が全行 0、走者なしが 77% で使えないと分かったため、データを作り直し、エンジンも置き換えました。旧エンジン（v1/v2・LightGBM・ベイズ）と、それに基づく検証値は削除しました
- 解説は Gemini から Claude に替えました。`/wp/commentary` と `/commentary/info` は無くなり、`/wp/advice` に替わっています
- `runs_per_game`・`batter_ops`・`pitcher_era` による補正は無くなりました（エンジンは全員をリーグ平均として扱います）

## テスト

```bash
pip install -r requirements.txt pytest httpx
pytest tests/
```

## License

MIT
