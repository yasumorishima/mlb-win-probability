"""
AI Commentary Generator using Gemini API (Free Tier)

Generates natural language game situation commentary from WP engine structured data.
Gemini 2.5 Flash + Cloud Run architecture for real-time game analysis.

MLOps Features (same philosophy as baseball-mlops):
- Prompt versioning with registry
- W&B experiment tracking (prompt version, latency, token usage, quality score)
- Automated quality evaluation (rule-based scoring)
- Session-based caching to avoid redundant API calls
"""

import logging
import os
import time

import google.genai as genai

from win_probability import (
    calculate_wp,
    RE24_MLB,
    MLB_RPG,
)

logger = logging.getLogger(__name__)

# ============================================================
# Prompt Registry — versioned prompts for A/B tracking
# ============================================================

PROMPT_VERSION = "v2"

PROMPT_REGISTRY: dict[str, dict] = {
    "v1": {
        "description": "Basic commentary — WP + LI + tactics only",
        "date": "2026-03-22",
    },
    "v2": {
        "description": "Rich context — WP + LI + RE24 + What-If + tactical reasoning",
        "date": "2026-03-22",
    },
}


# ============================================================
# Quality Evaluation — automated scoring of commentary output
# ============================================================

def evaluate_commentary_quality(
    commentary: str,
    result: dict,
    lang: str = "JA",
) -> dict:
    """
    Rule-based quality scoring of generated commentary.

    Checks whether the commentary properly incorporates the analytical data
    it was given, similar to how baseball-mlops evaluates model predictions
    against actual outcomes.

    Returns:
        dict with score (0-100), breakdown by criterion, and pass/fail.
    """
    checks = {}
    wp_pct = result["win_probability_pct"].replace("%", "")

    # 1. Does it mention win probability (with a number)?
    checks["mentions_wp"] = any(
        kw in commentary for kw in [wp_pct, "勝利確率", "Win Probability", "win probability", "WP"]
    )

    # 2. Does it mention leverage / pressure?
    checks["mentions_li"] = any(
        kw in commentary for kw in [
            "レバレッジ", "プレッシャー", "重要", "Leverage", "leverage", "pressure", "critical",
        ]
    )

    # 3. Does it mention run expectancy / scoring potential?
    checks["mentions_re24"] = any(
        kw in commentary for kw in [
            "期待得点", "得点期待", "Run Expectancy", "run expectancy", "expected runs", "scoring",
        ]
    )

    # 4. Does it reference next-play impact (What-If)?
    checks["mentions_whatif"] = any(
        kw in commentary for kw in [
            "一打", "ヒット", "本塁打", "三振", "single", "home run", "strikeout",
            "跳ね上が", "変わ", "swing", "push",
        ]
    )

    # 5. Appropriate length (not too short, not too long)?
    char_count = len(commentary)
    if lang == "JA":
        checks["appropriate_length"] = 100 <= char_count <= 600
    else:
        checks["appropriate_length"] = 150 <= char_count <= 800

    # 6. Does it mention tactics when tactics exist?
    tactics = result.get("tactics", [])
    recommended = [t for t in tactics if t["recommendation"] == "Recommended"]
    if recommended:
        tactic_keywords = []
        for t in recommended:
            tactic_keywords.append(t.get("tactic_ja", ""))
            tactic_keywords.append(t.get("tactic", ""))
        checks["mentions_tactics"] = any(
            kw in commentary for kw in tactic_keywords if kw
        )
    else:
        checks["mentions_tactics"] = True  # No tactics to mention = pass

    # Score calculation
    weights = {
        "mentions_wp": 25,
        "mentions_li": 20,
        "mentions_re24": 15,
        "mentions_whatif": 20,
        "appropriate_length": 10,
        "mentions_tactics": 10,
    }
    score = sum(weights[k] for k, v in checks.items() if v)

    return {
        "score": score,
        "max_score": 100,
        "checks": checks,
        "pass": score >= 60,
        "char_count": char_count,
    }


# ============================================================
# W&B Logging — experiment tracking for commentary generation
# ============================================================

def _log_to_wandb(
    result: dict,
    commentary: str,
    quality: dict,
    lang: str,
    latency_ms: float,
    token_count: int | None,
    prompt_version: str,
) -> None:
    """
    Log commentary generation metrics to W&B.

    Same pattern as baseball-mlops: each generation is a "run" with
    metrics that can be tracked over time to monitor quality drift.
    """
    try:
        import wandb

        wandb_key = os.environ.get("WANDB_API_KEY")
        if not wandb_key:
            return

        entity = os.environ.get("WANDB_ENTITY", "fw_yasu11-personal")
        run = wandb.init(
            project="mlb-win-probability",
            entity=entity,
            job_type="commentary",
            tags=["gemini", prompt_version, lang],
            config={
                "prompt_version": prompt_version,
                "model": "gemini-2.5-flash",
                "lang": lang,
                "game_state": result["game_state"],
            },
            reinit=True,
        )

        run.log({
            "quality_score": quality["score"],
            "quality_pass": int(quality["pass"]),
            "latency_ms": latency_ms,
            "commentary_length": quality["char_count"],
            "mentions_wp": int(quality["checks"]["mentions_wp"]),
            "mentions_li": int(quality["checks"]["mentions_li"]),
            "mentions_re24": int(quality["checks"]["mentions_re24"]),
            "mentions_whatif": int(quality["checks"]["mentions_whatif"]),
            "mentions_tactics": int(quality["checks"]["mentions_tactics"]),
            "appropriate_length": int(quality["checks"]["appropriate_length"]),
            "inning": result["game_state"]["inning"],
            "leverage_index": result["leverage_index"],
        })

        if token_count is not None:
            run.log({"token_count": token_count})

        # Log commentary as W&B Table for review
        table = wandb.Table(
            columns=["prompt_version", "lang", "game_state", "commentary", "quality_score"],
            data=[[
                prompt_version,
                lang,
                str(result["game_state"]),
                commentary,
                quality["score"],
            ]],
        )
        run.log({"commentary_samples": table})

        run.finish()

    except ImportError:
        logger.debug("wandb not installed, skipping logging")
    except Exception as e:
        logger.warning("W&B logging failed (non-blocking): %s", e)


# ============================================================
# What-If & RE24 helpers
# ============================================================

def _compute_whatif_outcomes(
    inning: int, top_bottom: str, outs: int,
    runners: tuple[int, int, int], score_diff: int,
    rpg: float,
) -> dict[str, dict]:
    """Compute WP after each possible next-play outcome."""
    outcomes = {
        "strikeout": {"runners": runners, "outs": min(outs + 1, 3), "runs": 0},
        "single": {"runners": (1, runners[0], runners[1]), "outs": outs, "runs": runners[2]},
        "double": {"runners": (0, 1, runners[0]), "outs": outs, "runs": runners[1] + runners[2]},
        "home_run": {"runners": (0, 0, 0), "outs": outs, "runs": 1 + sum(runners)},
        "walk": {
            "runners": (
                1,
                1 if runners[0] else runners[1],
                1 if (runners[0] and runners[1]) else runners[2],
            ),
            "outs": outs,
            "runs": 1 if (runners[0] and runners[1] and runners[2]) else 0,
        },
        "ground_out": {"runners": (0, runners[1], runners[2]), "outs": min(outs + 1, 3), "runs": 0},
    }

    base_wp = calculate_wp(inning, top_bottom, outs, runners, score_diff, rpg)
    results = {}
    for name, o in outcomes.items():
        new_outs = o["outs"]
        new_runners = o["runners"]
        runs_scored = o["runs"]

        if new_outs >= 3:
            if top_bottom == "top":
                wp = calculate_wp(inning, "bottom", 0, (0, 0, 0),
                                  score_diff - runs_scored, rpg)
            else:
                wp = calculate_wp(inning + 1, "top", 0, (0, 0, 0),
                                  score_diff + runs_scored, rpg)
        else:
            if top_bottom == "top":
                wp = calculate_wp(inning, top_bottom, new_outs, new_runners,
                                  score_diff - runs_scored, rpg)
            else:
                wp = calculate_wp(inning, top_bottom, new_outs, new_runners,
                                  score_diff + runs_scored, rpg)

        results[name] = {
            "wp": round(wp * 100, 1),
            "wpa": round((wp - base_wp) * 100, 1),
        }
    return results


def _get_current_re24(runners: tuple[int, int, int], outs: int, rpg: float) -> float:
    """Get RE24 expected runs for current base-out state."""
    key = (runners[0], runners[1], runners[2], outs)
    base = RE24_MLB.get(key, 0.0)
    scale = rpg / MLB_RPG
    return round(base * scale, 3)


# ============================================================
# Prompt Builder
# ============================================================

def _build_prompt(result: dict, lang: str = "JA") -> str:
    """Build a rich prompt from full_analysis() output with RE24 + What-If context."""
    gs = result["game_state"]
    inning = gs["inning"]
    top_bottom = gs["top_bottom"]
    half = "裏" if top_bottom == "bottom" else "表"
    outs = gs["outs"]
    runners = gs["runners"]
    score_diff = gs["score_diff"]
    rpg = gs.get("runs_per_game", MLB_RPG)
    wp_pct = result["win_probability_pct"]
    li = result["leverage_index"]
    li_lbl = result["leverage_label"]

    runner_tuple = (runners["1B"], runners["2B"], runners["3B"])

    # RE24
    re24 = _get_current_re24(runner_tuple, outs, rpg)

    # What-If
    whatif = _compute_whatif_outcomes(inning, top_bottom, outs, runner_tuple, score_diff, rpg)

    # Runner description
    runner_desc_parts = []
    if runners["1B"]:
        runner_desc_parts.append("1塁" if lang == "JA" else "1B")
    if runners["2B"]:
        runner_desc_parts.append("2塁" if lang == "JA" else "2B")
    if runners["3B"]:
        runner_desc_parts.append("3塁" if lang == "JA" else "3B")
    runner_desc = ("、".join(runner_desc_parts) + "にランナー") if runner_desc_parts else "ランナーなし"
    if lang == "EN":
        runner_desc = ("Runners on " + ", ".join(runner_desc_parts)) if runner_desc_parts else "Bases empty"

    # Tactics
    tactics = result.get("tactics", [])
    rec_tactics = [t for t in tactics if t["recommendation"] == "Recommended"]
    consider_tactics = [t for t in tactics if t["recommendation"] == "Consider"]

    if lang == "JA":
        if score_diff > 0:
            score_text = f"ホームが{score_diff}点リード"
        elif score_diff < 0:
            score_text = f"アウェイが{abs(score_diff)}点リード"
        else:
            score_text = "同点"

        tactic_lines = []
        for t in rec_tactics:
            tactic_lines.append(f"  - 【推奨】{t.get('tactic_ja', t['tactic'])}（RE24変化: {t['re24_delta']:+.3f}、理由: {t.get('reason', '')}）")
        for t in consider_tactics:
            tactic_lines.append(f"  - 【検討】{t.get('tactic_ja', t['tactic'])}（RE24変化: {t['re24_delta']:+.3f}、理由: {t.get('reason', '')}）")
        tactic_block = "\n".join(tactic_lines) if tactic_lines else "  特になし"

        whatif_lines = []
        labels_ja = {"strikeout": "三振", "single": "シングル", "double": "二塁打",
                     "home_run": "本塁打", "walk": "四球", "ground_out": "ゴロアウト"}
        for key, val in whatif.items():
            whatif_lines.append(f"  - {labels_ja.get(key, key)}: WP {val['wp']}%（{val['wpa']:+.1f}%）")
        whatif_block = "\n".join(whatif_lines)

        return f"""あなたはプロ野球の実況解説AIです。以下の試合状況データを分析し、野球ファンが読んで思わず唸るような解説を生成してください。

## 試合状況
- {inning}回{half} {outs}アウト
- {runner_desc}
- {score_text}

## 数理分析結果
- ホームチーム勝利確率: {wp_pct}
- レバレッジ指数（LI）: {li:.2f}（{li_lbl}）— LI=1.0が平均的場面、2.0以上は高プレッシャー、4.0以上は試合を左右する超重要場面
- 現在のRE24期待得点: {re24}点 — このイニングの残り打席で入ると期待される得点

## 作戦分析（RE24期待値ベース）
{tactic_block}

## 次のプレイ別WP変動予測
{whatif_block}

## 解説ルール
- 4〜6文で、データに基づいた深い解説を生成
- 勝利確率・レバレッジ指数・RE24の数字を自然に織り込む（ただし「RE24」は「期待得点」と言い換える）
- 次のプレイでWPがどう動くかに言及して臨場感を出す（例: 「ここで一打が出れば勝利確率は一気にX%まで跳ね上がります」）
- 作戦の推奨理由をデータで裏付ける
- 「です・ます」調、放送席の解説者のような口調
- 数字の羅列にならないこと。物語として語る"""

    else:
        if score_diff > 0:
            score_text = f"Home leads by {score_diff}"
        elif score_diff < 0:
            score_text = f"Away leads by {abs(score_diff)}"
        else:
            score_text = "Tied"

        tactic_lines = []
        for t in rec_tactics:
            tactic_lines.append(f"  - [Recommended] {t['tactic']} (RE24 delta: {t['re24_delta']:+.3f}, reason: {t.get('reason', '')})")
        for t in consider_tactics:
            tactic_lines.append(f"  - [Consider] {t['tactic']} (RE24 delta: {t['re24_delta']:+.3f}, reason: {t.get('reason', '')})")
        tactic_block = "\n".join(tactic_lines) if tactic_lines else "  None"

        whatif_lines = []
        for key, val in whatif.items():
            whatif_lines.append(f"  - {key.replace('_', ' ').title()}: WP {val['wp']}% ({val['wpa']:+.1f}%)")
        whatif_block = "\n".join(whatif_lines)

        return f"""You are a professional baseball commentary AI. Analyze the following game situation data and produce insightful, engaging commentary.

## Game State
- Inning: {half} of {inning}
- Outs: {outs}
- {runner_desc}
- {score_text}

## Analytics
- Home Win Probability: {wp_pct}
- Leverage Index (LI): {li:.2f} ({li_lbl}) — LI 1.0 = average, 2.0+ = high pressure, 4.0+ = game-defining moment
- Current RE24 (Run Expectancy): {re24} runs expected remaining this inning

## Tactical Analysis (RE24-based)
{tactic_block}

## Next-Play WP Projections
{whatif_block}

## Commentary Rules
- 4-6 sentences, data-driven depth
- Weave in WP, LI, and run expectancy naturally (use "run expectancy" not "RE24")
- Reference how the next play could swing WP (e.g., "A single here would push win probability to X%")
- Back up tactical recommendations with data
- Broadcast-booth tone — authoritative yet engaging
- Tell a story with the numbers, don't just list them"""


# ============================================================
# Cache
# ============================================================

def _cache_key(result: dict, lang: str) -> str:
    """Generate a cache key from game state."""
    gs = result["game_state"]
    return f"{gs['inning']}_{gs['top_bottom']}_{gs['outs']}_{gs['runners']}_{gs['score_diff']}_{lang}"


# ============================================================
# Main entry point
# ============================================================

def generate_commentary(
    result: dict,
    lang: str = "JA",
    api_key: str | None = None,
    cache: dict | None = None,
    track: bool = True,
) -> str | None:
    """
    Generate AI commentary from full_analysis() result using Gemini API.

    Follows the same MLOps pattern as baseball-mlops:
    - Versioned prompts (like versioned models)
    - Quality evaluation (like MAE evaluation)
    - W&B tracking (like training run tracking)
    - Caching (like model caching with APScheduler)

    Args:
        result: Output from full_analysis()
        lang: "JA" or "EN"
        api_key: Gemini API key (falls back to GEMINI_API_KEY env var)
        cache: Optional dict for caching (pass st.session_state for Streamlit)
        track: Whether to log metrics to W&B (default True)

    Returns:
        Commentary text, or None if API key is not configured.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        return None

    # Check cache
    ck = _cache_key(result, lang)
    if cache is not None and ck in cache:
        return cache[ck]

    # Generate with latency tracking
    client = genai.Client(api_key=key)
    prompt = _build_prompt(result, lang)

    t0 = time.monotonic()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    latency_ms = (time.monotonic() - t0) * 1000

    text = response.text

    # Extract token usage if available
    token_count = None
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        token_count = getattr(response.usage_metadata, "total_token_count", None)

    # Quality evaluation
    quality = evaluate_commentary_quality(text, result, lang) if text else None

    if quality and not quality["pass"]:
        logger.warning(
            "Commentary quality below threshold: score=%d/100, checks=%s",
            quality["score"], quality["checks"],
        )

    # W&B tracking (non-blocking, same pattern as baseball-mlops)
    if track and text and quality:
        _log_to_wandb(result, text, quality, lang, latency_ms, token_count, PROMPT_VERSION)

    # Store in cache
    if cache is not None and text:
        cache[ck] = text

    return text


def get_commentary_metadata() -> dict:
    """
    Return current commentary system metadata.
    Useful for /model/info style endpoint (like baseball-mlops Cloud Run).
    """
    return {
        "prompt_version": PROMPT_VERSION,
        "prompt_description": PROMPT_REGISTRY[PROMPT_VERSION]["description"],
        "model": "gemini-2.5-flash",
        "quality_threshold": 60,
        "quality_criteria": [
            "mentions_wp (25pt)",
            "mentions_li (20pt)",
            "mentions_re24 (15pt)",
            "mentions_whatif (20pt)",
            "appropriate_length (10pt)",
            "mentions_tactics (10pt)",
        ],
        "available_versions": list(PROMPT_REGISTRY.keys()),
    }
