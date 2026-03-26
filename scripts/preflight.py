"""Preflight checks before Statcast WP model training.

Validates BQ connectivity, all 7 required tables, column names
(including mlb-data-pipeline sanitization), and join key integrity.
Fails fast so 2+ hour training runs don't waste time.

Designed for maximum debuggability: every failure prints exactly what
was expected, what was found, and enough context to fix the issue.

Usage:
  python scripts/preflight.py
  python scripts/preflight.py --verbose

Exit code 0 = all checks passed, 1 = failure.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from difflib import get_close_matches
from pathlib import Path


def _log_elapsed(label: str, start: float, budget_min: int = 360):
    elapsed_min = (time.time() - start) / 60
    print(f"  [{label}] elapsed: {elapsed_min:.1f} min / {budget_min} min budget")
    if elapsed_min > budget_min * 0.8:
        print(f"  WARNING: {label} used {elapsed_min:.0f}/{budget_min} min "
              f"({elapsed_min / budget_min * 100:.0f}%) -- timeout risk!")

PROJECT = "data-platform-490901"
DATASET = "mlb_shared"

# All tables required by train_wp_statcast.py
REQUIRED_TABLES = {
    "statcast_pitches": {
        "min_rows": 5_000_000,
        "required_cols": [
            "game_pk", "pitcher", "batter", "events", "game_year", "game_type",
            "inning", "outs_when_up", "balls", "strikes",
            "home_score", "away_score", "on_1b", "on_2b", "on_3b",
            "release_speed", "launch_speed", "launch_angle",
            "estimated_woba_using_speedangle", "delta_run_exp",
            "home_team", "away_team", "inning_topbot",
            "bat_speed", "swing_length",  # 2024+
        ],
    },
    "park_factors": {
        "min_rows": 100,
        "required_cols": ["season", "team", "pf_5yr"],
    },
    "fg_pitching": {
        "min_rows": 1_000,
        "required_cols": ["player_id", "season", "name"],
        # Sanitized column names (mlb-data-pipeline convention):
        # % → _pct, / → _per_, + → _plus, trailing - → _minus
        "sanitized_cols": [
            "K_pct", "BB_pct", "ERA", "FIP", "xFIP", "WHIP", "BABIP",
            "SwStr_pct", "K_BB_pct", "CSW_pct",
            "ERA_minus", "FIP_minus", "xFIP_minus",
            "Stuff_plus", "Location_plus", "Pitching_plus",
            "K_per_9", "BB_per_9", "K_per_BB", "HR_per_9", "HR_per_FB",
            "GB_pct", "FB_pct", "LD_pct",
            "wFB_per_C", "wSL_per_C", "wCH_per_C",
            "GS", "Start_IP", "Relief_IP",
            "WAR",
        ],
    },
    "fg_batting": {
        "min_rows": 1_000,
        "required_cols": ["player_id", "season", "name"],
        "sanitized_cols": [
            "wOBA", "xwOBA", "wRC_plus", "OPS", "ISO", "BABIP",
            "K_pct", "BB_pct",
            "O_Swing_pct", "Z_Swing_pct", "O_Contact_pct", "Z_Contact_pct",
            "Zone_pct", "SwStr_pct", "Contact_pct",
            "GB_pct", "FB_pct", "LD_pct", "HR_per_FB",
            "HardHit_pct", "Pull_pct",
            "wFB_per_C", "wSL_per_C", "wCH_per_C",
            "Spd", "BsR", "WAR", "Off", "Def",
        ],
    },
    "sprint_speed": {
        "min_rows": 200,
        "required_cols": ["player_id", "season", "sprint_speed"],
    },
    "catcher": {
        "min_rows": 50,
        "required_cols": ["player_id", "season"],
    },
    "oaa_team": {
        "min_rows": 50,
        "required_cols": ["season"],
    },
}

errors: list[str] = []
warnings: list[str] = []


def _get_bq_client():
    from google.cloud import bigquery

    sa_key = os.environ.get("GCP_SA_KEY")
    if sa_key and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        key_path = Path("/tmp/gcp-sa-key.json")
        key_path.write_text(sa_key)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path)

    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not cred_path:
        local_key = Path(r"C:\Users\fw_ya\.claude\gcp-sa-key.json")
        if local_key.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(local_key)

    cred_used = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "(default)")
    print(f"  Credentials: {cred_used}")
    return bigquery.Client(project=PROJECT)


def _find_similar(name: str, candidates: set[str], n: int = 5) -> list[str]:
    """Find similar column names using fuzzy matching."""
    substr = [c for c in sorted(candidates) if name.lower() in c.lower() or c.lower() in name.lower()]
    fuzzy = get_close_matches(name, sorted(candidates), n=n, cutoff=0.4)
    seen = set()
    result = []
    for c in substr + fuzzy:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result[:n]


def check_bq_connection():
    """Check BQ connection and list all tables in dataset."""
    print("=" * 60)
    print("1. BQ Connection")
    print("=" * 60)
    try:
        client = _get_bq_client()
        print(f"  Project: {PROJECT}")
        print(f"  Dataset: {DATASET}")

        tables = list(client.list_tables(f"{PROJECT}.{DATASET}"))
        table_names = {t.table_id for t in tables}
        print(f"  Tables found ({len(tables)}):")
        for t in sorted(tables, key=lambda x: x.table_id):
            print(f"    - {t.table_id}")

        return client, table_names
    except Exception as e:
        errors.append(
            f"BQ connection failed.\n"
            f"  Error type: {type(e).__name__}\n"
            f"  Error:      {e}\n"
            f"  GOOGLE_APPLICATION_CREDENTIALS={os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'NOT SET')}\n"
            f"  Traceback:\n{traceback.format_exc()}"
        )
        return None, set()


def check_table_existence(client, existing_tables):
    """Check all required tables exist with sufficient rows."""
    print("\n" + "=" * 60)
    print("2. Table Existence & Row Counts")
    print("=" * 60)

    required_names = set(REQUIRED_TABLES.keys())
    missing_tables = required_names - existing_tables
    if missing_tables:
        errors.append(
            f"Missing tables: {sorted(missing_tables)}. "
            f"Available tables: {sorted(existing_tables)}. "
            f"Run mlb-data-pipeline weekly_refresh to create missing tables."
        )

    print(f"\n  {'Table':<22} {'Rows':>12} {'Cols':>6} {'Size':>10}  Status")
    print(f"  {'-'*22} {'-'*12} {'-'*6} {'-'*10}  {'-'*6}")

    for table_name, spec in REQUIRED_TABLES.items():
        if table_name not in existing_tables:
            print(f"  {table_name:<22} {'NOT FOUND':>12} {'':>6} {'':>10}  [FAIL]")
            continue
        try:
            t = client.get_table(f"{PROJECT}.{DATASET}.{table_name}")
            size_mb = t.num_bytes / 1024 / 1024
            status = "OK" if t.num_rows >= spec["min_rows"] else "LOW"
            print(f"  {table_name:<22} {t.num_rows:>12,} {len(t.schema):>6} "
                  f"{size_mb:>8.1f}MB  [{status}]")
            if t.num_rows < spec["min_rows"]:
                errors.append(
                    f"{table_name}: {t.num_rows:,} rows (expected {spec['min_rows']:,}+). "
                    f"Table may need refresh."
                )
        except Exception as e:
            errors.append(f"Failed to get table info for {table_name}: {e}")
            print(f"  {table_name:<22} {'ERROR':>12}")


def check_columns(client, existing_tables):
    """Check required and sanitized columns exist."""
    print("\n" + "=" * 60)
    print("3. Column Validation")
    print("=" * 60)
    for table_name, spec in REQUIRED_TABLES.items():
        if table_name not in existing_tables:
            continue

        try:
            t = client.get_table(f"{PROJECT}.{DATASET}.{table_name}")
        except Exception:
            continue

        schema_cols = {f.name for f in t.schema}
        schema_types = {f.name: f.field_type for f in t.schema}

        # Required columns
        req = spec.get("required_cols", [])
        missing_req = [c for c in req if c not in schema_cols]
        if missing_req:
            print(f"\n  FAIL: {table_name} — missing {len(missing_req)} required columns:")
            for mc in missing_req:
                similar = _find_similar(mc, schema_cols)
                print(f"    Expected: '{mc}'")
                if similar:
                    print(f"    Similar:  {similar}")
                    print(f"    (Possible rename or sanitization mismatch)")
                else:
                    print(f"    No similar columns — column may not exist in source data")
            errors.append(
                f"{table_name}: missing required columns {missing_req}. "
                f"Schema has {len(schema_cols)} columns: {sorted(schema_cols)}"
            )
        else:
            print(f"  OK: {table_name} — {len(req)} required columns present")

        # Sanitized columns
        san = spec.get("sanitized_cols", [])
        if san:
            missing_san = [c for c in san if c not in schema_cols]
            if missing_san:
                print(f"\n  FAIL: {table_name} — missing {len(missing_san)} sanitized columns:")
                print(f"  (These should follow mlb-data-pipeline naming: "
                      f"%→_pct, /→_per_, +→_plus, trailing -→_minus)")
                for mc in missing_san:
                    similar = _find_similar(mc, schema_cols)
                    print(f"    Expected: '{mc}'")
                    if similar:
                        print(f"    Similar:  {similar}")
                    else:
                        print(f"    No similar columns found")
                errors.append(
                    f"{table_name}: missing sanitized columns {missing_san}. "
                    f"This likely means mlb-data-pipeline sanitize_columns() rules "
                    f"don't match what train_wp_statcast.py expects. "
                    f"Full schema: {sorted(schema_cols)}"
                )
            else:
                print(f"  OK: {table_name} — {len(san)} sanitized columns present")


def check_statcast_year_coverage(client, existing_tables):
    """Check statcast_pitches has all years 2015-2024."""
    if "statcast_pitches" not in existing_tables:
        return

    print("\n" + "=" * 60)
    print("4. Statcast Year Coverage")
    print("=" * 60)
    q = f"""
        SELECT CAST(game_year AS INT64) AS yr, COUNT(*) AS n,
               COUNTIF(events IS NOT NULL) AS ab_outcomes,
               COUNT(DISTINCT game_pk) AS games,
               COUNT(DISTINCT pitcher) AS pitchers,
               COUNT(DISTINCT batter) AS batters
        FROM `{PROJECT}.{DATASET}.statcast_pitches`
        WHERE game_type = 'R'
        GROUP BY yr ORDER BY yr
    """
    try:
        rows = list(client.query(q).result())
        if not rows:
            errors.append("Statcast year coverage query returned 0 rows")
            return

        print(f"\n  {'Year':<6} {'Pitches':>12} {'ABs':>10} {'Games':>7} {'P':>5} {'B':>5}")
        print(f"  {'-'*6} {'-'*12} {'-'*10} {'-'*7} {'-'*5} {'-'*5}")
        years_found = set()
        for row in rows:
            years_found.add(row.yr)
            print(f"  {row.yr:<6} {row.n:>12,} {row.ab_outcomes:>10,} "
                  f"{row.games:>7,} {row.pitchers:>5,} {row.batters:>5,}")
        print(f"  {'TOTAL':<6} {sum(r.n for r in rows):>12,}")

        missing_years = set(range(2015, 2025)) - years_found
        if missing_years:
            errors.append(
                f"Missing years in statcast_pitches: {sorted(missing_years)}. "
                f"Years found: {sorted(years_found)}. "
                f"Run mlb-data-pipeline reload_statcast_bq workflow."
            )
    except Exception as e:
        errors.append(f"Year coverage query failed.\n  SQL: {q}\n  Error: {e}")


def check_join_keys(client, existing_tables):
    """Check join keys work across tables with diagnostic data."""
    print("\n" + "=" * 60)
    print("5. Join Key Integrity")
    print("=" * 60)

    # FG pitching → statcast pitcher join
    if "fg_pitching" in existing_tables and "statcast_pitches" in existing_tables:
        q = f"""
            WITH fg AS (
                SELECT DISTINCT player_id, season FROM `{PROJECT}.{DATASET}.fg_pitching`
                WHERE season >= 2015
            ),
            sc AS (
                SELECT DISTINCT pitcher AS player_id, CAST(game_year AS INT64) AS season
                FROM `{PROJECT}.{DATASET}.statcast_pitches`
                WHERE game_type = 'R'
            )
            SELECT
                (SELECT COUNT(*) FROM fg) AS fg_count,
                (SELECT COUNT(*) FROM sc) AS sc_count,
                (SELECT COUNT(*) FROM fg INNER JOIN sc USING(player_id, season)) AS matched,
                -- Sample unmatched FG entries for debugging
                ARRAY(
                    SELECT AS STRUCT f.player_id, f.season
                    FROM fg f LEFT JOIN sc s USING(player_id, season)
                    WHERE s.player_id IS NULL
                    LIMIT 5
                ) AS unmatched_fg,
                -- Sample unmatched SC entries
                ARRAY(
                    SELECT AS STRUCT s.player_id, s.season
                    FROM sc s LEFT JOIN fg f USING(player_id, season)
                    WHERE f.player_id IS NULL
                    LIMIT 5
                ) AS unmatched_sc
        """
        try:
            row = next(client.query(q).result())
            match_pct = row.matched / row.fg_count * 100 if row.fg_count else 0
            print(f"  fg_pitching ⟷ statcast (pitcher):")
            print(f"    FG entries: {row.fg_count:,}, SC entries: {row.sc_count:,}, "
                  f"Matched: {row.matched:,} ({match_pct:.0f}%)")
            if match_pct < 50:
                unmatched_fg = [(r.player_id, r.season) for r in row.unmatched_fg]
                unmatched_sc = [(r.player_id, r.season) for r in row.unmatched_sc]
                print(f"    Sample unmatched FG: {unmatched_fg}")
                print(f"    Sample unmatched SC: {unmatched_sc}")
                print(f"    (Check if player_id types match: FG may use MLBAM, SC uses pitcher column)")
                errors.append(
                    f"FG pitching join rate: {match_pct:.0f}%. "
                    f"Unmatched FG sample: {unmatched_fg}. "
                    f"Unmatched SC sample: {unmatched_sc}."
                )
            else:
                print(f"    OK")
        except Exception as e:
            warnings.append(f"Pitcher join check failed: {e}")

    # FG batting → statcast batter join
    if "fg_batting" in existing_tables and "statcast_pitches" in existing_tables:
        q = f"""
            WITH fg AS (
                SELECT DISTINCT player_id, season FROM `{PROJECT}.{DATASET}.fg_batting`
                WHERE season >= 2015
            ),
            sc AS (
                SELECT DISTINCT batter AS player_id, CAST(game_year AS INT64) AS season
                FROM `{PROJECT}.{DATASET}.statcast_pitches`
                WHERE game_type = 'R'
            )
            SELECT
                (SELECT COUNT(*) FROM fg) AS fg_count,
                (SELECT COUNT(*) FROM sc) AS sc_count,
                (SELECT COUNT(*) FROM fg INNER JOIN sc USING(player_id, season)) AS matched,
                ARRAY(
                    SELECT AS STRUCT f.player_id, f.season
                    FROM fg f LEFT JOIN sc s USING(player_id, season)
                    WHERE s.player_id IS NULL
                    LIMIT 5
                ) AS unmatched_fg
        """
        try:
            row = next(client.query(q).result())
            match_pct = row.matched / row.fg_count * 100 if row.fg_count else 0
            print(f"\n  fg_batting ⟷ statcast (batter):")
            print(f"    FG entries: {row.fg_count:,}, SC entries: {row.sc_count:,}, "
                  f"Matched: {row.matched:,} ({match_pct:.0f}%)")
            if match_pct < 50:
                unmatched_fg = [(r.player_id, r.season) for r in row.unmatched_fg]
                print(f"    Sample unmatched FG: {unmatched_fg}")
                errors.append(f"FG batting join rate: {match_pct:.0f}%. Unmatched: {unmatched_fg}")
            else:
                print(f"    OK")
        except Exception as e:
            warnings.append(f"Batter join check failed: {e}")

    # Park factors → statcast team join
    if "park_factors" in existing_tables and "statcast_pitches" in existing_tables:
        q = f"""
            WITH pf AS (
                SELECT DISTINCT team, CAST(season AS INT64) AS season
                FROM `{PROJECT}.{DATASET}.park_factors`
            ),
            sc AS (
                SELECT DISTINCT home_team AS team, CAST(game_year AS INT64) AS season
                FROM `{PROJECT}.{DATASET}.statcast_pitches`
                WHERE game_type = 'R'
            )
            SELECT
                (SELECT COUNT(*) FROM sc) AS sc_teams,
                (SELECT COUNT(*) FROM sc INNER JOIN pf USING(team, season)) AS matched,
                ARRAY(
                    SELECT AS STRUCT s.team, s.season
                    FROM sc s LEFT JOIN pf p USING(team, season)
                    WHERE p.team IS NULL
                    ORDER BY s.season DESC
                    LIMIT 5
                ) AS unmatched_teams
        """
        try:
            row = next(client.query(q).result())
            match_pct = row.matched / row.sc_teams * 100 if row.sc_teams else 0
            print(f"\n  park_factors ⟷ statcast (team):")
            print(f"    SC team-seasons: {row.sc_teams:,}, Matched: {row.matched:,} "
                  f"({match_pct:.0f}%)")
            if match_pct < 70:
                unmatched = [(r.team, r.season) for r in row.unmatched_teams]
                print(f"    Unmatched team-seasons: {unmatched}")
                print(f"    (Check team abbreviation format: SC uses 'NYY', PF may use 'NYA')")
                warnings.append(f"Park factors join rate: {match_pct:.0f}%. Unmatched: {unmatched}")
            else:
                print(f"    OK")
        except Exception as e:
            warnings.append(f"Park factors join check failed: {e}")

    # Sprint speed check
    if "sprint_speed" in existing_tables and "statcast_pitches" in existing_tables:
        q = f"""
            WITH ss AS (
                SELECT DISTINCT player_id, season FROM `{PROJECT}.{DATASET}.sprint_speed`
            ),
            sc AS (
                SELECT DISTINCT batter AS player_id, CAST(game_year AS INT64) AS season
                FROM `{PROJECT}.{DATASET}.statcast_pitches`
                WHERE game_type = 'R'
            )
            SELECT
                (SELECT COUNT(*) FROM ss) AS ss_count,
                (SELECT COUNT(*) FROM ss INNER JOIN sc USING(player_id, season)) AS matched
        """
        try:
            row = next(client.query(q).result())
            match_pct = row.matched / row.ss_count * 100 if row.ss_count else 0
            print(f"\n  sprint_speed ⟷ statcast (batter):")
            print(f"    SS entries: {row.ss_count:,}, Matched: {row.matched:,} ({match_pct:.0f}%)")
        except Exception as e:
            warnings.append(f"Sprint speed join check failed: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Preflight checks")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    t0 = time.time()

    print("=" * 60)
    print("mlb-win-probability Preflight Check")
    print("=" * 60)

    client, existing_tables = check_bq_connection()
    if not client:
        print(f"\n{'='*60}")
        print(f"PREFLIGHT FAILED — BQ connection error")
        print(f"{'='*60}")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)

    check_table_existence(client, existing_tables)
    _log_elapsed("table_existence", t0)
    check_columns(client, existing_tables)
    _log_elapsed("column_validation", t0)
    check_statcast_year_coverage(client, existing_tables)
    _log_elapsed("year_coverage", t0)
    check_join_keys(client, existing_tables)
    _log_elapsed("join_keys", t0)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if warnings:
        print(f"\n  Warnings ({len(warnings)}):")
        for i, w in enumerate(warnings, 1):
            print(f"    [{i}] {w}")
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for i, e in enumerate(errors, 1):
            print(f"    [{i}] {e}")
        print(f"\n{'='*60}")
        print(f"PREFLIGHT FAILED — {len(errors)} error(s), {len(warnings)} warning(s)")
        print(f"Fix the errors above before running training.")
        print(f"{'='*60}")
        sys.exit(1)
    else:
        print(f"\n  All checks passed ({len(warnings)} warnings)")
        print(f"\n{'='*60}")
        print(f"PREFLIGHT OK — ready to train")
        print(f"{'='*60}")
        sys.exit(0)


if __name__ == "__main__":
    main()
