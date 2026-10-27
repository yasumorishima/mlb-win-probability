"""
Amendment 1 (PREREG.md): the test calls as live Messages API calls instead of
Batches. Same requests as `run_claude.py submit` (its `params`, items and payloads,
imported unchanged), same raw record format, so `analyze.py` reads the output as is.

  python run_live.py --work W --arm B|C            # every non-excluded test item once (resumes)
  python run_live.py --work W --arm B|C --retry    # once: the items whose call errored

Writes W/raw_test_<arm>.jsonl (custom_id "<item id>|<arm>"). Resumes: an item that
already has a record is not called again. Errors:
- a stop error (authentication, permission, bad request, not found, payment 402) or
  STOP_AFTER consecutive errors of any kind stops the whole run and records nothing for
  those calls, so a systemic failure cannot turn items into misses; fix and resume;
- any other error (after the SDK's own 2 retries) is recorded as {"custom_id", "error",
  "message"}; `--retry` calls those items once more and appends (analyze.py keeps the
  last record per item). `--retry` needs a record for every item and calls each errored
  item at most once (an item with a retry record is never called again).
A lock file stops two processes on the same arm. The key is read from ANTHROPIC_API_KEY
only, set for the one command.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_claude import append, load, params, record  # noqa: E402

WORKERS = 4
STOP_AFTER = 5


def is_stop_error(e: Exception) -> bool:
    import anthropic
    if isinstance(e, (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                      anthropic.BadRequestError, anthropic.NotFoundError)):
        return True
    return getattr(e, "status_code", None) == 402


def run(inputs: Path, work: Path, arm: str, retry: bool) -> int:
    import anthropic
    client = anthropic.Anthropic(max_retries=2, timeout=300.0)
    out = work / f"raw_test_{arm}.jsonl"
    last: dict[str, dict] = {}
    n_rec: dict[str, int] = {}
    if out.exists():
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            k = r["custom_id"].rsplit("|", 1)[0]
            last[k] = r
            n_rec[k] = n_rec.get(k, 0) + 1
    items = [it for it in load(inputs, "test") if not it.get("excluded")]
    if retry:
        missing = [it["id"] for it in items if it["id"] not in last]
        if missing:
            print(f"main pass not complete ({len(missing)} items without a record); no retry")
            return 1
        # once per item: only items whose single record is an error (a retry record makes it 2)
        todo = [it for it in items if last[it["id"]].get("error") and n_rec[it["id"]] == 1]
    else:
        todo = [it for it in items if it["id"] not in last]
    print(arm, "items", len(items), "to call", len(todo), flush=True)

    lock = threading.Lock()
    stop = threading.Event()
    state = {"consecutive": 0, "reason": None}

    def one(it: dict) -> None:
        if stop.is_set():
            return
        cid = f"{it['id']}|{arm}"
        try:
            rec = record(client.messages.create(**params(it[f"payload_{arm}"])), cid)
            err = None
        except Exception as e:
            err = e
            rec = {"custom_id": cid, "error": type(e).__name__, "message": str(e)[:300]}
        with lock:
            if err is None:
                state["consecutive"] = 0
            else:
                state["consecutive"] += 1
                if is_stop_error(err) or state["consecutive"] >= STOP_AFTER:
                    state["reason"] = f"{type(err).__name__}: {str(err)[:300]}"
                    stop.set()
                    return          # not recorded: the item stays unanswered, resume after the fix
            if stop.is_set() and err is not None:
                return              # errors after a stop stay unrecorded; answers already paid for are kept
            append(out, rec)
            try:
                print(arm, it["id"], rec.get("stop_reason") or rec.get("error"), flush=True)
            except OSError:
                pass

    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(one, todo))
    if stop.is_set():
        print("STOPPED:", state["reason"], flush=True)
        return 2
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, default=Path(__file__).resolve().parent / "inputs")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--arm", choices=["B", "C"], required=True)
    ap.add_argument("--retry", action="store_true")
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    lockfile = a.work / f"run_live_{a.arm}.lock"
    try:
        fd = os.open(lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        sys.exit(f"{lockfile} exists: another run on arm {a.arm} (remove it only if none is running)")
    try:
        code = run(a.inputs, a.work, a.arm, a.retry)
    finally:
        os.close(fd)
        lockfile.unlink()
    sys.exit(code)


if __name__ == "__main__":
    main()
