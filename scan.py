#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch prompt-injection scan over a local corpus.

The MCP tool checks one string per call. Pointing an agent at a real corpus
that way does not work: every verdict lands in the agent's context, and the
API has been observed taking anywhere from 2 to 174 seconds per request. So
the loop lives here, in ordinary code, and the agent reads the summary.

    export LOVEC_KEY=aig_...
    lovec-scan ./docs --out lovec-scan-out

Writes two files into --out:

    results.jsonl   one line per chunk, appended as it goes (resumable)
    summary.json    aggregates, written at the end

Then ask your agent for the `injection_scan_report` prompt from the lovec MCP
server, which reads summary.json and writes the report.

Every chunk costs one request against your key's balance. Use --dry-run first
to see how many that will be.
"""
import argparse
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path

import httpx
from pydantic import ValidationError

from lovec_api import BASE, ENDPOINT, MAX_CHARS, TOTAL_TIMEOUT, CheckResult

DEFAULT_EXTS = [".txt", ".md", ".markdown", ".rst"]
RETRY_STATUSES = {429, 500, 502, 503, 504}
BACKOFF_SECONDS = [3, 9]  # transient server errors and timeouts
# Rate limiting needs a different shape of patience: backing off 3s from a 429
# just spends another request on the same refusal. Measured on a real scan, 4
# workers drew 429s on a third of the corpus with the short schedule.
BACKOFF_429_SECONDS = [15, 45]
EXCERPT_CHARS = 300


class FatalStop(Exception):
    """Something that makes the rest of the run pointless (no key, no balance)."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


# ---------- input ----------
def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split on paragraph boundaries, staying under max_chars."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks, buf = [], ""
    for para in text.split("\n"):
        if len(buf) + len(para) + 1 <= max_chars:
            buf = (buf + "\n" + para) if buf else para
        else:
            if buf:
                chunks.append(buf)
            while len(para) > max_chars:
                chunks.append(para[:max_chars])
                para = para[max_chars:]
            buf = para
    if buf:
        chunks.append(buf)
    return chunks


def discover_docs(paths: list[str], exts: list[str], jsonl: str | None) -> tuple[list[dict], list[str]]:
    """Return (documents, skipped_notes). A document is {id, path, text}."""
    docs, skipped = [], []
    if jsonl:
        with open(jsonl, encoding="utf-8") as fh:
            for n, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = (row.get("text") or "").strip()
                if not text:
                    continue
                docs.append({
                    "id": str(row.get("id") or f"{Path(jsonl).name}:{n}"),
                    "path": row.get("path") or jsonl,
                    "text": text,
                })
        return docs, skipped

    ext_set = {e.lower() for e in exts}
    other_ext: dict[str, int] = {}
    for p in paths:
        root = Path(p)
        if root.is_file():
            candidates = [root]
        elif root.is_dir():
            candidates = sorted(f for f in root.rglob("*") if f.is_file())
        else:
            skipped.append(f"{p}: no such file or directory")
            continue
        for f in candidates:
            if f.suffix.lower() not in ext_set:
                if root.is_dir():
                    other_ext[f.suffix.lower() or "(no extension)"] = (
                        other_ext.get(f.suffix.lower() or "(no extension)", 0) + 1
                    )
                else:
                    skipped.append(f"{f}: extension not in --ext")
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace").strip()
            except OSError as e:
                skipped.append(f"{f}: {e}")
                continue
            if not text:
                continue
            docs.append({"id": str(f), "path": str(f), "text": text})

    for ext, count in sorted(other_ext.items(), key=lambda kv: -kv[1]):
        skipped.append(f"{count} file(s) with extension {ext} — not in --ext, not read")
    return docs, skipped


def build_units(docs: list[dict], max_chars: int) -> list[dict]:
    """One unit per chunk. Chunks carry their doc id so results roll back up."""
    units = []
    for d in docs:
        chunks = chunk_text(d["text"], max_chars)
        for i, ch in enumerate(chunks):
            units.append({
                "unit_id": f"{d['id']}#c{i}",
                "doc_id": d["id"],
                "path": d["path"],
                "chunk_index": i,
                "n_chunks": len(chunks),
                "text": ch,
            })
    return units


# ---------- API ----------
async def check_once(client: httpx.AsyncClient, text: str, key: str) -> tuple[dict | None, str | None]:
    """One request. Returns (payload, error_kind). Raises FatalStop for run-enders."""
    try:
        resp = await asyncio.wait_for(
            client.post(
                BASE + ENDPOINT,
                json={"text": text},
                headers={"Authorization": f"Bearer {key}"},
            ),
            timeout=TOTAL_TIMEOUT,
        )
    except (asyncio.TimeoutError, httpx.TimeoutException):
        return None, "timeout"
    except httpx.RequestError as e:
        return None, f"network:{type(e).__name__}"

    if resp.status_code in (401, 403):
        raise FatalStop("auth_rejected", "API key rejected (invalid or revoked)")
    if resp.status_code == 402:
        raise FatalStop("insufficient_balance", "API key ran out of balance mid-scan")
    if resp.status_code == 429:
        # Server-supplied delay wins over our guess when it sends one.
        retry_after = resp.headers.get("Retry-After", "")
        suffix = f":{retry_after}" if retry_after.strip().isdigit() else ""
        return None, f"http:429{suffix}"
    if resp.status_code >= 400:
        return None, f"http:{resp.status_code}"

    try:
        return resp.json(), None
    except ValueError:
        return None, "invalid_json"


async def scan_unit(unit: dict, client: httpx.AsyncClient, key: str, sem: asyncio.Semaphore,
                    state: dict) -> dict | None:
    async with sem:
        if state["stop"]:
            return None
        started = time.time()
        error = None
        for attempt in range(len(BACKOFF_SECONDS) + 1):
            try:
                payload, error = await check_once(client, unit["text"], key)
            except FatalStop as e:
                state["stop"] = True
                state["stop_reason"] = {"reason": e.reason, "detail": e.detail}
                return None
            if error is None:
                try:
                    result = CheckResult.model_validate(payload)
                except ValidationError:
                    error = "unexpected_shape"
                    break
                return {
                    "unit_id": unit["unit_id"],
                    "doc_id": unit["doc_id"],
                    "path": unit["path"],
                    "chunk_index": unit["chunk_index"],
                    "n_chunks": unit["n_chunks"],
                    "n_chars": len(unit["text"]),
                    "is_injection": result.is_injection,
                    "score": result.score,
                    "lang_tag": result.lang_tag,
                    "detector_version": result.version,
                    "excerpt": " ".join(unit["text"][:EXCERPT_CHARS].split()),
                    "latency_ms": round((time.time() - started) * 1000),
                    "error": None,
                }
            parts = error.split(":")
            status = int(parts[1]) if error.startswith("http:") and parts[1].isdigit() else None
            retryable = (
                error.startswith("timeout")
                or error.startswith("network")
                or (status is not None and status in RETRY_STATUSES)
            )
            schedule = BACKOFF_429_SECONDS if status == 429 else BACKOFF_SECONDS
            if not retryable or attempt >= len(schedule):
                break
            delay = schedule[attempt]
            if status == 429 and len(parts) > 2 and parts[2].isdigit():
                delay = max(delay, int(parts[2]))
            await asyncio.sleep(delay)

        if error and error.startswith("http:429"):
            error = "http:429"  # drop any Retry-After suffix so counts don't fragment

        return {
            "unit_id": unit["unit_id"],
            "doc_id": unit["doc_id"],
            "path": unit["path"],
            "chunk_index": unit["chunk_index"],
            "n_chunks": unit["n_chunks"],
            "n_chars": len(unit["text"]),
            "is_injection": None,
            "score": None,
            "latency_ms": round((time.time() - started) * 1000),
            "error": error,
        }


async def run_scan(units: list[dict], key: str, workers: int, results_path: Path,
                   state: dict) -> list[dict]:
    sem = asyncio.Semaphore(workers)
    written: list[dict] = []
    lock = asyncio.Lock()
    done_count = {"n": 0}

    async with httpx.AsyncClient(timeout=TOTAL_TIMEOUT, follow_redirects=True) as client:
        with open(results_path, "a", encoding="utf-8") as fh:

            async def one(unit):
                row = await scan_unit(unit, client, key, sem, state)
                if row is None:
                    return
                async with lock:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                    written.append(row)
                    done_count["n"] += 1
                    if done_count["n"] % 10 == 0 or done_count["n"] == len(units):
                        print(f"  {done_count['n']}/{len(units)} chunks", file=sys.stderr)

            await asyncio.gather(*(one(u) for u in units))
    return written


# ---------- summary ----------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% CI for a proportion. Correct at k=0, which is the case that matters."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (centre - spread) / denom), min(1.0, (centre + spread) / denom))


def summarize(docs: list[dict], rows: list[dict], args, state: dict, skipped: list[str],
              started_at: float) -> dict:
    by_doc: dict[str, list[dict]] = {}
    for r in rows:
        by_doc.setdefault(r["doc_id"], []).append(r)

    def flagged(r):
        if r["error"]:
            return False
        if args.threshold is not None:
            return r["score"] is not None and r["score"] >= args.threshold
        return bool(r["is_injection"])

    docs_full, docs_partial, docs_flagged = [], [], []
    for doc_id, rs in by_doc.items():
        expected = rs[0]["n_chunks"]
        ok = [r for r in rs if not r["error"]]
        if any(flagged(r) for r in rs):
            docs_flagged.append(doc_id)
        if len(ok) == expected:
            docs_full.append(doc_id)
        else:
            docs_partial.append(doc_id)

    errors_by_kind: dict[str, int] = {}
    for r in rows:
        if r["error"]:
            errors_by_kind[r["error"]] = errors_by_kind.get(r["error"], 0) + 1

    # Flag rate denominator = fully covered documents only. A document whose
    # chunks errored was not checked and cannot count as clean.
    n_den = len(docs_full)
    n_flagged_full = len([d for d in docs_flagged if d in set(docs_full)])
    lo, hi = wilson(n_flagged_full, n_den)

    bins = [0.0] * 10
    for r in rows:
        if r["score"] is not None:
            bins[min(9, int(r["score"] * 10))] += 1

    top = sorted(
        [r for r in rows if flagged(r)],
        key=lambda r: (r["score"] if r["score"] is not None else 0),
        reverse=True,
    )[: args.top]

    latencies = sorted(r["latency_ms"] for r in rows if not r["error"])
    scanned_doc_ids = set(by_doc)

    notes = []
    if state.get("stop_reason"):
        notes.append(
            f"Scan stopped early ({state['stop_reason']['reason']}): "
            f"{state['stop_reason']['detail']}. Documents not reached were not checked."
        )
    if docs_partial:
        notes.append(
            f"{len(docs_partial)} document(s) had at least one chunk fail — those "
            "documents are only partly checked and are excluded from the flag-rate "
            "denominator."
        )
    unreached = len(docs) - len(scanned_doc_ids)
    if unreached > 0:
        notes.append(f"{unreached} document(s) were never attempted.")
    if n_flagged_full == 0 and n_den > 0:
        notes.append(
            f"Zero flags is an upper bound, not a clean bill: consistent with a true "
            f"rate up to {hi * 100:.2f}% of documents."
        )

    return {
        "schema": "lovec-scan/1",
        "run": {
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started_at)),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "base": BASE,
            "workers": args.workers,
            "max_chars": args.max_chars,
            "decision": (
                {"mode": "score_threshold", "threshold": args.threshold}
                if args.threshold is not None
                else {"mode": "is_injection", "threshold": None}
            ),
            "detector_versions": sorted(
                {r["detector_version"] for r in rows if r.get("detector_version")}
            ),
            "stopped_early": state.get("stop_reason"),
        },
        "coverage": {
            "docs_total": len(docs),
            "docs_attempted": len(scanned_doc_ids),
            "docs_fully_covered": len(docs_full),
            "docs_partial": len(docs_partial),
            "chunks_total": len(rows),
            "chunks_ok": len([r for r in rows if not r["error"]]),
            "chunks_errored": len([r for r in rows if r["error"]]),
            "errors_by_kind": errors_by_kind,
            "inputs_skipped": skipped,
        },
        "flags": {
            "docs_flagged": len(docs_flagged),
            "docs_flagged_within_full_coverage": n_flagged_full,
            "chunks_flagged": len([r for r in rows if flagged(r)]),
            "doc_flag_rate": (n_flagged_full / n_den) if n_den else None,
            "doc_flag_rate_ci95": [lo, hi],
            "denominator": n_den,
            "denominator_note": "fully covered documents only",
        },
        "score_histogram": {
            "bin_edges": [round(i / 10, 1) for i in range(11)],
            "counts": [int(b) for b in bins],
        },
        "latency_ms": {
            "p50": latencies[len(latencies) // 2] if latencies else None,
            "p95": latencies[int(len(latencies) * 0.95)] if latencies else None,
            "max": latencies[-1] if latencies else None,
        },
        "top_flags": [
            {
                "doc_id": r["doc_id"],
                "path": r["path"],
                "chunk_index": r["chunk_index"],
                "score": r["score"],
                "lang_tag": r.get("lang_tag"),
                "excerpt": r["excerpt"],
            }
            for r in top
        ],
        "notes": notes,
    }


# ---------- cli ----------
def load_done(results_path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if not results_path.exists():
        return done
    with open(results_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue  # torn last line from a killed run
            if row.get("error") is None and "unit_id" in row:
                done[row["unit_id"]] = row
    return done


def collapse(rows: list[dict]) -> list[dict]:
    """One row per unit_id across all passes of a resumed run.

    A unit that errored on an earlier pass and succeeded on a later one appears
    twice in the file. Success always wins over error, and between two of the
    same kind the later line wins — otherwise a retry that fixed a coverage gap
    would still be reported as a gap.
    """
    best: dict[str, dict] = {}
    for r in rows:
        uid = r.get("unit_id")
        if not uid:
            continue
        prev = best.get(uid)
        if prev is None or prev.get("error") is not None or r.get("error") is None:
            best[uid] = r
    return list(best.values())


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="lovec-scan",
        description="Scan a local corpus for prompt injections via the lovec.tech API.",
    )
    ap.add_argument("paths", nargs="*", help="files or directories to scan")
    ap.add_argument("--jsonl", help="read documents from a JSONL file with {id, text} instead")
    ap.add_argument("--out", default="lovec-scan-out", help="output directory")
    ap.add_argument("--ext", default=",".join(DEFAULT_EXTS),
                    help=f"comma-separated extensions to read (default: {','.join(DEFAULT_EXTS)})")
    ap.add_argument("--workers", type=int, default=2,
                    help="concurrent requests (default 2; 4 drew 429s on a real scan)")
    ap.add_argument("--max-chars", type=int, default=4500,
                    help=f"chunk size, must be <= API limit of {MAX_CHARS}")
    ap.add_argument("--threshold", type=float, default=None,
                    help="flag on score >= THRESHOLD instead of the API's is_injection")
    ap.add_argument("--limit", type=int, default=None, help="scan at most N documents")
    ap.add_argument("--top", type=int, default=25, help="how many flags to inline in summary.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="count documents and chunks, then exit without calling the API")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if not args.paths and not args.jsonl:
        build_parser().error("give at least one path, or --jsonl FILE")
    if args.max_chars > MAX_CHARS:
        build_parser().error(f"--max-chars cannot exceed the API limit of {MAX_CHARS}")
    if args.threshold is not None and not (0 <= args.threshold <= 1):
        build_parser().error("--threshold must be between 0 and 1")

    started_at = time.time()
    docs, skipped = discover_docs(args.paths, [e.strip() for e in args.ext.split(",")], args.jsonl)
    if args.limit:
        docs = docs[: args.limit]
    if not docs:
        print("No readable documents found.", file=sys.stderr)
        for s in skipped:
            print(f"  {s}", file=sys.stderr)
        sys.exit(1)

    units = build_units(docs, args.max_chars)
    print(f"{len(docs)} document(s) -> {len(units)} chunk(s) of <= {args.max_chars} chars",
          file=sys.stderr)
    for s in skipped:
        print(f"  skipped: {s}", file=sys.stderr)

    if args.dry_run:
        print(f"Dry run: this scan would cost {len(units)} request(s). Nothing was sent.",
              file=sys.stderr)
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    done = load_done(results_path)
    todo = [u for u in units if u["unit_id"] not in done]
    if done:
        print(f"Resuming: {len(done)} chunk(s) already done, {len(todo)} to go", file=sys.stderr)

    # Checked here, not earlier: a fully resumed run makes no requests, and
    # should still be able to rebuild summary.json from existing results.
    key = os.environ.get("LOVEC_KEY")
    if todo and not key:
        print("LOVEC_KEY is not set. Get a key at https://lovec.tech", file=sys.stderr)
        sys.exit(2)

    state: dict = {"stop": False, "stop_reason": None}
    if todo:
        asyncio.run(run_scan(todo, key, args.workers, results_path, state))

    all_rows = collapse(_read_all(results_path))
    summary = summarize(docs, all_rows, args, state, skipped, started_at)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    f = summary["flags"]
    c = summary["coverage"]
    # Coverage and flags are printed as separate lines on purpose. Putting them
    # in one sentence reads as a rate, and a document can be flagged *and* only
    # partly checked — it would look like it sat inside the covered set.
    print(
        f"\nDocuments: {c['docs_total']} found, {c['docs_fully_covered']} fully checked, "
        f"{c['docs_partial']} partly checked.",
        file=sys.stderr,
    )
    print(f"Flagged for review: {f['docs_flagged']} document(s).", file=sys.stderr)
    for n in summary["notes"]:
        print(f"  note: {n}", file=sys.stderr)
    print(f"\nWrote {results_path} and {summary_path}", file=sys.stderr)
    if state.get("stop_reason"):
        sys.exit(3)


def _read_all(results_path: Path) -> list[dict]:
    out = []
    if not results_path.exists():
        return out
    with open(results_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


if __name__ == "__main__":
    main()
