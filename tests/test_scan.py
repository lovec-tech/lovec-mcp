import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scan  # noqa: E402


def _args(**over):
    base = dict(threshold=None, top=25, workers=4, max_chars=4500)
    base.update(over)
    return argparse.Namespace(**base)


# ---------- chunking ----------
def test_short_text_is_one_chunk():
    assert scan.chunk_text("hello", 100) == ["hello"]


def test_empty_text_yields_no_chunks():
    assert scan.chunk_text("   \n  ", 100) == []


def test_long_text_splits_under_limit():
    text = "\n".join(f"paragraph {i} " + "word " * 40 for i in range(50))
    chunks = scan.chunk_text(text, 500)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)


def test_single_paragraph_longer_than_limit_is_hard_split():
    chunks = scan.chunk_text("x" * 1000, 300)
    assert all(len(c) <= 300 for c in chunks)
    assert "".join(chunks) == "x" * 1000


def test_build_units_labels_chunks_with_doc_id():
    docs = [{"id": "d1", "path": "d1.md", "text": "x" * 1000}]
    units = scan.build_units(docs, 300)
    assert {u["doc_id"] for u in units} == {"d1"}
    assert [u["chunk_index"] for u in units] == list(range(len(units)))
    assert all(u["n_chunks"] == len(units) for u in units)


# ---------- discovery ----------
def test_discover_filters_by_extension_and_reports_skips(tmp_path):
    (tmp_path / "a.md").write_text("kept", encoding="utf-8")
    (tmp_path / "b.pdf").write_text("dropped", encoding="utf-8")
    docs, skipped = scan.discover_docs([str(tmp_path)], [".md"], None)
    assert [d["text"] for d in docs] == ["kept"]
    assert any(".pdf" in s for s in skipped)


def test_discover_reports_missing_path(tmp_path):
    docs, skipped = scan.discover_docs([str(tmp_path / "nope")], [".md"], None)
    assert docs == []
    assert any("no such file" in s for s in skipped)


def test_discover_from_jsonl(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_text(
        json.dumps({"id": "x1", "text": "hello"}) + "\n" + json.dumps({"id": "x2", "text": ""}) + "\n",
        encoding="utf-8",
    )
    docs, _ = scan.discover_docs([], [".md"], str(p))
    assert [d["id"] for d in docs] == ["x1"]  # empty text dropped


# ---------- resume bookkeeping ----------
def test_load_done_skips_errored_rows_so_they_retry(tmp_path):
    p = tmp_path / "results.jsonl"
    p.write_text(
        json.dumps({"unit_id": "u1", "error": None, "score": 0.1}) + "\n"
        + json.dumps({"unit_id": "u2", "error": "http:504"}) + "\n",
        encoding="utf-8",
    )
    done = scan.load_done(p)
    assert set(done) == {"u1"}


def test_load_done_tolerates_torn_last_line(tmp_path):
    p = tmp_path / "results.jsonl"
    p.write_text(json.dumps({"unit_id": "u1", "error": None}) + "\n{\"unit_id\": \"u2\"", encoding="utf-8")
    assert set(scan.load_done(p)) == {"u1"}


def test_collapse_prefers_success_over_earlier_error():
    rows = [
        {"unit_id": "u1", "error": "http:504", "score": None},
        {"unit_id": "u1", "error": None, "score": 0.9},
    ]
    assert scan.collapse(rows)[0]["score"] == 0.9


def test_collapse_does_not_let_a_later_error_overwrite_success():
    rows = [
        {"unit_id": "u1", "error": None, "score": 0.9},
        {"unit_id": "u1", "error": "timeout", "score": None},
    ]
    assert scan.collapse(rows)[0]["error"] is None


# ---------- statistics ----------
def test_wilson_at_zero_events_has_nonzero_upper_bound():
    lo, hi = scan.wilson(0, 100)
    assert lo == 0.0
    assert 0.0 < hi < 0.05


def test_wilson_empty_denominator_is_fully_uncertain():
    assert scan.wilson(0, 0) == (0.0, 1.0)


# ---------- summary contract ----------
def _row(doc, idx, n_chunks, score=0.0, is_inj=False, error=None):
    return {
        "unit_id": f"{doc}#c{idx}", "doc_id": doc, "path": doc, "chunk_index": idx,
        "n_chunks": n_chunks, "n_chars": 10, "is_injection": is_inj, "score": score,
        "lang_tag": "ru", "detector_version": "1.0", "excerpt": "text",
        "latency_ms": 100, "error": error,
    }


def test_partially_covered_doc_is_excluded_from_denominator():
    docs = [{"id": "d1"}, {"id": "d2"}]
    rows = [
        _row("d1", 0, 1),
        _row("d2", 0, 2),
        _row("d2", 1, 2, error="http:504"),
    ]
    s = scan.summarize(docs, rows, _args(), {}, [], 0)
    assert s["coverage"]["docs_fully_covered"] == 1
    assert s["coverage"]["docs_partial"] == 1
    assert s["flags"]["denominator"] == 1
    assert any("only partly checked" in n for n in s["notes"])


def test_flagged_but_partial_doc_counts_as_a_flag_yet_not_in_the_denominator():
    """A document can be worth reviewing and still not be a valid observation.

    d2 has one flagged chunk and one that errored, so it is a real flag but was
    never fully checked. It must show up in docs_flagged (someone should look at
    it) while staying out of the rate denominator.
    """
    docs = [{"id": "d1"}, {"id": "d2"}]
    rows = [
        _row("d1", 0, 1),
        _row("d2", 0, 2, score=1.0, is_inj=True),
        _row("d2", 1, 2, error="http:504"),
    ]
    s = scan.summarize(docs, rows, _args(), {}, [], 0)
    assert s["flags"]["docs_flagged"] == 1
    assert s["flags"]["docs_flagged_within_full_coverage"] == 0
    assert s["flags"]["denominator"] == 1
    assert s["coverage"]["docs_partial"] == 1


def test_zero_flags_produces_an_upper_bound_note():
    docs = [{"id": f"d{i}"} for i in range(50)]
    rows = [_row(f"d{i}", 0, 1) for i in range(50)]
    s = scan.summarize(docs, rows, _args(), {}, [], 0)
    assert s["flags"]["docs_flagged"] == 0
    assert s["flags"]["doc_flag_rate_ci95"][1] > 0
    assert any("upper bound" in n for n in s["notes"])


def test_threshold_mode_overrides_is_injection():
    docs = [{"id": "d1"}]
    rows = [_row("d1", 0, 1, score=0.8, is_inj=False)]
    default = scan.summarize(docs, rows, _args(), {}, [], 0)
    thresholded = scan.summarize(docs, rows, _args(threshold=0.7), {}, [], 0)
    assert default["flags"]["docs_flagged"] == 0
    assert thresholded["flags"]["docs_flagged"] == 1
    assert thresholded["run"]["decision"]["mode"] == "score_threshold"


def test_unreached_documents_are_called_out():
    docs = [{"id": "d1"}, {"id": "d2"}]
    rows = [_row("d1", 0, 1)]
    s = scan.summarize(docs, rows, _args(), {}, [], 0)
    assert any("never attempted" in n for n in s["notes"])


def test_stopped_early_is_surfaced_in_run_and_notes():
    docs = [{"id": "d1"}]
    rows = [_row("d1", 0, 1)]
    state = {"stop_reason": {"reason": "insufficient_balance", "detail": "ran out"}}
    s = scan.summarize(docs, rows, _args(), state, [], 0)
    assert s["run"]["stopped_early"]["reason"] == "insufficient_balance"
    assert any("stopped early" in n.lower() for n in s["notes"])


# ---------- network behaviour ----------
async def _run_unit(handler, monkeypatch, text="hello"):
    monkeypatch.setattr(scan, "BACKOFF_SECONDS", [0, 0])
    monkeypatch.setattr(scan, "BACKOFF_429_SECONDS", [0, 0])
    # Retry-After can still push the wait back up past the zeroed schedule, so
    # neutralise the sleep itself rather than just the constants.
    real_sleep = asyncio.sleep
    monkeypatch.setattr(scan.asyncio, "sleep", lambda _d: real_sleep(0))
    unit = {"unit_id": "u1", "doc_id": "d1", "path": "d1", "chunk_index": 0,
            "n_chunks": 1, "text": text}
    state = {"stop": False, "stop_reason": None}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        row = await scan.scan_unit(unit, client, "aig_test", asyncio.Semaphore(2), state)
    return row, state


async def test_scan_unit_happy_path(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={"is_injection": True, "score": 1.0, "lang_tag": "ru"})

    row, _ = await _run_unit(handler, monkeypatch)
    assert row["is_injection"] is True and row["error"] is None


async def test_scan_unit_retries_transient_504_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(504, json={"error": "timeout"})
        return httpx.Response(200, json={"is_injection": False, "score": 0.0})

    row, _ = await _run_unit(handler, monkeypatch)
    assert calls["n"] == 2
    assert row["error"] is None


async def test_scan_unit_gives_up_and_records_the_error(monkeypatch):
    def handler(request):
        return httpx.Response(504, json={"error": "timeout"})

    row, _ = await _run_unit(handler, monkeypatch)
    assert row["error"] == "http:504"
    assert row["is_injection"] is None  # never a clean verdict


async def test_insufficient_balance_stops_the_run(monkeypatch):
    def handler(request):
        return httpx.Response(402, json={"error": "insufficient_balance"})

    row, state = await _run_unit(handler, monkeypatch)
    assert row is None
    assert state["stop"] is True
    assert state["stop_reason"]["reason"] == "insufficient_balance"


async def test_rejected_key_stops_the_run(monkeypatch):
    def handler(request):
        return httpx.Response(401, json={"error": "unauthorized"})

    _, state = await _run_unit(handler, monkeypatch)
    assert state["stop_reason"]["reason"] == "auth_rejected"


async def test_rate_limit_retries_then_records_a_clean_429(monkeypatch):
    """A 429 is retried on its own slower schedule and recorded without the
    Retry-After suffix, so errors_by_kind does not fragment into http:429:30."""
    def handler(request):
        return httpx.Response(429, headers={"Retry-After": "30"}, json={"error": "rate_limited"})

    row, _ = await _run_unit(handler, monkeypatch)
    assert row["error"] == "http:429"


async def test_rate_limit_recovers_on_retry(monkeypatch):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, json={"error": "rate_limited"})
        return httpx.Response(200, json={"is_injection": False, "score": 0.1})

    row, _ = await _run_unit(handler, monkeypatch)
    assert calls["n"] == 3
    assert row["error"] is None


async def test_retry_after_extends_but_never_shortens_the_wait(monkeypatch):
    """Retry-After is honoured when it is longer than our own backoff."""
    slept = []
    real_sleep = asyncio.sleep  # bind before patching, or the stub calls itself

    async def fake_sleep(d):
        slept.append(d)
        await real_sleep(0)

    monkeypatch.setattr(scan.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(scan, "BACKOFF_429_SECONDS", [5, 5])

    def handler(request):
        return httpx.Response(429, headers={"Retry-After": "40"}, json={})

    unit = {"unit_id": "u1", "doc_id": "d1", "path": "d1", "chunk_index": 0,
            "n_chunks": 1, "text": "hi"}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await scan.scan_unit(unit, client, "k", asyncio.Semaphore(1),
                             {"stop": False, "stop_reason": None})
    assert slept == [40, 40]


async def test_bad_shape_is_an_error_not_a_verdict(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={"is_injection": False, "score": 7})

    row, _ = await _run_unit(handler, monkeypatch)
    assert row["error"] == "unexpected_shape"


@pytest.mark.parametrize("max_chars", [4500, scan.MAX_CHARS])
def test_chunks_never_exceed_the_api_limit(max_chars):
    docs = [{"id": "d", "path": "d", "text": "слово " * 20000}]
    units = scan.build_units(docs, max_chars)
    assert all(len(u["text"]) <= scan.MAX_CHARS for u in units)
