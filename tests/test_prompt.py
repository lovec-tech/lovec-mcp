import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import server  # noqa: E402


def test_missing_summary_tells_you_how_to_make_one(tmp_path):
    out = server.injection_scan_report(str(tmp_path / "nope.json"))
    assert "lovec-scan" in out
    assert "No scan summary" in out


def test_unreadable_summary_does_not_pretend_to_have_data(tmp_path):
    p = tmp_path / "summary.json"
    p.write_text("{not json", encoding="utf-8")
    out = server.injection_scan_report(str(p))
    assert "Could not read" in out
    assert "How to report" not in out


def test_valid_summary_is_inlined_with_the_rules(tmp_path):
    p = tmp_path / "summary.json"
    payload = {"schema": "lovec-scan/1", "flags": {"docs_flagged": 0}}
    p.write_text(json.dumps(payload), encoding="utf-8")
    out = server.injection_scan_report(str(p))
    assert "lovec-scan/1" in out
    assert "results.jsonl" in out


def test_rules_forbid_the_claims_that_would_make_the_report_wrong(tmp_path):
    p = tmp_path / "summary.json"
    p.write_text(json.dumps({"flags": {}}), encoding="utf-8")
    out = server.injection_scan_report(str(p))
    lowered = out.lower()
    # the three failure modes this prompt exists to prevent
    assert "never call the corpus clean" in lowered
    assert "never compute precision, recall" in lowered
    assert "untrusted text" in lowered


def test_rules_cover_plain_language_and_the_business_reader(tmp_path):
    """A report nobody can read, or that routes to nobody, does not get acted on."""
    p = tmp_path / "summary.json"
    p.write_text(json.dumps({"flags": {}}), encoding="utf-8")
    lowered = server.injection_scan_report(str(p)).lower()
    assert "active voice" in lowered
    assert "not just x, but y" in lowered
    assert "routing" in lowered
    assert "previous scan" in lowered
    assert "reproducibility" in lowered


async def test_prompt_is_exposed_over_the_protocol():
    from mcp.client import Client

    async with Client(server.mcp) as client:
        prompts = await client.list_prompts()
        names = [p.name for p in prompts.prompts]
        assert "injection_scan_report" in names


async def test_tool_surface_is_unchanged_by_adding_a_prompt():
    from mcp.client import Client

    async with Client(server.mcp) as client:
        tools = await client.list_tools()
        assert [t.name for t in tools.tools] == ["check_prompt_injection"]
