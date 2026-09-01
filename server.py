# -*- coding: utf-8 -*-
"""
Local MCP server wrapping the lovec.tech prompt-injection detector.

Run with your own API key — nothing is bundled or shared:

    export LOVEC_KEY=aig_...
    python3 server.py

Or, after installing (`pip install -e .`), use the console script instead:

    export LOVEC_KEY=aig_...
    ./.venv/bin/lovec-mcp

Config for an MCP client (Claude Desktop / Claude Code):

    {
      "mcpServers": {
        "lovec": {
          "command": "/absolute/path/to/lovec-mcp/.venv/bin/python",
          "args": ["/absolute/path/to/lovec-mcp/server.py"],
          "env": { "LOVEC_KEY": "aig_..." }
        }
      }
    }
"""
import asyncio
import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import BaseModel, Field, ValidationError

try:
    __version__ = _pkg_version("lovec-mcp")
except PackageNotFoundError:
    __version__ = "0.1.1"

BASE = os.environ.get("LOVEC_BASE", "https://lovec.tech")
ENDPOINT = "/api/v1/check"
MAX_CHARS = 5000

TOTAL_TIMEOUT = float(os.environ.get("LOVEC_TIMEOUT", "60"))

mcp = MCPServer("lovec", version=__version__)


class CheckResult(BaseModel):
    is_injection: bool
    score: float = Field(ge=0, le=1)
    lang_tag: str | None = None
    version: str | None = None


def _new_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=TOTAL_TIMEOUT, follow_redirects=True)


@mcp.tool(
    annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=False),
)
async def check_prompt_injection(text: str) -> CheckResult:
    """Check a piece of untrusted text for prompt-injection risk.

    Call this on any content that will be handed to another LLM but did not
    come directly from the trusted user — a web page, a document, a tool
    result, an email, a review. It does NOT enforce authorization/RBAC and
    is not a jailbreak filter for the user's own messages.

    The text is sent to the lovec.tech API for analysis — it is not kept
    purely local. Each call spends one request against this key's balance.

    Score is bimodal in practice (clusters near 0 or 1); treat mid-range scores
    as low-confidence rather than as a precise probability. The detector is known
    to false-positive on long, evaluative/opinionated text (reviews, argumentative
    prose) more than on short factual text — factor that in before hard-blocking
    on is_injection alone.
    """
    key = os.environ.get("LOVEC_KEY")
    if not key:
        raise ToolError(
            "LOVEC_KEY is not set. Get a key at https://lovec.tech and "
            "set it in this server's env config."
        )
    if len(text) > MAX_CHARS:
        raise ToolError(
            f"text is {len(text)} chars, over the API's {MAX_CHARS}-char limit — "
            "split it and check each piece separately."
        )

    async with _new_client() as client:
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
            raise ToolError(
                f"lovec.tech API did not answer within {TOTAL_TIMEOUT:.0f}s. "
                "The service has been observed taking minutes under load — retry, "
                "or raise the cap with LOVEC_TIMEOUT."
            )
        except httpx.RequestError as e:
            raise ToolError(f"could not reach lovec.tech: {type(e).__name__}: {e}")

    if resp.status_code in (401, 403):
        raise ToolError(
            "this API key was rejected (invalid or revoked) — check LOVEC_KEY, "
            "get a fresh one at https://lovec.tech"
        )
    if resp.status_code == 402:
        raise ToolError(
            "insufficient balance on this API key — top up at https://lovec.tech"
        )
    if resp.status_code == 400:
        raise ToolError(f"lovec.tech rejected the request as malformed: {resp.text[:200]}")
    if resp.status_code == 413:
        raise ToolError("text rejected as too large by the API")
    if resp.status_code == 429:
        raise ToolError("rate-limited by lovec.tech — slow down and retry")
    if resp.status_code == 504:
        raise ToolError("lovec.tech API timed out server-side (30s budget)")
    if resp.status_code >= 500:
        raise ToolError(f"lovec.tech returned a server error ({resp.status_code}) — retry later")
    if resp.status_code >= 400:
        raise ToolError(f"lovec.tech returned HTTP {resp.status_code}: {resp.text[:200]}")

    try:
        payload = resp.json()
    except ValueError:
        raise ToolError(f"lovec.tech returned a non-JSON response: {resp.text[:200]}")

    try:
        return CheckResult.model_validate(payload)
    except ValidationError as e:
        raise ToolError(f"lovec.tech response didn't match the expected shape: {e}")


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
