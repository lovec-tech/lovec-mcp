# -*- coding: utf-8 -*-
"""Shared endpoint contract for the lovec.tech detector.

Both the MCP server (`server.py`) and the batch scanner (`scan.py`) import
from here, so the base URL, path and size limit have exactly one definition.
This product's domain has already moved more than once, and a scanner that
kept its own copy of the URL is how you ship a half-migrated client.
"""
import os

from pydantic import BaseModel, Field

BASE = os.environ.get("LOVEC_BASE", "https://lovec.tech")
ENDPOINT = "/api/v1/check"
MAX_CHARS = 5000

TOTAL_TIMEOUT = float(os.environ.get("LOVEC_TIMEOUT", "60"))


class CheckResult(BaseModel):
    """Response shape of POST /api/v1/check.

    `lang_tag` and `version` showed up in live responses without being
    announced, so they are modelled as optional rather than dropped.
    """

    is_injection: bool
    score: float = Field(ge=0, le=1)
    lang_tag: str | None = None
    version: str | None = None
