"""CUAD parser.

CUAD's primary distribution is `CUAD_v1.json` — a SQuAD-style file where:
  - `data[i].title`   = contract filename (e.g. "BlackRockFundsIII_...pdf")
  - `data[i].paragraphs[j].context` = a contract chunk
  - `data[i].paragraphs[j].qas[k]`  = one of 41 clause-category questions
        - `qas[k].question` includes the category name
        - `qas[k].answers`  = list of {text, answer_start} spans

We don't need the QA framing for ingestion. We just walk every contract,
collect its answer spans grouped by category, and emit ParsedClause rows.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path

import structlog
from pydantic import BaseModel

log = structlog.get_logger()


# CUAD encodes the category as a long sentence in the question. We extract
# the canonical short tag via a regex on the trailing parenthetical.
# Example: "Highlight the parts (if any) of this contract related to
#   \"Document Name\" that should be reviewed by a lawyer..."
_CATEGORY_RE = re.compile(r'related to "([^"]+)"', re.IGNORECASE)


def _normalize_category(question: str) -> str:
    """Pull the canonical category tag out of CUAD's verbose question text."""
    m = _CATEGORY_RE.search(question)
    raw = m.group(1) if m else question[:60]
    return re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")


def _contract_id(title: str) -> str:
    """Stable hash so re-ingesting the same file produces the same ID."""
    return hashlib.sha1(title.encode("utf-8")).hexdigest()[:16]


def _clause_id(contract_id: str, category: str, text: str) -> str:
    digest = hashlib.sha1(f"{contract_id}|{category}|{text}".encode("utf-8")).hexdigest()
    return f"{contract_id}-{digest[:10]}"


class ParsedClause(BaseModel):
    clause_id: str
    contract_id: str
    category: str
    text: str
    answer_start: int | None = None


class ParsedContract(BaseModel):
    contract_id: str
    filename: str
    clauses: list[ParsedClause]


def parse_cuad(cuad_json_path: Path) -> Iterator[ParsedContract]:
    """Stream parsed contracts. Don't load the whole result list into memory;
    the dispatcher pipes them straight to workers."""
    with cuad_json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    for entry in raw["data"]:
        title = entry["title"]
        cid = _contract_id(title)
        clauses: list[ParsedClause] = []

        for paragraph in entry["paragraphs"]:
            for qa in paragraph["qas"]:
                category = _normalize_category(qa["question"])
                for ans in qa.get("answers", []):
                    text = (ans.get("text") or "").strip()
                    if not text:
                        continue
                    clauses.append(
                        ParsedClause(
                            clause_id=_clause_id(cid, category, text),
                            contract_id=cid,
                            category=category,
                            text=text,
                            answer_start=ans.get("answer_start"),
                        )
                    )

        # Deduplicate identical spans (CUAD has some).
        seen: set[str] = set()
        unique: list[ParsedClause] = []
        for c in clauses:
            if c.clause_id in seen:
                continue
            seen.add(c.clause_id)
            unique.append(c)

        log.info("parser.contract", contract_id=cid, filename=title, n_clauses=len(unique))
        yield ParsedContract(contract_id=cid, filename=title, clauses=unique)
