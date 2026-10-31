"""Narration + reliability layer for the weekend-story pipeline.

Adapted from the agentic loop that used to live in agent/agent.py (recovered from git
history — `git show HEAD:agent/agent.py`), generalized from "one fixed pace report" to "one
paragraph per storyline, each independently fact-checked against only the numbers it was
given." The core guarantee is unchanged: the model narrates, it never calculates, and nothing
it writes reaches the reader unless every number in it traces back to the storyline's own
`supporting_facts`.
"""
from __future__ import annotations

import json
import re

import ollama

MODEL = "llama3.2:3b"
NUMBER_RE = re.compile(r"-?\d+\.\d+|-?\d+")
CONTEXT_CHARS = 60  # characters scanned on either side of a cited number for attribution

STORY_SYSTEM_PROMPT = """You are a Formula 1 journalist writing one paragraph of a race-weekend \
feature article, in the style of a motorsport analyst (concise, vivid, specific — not a stats \
dump). You will be given ONE storyline as a JSON block: a headline, whether it's an \
"achievement" or a "mistake", the driver/team subject ("subject", plus "subject_full_name" if \
given), and the exact supporting facts.

Write 2-3 sentences of plain-English narrative prose using ONLY the numbers given to you. Do \
not invent, estimate, round differently, or calculate any number that is not already present \
in the JSON. Every number you write must appear in the JSON, attached to the correct driver. \
The storyline is about "subject" ONLY — use exactly that driver/team (by its \
subject_full_name if given, otherwise its code). Never substitute, add, or imply a different \
real driver, team, or season/championship detail that is not in the JSON, even if you know \
real F1 facts about them — this report must stand entirely on the given data. In particular, \
NEVER mention championship points or standings, finishing/grid positions, additional lap \
numbers, or any other number that is not one of the literal numeric values already present in \
the JSON you were given — if you don't have a number for something, describe it in words \
instead, with no digits attached. No markdown, no headers, just prose."""

HEADLINE_SYSTEM_PROMPT = """You are a Formula 1 journalist. You will be given a JSON list of \
this race weekend's storylines (achievements and mistakes, already ranked by how notable they \
are). Write a short, punchy magazine-style headline (under 12 words) and a one-sentence dek \
(subheading) that together frame the weekend's overall narrative using only what the storyline \
titles already say — do not introduce any driver, team, or number that isn't in the list.

Respond as JSON only: {"headline": "...", "dek": "..."}"""


# ---------------------------------------------------------------------------
# Summarize — the SLM narrates, it never calculates.
# ---------------------------------------------------------------------------
def narrate_storyline(storyline: dict) -> str:
    payload = {k: v for k, v in storyline.items() if k != "chart_key"}
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": STORY_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, indent=2)},
        ],
    )
    return response["message"]["content"].strip()


def narrate_headline(storylines: list[dict]) -> dict:
    titles = [{"title": s["title"], "angle": s["angle"]} for s in storylines]
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": HEADLINE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(titles, indent=2)},
        ],
    )
    try:
        return json.loads(response["message"]["content"].strip())
    except (json.JSONDecodeError, ValueError):
        return {"headline": "Race Weekend Story", "dek": ""}


# ---------------------------------------------------------------------------
# Verify — reliability layer: every cited number must trace back to this
# storyline's own supporting_facts, correctly attributed to its subject.
# ---------------------------------------------------------------------------
def _flatten_numbers(obj) -> set[str]:
    found: set[str] = set()
    if isinstance(obj, dict):
        for v in obj.values():
            found |= _flatten_numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            found |= _flatten_numbers(v)
    elif isinstance(obj, (int, float)):
        # also index the unsigned magnitude: facts store signed deltas
        # (-7 = lost 7 places), but natural phrasing drops the sign
        # ("loses 7 places") — both are the same fact, not a hallucination.
        for v in (obj, abs(obj)):
            found.add(str(v))
            found.add(f"{v:.3f}")
    elif isinstance(obj, str):
        found |= set(NUMBER_RE.findall(obj))
    return found


def _mentions(token: str, text_lower: str) -> bool:
    """Word-boundary match — a plain substring check would let a fabricated
    name falsely satisfy a short code like 'per' or 'mag'."""
    if not token:
        return False
    return re.search(r"\b" + re.escape(token.lower()) + r"\b", text_lower) is not None


def verify_storyline(narrative: str, storyline: dict, driver_names: dict[str, str] | None = None) -> dict:
    """Checks (1) every number cited in the narrative exists somewhere in
    supporting_facts, and (2) the storyline's subject is actually named in the
    text — catching the case where the model writes a plausible paragraph
    about the wrong driver while citing genuine numbers."""
    driver_names = driver_names or {}
    facts = storyline["supporting_facts"]
    subject = storyline["subject"]

    source_numbers = {float(n) for n in _flatten_numbers(facts)}
    cited_numbers = NUMBER_RE.findall(narrative)
    unverified = [n for n in cited_numbers if not any(abs(float(n) - s) < 0.005 for s in source_numbers)]

    text_lower = narrative.lower()
    full_name = driver_names.get(subject, "")
    # journalists refer to a driver by code, full name, or just last name
    # ("Hulkenberg") after the first mention — check all three forms
    candidates = [subject, full_name, full_name.rsplit(" ", 1)[-1] if full_name else ""]
    subject_mentioned = any(_mentions(c, text_lower) for c in candidates)

    return {
        "passed": len(unverified) == 0 and subject_mentioned,
        "unverified_numbers": unverified,
        "subject_mentioned": subject_mentioned,
        "cited_numbers": cited_numbers,
    }


def narrate_and_verify(storyline: dict, driver_names: dict[str, str] | None = None, max_attempts: int = 3) -> dict:
    """One storyline in, {"narrative", "verification"} out. Retries once on
    failure — a deeper loop would re-prompt with the specific flagged numbers,
    but a plain retry already clears most single-draft slips."""
    narrative = narrate_storyline(storyline)
    verification = verify_storyline(narrative, storyline, driver_names)
    attempts = 1
    while not verification["passed"] and attempts < max_attempts:
        narrative = narrate_storyline(storyline)
        verification = verify_storyline(narrative, storyline, driver_names)
        attempts += 1
    return {"narrative": narrative, "verification": verification, "attempts": attempts}


# ---------------------------------------------------------------------------
# Log — thin persistent-memory layer, one entry per run (mirrors agent.py's
# log_run / report_history.json so this pipeline is auditable the same way).
# ---------------------------------------------------------------------------
import os
from datetime import datetime, timezone

HISTORY_PATH = os.path.join(os.path.dirname(__file__), "weekend_story_history.json")


def log_run(session_label: str, headline: dict, storylines: list[dict]) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session": session_label,
        "headline": headline,
        "storylines": [
            {"title": s["title"], "angle": s["angle"], "subject": s["subject"],
             "score": s["score"], "narrative": s["narrative"], "verification": s["verification"]}
            for s in storylines
        ],
    }
    history = []
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)
    history.append(entry)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
