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
dump). You will be given ONE storyline as a JSON block: a headline, an angle — "achievement" \
(something went well), "mistake" (something went wrong), or "insight" (notable, but not \
clearly good or bad — write it as neutral context/color, not forced praise or criticism) — the \
driver/team subject ("subject", plus "subject_full_name" if given), and the exact supporting facts.

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
this race weekend's storylines — achievements, mistakes, and neutral insights — already \
ranked by how notable they are. Write a short, punchy magazine-style headline (under 12 words) \
and a one-sentence dek \
(subheading) that together frame the weekend's overall narrative using only what the storyline \
titles already say — do not introduce any driver, team, or number that isn't in the list.

Respond as JSON only: {"headline": "...", "dek": "..."}"""


# ---------------------------------------------------------------------------
# Summarize — the SLM narrates, it never calculates.
# ---------------------------------------------------------------------------
def narrate_storyline(storyline: dict, feedback: str | None = None) -> str:
    """Draft (or redraft) this storyline's paragraph. Pass `feedback` (from
    build_correction_feedback, applied to the previous draft's verification)
    to turn a redraft into a targeted correction instead of a blind retry."""
    payload = {k: v for k, v in storyline.items()
               if k not in ("chart_key", "narrative", "verification", "attempts", "draft_history")}
    user_content = json.dumps(payload, indent=2)
    if feedback:
        user_content += (
            f"\n\nYour previous draft had problems and must be corrected:\n{feedback}\n"
            "Rewrite the paragraph fixing exactly these issues. Keep everything else that was "
            "already accurate."
        )
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": STORY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return response["message"]["content"].strip()


def build_correction_feedback(verification: dict, storyline: dict, driver_names: dict[str, str] | None = None) -> str | None:
    """Turns a failed verify_storyline() result into a specific instruction for the next
    narrate_storyline() call, so a redraft fixes exactly what was flagged instead of the model
    guessing again from scratch."""
    if verification["passed"]:
        return None
    driver_names = driver_names or {}
    lines = []
    if verification["unverified_numbers"]:
        bad = ", ".join(sorted(set(verification["unverified_numbers"])))
        lines.append(
            f"- These numbers you wrote do NOT appear in the supporting facts and must be "
            f"removed or replaced: {bad}. Do not substitute a different invented number either "
            f"— only use numbers that are literally in the JSON, or describe it in words with "
            f"no digit at all."
        )
    if not verification["subject_mentioned"]:
        subject = storyline["subject"]
        name = driver_names.get(subject, storyline.get("subject_full_name", ""))
        who = f'"{name}" (code {subject})' if name else f'"{subject}"'
        lines.append(f"- You must explicitly name the subject, {who}, in the paragraph — "
                      f"do not write about any other driver or team.")
    return "\n".join(lines)


def _fallback_headline(storylines: list[dict]) -> dict:
    """Used only if the model's headline response isn't valid JSON — still built from this
    run's actual top storyline, not a generic placeholder string, so even a fallback headline
    reflects real graph output."""
    if not storylines:
        return {"headline": "Race Weekend Story", "dek": ""}
    top = storylines[0]
    achievements = sum(1 for s in storylines if s["angle"] == "achievement")
    mistakes = sum(1 for s in storylines if s["angle"] == "mistake")
    insights = sum(1 for s in storylines if s["angle"] == "insight")
    insight_clause = f", {insights} storyline(s) worth watching" if insights else ""
    return {
        "headline": top["title"],
        "dek": f"{achievements} standout drive(s), {mistakes} costly mistake(s){insight_clause} this weekend.",
    }


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
        parsed = json.loads(response["message"]["content"].strip())
        if not parsed.get("headline"):
            raise ValueError("empty headline")
        return parsed
    except (json.JSONDecodeError, ValueError, AttributeError):
        return _fallback_headline(storylines)


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
    """One storyline in, {"narrative", "verification", "draft_history"} out. On failure,
    redrafts with targeted correction feedback (build_correction_feedback) rather than a
    blind retry — each attempt is recorded in draft_history so the flag -> correction cycle
    is visible in the run log, not just the final result."""
    draft_history = []
    feedback = None
    narrative = narrate_storyline(storyline, feedback)
    verification = verify_storyline(narrative, storyline, driver_names)
    draft_history.append({"attempt": 1, "narrative": narrative, "verification": verification,
                           "correction_feedback": None})
    attempts = 1
    while not verification["passed"] and attempts < max_attempts:
        feedback = build_correction_feedback(verification, storyline, driver_names)
        narrative = narrate_storyline(storyline, feedback)
        verification = verify_storyline(narrative, storyline, driver_names)
        attempts += 1
        draft_history.append({"attempt": attempts, "narrative": narrative, "verification": verification,
                               "correction_feedback": feedback})
    return {"narrative": narrative, "verification": verification, "attempts": attempts,
            "draft_history": draft_history}


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
             "score": s["score"], "narrative": s["narrative"], "verification": s["verification"],
             "draft_history": s.get("draft_history", [])}
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
