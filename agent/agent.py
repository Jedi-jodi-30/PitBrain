"""PitBrain agentic loop — pace-analysis case.

Instantiates the architecture from the project doc for one concrete analysis:

    fetch/compute (deterministic, analysis_suite/pace_analysis.py)
        -> summarize (local SLM narrates, never calculates)
        -> verify (every number the SLM wrote must trace back to the facts)
        -> log (thin persistent-memory layer)

The model only ever sees a small JSON fact table built from pace_analysis.py's
already-computed dataframes — it is not allowed to compute anything itself,
which is what keeps numeric hallucination out of the report.
"""
import json
import os
import re
import sys
from datetime import datetime, timezone

import ollama

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "analysis_suite"))  # pace_analysis's own `from basic_imports import *`

import analysis_suite.pace_analysis as pace

MODEL = "gemma2:2b"
HISTORY_PATH = os.path.join(os.path.dirname(__file__), "report_history.json")
NUMBER_RE = re.compile(r"-?\d+\.\d+|-?\d+")


# ---------------------------------------------------------------------------
# Fetch/compute — deterministic. The SLM below is only ever handed this dict.
# ---------------------------------------------------------------------------
def build_pace_facts() -> dict:
    top5 = pace.median_pace_drivers_df.head(5)
    bottom3 = pace.median_pace_drivers_df.tail(3)

    skew = pace.mean_lap_time_per_driver_df.merge(
        pace.median_pace_drivers_df, on="Driver", suffixes=("_mean", "_median")
    )
    skew["skew_seconds"] = skew["LapTimeSeconds_mean"] - skew["LapTimeSeconds_median"]
    most_inconsistent = skew.sort_values("skew_seconds", ascending=False).iloc[0]

    return {
        "session": "2023 Italian Grand Prix (Monza) — Race",
        "field_size": int(pace.median_pace_drivers_df.shape[0]),
        "fastest_lap": {
            "driver": pace.fastest_lap_driver,
            "time_seconds": round(float(pace.fastest_lap_time), 3),
            "lap_number": int(pace.fastest_lap_lap_number),
            "compound": pace.fastest_lap_compound,
        },
        "top5_median_pace": [
            {"driver": row.Driver, "median_seconds": round(float(row.LapTimeSeconds), 3)}
            for row in top5.itertuples()
        ],
        "bottom3_median_pace": [
            {"driver": row.Driver, "median_seconds": round(float(row.LapTimeSeconds), 3)}
            for row in bottom3.itertuples()
        ],
        "most_inconsistent_driver": {
            "driver": most_inconsistent["Driver"],
            "mean_seconds": round(float(most_inconsistent["LapTimeSeconds_mean"]), 3),
            "median_seconds": round(float(most_inconsistent["LapTimeSeconds_median"]), 3),
            "skew_seconds": round(float(most_inconsistent["skew_seconds"]), 3),
        },
    }


# ---------------------------------------------------------------------------
# Summarize — the SLM narrates the facts, it never calculates anything.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a Formula 1 race strategist writing a short analyst report. "
    "You will be given a JSON block of pre-computed race statistics. "
    "Write 3-4 sentences of plain-English analysis using ONLY the numbers given to you. "
    "Do not invent, estimate, round differently, or calculate any number that is not "
    "already present in the JSON. Every number you write must appear in the JSON."
)


def summarize(facts: dict) -> str:
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(facts, indent=2)},
        ],
    )
    return response["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Verify — reliability layer: flag any number in the narrative that doesn't
# trace back to the fact table. Deliberately simple (float-tolerant string
# matching, no NLP) — a thin first pass per the project's build approach.
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
        found.add(str(obj))
        found.add(f"{obj:.3f}")
    elif isinstance(obj, str):
        found |= set(NUMBER_RE.findall(obj))  # e.g. the "2023" inside the session name
    return found


def _spotlight_facts(facts: dict) -> list[tuple[float, str]]:
    """(number, correct_driver_code) pairs the narrative must attribute correctly."""
    pairs = [(facts["fastest_lap"]["time_seconds"], facts["fastest_lap"]["driver"])]
    for row in facts["top5_median_pace"]:
        pairs.append((row["median_seconds"], row["driver"]))
    for row in facts["bottom3_median_pace"]:
        pairs.append((row["median_seconds"], row["driver"]))
    mi = facts["most_inconsistent_driver"]
    pairs.append((mi["mean_seconds"], mi["driver"]))
    pairs.append((mi["median_seconds"], mi["driver"]))
    return list(dict.fromkeys(pairs))  # de-dupe: mi's median can coincide with a bottom3 entry


def _driver_names(codes: set[str]) -> dict[str, str]:
    """Map FastF1 driver codes to lowercase last names, via the already-loaded session."""
    names = {}
    for code in codes:
        try:
            names[code] = pace.session.get_driver(code)["LastName"].lower()
        except Exception:
            names[code] = code.lower()
    return names


CONTEXT_CHARS = 60  # characters of context scanned on either side of a cited number


def _mentions(token: str, text_lower: str) -> bool:
    """Word-boundary match — a plain substring check would let a fabricated
    name like 'Strakman' falsely satisfy the 3-letter code 'str'."""
    return re.search(r"\b" + re.escape(token.lower()) + r"\b", text_lower) is not None


def _check_attribution(narrative: str, facts: dict) -> list[dict]:
    """Catch right-number-wrong-driver drift: verify() alone can't see this
    because the number itself is genuine, it's just attached to the wrong name.

    Scans a character window around each cited number rather than splitting
    into sentences — a naive sentence-splitter treats every decimal point
    (".072", ".831", ...) as a sentence boundary, which would silently break
    this check on exactly the lap-time numbers it needs to look at.
    """
    pairs = _spotlight_facts(facts)
    known_codes = {code for _, code in pairs}
    names = _driver_names(known_codes)

    errors = []
    for match in NUMBER_RE.finditer(narrative):
        try:
            cited_value = float(match.group())
        except ValueError:
            continue

        matching_pairs = [p for p in pairs if abs(cited_value - p[0]) < 0.005]
        if not matching_pairs:
            continue  # not a number tied to a specific driver

        start = max(0, match.start() - CONTEXT_CHARS)
        end = min(len(narrative), match.end() + CONTEXT_CHARS)
        # never slice a word in half — walk out to the nearest whitespace
        while start > 0 and not narrative[start - 1].isspace():
            start -= 1
        while end < len(narrative) and not narrative[end].isspace():
            end += 1
        context = narrative[start:end]
        context_lower = context.lower()

        for number, correct_code in matching_pairs:
            correct_mentioned = _mentions(correct_code, context_lower) or _mentions(names[correct_code], context_lower)
            if correct_mentioned:
                continue

            wrong_code = next(
                (c for c in known_codes
                 if c != correct_code and (_mentions(c, context_lower) or _mentions(names[c], context_lower))),
                None,
            )
            errors.append({
                "number": number,
                "expected_driver": correct_code,
                "found_driver": wrong_code,
                "context": context.strip(),
                "span": [match.start(), match.end()],  # narrative[start:end] == the literal cited number
            })
    return errors


def verify(narrative: str, facts: dict) -> dict:
    source_numbers = {float(n) for n in _flatten_numbers(facts)}
    cited_numbers = NUMBER_RE.findall(narrative)

    unverified = [
        n for n in cited_numbers
        if not any(abs(float(n) - s) < 0.005 for s in source_numbers)
    ]
    attribution_errors = _check_attribution(narrative, facts)

    return {
        "passed": len(unverified) == 0 and len(attribution_errors) == 0,
        "unverified_numbers": unverified,
        "attribution_errors": attribution_errors,
        "cited_numbers": cited_numbers,
    }


# ---------------------------------------------------------------------------
# Log — thin persistent-memory layer: append every run to a local JSON file.
# ---------------------------------------------------------------------------
def log_run(facts: dict, narrative: str, verification: dict) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "facts": facts,
        "narrative": narrative,
        "verification": verification,
    }
    history = []
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r") as f:
            history = json.load(f)
    history.append(entry)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


# ---------------------------------------------------------------------------
# The agentic loop itself: compute -> summarize -> verify -> (retry once) -> log
# ---------------------------------------------------------------------------
def run_pace_report() -> dict:
    facts = build_pace_facts()
    narrative = summarize(facts)
    verification = verify(narrative, facts)

    if not verification["passed"]:
        narrative = summarize(facts)  # one retry; a deeper loop would re-prompt with the flagged numbers
        verification = verify(narrative, facts)

    log_run(facts, narrative, verification)
    return {"facts": facts, "narrative": narrative, "verification": verification}


if __name__ == "__main__":
    result = run_pace_report()

    print("\n=== PitBrain Pace Report ===\n")
    print(result["narrative"])

    print("\n--- Reliability check ---")
    v = result["verification"]
    if v["passed"]:
        print("All cited numbers verified, correctly attributed to their drivers.")
    else:
        if v["unverified_numbers"]:
            print(f"UNVERIFIED numbers (not in source data): {v['unverified_numbers']}")
        for err in v["attribution_errors"]:
            found = err["found_driver"] or "no known driver"
            print(f"ATTRIBUTION ERROR: {err['number']} belongs to {err['expected_driver']}, "
                  f"but nearby text names {found} instead -> \"{err['context']}\"")

    # Publish the report as the final step of the loop, reusing this run's
    # result so the report reflects exactly what was just printed above
    # (rather than calling the model a second time).
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "report_suite"))
    import report_suite.generate_report
    report_path = report_suite.generate_report.generate(result=result)
    print(f"\nReport written to {report_path}")
