"""Storyline mining over the deterministic fact catalog from new_analysis.py.

run_all_analysis.py computes every metric in the catalog and dumps them as one flat list of
facts — useful as ground truth, useless as a magazine feature (nobody wants 6,000 numbers).
This module is the layer in between: a handful of independent detector functions, each
scanning the fact list for something that's actually a *story* — a big mover, a strategy that
paid off or fell off a cliff, a teammate getting beaten — and scoring how notable it is.

Each detector returns zero or more candidate dicts:
    {
        "title": str,                  # short, specific — not a metric name
        "angle": "achievement" | "mistake",
        "subject": str,                 # driver code or team name
        "score": float,                 # magnitude, for ranking across detectors
        "chart_key": str | None,        # key into charts_manifest.json, if one exists
        "supporting_facts": [dict, ...] # the exact facts backing this storyline — narration
                                         # and the reliability layer are both scoped to these,
                                         # not the full 6000-fact catalog
    }

select_storylines() runs every detector and returns the top N by score. Pure, deterministic,
no LLM involved — the only thing allowed to decide "what's a story" is the data.
"""
from __future__ import annotations

ACHIEVEMENT = "achievement"
MISTAKE = "mistake"


def _rows(facts: list[dict], metric: str) -> list[dict]:
    return [f for f in facts if f.get("metric") == metric]


def _driver_pace_lookup(facts: list[dict]) -> dict[str, float]:
    return {r["driver"]: r["value"] for r in _rows(facts, "median_pace_seconds")}


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
def position_swings(facts: list[dict], top_n: int = 2) -> list[dict]:
    rows = sorted(_rows(facts, "net_position_change"), key=lambda r: r["value"])
    out = []
    for r in rows[:top_n]:
        if r["value"] >= 0:
            continue
        out.append({
            "title": f"{r['driver']} loses {abs(r['value'])} places",
            "angle": MISTAKE, "subject": r["driver"], "score": abs(r["value"]) * 12,
            "chart_key": "net_position_change", "supporting_facts": [r],
        })
    for r in rows[-top_n:][::-1]:
        if r["value"] <= 0:
            continue
        out.append({
            "title": f"{r['driver']} gains {r['value']} places",
            "angle": ACHIEVEMENT, "subject": r["driver"], "score": r["value"] * 12,
            "chart_key": "net_position_change", "supporting_facts": [r],
        })
    return out


def tyre_cliff(facts: list[dict], top_n: int = 1) -> list[dict]:
    rows = sorted(_rows(facts, "cliff_detection"), key=lambda r: r["value"], reverse=True)
    out = []
    for r in rows[:top_n]:
        out.append({
            "title": f"{r['driver']} hits a tyre cliff on lap {r['lap_number']}",
            "angle": MISTAKE, "subject": r["driver"], "score": r["value"] * 15,
            "chart_key": "cliff_detection", "supporting_facts": [r],
        })
    return out


def degradation_contrast(facts: list[dict], min_laps: int = 15) -> list[dict]:
    """Best and worst sustained tyre management — long stints only, so a 4-lap
    splash-and-dash doesn't masquerade as a degradation story."""
    rows = [r for r in _rows(facts, "degradation_slope_sec_per_lap") if r.get("n_laps", 0) >= min_laps]
    if not rows:
        return []
    out = []
    best = min(rows, key=lambda r: r["value"])
    worst = max(rows, key=lambda r: r["value"])
    out.append({
        "title": f"{best['driver']} manages the {best['compound'].lower()} tyre best",
        "angle": ACHIEVEMENT, "subject": best["driver"], "score": max(0.0, -best["value"]) * 200 + 5,
        "chart_key": "degradation_slope", "supporting_facts": [best],
    })
    if worst["value"] > 0:
        out.append({
            "title": f"{worst['driver']}'s {worst['compound'].lower()} tyre falls off over a long stint",
            "angle": MISTAKE, "subject": worst["driver"], "score": worst["value"] * 200,
            "chart_key": "degradation_slope", "supporting_facts": [worst],
        })
    return out


def dominant_leader(facts: list[dict]) -> list[dict]:
    rows = sorted(_rows(facts, "laps_led"), key=lambda r: r["value"], reverse=True)
    if not rows or rows[0]["value"] < 5:
        return []
    r = rows[0]
    return [{
        "title": f"{r['driver']} controls the race from the front",
        "angle": ACHIEVEMENT, "subject": r["driver"], "score": r["value"] * 1.5,
        "chart_key": "laps_led", "supporting_facts": [r],
    }]


def fastest_lap_spotlight(facts: list[dict]) -> list[dict]:
    rows = _rows(facts, "fastest_lap_headline")
    if not rows:
        return []
    r = rows[0]
    return [{
        "title": f"{r['driver']} sets the fastest lap of the session",
        "angle": ACHIEVEMENT, "subject": r["driver"], "score": 25,
        "chart_key": "fastest_lap_headline", "supporting_facts": [r],
    }]


def teammate_mismatch(facts: list[dict], top_n: int = 1) -> list[dict]:
    """Biggest intra-team pace gap, attributed to whichever of the pair was slower."""
    rows = sorted(_rows(facts, "intra_team_teammate_gap"), key=lambda r: r["value"], reverse=True)
    pace = _driver_pace_lookup(facts)
    out = []
    for r in rows[:top_n]:
        pair = [d.strip() for d in r["drivers"].split(",")]
        if len(pair) != 2 or not all(d in pace for d in pair):
            continue
        slower = max(pair, key=lambda d: pace[d])
        faster = pair[0] if slower == pair[1] else pair[1]
        out.append({
            "title": f"{faster} outpaces teammate {slower} by {r['value']:.3f}s at {r['team']}",
            "angle": MISTAKE, "subject": slower, "score": r["value"] * 8,
            "chart_key": "intra_team_teammate_gap", "supporting_facts": [r],
        })
    return out


def track_discipline(facts: list[dict], min_deleted: int = 2) -> list[dict]:
    rows = sorted(
        [r for r in _rows(facts, "deleted_lap_count") if isinstance(r.get("value"), int)],
        key=lambda r: r["value"], reverse=True,
    )
    if not rows or rows[0]["value"] < min_deleted:
        return []
    r = rows[0]
    return [{
        "title": f"{r['driver']} repeatedly runs beyond track limits",
        "angle": MISTAKE, "subject": r["driver"], "score": r["value"] * 4,
        "chart_key": "deleted_lap_count", "supporting_facts": [r],
    }]


DETECTORS = [
    position_swings, tyre_cliff, degradation_contrast, dominant_leader,
    fastest_lap_spotlight, teammate_mismatch, track_discipline,
]


def select_storylines(facts: list[dict], top_n: int = 6) -> list[dict]:
    """Run every detector, de-duplicate by (subject, angle) keeping the higher score,
    and return the top N candidates ranked by score."""
    candidates: dict[tuple[str, str], dict] = {}
    for detector in DETECTORS:
        for story in detector(facts):
            key = (story["subject"], story["angle"])
            if key not in candidates or story["score"] > candidates[key]["score"]:
                candidates[key] = story
    ranked = sorted(candidates.values(), key=lambda s: s["score"], reverse=True)
    return ranked[:top_n]
