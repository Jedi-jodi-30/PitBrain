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
INSIGHT = "insight"  # notable, but not clearly good or bad on its own — context/strategy color


def _rows(facts: list[dict], metric: str) -> list[dict]:
    return [f for f in facts if f.get("metric") == metric]


def _driver_pace_lookup(facts: list[dict]) -> dict[str, float]:
    return {r["driver"]: r["value"] for r in _rows(facts, "median_pace_seconds")}


def _top_pace_leaders(facts: list[dict], n: int = 6) -> set[str]:
    """Same top-N-by-median-pace set new_analysis.py's _top_leaders() uses to decide which
    drivers get highlighted (in color, named) on position_per_lap / rolling_pace, vs. drawn as
    unlabeled gray lines. Replicated from facts rather than recomputed from clean_laps so a
    storyline can check *before* claiming one of those charts whether its subject will actually
    be one of the highlighted lines — otherwise the chart would show the story's subject as
    just another gray line among a dozen others, which isn't a usable illustration of the story."""
    pace = _driver_pace_lookup(facts)
    return {d for d, _ in sorted(pace.items(), key=lambda kv: kv[1])[:n]}


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
def position_swings(facts: list[dict], top_n: int = 1) -> list[dict]:
    """The single biggest mover each direction — not the top N losers, just the ONE most
    dramatic fall and the ONE most dramatic climb. Two different stories (opposite extremes
    of the same metric), not two instances of the same story."""
    rows = sorted(_rows(facts, "net_position_change"), key=lambda r: r["value"])
    out = []
    for r in rows[:top_n]:
        if r["value"] >= 0:
            continue
        out.append({
            "title": f"{r['driver']} suffers the biggest fall of the race — down {abs(r['value'])} places",
            "angle": MISTAKE, "subject": r["driver"], "score": abs(r["value"]) * 12,
            "chart_key": "net_position_change", "supporting_facts": [r],
        })
    for r in rows[-top_n:][::-1]:
        if r["value"] <= 0:
            continue
        out.append({
            "title": f"{r['driver']} delivers the biggest climb of the race — up {r['value']} places",
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
            # NOT "cliff_detection": that catalog chart illustrates whichever
            # driver/stint has the most flagged cliff *events* field-wide, which is
            # frequently a different driver than the single biggest jump picked here
            # (e.g. this storyline could be about STR's one 4s cliff while the stock
            # chart shows SAR, who merely had more, smaller ones) — showing it would
            # be a wrong, misleading chart under a correct headline, so show none.
            "chart_key": None, "supporting_facts": [r],
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
    # position_per_lap is the far more dramatic chart for this story (the actual race
    # trajectory, not just a laps-led bar) — but it only highlights the top-6-by-pace
    # drivers in color, everyone else is an unlabeled gray line. Use it only when this
    # storyline's own subject is confirmed to be one of those six; otherwise fall back to
    # the always-safe laps_led bar rather than risk an illustration that doesn't show them.
    chart_key = "position_per_lap" if r["driver"] in _top_pace_leaders(facts) else "laps_led"
    return [{
        "title": f"{r['driver']} controls the race from the front",
        "angle": ACHIEVEMENT, "subject": r["driver"], "score": r["value"] * 1.5,
        "chart_key": chart_key, "supporting_facts": [r],
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


def consistency_standout(facts: list[dict]) -> list[dict]:
    """Coefficient of variation (std/mean lap time) — low means metronomic, high means
    erratic (traffic, a struggle, a problem on track)."""
    rows = sorted(_rows(facts, "coefficient_of_variation"), key=lambda r: r["value"])
    if not rows:
        return []
    most, least = rows[0], rows[-1]
    out = [{
        "title": f"{most['driver']} is the most metronomic driver on track",
        "angle": ACHIEVEMENT, "subject": most["driver"], "score": (least["value"] - most["value"]) * 400 + 5,
        "chart_key": "coefficient_of_variation", "supporting_facts": [most],
    }]
    if least["driver"] != most["driver"]:
        out.append({
            "title": f"{least['driver']}'s pace is the most erratic in the field",
            "angle": MISTAKE, "subject": least["driver"], "score": (least["value"] - most["value"]) * 400,
            "chart_key": "coefficient_of_variation", "supporting_facts": [least],
        })
    return out


def pace_skew_standout(facts: list[dict]) -> list[dict]:
    """mean − median lap time: a positive skew means a handful of slow outlier laps (traffic,
    a lock-up, a scare) dragged the average up well above the typical lap — a different signal
    from coefficient_of_variation, which measures spread throughout, not a few one-off bad
    laps on an otherwise solid race."""
    rows = sorted(_rows(facts, "mean_pace_seconds"), key=lambda r: r["skew_seconds"], reverse=True)
    if not rows:
        return []
    r = rows[0]
    if r["skew_seconds"] < 0.05:
        return []
    return [{
        "title": f"{r['driver']}'s average is dragged up by a handful of slow laps",
        "angle": MISTAKE, "subject": r["driver"], "score": r["skew_seconds"] * 60,
        "chart_key": "mean_pace", "supporting_facts": [r],
    }]


def theoretical_lap_efficiency(facts: list[dict]) -> list[dict]:
    """Gap between a driver's actual best lap and their own theoretical best (sum of their
    three best individual sector times) — how much they left on the table across the lap."""
    rows = sorted(_rows(facts, "theoretical_best_lap_gap"), key=lambda r: r["value"])
    if not rows:
        return []
    best, worst = rows[0], rows[-1]
    out = [{
        "title": f"{best['driver']} nearly extracts the perfect lap",
        "angle": ACHIEVEMENT, "subject": best["driver"], "score": max(0.0, 0.4 - best["value"]) * 60 + 5,
        "chart_key": "theoretical_best_lap", "supporting_facts": [best],
    }]
    if worst["driver"] != best["driver"] and worst["value"] > 0:
        out.append({
            "title": f"{worst['driver']} leaves the most time on the table across the lap",
            "angle": MISTAKE, "subject": worst["driver"], "score": worst["value"] * 60,
            "chart_key": "theoretical_best_lap", "supporting_facts": [worst],
        })
    return out


def speed_trap_standout(facts: list[dict]) -> list[dict]:
    """Highest average speed-trap reading — straightline/tow/low-drag strength. Achievement
    only: a slower trap speed is a car trait, not a driving mistake."""
    rows = sorted(_rows(facts, "avg_speed_trap"), key=lambda r: r["value"], reverse=True)
    if not rows:
        return []
    r = rows[0]
    return [{
        "title": f"{r['driver']} is the fastest through the speed trap",
        "angle": ACHIEVEMENT, "subject": r["driver"], "score": 10,
        "chart_key": "avg_speed_trap", "supporting_facts": [r],
    }]


def purple_sector_count(facts: list[dict]) -> list[dict]:
    """Who banked the most session-best ("purple") sectors — a sharper signal of raw pace
    than a single fastest lap, since it's earned across the whole session."""
    rows = _rows(facts, "session_best_sector")
    if not rows:
        return []
    counts: dict[str, list[dict]] = {}
    for r in rows:
        counts.setdefault(r["driver"], []).append(r)
    driver, driver_facts = max(counts.items(), key=lambda kv: len(kv[1]))
    if len(driver_facts) < 2:
        return []
    return [{
        "title": f"{driver} banks {len(driver_facts)} session-best sectors",
        "angle": ACHIEVEMENT, "subject": driver, "score": len(driver_facts) * 9,
        "chart_key": "session_best_sector_moments", "supporting_facts": driver_facts,
    }]


def position_volatility_standout(facts: list[dict]) -> list[dict]:
    """Std dev of position held across the race — a different lens on race control than lap
    time: this catches a driver whose *position* bounced around (traffic, incidents nearby,
    a scrappy race) even if their own pace was fine."""
    rows = sorted(_rows(facts, "position_volatility"), key=lambda r: r["value"])
    if not rows:
        return []
    stable, chaotic = rows[0], rows[-1]
    out = [{
        "title": f"{stable['driver']} holds position like a metronome all race",
        "angle": ACHIEVEMENT, "subject": stable["driver"], "score": max(0.0, 2.0 - stable["value"]) * 8 + 3,
        "chart_key": "position_volatility", "supporting_facts": [stable],
    }]
    if chaotic["driver"] != stable["driver"]:
        out.append({
            "title": f"{chaotic['driver']} has the most turbulent race on track",
            "angle": INSIGHT, "subject": chaotic["driver"], "score": chaotic["value"] * 8,
            "chart_key": "position_volatility", "supporting_facts": [chaotic],
        })
    return out


# ---------------------------------------------------------------------------
# Team-level detectors — same idea, but the subject is a constructor, not a driver.
# team_pace_leader / team_balance use the catalog's own team-grain metrics
# (team_median_pace, intra_team_teammate_gap). team_strategy_efficiency uses
# "pit_stops_vs_position_change" — not a catalog metric; generate_weekend_story.py
# computes it fresh from session.laps (raw, unfiltered) since clean_laps has every
# pit in/out lap filtered out by definition, same gap new_analysis.py's own
# pit_stop_count()/team_pit_stop_timing() explicitly punt on.
# ---------------------------------------------------------------------------
def team_pace_leader(facts: list[dict]) -> list[dict]:
    rows = sorted(_rows(facts, "team_median_pace"), key=lambda r: r["value"])
    if not rows:
        return []
    r = rows[0]
    return [{
        "title": f"{r['team']} fields the fastest car of the weekend",
        "angle": ACHIEVEMENT, "subject": r["team"], "score": 20,
        "chart_key": "team_median_pace", "supporting_facts": [r], "is_team_story": True,
    }]


def team_balance(facts: list[dict]) -> list[dict]:
    """Smallest intra-team teammate pace gap — both cars extracting the same lap time is a
    genuine team-level achievement (car + both drivers all performing to the same level),
    distinct from (and the opposite extreme of) the individual teammate_mismatch story."""
    rows = sorted(_rows(facts, "intra_team_teammate_gap"), key=lambda r: r["value"])
    if not rows:
        return []
    r = rows[0]
    return [{
        "title": f"{r['team']} runs the most evenly matched lineup on the grid",
        "angle": ACHIEVEMENT, "subject": r["team"], "score": max(0.0, 0.3 - r["value"]) * 40 + 3,
        "chart_key": "intra_team_teammate_gap", "supporting_facts": [r], "is_team_story": True,
    }]


def team_strategy_efficiency(facts: list[dict]) -> list[dict]:
    """Positions gained (or lost) per pit stop, by team — who actually converted pit strategy
    into track position vs. who paid a price for theirs. Illustrated with a dedicated scatter
    chart built just for this (see generate_weekend_story._team_strategy_facts_and_chart) —
    not reused anywhere else in the report."""
    rows = sorted(_rows(facts, "pit_stops_vs_position_change"), key=lambda r: r["value"], reverse=True)
    if not rows:
        return []
    out = []
    best = rows[0]
    if best["value"] > 0:
        out.append({
            "title": f"{best['team']} extracts the most track position per pit stop",
            "angle": ACHIEVEMENT, "subject": best["team"], "score": best["value"] * 15 + 8,
            "chart_key": "team_strategy_efficiency", "supporting_facts": [best], "is_team_story": True,
        })
    worst = rows[-1]
    if worst["team"] != best["team"] and worst["value"] < 0:
        out.append({
            "title": f"{worst['team']}'s pit strategy costs them track position",
            "angle": MISTAKE, "subject": worst["team"], "score": abs(worst["value"]) * 15,
            "chart_key": "team_strategy_efficiency", "supporting_facts": [worst], "is_team_story": True,
        })
    return out


def strategy_overview(facts: list[dict]) -> list[dict]:
    """Who ran the most stints (tyre changes) of the weekend — the standout alternate or
    aggressive strategy call, illustrated with the full-field tyre stint usage chart (already
    in the catalog as stint_composition — a Gantt of every driver's compound/stint/lap range),
    shown here as its own exhibit rather than silently sitting unused in analysis_output."""
    rows = _rows(facts, "stint_composition")
    if not rows:
        return []
    counts: dict[str, list[dict]] = {}
    for r in rows:
        counts.setdefault(r["driver"], []).append(r)
    driver, driver_facts = max(counts.items(), key=lambda kv: len(kv[1]))
    if len(driver_facts) < 3:
        return []
    return [{
        "title": f"{driver} runs the most varied tyre strategy of the weekend — {len(driver_facts)} stints",
        "angle": INSIGHT, "subject": driver, "score": len(driver_facts) * 6,
        "chart_key": "stint_composition", "supporting_facts": driver_facts, "is_strategy_story": True,
    }]


def team_quali_pace_standout(facts: list[dict]) -> list[dict]:
    """Qualifying pace vs top speed, by team — a different session and a different question
    than every race-grain detector above: who actually built the best-rounded car (fast
    through the lap AND fast on the straights) in qualifying trim. Optional: generate_weekend_
    story.py only adds these facts when a qualifying session was fetchable for this event; if
    it wasn't (or this is already a qualifying-session report), there's nothing to detect."""
    rows = _rows(facts, "quali_pace_vs_topspeed")
    if not rows:
        return []
    # best-rounded = good rank on both axes (low quali-pace rank, high top-speed rank)
    by_pace = sorted(rows, key=lambda r: r["median_laptime"])
    by_speed = sorted(rows, key=lambda r: r["top_speed"], reverse=True)
    pace_rank = {r["team"]: i for i, r in enumerate(by_pace)}
    speed_rank = {r["team"]: i for i, r in enumerate(by_speed)}
    best = min(rows, key=lambda r: pace_rank[r["team"]] + speed_rank[r["team"]])
    return [{
        "title": f"{best['team']} builds the most complete qualifying package — quick and fast in a straight line",
        "angle": ACHIEVEMENT, "subject": best["team"],
        "score": 10 + max(0, 10 - pace_rank[best["team"]] - speed_rank[best["team"]]),
        "chart_key": "team_quali_pace_speed", "supporting_facts": [best], "is_team_story": True,
    }]


DETECTORS = [
    position_swings, tyre_cliff, degradation_contrast, dominant_leader,
    fastest_lap_spotlight, teammate_mismatch, track_discipline,
    consistency_standout, pace_skew_standout, theoretical_lap_efficiency, speed_trap_standout,
    purple_sector_count, position_volatility_standout, team_pace_leader, team_balance,
    team_strategy_efficiency, strategy_overview, team_quali_pace_standout,
]

GUARANTEED_TAGS = ("is_team_story", "is_strategy_story")


def select_storylines(facts: list[dict], top_n: int = 8) -> list[dict]:
    """Run every detector, de-duplicate by (subject, angle) keeping the higher score, and
    return the top N candidates ranked by score — with coverage guaranteed for each tag in
    GUARANTEED_TAGS (team-level stories, strategy stories): a weekend report is driver-pace
    heavy by nature (far more driver-grain metrics exist in the catalog than team/strategy
    ones), so a pure score cut can crowd that coverage out even though it's exactly the kind
    of variety worth keeping. If no story carrying a given tag makes the natural cut, the
    best-scoring one swaps in for the weakest untagged story instead of being silently
    dropped."""
    candidates: dict[tuple[str, str], dict] = {}
    for detector in DETECTORS:
        for story in detector(facts):
            key = (story["subject"], story["angle"])
            if key not in candidates or story["score"] > candidates[key]["score"]:
                candidates[key] = story
    ranked = sorted(candidates.values(), key=lambda s: s["score"], reverse=True)

    selected = ranked[:top_n]
    if len(ranked) > top_n:
        for tag in GUARANTEED_TAGS:
            if any(s.get(tag) for s in selected):
                continue
            best_tagged = next((s for s in ranked if s.get(tag)), None)
            if not best_tagged:
                continue
            for i in range(len(selected) - 1, -1, -1):
                if not any(selected[i].get(t) for t in GUARANTEED_TAGS):
                    selected[i] = best_tagged
                    break
        selected.sort(key=lambda s: s["score"], reverse=True)
    return selected
