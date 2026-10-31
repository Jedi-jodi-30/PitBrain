"""Runs every function in new_analysis.py against a loaded session and persists the output:
all facts to one JSON file, each chart to its own PNG, plus a manifest mapping function name
to its chart file (for a future report-assembly step to pick up without re-running anything).
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # this script only saves files, never shows a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ANALYSIS_SUITE_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_ANALYSIS_SUITE_DIR, ".."))  # pace_analysis's own `from analysis_suite... import`
sys.path.insert(0, _ANALYSIS_SUITE_DIR)
import pace_analysis as pa
import new_analysis as na

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "analysis_output")
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")
FACTS_PATH = os.path.join(OUTPUT_DIR, "facts.json")
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "charts_manifest.json")

# Every catalog function that takes just clean_laps
SINGLE_ARG_FUNCTIONS = [
    na.median_pace, na.fastest_lap, na.mean_pace, na.pace_delta_to_fastest, na.rolling_pace,
    na.lap_time_std_dev, na.coefficient_of_variation, na.interquartile_range,
    na.consistency_band_pct, na.lap_time_range,
    na.stint_composition, na.pit_stop_count, na.compound_usage_split, na.degradation_slope,
    na.early_vs_late_stint_pace, na.fresh_vs_used_tyre_delta, na.cliff_detection,
    na.median_sector_times, na.theoretical_best_lap, na.sector_delta_to_fastest, na.sector_trend_over_race,
    na.max_speed_by_trap, na.avg_speed_trap, na.speed_trap_trend,
    na.position_per_lap, na.net_position_change, na.laps_led, na.position_volatility,
    na.team_median_pace, na.intra_team_teammate_gap, na.team_pit_stop_timing,
    na.personal_best_timing, na.fastest_lap_headline, na.session_best_sector_moments,
    na.excluded_laps_note, na.deleted_lap_count, na.accuracy_ratio,
]

# The three catalog functions that compare a specific pair of drivers
HEAD_TO_HEAD_FUNCTIONS = [na.head_to_head_lap_delta, na.cumulative_gap, na.lap_win_count]


def _json_default(obj):
    """json.dump doesn't know numpy scalar types — pandas itertuples() hands them out for
    every numeric column, and round() on one returns another numpy scalar, not a float."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    return str(obj)


def _default_pair(clean_laps: pd.DataFrame) -> tuple:
    """Falls back to the two closest drivers by median pace when no pair is given."""
    top2 = clean_laps.groupby("Driver")["LapTimeSeconds"].median().sort_values().head(2).index.tolist()
    return top2[0], top2[1]


def run_all(clean_laps: pd.DataFrame, driver_pair: tuple | None = None) -> dict:
    driver_a, driver_b = driver_pair or _default_pair(clean_laps)

    os.makedirs(CHARTS_DIR, exist_ok=True)
    all_facts = []
    chart_paths = {}
    failures = []

    def _store(name: str, result: dict):
        all_facts.extend(result["facts"])
        if result["chart"] is not None:
            path = os.path.join(CHARTS_DIR, f"{name}.png")
            result["chart"].savefig(path, dpi=150, bbox_inches="tight")
            plt.close(result["chart"])
            chart_paths[name] = os.path.relpath(path, OUTPUT_DIR)

    for fn in SINGLE_ARG_FUNCTIONS:
        name = fn.__name__
        try:
            _store(name, fn(clean_laps))
        except Exception as e:
            failures.append(name)
            print(f"FAILED {name}: {type(e).__name__}: {e}")

    for fn in HEAD_TO_HEAD_FUNCTIONS:
        name = fn.__name__
        try:
            _store(name, fn(clean_laps, driver_a, driver_b))
        except Exception as e:
            failures.append(name)
            print(f"FAILED {name}: {type(e).__name__}: {e}")

    with open(FACTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_facts, f, indent=2, default=_json_default)

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(chart_paths, f, indent=2)

    return {
        "facts_count": len(all_facts),
        "charts_count": len(chart_paths),
        "functions_run": len(SINGLE_ARG_FUNCTIONS) + len(HEAD_TO_HEAD_FUNCTIONS),
        "failures": failures,
        "driver_pair": (driver_a, driver_b),
        "facts_path": FACTS_PATH,
        "charts_dir": CHARTS_DIR,
    }


if __name__ == "__main__":
    session_analysis = pa.PaceAnalysis()
    summary = run_all(session_analysis.clean_laps)

    print(f"\nRan {summary['functions_run']} functions "
          f"({summary['functions_run'] - len(summary['failures'])} succeeded, "
          f"{len(summary['failures'])} failed).")
    print(f"Head-to-head pair: {summary['driver_pair'][0]} vs {summary['driver_pair'][1]}")
    print(f"Wrote {summary['facts_count']} facts to {summary['facts_path']}")
    print(f"Wrote {summary['charts_count']} charts to {summary['charts_dir']}")
    if summary["failures"]:
        print(f"Failures: {summary['failures']}")
