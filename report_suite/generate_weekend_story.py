"""Builds the retrospective "weekend story" report: deterministic facts (analysis_suite) ->
storyline mining (analysis_suite/storylines.py) -> narration + reliability check
(report_suite/narrative.py) -> one self-contained HTML article.

Usage:
    python report_suite/generate_weekend_story.py --year 2023 --event Monza --session R

Reuses analysis_output/facts.json + charts_manifest.json when they're already on disk for the
requested session (so re-running this doesn't recompute the whole catalog); otherwise runs
analysis_suite/run_all_analysis.py once to produce them.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import threading
from html import escape as esc

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "analysis_suite"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_suite.basic_imports import get_session, get_clean_laps
from analysis_suite.storylines import select_storylines
import analysis_suite.run_all_analysis as run_all_analysis
from report_suite.narrative import narrate_and_verify, narrate_headline, log_run

OUTPUT_DIR = os.path.join(_REPO_ROOT, "reports")
ANALYSIS_OUTPUT_DIR = os.path.join(_REPO_ROOT, "analysis_output")


def _session_slug(year: int, event: str, session_type: str) -> str:
    return f"{year}_{event.lower().replace(' ', '_')}_{session_type}"


def _session_output_dir(year: int, event: str, session_type: str) -> str:
    """Every session gets its own facts.json/charts_manifest.json/charts/ under here. Must
    never be a single fixed path shared across sessions — see run_all.run_all()'s docstring
    for why that was a real bug (every race after the first silently served back the first
    race's cached analysis)."""
    return os.path.join(ANALYSIS_OUTPUT_DIR, "sessions", _session_slug(year, event, session_type))


def _driver_names(session) -> dict[str, str]:
    names = {}
    for code in session.drivers:
        try:
            names[code] = session.get_driver(code)["FullName"]
        except Exception:
            continue
    return names


def _load_or_build_facts(clean_laps, output_dir: str, force: bool = False) -> tuple[list[dict], dict[str, str]]:
    facts_path = os.path.join(output_dir, "facts.json")
    manifest_path = os.path.join(output_dir, "charts_manifest.json")
    if force or not (os.path.exists(facts_path) and os.path.exists(manifest_path)):
        run_all_analysis.run_all(clean_laps, output_dir=output_dir)
    with open(facts_path, "r", encoding="utf-8") as f:
        facts = json.load(f)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return facts, manifest


def _team_strategy_facts_and_chart(facts: list[dict], session, clean_laps, output_dir: str) -> tuple[list[dict], str | None]:
    """Pit-stop count isn't in the catalog at all — new_analysis.py's own pit_stop_count()
    explicitly punts on it, because clean_laps has every pit in/out lap filtered out by
    definition. This pulls real counts from session.laps (the raw, unfiltered laps) instead,
    aggregates to team level, and cross-references against each team's net position change
    (reusing the already-computed per-driver facts, not recomputing position logic) to surface
    which team actually converted pit stops into track position and which didn't. A genuinely
    new chart — nothing else in the report reuses a scatter plot."""
    raw_laps = session.laps
    pit_counts = raw_laps.groupby("Driver")["PitInTime"].apply(lambda s: int(s.notna().sum()))
    driver_team = clean_laps.drop_duplicates("Driver").set_index("Driver")["Team"]
    net_change = {r["driver"]: r["value"] for r in facts if r.get("metric") == "net_position_change"}

    rows = [
        {"Driver": driver, "Team": team, "PitStops": pit_counts.get(driver, 0), "NetChange": net_change[driver]}
        for driver, team in driver_team.items() if driver in net_change
    ]
    if not rows:
        return [], None

    df = pd.DataFrame(rows)
    team_df = (df.groupby("Team")
               .agg(AvgPitStops=("PitStops", "mean"), NetPositionChange=("NetChange", "sum"))
               .reset_index())
    team_df["PositionsPerStop"] = team_df["NetPositionChange"] / team_df["AvgPitStops"].replace(0, np.nan)
    team_df = team_df.dropna(subset=["PositionsPerStop"])
    if team_df.empty:
        return [], None

    facts_out = [
        {"category": "team_strategy", "metric": "pit_stops_vs_position_change", "team": r.Team,
         "avg_pit_stops": round(r.AvgPitStops, 2), "net_position_change": int(r.NetPositionChange),
         "value": round(r.PositionsPerStop, 3), "chart_hint": "scatter"}
        for r in team_df.itertuples()
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhline(0, color="#999999", linewidth=1, zorder=1)
    ax.grid(color="#e5e5e5", linewidth=0.7, zorder=0)

    _plot_team_scatter(ax, team_df["AvgPitStops"], team_df["NetPositionChange"], team_df["Team"])

    ax.set_xlabel("Average pit stops per driver")
    ax.set_ylabel("Net position change (team total)")
    ax.set_title("Strategic Efficiency — Pit Stops vs Net Position Change, by Team",
                 loc="left", fontsize=12, fontweight="bold")
    ax.margins(x=0.18)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    chart_abs_path = os.path.join(output_dir, "charts", "team_strategy_efficiency.png")
    os.makedirs(os.path.dirname(chart_abs_path), exist_ok=True)
    fig.savefig(chart_abs_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return facts_out, chart_abs_path


def _fetch_quali_laps(year: int, event: str):
    quali_session = get_session(year, event, "Q")
    quali_session.load()  # get_session() doesn't load — only get_clean_laps() does that
    return quali_session.laps.pick_quicklaps()


def _team_quali_pace_facts_and_chart(year: int, event: str, output_dir: str,
                                      timeout_s: float = 25.0) -> tuple[list[dict], str | None]:
    """Qualifying pace vs top speed, by team — a second, optional session fetch (this is a
    *race*-weekend report; qualifying isn't loaded otherwise). Built from lap-level speed-trap
    columns (SpeedST) already on every lap, same as new_analysis.py's avg_speed_trap/
    max_speed_by_trap — no telemetry fetch needed.

    Run under a hard wall-clock timeout, not just a try/except: qualifying might not be
    fetchable (cache miss, API hiccup, no separate quali session for this event) and a plain
    except handles all of those — but a network call that hangs instead of raising (seen in
    practice: FastF1's HTTP calls have no timeout of their own, and a degraded connection can
    silently drop a request instead of erroring) would never hit that except at all, and this
    "optional exhibit" would stall the entire report indefinitely. The fetch runs in a plain
    daemon thread (not a ThreadPoolExecutor — its context manager's shutdown(wait=True) would
    block on exactly the hung thread we're trying to time out) so a stuck fetch can never block
    the pipeline past `timeout_s`, nor the process from exiting even if the thread never
    returns at all."""
    result: dict = {}

    def _worker():
        try:
            result["laps"] = _fetch_quali_laps(year, event)
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=_worker, daemon=True, name="quali-fetch")
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive() or "laps" not in result:
        return [], None  # timed out, or the fetch itself raised — either way, skip this exhibit

    quali_laps = result["laps"]
    if quali_laps is None or quali_laps.empty:
        return [], None

    quali_laps = quali_laps.copy()
    quali_laps["LapTimeSeconds"] = quali_laps["LapTime"].dt.total_seconds()
    team_df = (quali_laps.dropna(subset=["SpeedST"])
               .groupby("Team")
               .agg(MedianLaptime=("LapTimeSeconds", "median"), TopSpeed=("SpeedST", "max"))
               .dropna().reset_index())
    if team_df.empty:
        return [], None

    facts_out = [
        {"category": "team_quali", "metric": "quali_pace_vs_topspeed", "team": r.Team,
         "median_laptime": round(r.MedianLaptime, 3), "top_speed": round(r.TopSpeed, 1),
         "value": round(r.MedianLaptime, 3), "chart_hint": "scatter"}
        for r in team_df.itertuples()
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.grid(color="#e5e5e5", linewidth=0.7, zorder=0)
    _plot_team_scatter(ax, team_df["MedianLaptime"], team_df["TopSpeed"], team_df["Team"])
    ax.invert_xaxis()  # faster (lower) lap time reads to the right — "further right = better" on both axes
    ax.set_xlabel("Median qualifying lap time, s (further right is faster)")
    ax.set_ylabel("Top speed — SpeedST trap (km/h)")
    ax.set_title("Qualifying Pace vs Top Speed, by Team", loc="left", fontsize=12, fontweight="bold")
    ax.margins(x=0.18)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    chart_abs_path = os.path.join(output_dir, "charts", "team_quali_pace_speed.png")
    os.makedirs(os.path.dirname(chart_abs_path), exist_ok=True)
    fig.savefig(chart_abs_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return facts_out, chart_abs_path


def _plot_team_scatter(ax, x_vals, y_vals, labels, color: str = "#2a78d6") -> None:
    """Shared by every team-level scatter chart in this module: plots the points, numbers them,
    and lists the numbered names in a legend beside the chart rather than as inline labels.

    Inline labels went through two failed rounds first — teams cluster tightly (close lap
    times, close speeds) *and* team names are long ("Red Bull Racing"), so no data-space
    collision heuristic (exact-match grouping, then proximity binning) reliably kept long
    labels apart at small real separations; text width just doesn't scale with the data's
    numeric span. A numbered legend sidesteps that entirely — but two teams can still land on
    the *exact* same point (same avg pit stops and same net position change is common, being
    small integers), which would stack their number markers directly on each other instead of
    just their labels. Exact-duplicate points are nudged apart in a small circle before
    plotting so every marker — and its number — stays independently visible; this is a purely
    visual spread for legibility, not a data change (the legend/facts still carry the real
    values)."""
    x_vals, y_vals, labels = list(x_vals), list(y_vals), list(labels)
    x_span = (max(x_vals) - min(x_vals)) or 1
    y_span = (max(y_vals) - min(y_vals)) or 1

    groups: dict[tuple[float, float], list[int]] = {}
    for i, (x, y) in enumerate(zip(x_vals, y_vals)):
        groups.setdefault((x, y), []).append(i)
    x_plot, y_plot = list(x_vals), list(y_vals)
    for (x, y), idxs in groups.items():
        n = len(idxs)
        if n == 1:
            continue
        for k, i in enumerate(idxs):
            angle = 2 * math.pi * k / n
            x_plot[i] = x + 0.045 * x_span * math.cos(angle)
            y_plot[i] = y + 0.045 * y_span * math.sin(angle)

    ax.scatter(x_plot, y_plot, s=220, color=color, zorder=3)

    order = sorted(range(len(labels)), key=lambda i: labels[i])
    rank_of = {i: rank for rank, i in enumerate(order, start=1)}
    for i, (x, y) in enumerate(zip(x_plot, y_plot)):
        ax.annotate(str(rank_of[i]), (x, y), ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white", zorder=4)

    legend_lines = "\n".join(f"{rank_of[i]}.  {labels[i]}" for i in order)
    ax.text(1.04, 0.98, legend_lines, transform=ax.transAxes, fontsize=9.5,
            va="top", ha="left", linespacing=1.7)


def _chart_data_uri(manifest: dict, chart_key: str | None) -> str | None:
    # manifest values are absolute paths (see run_all_analysis.run_all) — no join against a
    # shared base dir, so a manifest from one session's output_dir can never be misread
    # against another session's directory
    if not chart_key or chart_key not in manifest:
        return None
    path = manifest[chart_key]
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _mini_stat_chart(storyline: dict) -> str | None:
    """A text-forward stat card built fresh from THIS storyline's own supporting facts — used
    whenever the catalog's field-wide chart for this metric has already been claimed by an
    earlier, higher-scored storyline (or there wasn't a safe one to begin with, e.g.
    tyre_cliff). Deliberately NOT another bar chart: every ranked/diverging catalog chart in
    this report already looks like a bar chart, so a fallback bar chart reads as "the same
    chart again" even with different numbers in it. A stat card is visually unmistakable from
    any of them, on top of showing a distinct image each time."""
    facts = [f for f in storyline.get("supporting_facts") or [] if isinstance(f.get("value"), (int, float))]
    if not facts:
        return None

    color = {"achievement": "#1a7a43", "mistake": "#a8501a", "insight": "#6b3fa0"}.get(storyline["angle"], "#2a78d6")
    fig, ax = plt.subplots(figsize=(6.4, 0.6 * len(facts) + 1.8))
    ax.axis("off")

    ax.text(0.0, 0.94, storyline["subject"], fontsize=15, fontweight="bold",
            color="#14171a", transform=ax.transAxes, va="top")

    for i, f in enumerate(facts):
        if "sector" in f:
            label = f"Sector {f['sector']}"
        elif "stint" in f and "compound" in f:
            label = f"Stint {int(f['stint'])} · {f['compound'].title()}"
        elif "lap_number" in f:
            label = f"Lap {f['lap_number']}"
        else:
            label = f.get("metric", "").replace("_", " ")
        y = 0.68 - i * 0.30
        ax.text(0.0, y, f"{f['value']:g}", fontsize=26, fontweight="bold",
                color=color, transform=ax.transAxes, va="top")
        ax.text(0.0, y - 0.15, label, fontsize=10, color="#767f86",
                transform=ax.transAxes, va="top")
    fig.tight_layout()

    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# ---------------------------------------------------------------------------
# HTML rendering — a single self-contained file, same newspaper register as
# report_suite/generate_report.py's CSS (copied here rather than imported,
# since that module currently fails to import — see docs/analysis_catalog.md
# audit notes).
# ---------------------------------------------------------------------------
CSS = """
:root {
  --paper: #e8eaed; --ink: #14171a; --ink-secondary: #454e55; --ink-muted: #767f86;
  --rule: #c3c8ce; --accent: #6b3fa0; --accent-soft: #ece4f5;
  --good: #1a7a43; --good-soft: #e1f2e8; --bad: #a8501a; --bad-soft: #fae3d3;
  --warn: #b8860b; --warn-soft: #fdf3d6;
  --card: #fcfcfb; --card-border: rgba(20,23,26,0.14);
  --shadow: 0 1px 2px rgba(20,23,26,0.06), 0 6px 16px rgba(20,23,26,0.08);
  --sans: -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, "Cascadia Code", "SF Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper: #0b0d10; --ink: #f3f4f5; --ink-secondary: #a9b1b7; --ink-muted: #767f86;
    --rule: #2b3036; --accent: #b58af0; --accent-soft: #241a30;
    --good: #4ade80; --good-soft: #10301f; --bad: #f0a459; --bad-soft: #3a2413;
    --warn: #e0b84d; --warn-soft: #3a3012;
    --card: #15181c; --card-border: rgba(255,255,255,0.14);
  }
}
:root[data-theme="dark"] {
  --paper: #0b0d10; --ink: #f3f4f5; --ink-secondary: #a9b1b7; --ink-muted: #767f86;
  --rule: #2b3036; --accent: #b58af0; --accent-soft: #241a30;
  --good: #4ade80; --good-soft: #10301f; --bad: #f0a459; --bad-soft: #3a2413;
  --warn: #e0b84d; --warn-soft: #3a3012;
  --card: #15181c; --card-border: rgba(255,255,255,0.14);
}
* { box-sizing: border-box; }
html { color-scheme: light dark; }
body { margin: 0; background: var(--paper); color: var(--ink); font-family: var(--sans); -webkit-font-smoothing: antialiased; }
.masthead { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; flex-wrap: wrap; padding: 18px 24px; border-bottom: 2px solid var(--ink); }
.wordmark { font-style: italic; font-weight: 800; font-size: 1.3rem; letter-spacing: -.01em; text-transform: uppercase; }
.wordmark span { color: var(--accent); }
.session-readout { font-family: var(--mono); font-size: .76rem; letter-spacing: .06em; color: var(--ink-secondary); text-transform: uppercase; }
.hero { max-width: 760px; margin: 0 auto; padding: 44px 24px 8px; }
.eyebrow { font-family: var(--mono); font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: var(--accent); font-weight: 600; }
.headline { font-style: italic; font-weight: 800; font-size: clamp(2rem, 5vw, 2.9rem); line-height: 1.05; letter-spacing: -.015em; margin: 12px 0 10px; text-wrap: balance; }
.dek { font-size: 1.06rem; color: var(--ink-secondary); max-width: 58ch; line-height: 1.5; margin: 0 0 28px; }
.story { max-width: 760px; margin: 0 auto; padding: 0 24px 48px; }
.story-card { background: var(--card); border: 1px solid var(--card-border); border-radius: 10px; box-shadow: var(--shadow); padding: 24px 26px; margin-bottom: 26px; }
.badge { display: inline-block; font-family: var(--mono); font-size: .68rem; letter-spacing: .08em; text-transform: uppercase; font-weight: 700; padding: 3px 9px; border-radius: 999px; margin-bottom: 10px; }
.badge.achievement { color: var(--good); background: var(--good-soft); }
.badge.mistake { color: var(--bad); background: var(--bad-soft); }
.badge.insight { color: var(--accent); background: var(--accent-soft); }
.story-title { font-size: 1.3rem; font-weight: 800; margin: 0 0 10px; letter-spacing: -.01em; }
.story-text { font-size: 1.02rem; line-height: 1.6; color: var(--ink); margin: 0 0 14px; }
.story-chart { width: 100%; border-radius: 6px; border: 1px solid var(--card-border); margin-top: 6px; }
.flag { background: var(--warn-soft); color: var(--warn); border: 1px dashed var(--warn); border-radius: 6px; padding: 10px 12px; font-size: .84rem; font-family: var(--mono); margin-top: 10px; }
.stats-table { width: 100%; border-collapse: collapse; font-size: .92rem; margin-top: 8px; }
.stats-table th, .stats-table td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--rule); }
.stats-table th { font-family: var(--mono); font-size: .7rem; letter-spacing: .06em; text-transform: uppercase; color: var(--ink-muted); }
.footer { max-width: 760px; margin: 0 auto; padding: 0 24px 48px; font-size: .8rem; color: var(--ink-muted); border-top: 1px solid var(--rule); padding-top: 16px; }
"""


def _story_section(storyline: dict, manifest: dict, claimed_charts: set[str]) -> str:
    # only ever called with storylines that passed verification — see build_html — so
    # there is no flagged/unverified-content case to render here at all
    angle_class = storyline["angle"]  # "achievement" | "mistake" | "insight" — same strings as the CSS classes

    chart_key = storyline.get("chart_key")
    chart_uri = None
    if chart_key and chart_key not in claimed_charts:
        chart_uri = _chart_data_uri(manifest, chart_key)
        if chart_uri:
            claimed_charts.add(chart_key)
    if chart_uri is None:
        # either there was no safe catalog chart, or another storyline already used it —
        # build a one-off chart from this storyline's own facts instead of reusing an image
        chart_uri = _mini_stat_chart(storyline)
    chart_html = f'<img class="story-chart" src="{chart_uri}" alt="{esc(storyline["title"])}">' if chart_uri else ""

    return f"""
<div class="story-card">
  <span class="badge {angle_class}">{storyline["angle"]}</span>
  <h2 class="story-title">{esc(storyline["title"])}</h2>
  <p class="story-text">{esc(storyline["narrative"])}</p>
  {chart_html}
</div>"""


def build_html(session_label: str, headline: dict, storylines: list[dict], manifest: dict) -> str:
    """`storylines` may include ones that failed verification — only the ones that passed are
    ever rendered into the page body; this function does the filtering so every caller gets
    the same "never publish unverified content" guarantee for free."""
    published = [s for s in storylines if s["verification"]["passed"]]
    excluded = len(storylines) - len(published)
    claimed_charts: set[str] = set()
    sections = "\n".join(_story_section(s, manifest, claimed_charts) for s in published)
    excluded_note = (
        f" {excluded} storyline(s) were mined but excluded after failing the reliability "
        f"check (not shown — see report_suite/weekend_story_history.json for the full draft "
        f"history)." if excluded else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(headline.get("headline", "Race Weekend Story"))}</title>
<style>{CSS}</style>
</head>
<body>
<div class="masthead">
  <div class="wordmark">Pit<span>Brain</span></div>
  <div class="session-readout">{esc(session_label)}</div>
</div>
<div class="hero">
  <div class="eyebrow">Weekend Retrospective</div>
  <h1 class="headline">{esc(headline.get("headline", "Race Weekend Story"))}</h1>
  <p class="dek">{esc(headline.get("dek", ""))}</p>
</div>
<div class="story">
{sections}
</div>
<div class="footer">
  DATA &mdash; FastF1 ({esc(session_label)}), clean laps only, no pit in/out.<br>
  {len(published)}/{len(storylines)} storylines passed the reliability layer and are shown above.{excluded_note}
</div>
</body>
</html>"""


def build_storylines(year: int, event: str, session_type: str = "R", top_n: int = 8,
                      force_recompute: bool = False) -> dict:
    """The analysis half only: fetch, facts, storyline mining. Never raises — any fetch or
    compute failure comes back as {"error": "..."} so callers (the LangGraph `analysis` node
    in particular) have one uniform failure contract to check instead of a try/except per
    caller. On success, storylines carry `subject_full_name` but no `narrative` yet — drafting
    and verifying is a separate step (see report_suite/narrative.py), deliberately kept out of
    this function so a graph can insert a reliability gate in between.
    """
    try:
        session = get_session(year, event, session_type)
        clean_laps = get_clean_laps(session)
        output_dir = _session_output_dir(year, event, session_type)
        facts, manifest = _load_or_build_facts(clean_laps, output_dir, force=force_recompute)
        strategy_facts, strategy_chart_abs = _team_strategy_facts_and_chart(facts, session, clean_laps, output_dir)
        if strategy_facts:
            facts = facts + strategy_facts
            manifest = {**manifest, "team_strategy_efficiency": strategy_chart_abs}

        # TEMPORARILY DISABLED: this second session fetch was still stalling the whole
        # pipeline even with the thread+timeout guard in _team_quali_pace_facts_and_chart
        # (see its docstring) — the daemon-thread join itself wasn't enough to unblock
        # things in practice, root cause not yet confirmed. Re-enable once that's
        # understood; the function below is untouched and ready to use.
        # if session_type == "R":
        #     quali_facts, quali_chart_abs = _team_quali_pace_facts_and_chart(year, event, output_dir)
        #     if quali_facts:
        #         facts = facts + quali_facts
        #         manifest = {**manifest, "team_quali_pace_speed": quali_chart_abs}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    storylines = select_storylines(facts, top_n=top_n)
    if not storylines:
        return {"error": "No storylines found in the analysis facts for this session."}

    driver_names = _driver_names(session)
    for s in storylines:
        # team-subject storylines (team_pace_leader, team_balance) aren't in driver_names —
        # the subject itself (e.g. "Red Bull Racing") already is its own full name
        s["subject_full_name"] = driver_names.get(s["subject"], s["subject"])

    return {
        "session_label": f"{session.event['EventName']} {session.event.year} — {session.name}",
        "storylines": storylines,
        "manifest": manifest,
        "driver_names": driver_names,
    }


def render_and_publish(event: str, year: int, session_label: str, headline: dict,
                        storylines: list[dict], manifest: dict) -> str:
    """The publish half only: render the HTML and write it to reports/. Assumes every
    storyline already has `narrative`/`verification`/`attempts` set (narrate_and_publish, or a
    graph's own narration+reliability nodes, does that first)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    slug = f"{event.lower().replace(' ', '_')}_{year}_weekend_story.html"
    out_path = os.path.join(OUTPUT_DIR, slug)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(build_html(session_label, headline, storylines, manifest))
    return out_path


def narrate_and_publish(year: int, event: str, bundle: dict) -> dict:
    """The reporting half only: narrate+verify every storyline in `bundle` (as returned by
    build_storylines on success — never pass it an {"error": ...} result), draft a headline,
    render, log, and return a summary. Split out of generate() so a caller (or a graph) can
    run build_storylines() and gate on its result before paying for any LLM calls at all."""
    storylines = bundle["storylines"]
    driver_names = bundle["driver_names"]

    for s in storylines:
        result = narrate_and_verify(s, driver_names=driver_names)
        s["narrative"] = result["narrative"]
        s["verification"] = result["verification"]
        s["attempts"] = result["attempts"]

    # headline is drafted from only the storylines that will actually be published —
    # it shouldn't frame the weekend around something that got excluded
    published = [s for s in storylines if s["verification"]["passed"]]
    headline = narrate_headline(published)
    log_run(bundle["session_label"], headline, storylines)
    out_path = render_and_publish(event, year, bundle["session_label"], headline,
                                   storylines, bundle["manifest"])

    return {"report_path": out_path, "session_label": bundle["session_label"],
            "headline": headline, "storylines": storylines}


def generate(year: int, event: str, session_type: str = "R", top_n: int = 8, force_recompute: bool = False) -> str:
    """Backward-compatible convenience wrapper: build_storylines() then narrate_and_publish()
    in sequence, no gating in between. Used by the CLI below; the graph in supervisor/graph.py
    calls the two halves separately instead, with a reliability node in between."""
    bundle = build_storylines(year, event, session_type, top_n, force_recompute)
    if "error" in bundle:
        raise RuntimeError(bundle["error"])
    return narrate_and_publish(year, event, bundle)["report_path"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a PitBrain weekend-story report.")
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--event", type=str, default="Monza")
    parser.add_argument("--session", type=str, default="R", help="Session type, e.g. R, Q")
    parser.add_argument("--top-n", type=int, default=6)
    parser.add_argument("--force-recompute", action="store_true",
                         help="Re-run the full analysis catalog even if cached output exists.")
    args = parser.parse_args()

    path = generate(args.year, args.event, args.session, args.top_n, args.force_recompute)
    print(f"\nWeekend story written to {path}")
