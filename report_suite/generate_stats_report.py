"""Publishes a real report package for the pace_analysis and driver_comparison routes.

Before this, those two routes out of the supervisor graph just returned their raw analysis
dict as JSON — not a report, a data dump. This module assembles an actual small HTML report
from that same analysis output (same CSS/visual register as the weekend-story report), so
every route the supervisor can take produces a published report, not a placeholder.
"""
from __future__ import annotations

import os
from html import escape as esc

import pandas as pd

from report_suite.generate_weekend_story import CSS, OUTPUT_DIR


def _page(title: str, session_label: str, body_html: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="masthead">
  <div class="wordmark">Pit<span>Brain</span></div>
  <div class="session-readout">{esc(session_label)}</div>
</div>
<div class="hero">
  <div class="eyebrow">Stats Report</div>
  <h1 class="headline">{esc(title)}</h1>
</div>
<div class="story">
{body_html}
</div>
<div class="footer">DATA &mdash; FastF1 ({esc(session_label)}), clean laps only, no pit in/out.</div>
</body>
</html>"""


def _write(html: str, slug: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{slug}_stats_report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


def _ranked_table(df: pd.DataFrame, value_col: str, label: str, precision: int = 3) -> str:
    df = df.sort_values(value_col)
    rows = "".join(
        f"<tr><td>{esc(str(r['Driver']))}</td><td>{r[value_col]:.{precision}f}</td></tr>"
        for _, r in df.iterrows()
    )
    return (f"<table class='stats-table'><thead><tr><th>Driver</th><th>{esc(label)}</th></tr>"
            f"</thead><tbody>{rows}</tbody></table>")


def render_pace_report(session_label: str, analysis: dict) -> str:
    fastest = analysis["fastest_lap"]
    median = analysis["median_pace"]

    fastest_card = (
        "<div class='story-card'><span class='badge achievement'>fastest lap</span>"
        f"<h2 class='story-title'>{esc(str(fastest['Driver']))} — {fastest['LapTimeSeconds']:.3f}s</h2>"
        f"<p class='story-text'>Lap {int(fastest['LapNumber'])}, "
        f"{esc(str(fastest.get('Compound', 'unknown')))} tyre.</p></div>"
    )
    pace_card = (
        "<div class='story-card'><h2 class='story-title'>Median Race Pace (all drivers)</h2>"
        f"{_ranked_table(median, 'LapTimeSeconds', 'Median lap (s)')}</div>"
    )
    html = _page("Race Pace Report", session_label, fastest_card + pace_card)
    return _write(html, "pace_report")


def render_comparison_report(session_label: str, analysis: dict) -> str:
    summary = analysis["head_to_head_summary"]
    a, b = summary["driver_a"], summary["driver_b"]
    gap = summary["final_gap_seconds"]
    gap_text = f"{gap:+.3f}s" if gap is not None else "n/a"

    h2h_card = (
        "<div class='story-card'><span class='badge achievement'>head-to-head</span>"
        f"<h2 class='story-title'>{esc(a)} vs {esc(b)}</h2>"
        f"<p class='story-text'>{esc(a)} was faster on {summary['laps_a_faster']} laps, "
        f"{esc(b)} on {summary['laps_b_faster']}, out of {summary['laps_compared']} laps both "
        f"completed. Final cumulative gap ({esc(a)} − {esc(b)}): {gap_text}.</p></div>"
    )
    pace_card = (
        "<div class='story-card'><h2 class='story-title'>Median Pace</h2>"
        f"{_ranked_table(analysis['median_pace'], 'LapTimeSeconds', 'Median lap (s)')}</div>"
    )
    html = _page(f"{a} vs {b} — Head to Head", session_label, h2h_card + pace_card)
    return _write(html, f"{a.lower()}_{b.lower()}_comparison")


def render_stats_report(route: str, session_label: str, analysis: dict) -> str:
    if route == "pace_analysis":
        return render_pace_report(session_label, analysis)
    if route == "driver_comparison":
        return render_comparison_report(session_label, analysis)
    raise ValueError(f"render_stats_report doesn't handle route {route!r}")
