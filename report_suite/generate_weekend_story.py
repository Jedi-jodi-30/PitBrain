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
import os
import sys
from html import escape as esc

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "analysis_suite"))

from analysis_suite.basic_imports import get_session, get_clean_laps
from analysis_suite.storylines import select_storylines, ACHIEVEMENT
import analysis_suite.run_all_analysis as run_all_analysis
from report_suite.narrative import narrate_and_verify, narrate_headline, log_run

OUTPUT_DIR = os.path.join(_REPO_ROOT, "reports")
ANALYSIS_OUTPUT_DIR = os.path.join(_REPO_ROOT, "analysis_output")
FACTS_PATH = os.path.join(ANALYSIS_OUTPUT_DIR, "facts.json")
MANIFEST_PATH = os.path.join(ANALYSIS_OUTPUT_DIR, "charts_manifest.json")


def _driver_names(session) -> dict[str, str]:
    names = {}
    for code in session.drivers:
        try:
            names[code] = session.get_driver(code)["FullName"]
        except Exception:
            continue
    return names


def _load_or_build_facts(clean_laps, force: bool = False) -> tuple[list[dict], dict[str, str]]:
    if force or not (os.path.exists(FACTS_PATH) and os.path.exists(MANIFEST_PATH)):
        run_all_analysis.run_all(clean_laps)
    with open(FACTS_PATH, "r", encoding="utf-8") as f:
        facts = json.load(f)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return facts, manifest


def _chart_data_uri(manifest: dict, chart_key: str | None) -> str | None:
    if not chart_key or chart_key not in manifest:
        return None
    path = os.path.join(ANALYSIS_OUTPUT_DIR, manifest[chart_key])
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
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
.story-title { font-size: 1.3rem; font-weight: 800; margin: 0 0 10px; letter-spacing: -.01em; }
.story-text { font-size: 1.02rem; line-height: 1.6; color: var(--ink); margin: 0 0 14px; }
.story-chart { width: 100%; border-radius: 6px; border: 1px solid var(--card-border); margin-top: 6px; }
.flag { background: var(--warn-soft); color: var(--warn); border: 1px dashed var(--warn); border-radius: 6px; padding: 10px 12px; font-size: .84rem; font-family: var(--mono); margin-top: 10px; }
.footer { max-width: 760px; margin: 0 auto; padding: 0 24px 48px; font-size: .8rem; color: var(--ink-muted); border-top: 1px solid var(--rule); padding-top: 16px; }
"""


def _story_section(storyline: dict, manifest: dict) -> str:
    angle_class = "achievement" if storyline["angle"] == ACHIEVEMENT else "mistake"
    chart_uri = _chart_data_uri(manifest, storyline.get("chart_key"))
    chart_html = f'<img class="story-chart" src="{chart_uri}" alt="{esc(storyline["title"])}">' if chart_uri else ""

    flag_html = ""
    v = storyline["verification"]
    if not v["passed"]:
        reasons = []
        if v["unverified_numbers"]:
            reasons.append(f"unverified numbers: {', '.join(v['unverified_numbers'])}")
        if not v["subject_mentioned"]:
            reasons.append(f"subject '{storyline['subject']}' not clearly named in the text")
        flag_html = (
            f'<div class="flag">&#9888; Needs review after {storyline["attempts"]} draft(s) — '
            f'{esc("; ".join(reasons))}. Supporting facts: '
            f'{esc(json.dumps(storyline["supporting_facts"]))}</div>'
        )

    return f"""
<div class="story-card">
  <span class="badge {angle_class}">{storyline["angle"]}</span>
  <h2 class="story-title">{esc(storyline["title"])}</h2>
  <p class="story-text">{esc(storyline["narrative"])}</p>
  {chart_html}
  {flag_html}
</div>"""


def build_html(session_label: str, headline: dict, storylines: list[dict], manifest: dict) -> str:
    sections = "\n".join(_story_section(s, manifest) for s in storylines)
    passed = sum(1 for s in storylines if s["verification"]["passed"])
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
  {passed}/{len(storylines)} storylines passed the reliability layer on first or second draft; flagged ones are marked above with their raw supporting facts.
</div>
</body>
</html>"""


def generate(year: int, event: str, session_type: str = "R", top_n: int = 6, force_recompute: bool = False) -> str:
    session = get_session(year, event, session_type)
    clean_laps = get_clean_laps(session)
    facts, manifest = _load_or_build_facts(clean_laps, force=force_recompute)

    storylines = select_storylines(facts, top_n=top_n)
    driver_names = _driver_names(session)
    for s in storylines:
        s["subject_full_name"] = driver_names.get(s["subject"])

    for s in storylines:
        result = narrate_and_verify(s, driver_names=driver_names)
        s["narrative"] = result["narrative"]
        s["verification"] = result["verification"]
        s["attempts"] = result["attempts"]

    headline = narrate_headline(storylines)

    session_label = f"{session.event['EventName']} {session.event.year} — {session.name}"
    log_run(session_label, headline, storylines)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    slug = f"{event.lower().replace(' ', '_')}_{year}_weekend_story.html"
    out_path = os.path.join(OUTPUT_DIR, slug)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(build_html(session_label, headline, storylines, manifest))

    return out_path


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
