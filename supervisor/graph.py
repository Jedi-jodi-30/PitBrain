"""The Supervision Agent, as an actual LangGraph StateGraph.

    query -> supervisor (LLM tool-calling: reasons about the query, picks ONE capability
                          as a tool call instead of matching a rigid intent enum)
          -> analysis (runs the chosen capability's deterministic half; any fetch/compute
                        failure comes back as {"error": ...}, never an exception)
          -> narration (drafts prose per storyline — weekend-rundown route only)
          -> reliability (ONE agent, two checks: analysis-level error/empty gate, which
                           blocks the whole run; and per-storyline narration-number checks,
                           which loop back to narration for a redraft, then fall through as a
                           visibly flagged storyline rather than blocking everything)
          -> report (publish) | blocked (no report written)

The old intent_paser_agent.parse_user_query is demoted from hard router to an optional helper
nothing here actually calls yet — the supervisor's own tool choice is what decides routing,
which is the point: it reasons about phrasing ("complete race analysis", "what happened this
weekend", ...) instead of needing to match a fixed analysis_type string.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import TypedDict

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _REPO_ROOT)

from ollama import ChatResponse, chat
from langgraph.graph import StateGraph, END

from analysis_agent.analysis_agent import fetch_data_analysis, _json_default
from report_suite.generate_weekend_story import build_storylines, render_and_publish
from report_suite.narrative import (
    narrate_storyline, verify_storyline, narrate_headline, log_run, build_correction_feedback,
)
from report_suite.generate_stats_report import render_stats_report

MODEL = "llama3.2:3b"
MAX_NARRATION_ROUNDS = 2
RUN_LOG_PATH = os.path.join(os.path.dirname(__file__), "run_log.json")


class PitBrainState(TypedDict, total=False):
    query: str
    route: str | None
    route_args: dict
    analysis: dict
    narration_round: int
    blocked: bool
    blocked_reason: str
    result: dict


# ---------------------------------------------------------------------------
# Supervisor — routing tools. These do no real work; they just carry the
# model's chosen route + arguments back out. The *analysis* node does the
# actual fetch/compute, after routing has already been decided.
# ---------------------------------------------------------------------------
def run_weekend_rundown(year: int, event: str, session_type: str = "R") -> dict:
    """Give a narrative rundown, recap, or "what happened" story of a race weekend —
    achievements and mistakes, told as a short magazine-style feature article with
    supporting charts. Use this for any request asking for a story, summary, recap, or
    complete analysis of a weekend, not just raw numbers.

    Parameters:
    year (int): The year of the session.
    event (str): The event name, e.g. 'Monza'.
    session_type (str): 'R' for race, 'Q' for qualifying, 'P' for practice. Default 'R'.
    """
    return {"route": "weekend_rundown", "year": year, "event": event, "session_type": session_type}


def run_pace_analysis(year: int, event: str, session_type: str = "R") -> dict:
    """Give raw race-pace statistics for the whole field — median/mean lap times, fastest
    lap, pace delta to the leader. Use this only when the user wants plain numbers, not a
    narrative story.

    Parameters:
    year (int): The year of the session.
    event (str): The event name, e.g. 'Monza'.
    session_type (str): 'R' for race, 'Q' for qualifying, 'P' for practice. Default 'R'.
    """
    return {"route": "pace_analysis", "year": year, "event": event, "session_type": session_type}


def run_driver_comparison(year: int, event: str, driver_a: str, driver_b: str, session_type: str = "R") -> dict:
    """Compare the pace of exactly two named drivers head-to-head for one session.

    Parameters:
    year (int): The year of the session.
    event (str): The event name, e.g. 'Monza'.
    driver_a (str): First driver's name or 3-letter code.
    driver_b (str): Second driver's name or 3-letter code.
    session_type (str): 'R' for race, 'Q' for qualifying, 'P' for practice. Default 'R'.
    """
    return {"route": "driver_comparison", "year": year, "event": event,
            "session_type": session_type, "driver_a": driver_a, "driver_b": driver_b}


ROUTER_TOOLS = {
    "run_weekend_rundown": run_weekend_rundown,
    "run_pace_analysis": run_pace_analysis,
    "run_driver_comparison": run_driver_comparison,
}

SUPERVISOR_PROMPT = """You are the PitBrain supervision agent. A user will ask you something
about a Formula 1 session in their own words — it may not use any particular keywords. Reason
about what they actually want and call exactly ONE of the tools you're given to route the
request. Prefer run_weekend_rundown whenever the user wants a story, recap, summary,
"what happened", or a complete/full analysis of a weekend — that is the richest capability and
the right default for anything that isn't clearly asking for bare numbers or a two-driver
comparison. Fill in year/event/session_type (and driver_a/driver_b for a comparison) from the
query as best you can; default session_type to "R" (race) unless qualifying or practice is
clearly meant."""


def supervisor_node(state: PitBrainState) -> dict:
    response: ChatResponse = chat(
        MODEL,
        messages=[
            {"role": "system", "content": SUPERVISOR_PROMPT},
            {"role": "user", "content": state["query"]},
        ],
        tools=[run_weekend_rundown, run_pace_analysis, run_driver_comparison],
    )

    if not response.message.tool_calls:
        return {"route": None, "blocked": True,
                "blocked_reason": "Could not determine which analysis to run from the query."}

    call = response.message.tool_calls[0]
    fn = ROUTER_TOOLS.get(call.function.name)
    if fn is None:
        return {"route": None, "blocked": True,
                "blocked_reason": f"Supervisor picked an unknown tool: {call.function.name}"}

    routed = fn(**call.function.arguments)
    route = routed.pop("route")
    return {"route": route, "route_args": routed}


def route_after_supervisor(state: PitBrainState) -> str:
    return "blocked" if state.get("blocked") else "analysis"


# ---------------------------------------------------------------------------
# Analysis — the deterministic half of whichever capability was routed to.
# Uniform failure contract: {"error": "..."} in state["analysis"], never a
# raised exception reaching the graph.
# ---------------------------------------------------------------------------
def analysis_node(state: PitBrainState) -> dict:
    route = state.get("route")
    args = state.get("route_args", {})

    try:
        year = int(args["year"])
        event = str(args["event"])
        session_type = args.get("session_type") or "R"
    except Exception as e:
        return {"analysis": {"error": f"Invalid routed arguments {args!r}: {type(e).__name__}: {e}"}}

    if route == "weekend_rundown":
        return {"analysis": build_storylines(year, event, session_type)}

    if route in ("pace_analysis", "driver_comparison"):
        intent = {
            "year": year, "event": event, "session_type": session_type,
            "analysis_type": ("driver_pace_comparison_analysis" if route == "driver_comparison"
                               else "global_pace_analysis"),
            "drivers": [args.get("driver_a"), args.get("driver_b")] if route == "driver_comparison" else [],
        }
        try:
            result = fetch_data_analysis(intent)
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}
        return {"analysis": result}

    return {"analysis": {"error": f"Unknown route: {route!r}"}}


def route_after_analysis(state: PitBrainState) -> str:
    analysis = state.get("analysis", {})
    if state.get("route") == "weekend_rundown" and "error" not in analysis:
        return "narration"
    return "reliability"


# ---------------------------------------------------------------------------
# Narration — drafts prose per storyline (weekend-rundown route only). Skips
# any storyline that's already passed verification, so a redraft loop only
# re-spends an LLM call on the storylines that actually need it.
# ---------------------------------------------------------------------------
def narration_node(state: PitBrainState) -> dict:
    analysis = state.get("analysis", {})
    driver_names = analysis.get("driver_names", {})
    round_num = state.get("narration_round", 0) + 1
    for s in analysis.get("storylines", []):
        prior_verification = s.get("verification")
        if prior_verification and prior_verification.get("passed"):
            continue
        # a flagged storyline gets corrective feedback built from exactly what reliability
        # found wrong with it, not a blind redraft — see build_correction_feedback
        feedback = (build_correction_feedback(prior_verification, s, driver_names)
                    if prior_verification else None)
        s["narrative"] = narrate_storyline(s, feedback)
        s["attempts"] = round_num
        s.setdefault("draft_history", []).append(
            {"attempt": round_num, "narrative": s["narrative"], "correction_feedback": feedback})
    return {"analysis": analysis, "narration_round": round_num}


# ---------------------------------------------------------------------------
# Reliability — the unified agent. (a) analysis-level: did analysis error or
# come back empty? Hard-blocks the whole run. (b) narration-level: does every
# cited number in each storyline's drafted prose trace back to that
# storyline's own supporting facts? Loops back to narration for a redraft
# while attempts remain; otherwise the storyline is reported but visibly
# flagged, never silently passed off as verified.
# ---------------------------------------------------------------------------
def reliability_node(state: PitBrainState) -> dict:
    analysis = state.get("analysis")
    if not analysis or "error" in analysis:
        reason = analysis.get("error", "Empty analysis result") if analysis else "Empty analysis result"
        return {"blocked": True, "blocked_reason": reason}

    if state.get("route") != "weekend_rundown":
        return {"blocked": False}

    driver_names = analysis.get("driver_names", {})
    for s in analysis["storylines"]:
        v = verify_storyline(s["narrative"], s, driver_names)
        s["verification"] = v
        # fill in the result for the draft narration_node just logged, so the
        # flag -> correction cycle is fully visible per attempt, not just the final one
        if s.get("draft_history"):
            s["draft_history"][-1]["verification"] = v

    return {"blocked": False, "analysis": analysis}


def route_after_reliability(state: PitBrainState) -> str:
    if state.get("blocked"):
        return "blocked"
    if state.get("route") == "weekend_rundown":
        storylines = state["analysis"]["storylines"]
        still_failing = [s for s in storylines if not s["verification"]["passed"]]
        if still_failing and state.get("narration_round", 0) < MAX_NARRATION_ROUNDS:
            return "narration"
    return "report"


# ---------------------------------------------------------------------------
# Report / blocked — terminal nodes.
# ---------------------------------------------------------------------------
def report_node(state: PitBrainState) -> dict:
    route = state["route"]
    analysis = state["analysis"]
    args = state["route_args"]

    if route == "weekend_rundown":
        storylines = analysis["storylines"]
        # headline is drafted from only what will actually be published — see
        # report_suite/generate_weekend_story.py::narrate_and_publish for the same rule
        published = [s for s in storylines if s["verification"]["passed"]]
        headline = narrate_headline(published)
        log_run(analysis["session_label"], headline, storylines)
        out_path = render_and_publish(args["event"], int(args["year"]), analysis["session_label"],
                                       headline, storylines, analysis["manifest"])
        result = {
            "report_path": out_path,
            "session": analysis["session_label"],
            "headline": headline.get("headline"),
            "dek": headline.get("dek"),
            "storylines": [{"title": s["title"], "angle": s["angle"],
                             "verified": s["verification"]["passed"]} for s in published],
            "excluded_unverified": len(storylines) - len(published),
        }
    else:
        # publish an actual report package built from this run's analysis output — not a
        # raw-dict placeholder — same as the weekend-rundown route does above
        session_label = f"{args['event']} {args['year']} — {route}"
        out_path = render_stats_report(route, session_label, analysis)
        result = {k: v for k, v in analysis.items() if k != "clean_laps"}
        result["route"] = route
        result["report_path"] = out_path
        result["session"] = session_label

    return {"result": result}


def blocked_node(state: PitBrainState) -> dict:
    return {"result": {
        "error": state.get("blocked_reason", "Unknown failure"),
        "route": state.get("route"),
        "route_args": state.get("route_args"),
    }}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def _build_graph():
    graph = StateGraph(PitBrainState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("narration", narration_node)
    graph.add_node("reliability", reliability_node)
    graph.add_node("report", report_node)
    graph.add_node("blocked", blocked_node)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges("supervisor", route_after_supervisor,
                                 {"analysis": "analysis", "blocked": "blocked"})
    graph.add_conditional_edges("analysis", route_after_analysis,
                                 {"narration": "narration", "reliability": "reliability"})
    graph.add_edge("narration", "reliability")
    graph.add_conditional_edges("reliability", route_after_reliability,
                                 {"blocked": "blocked", "narration": "narration", "report": "report"})
    graph.add_edge("report", END)
    graph.add_edge("blocked", END)
    return graph.compile()


_compiled = _build_graph()


def _log(query: str, final_state: dict) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "route": final_state.get("route"),
        "blocked": bool(final_state.get("blocked")),
        "result": final_state.get("result"),
    }
    history = []
    if os.path.exists(RUN_LOG_PATH):
        with open(RUN_LOG_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)
    history.append(entry)
    with open(RUN_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=_json_default)


def run_query(query: str) -> dict:
    final_state = _compiled.invoke({"query": query})
    _log(query, final_state)
    return final_state.get("result", {"error": "No result produced."})


if __name__ == "__main__":
    import sys as _sys
    q = " ".join(_sys.argv[1:]) or "give me a rundown of monza race 2023"
    print(json.dumps(run_query(q), indent=2, default=_json_default))
