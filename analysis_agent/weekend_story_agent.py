"""Tool-calling agent for the "race_weekend_rundown" intent.

Mirrors analysis_agent.py's pattern (bind a real function as an Ollama tool, let the model
pick it and fill in the arguments from the parsed intent) but wired to the storyline-driven
weekend report pipeline instead of the pace-analysis one: fetch -> deterministic facts ->
storyline mining -> narrate + verify -> publish HTML. See report_suite/generate_weekend_story.py
for that pipeline itself; this module is only the tool-calling wrapper around it.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ollama import ChatResponse, chat
from intent_paser_agent.intent_paser_agent import parse_user_query
from report_suite.generate_weekend_story import generate as generate_weekend_story


def generate_weekend_rundown(year: int, event: str, session_type: str = "R") -> dict:
    """
    Generate and publish a narrative race-weekend rundown report: fetches the session,
    runs the full deterministic analysis catalog, mines it for the weekend's standout
    achievements and mistakes, narrates each one with a local model, fact-checks every
    narrated number against the source data, and writes a self-contained HTML report.

    Parameters:
    year (int): The year of the session.
    event (str): The event name (e.g., 'Monza').
    session_type (str): The session type, e.g. 'R' for race, 'Q' for qualifying. Defaults to 'R'.

    Returns:
    dict: A summary of the published report — path, headline, and each storyline's title,
    angle, and whether it passed the reliability check.
    """
    # Tool-call arguments come back as whatever the model actually sent — coerce
    # defensively rather than trust it (same reasoning as data_fetcher.get_clean_laps).
    year = int(year)
    session_type = session_type or "R"

    report_path = generate_weekend_story(year, event, session_type)

    history_path = os.path.join(os.path.dirname(__file__), "..", "report_suite", "weekend_story_history.json")
    with open(history_path, "r", encoding="utf-8") as f:
        last_run = json.load(f)[-1]

    return {
        "report_path": os.path.abspath(report_path),
        "session": last_run["session"],
        "headline": last_run["headline"].get("headline"),
        "dek": last_run["headline"].get("dek"),
        "storylines": [
            {"title": s["title"], "angle": s["angle"], "verified": s["verification"]["passed"]}
            for s in last_run["storylines"]
        ],
    }


available_tools = {"generate_weekend_rundown": generate_weekend_rundown}

prompt = """You are an analysis agent. You will be given a structured intent dict describing
what the user asked for, with keys "year", "event", "session_type", "analysis_type", "drivers".
When analysis_type is "race_weekend_rundown", call generate_weekend_rundown with year, event,
and session_type taken directly from the intent (default session_type to "R" if it is null).
Do not call any tool for any other analysis_type."""


def run(intent: dict) -> dict:
    """
    Run the weekend-story tool-calling agent on an already-parsed intent.

    Parameters:
    intent (dict): Structured intent from intent_paser_agent.parse_user_query.

    Returns:
    dict: The tool's result (see generate_weekend_rundown), or an error dict.
    """
    if intent.get("analysis_type") != "race_weekend_rundown":
        return {"error": f"weekend_story_agent only handles 'race_weekend_rundown', got "
                          f"{intent.get('analysis_type')!r}."}
    if not all([intent.get("year"), intent.get("event")]):
        return {"error": "Missing required parameters (year, event) in the intent."}

    chat_response: ChatResponse = chat(
        "llama3.2:3b",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(intent)},
        ],
        tools=[generate_weekend_rundown],
    )

    if not chat_response.message.tool_calls:
        return {"error": "Model did not call the weekend-rundown tool."}

    tool_call = chat_response.message.tool_calls[0]
    function_to_call = available_tools.get(tool_call.function.name)
    if function_to_call is None:
        return {"error": f"Unknown tool requested: {tool_call.function.name}"}

    return function_to_call(**tool_call.function.arguments)


if __name__ == "__main__":
    intent = parse_user_query("Give me a rundown of the Monza race 2023")
    print("Parsed intent:", intent)
    result = run(intent)
    print(json.dumps(result, indent=2))
