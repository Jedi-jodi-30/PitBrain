"""
Your job is fetch data from fastf1 api and return analysis results.
Intent will be given to you in a structured format as a dictionary with the following keys:
- "year": The year of the session (int or None if not found).
- "event": The event name (str or None if not found).
- "session_type": The session type (str or None if not found, e.g., "R" for race, "Q" for qualifying, "P" for practice).
- "analysis_type": The type of analysis requested (str, either "global_pace_analysis" or "driver_pace_comparison_analysis", or None if not found).
- "drivers": Only when analysis_type is "driver_pace_comparison_analysis": a list of the driver names or codes the user named for comparison (e.g. ["Verstappen", "Piastri"] or ["VER", "PIA"]). Otherwise an empty list.
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis_suite.pace_analysis import GlobalPaceAnalysis, DriverComparisonAnalysis
from analysis_agent.tools.data_fetcher import get_clean_laps, get_drivers_data_only, resolve_driver_code
from ollama import ChatResponse, chat
from intent_paser_agent.intent_paser_agent import parse_user_query


def fetch_data_analysis(intent: dict) -> dict:
    """
    Fetch data based on the provided intent.

    Parameters:
    intent (dict): A dictionary containing the structured intent.

    Returns:
    dict: A dictionary containing the fetched data and analysis results.
    """
    year = intent.get("year")
    event = intent.get("event")
    session_type = intent.get("session_type")
    analysis_type = intent.get("analysis_type")
    drivers = intent.get("drivers", [])

    if not all([year, event, session_type, analysis_type]):
        return {"error": "Missing required parameters in the intent."}

    # Fetch clean laps for the specified session
    clean_laps, session = get_clean_laps(year, event, session_type)

    if analysis_type == "global_pace_analysis":
        analysis_result = GlobalPaceAnalysis(clean_laps)
        return {
            "clean_laps": clean_laps,
            "median_pace": analysis_result.median_pace(),
            "mean_pace": analysis_result.mean_pace(),
            "rolling_pace": analysis_result.rolling_pace(),
            "fastest_lap": analysis_result.fastest_lap(),
            "fastest_lap_per_driver": analysis_result.fastest_lap_per_driver(),
            "pace_delta": analysis_result.pace_delta()
        }
    elif analysis_type == "driver_pace_comparison_analysis":
        if len(drivers) != 2:
            return {"error": "Driver comparison analysis requires exactly two drivers."}

        # The intent parser only ever extracts free-text names ("Verstappen"),
        # but everything downstream filters on the Driver column, which holds
        # 3-letter codes ("VER") — resolve before doing anything else.
        resolved = [resolve_driver_code(session, d) for d in drivers]
        if any(code is None for code in resolved):
            unresolved = [d for d, code in zip(drivers, resolved) if code is None]
            return {"error": f"Could not resolve driver(s) to a session code: {unresolved}"}
        drivers = resolved

        driver_a_data = get_drivers_data_only(clean_laps, drivers[0])
        driver_b_data = get_drivers_data_only(clean_laps, drivers[1])
        comparison_result = DriverComparisonAnalysis(clean_laps, drivers)
        return {
            "head_to_head_summary": comparison_result.head_to_head_summary(drivers[0], drivers[1]),
            "median_pace": comparison_result.median_pace(),
            "mean_pace": comparison_result.mean_pace(),
            "rolling_pace": comparison_result.rolling_pace(),
            "head_to_head": comparison_result.head_to_head(drivers[0], drivers[1]),
        }
    else:
        return {"error": f"Unknown analysis type: {analysis_type}."}

available_tools ={'fetch_data_analysis': fetch_data_analysis}

prompt = """ 
You are an analysis agent that fetches data from the fastf1 api and returns analysis results based on the user's intent. The intent will be provided to you in a structured format as a dictionary with the following keys:
- "year": The year of the session (int or None if not found).
- "event": The name of the event (str or None if not found).
- "session_type": The type of the session (str or None if not found).
- "analysis_type": The type of analysis to perform (str or None if not found).
- "drivers": A list of drivers for comparison analysis (list of str or None if not found).
Present the analysis results in a structured format as a dictionary. If any required parameters are missing or invalid, return an error message in the dictionary.
"""
def analyze_intent(intent: dict) -> dict:
    """
    Analyze the provided intent and return the analysis results.

    Parameters:
    intent (dict): A dictionary containing the structured intent.

    Returns:
    dict: A dictionary containing the analysis results or an error message.
    """
    chat_response: ChatResponse = chat(
        'llama3.2:3b',
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': json.dumps(intent)}
        ],
        tools=[fetch_data_analysis],  # bind the actual function, not its name —
        # ollama introspects the callable's signature/docstring to build the
        # tool's JSON schema; a bare string has no schema for the model to use.
    )

    if not chat_response.message.tool_calls:
        return {"error": "Model did not call the analysis tool."}

    # Only one tool exists and its result is already the final, deterministic
    # answer, so run whichever call the model made and return that directly —
    # there's nothing here that needs the model to narrate or re-interpret.
    tool_call = chat_response.message.tool_calls[0]
    function_to_call = available_tools.get(tool_call.function.name)
    if function_to_call is None:
        return {"error": f"Unknown tool requested: {tool_call.function.name}"}

    return function_to_call(**tool_call.function.arguments)

def _json_default(obj):
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return json.loads(obj.to_json(orient="records" if isinstance(obj, pd.DataFrame) else "index"))
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

intent = parse_user_query("Compare the pace of Verstappen and Piastri in the 2023 Monza race")
analysis_results = analyze_intent(intent)
print(json.dumps(analysis_results, indent=4, default=_json_default))