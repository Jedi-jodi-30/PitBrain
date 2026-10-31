"""
From users' query need to extract the year, event, and session type (e.g., "2023 Hungary Grand Prix race session")
and return them in a structured format. 
The output should be a dictionary with keys "year", "event", and "session_type". If any of these elements are missing or cannot be determined, 
the corresponding value should be None.
And also get what type of analysis the user wants to perform like global pace analysis or comparison between two drivers and return it in a structured format.
The output should be a dictionary with keys "analysis_type" and "drivers" (if applicable
"""

import json

from ollama import ChatResponse, chat

prompt = """
You are an intent parser agent. Your task is to extract structured information from the user's query regarding Formula 1 race sessions and analysis requests.
The user may ask for:
- race pace analysis ("global_pace_analysis") — raw pace numbers/statistics for the field
- a comparison between two drivers ("driver_pace_comparison_analysis")
- a narrative rundown / recap / summary / "what happened" story of a race weekend
  ("race_weekend_rundown") — phrases like "give me a rundown of...", "recap...",
  "summarize the race...", "what happened at...", "tell me the story of..."
Output the extracted information in a structured format as a dictionary with the following keys:
- "year": The year of the session (int or None if not found).
- "event": The event name (str or None if not found).
- "session_type": The session type (str or None if not found, e.g., "R" for race, "Q" for qualifying, "P" for practice). Default to "R" for a rundown/recap request unless the user clearly asks about qualifying or practice.
- "analysis_type": The type of analysis requested (str, one of "global_pace_analysis", "driver_pace_comparison_analysis", "race_weekend_rundown", or None if not found).
- "drivers": Only when analysis_type is "driver_pace_comparison_analysis": a list of the driver names or codes the user named for comparison (e.g. ["Verstappen", "Piastri"] or ["VER", "PIA"]). Otherwise an empty list.
"""

# Constrains the model to emit exactly this shape as its entire response —
# no prose, no markdown fence — so the output is always valid JSON rather
# than something we have to hope the model formatted correctly.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "year": {"type": ["integer", "null"]},
        "event": {"type": ["string", "null"]},
        "session_type": {"type": ["string", "null"]},
        "analysis_type": {
            "type": ["string", "null"],
            "enum": ["global_pace_analysis", "driver_pace_comparison_analysis",
                     "race_weekend_rundown", None],
        },
        "drivers": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["year", "event", "session_type", "analysis_type", "drivers"],
}

def parse_user_query(user_query: str) -> dict:
    """
    Parse the user's query to extract structured information.

    Parameters:
    user_query (str): The user's query string.

    Returns:
    dict: A dictionary containing the extracted information.
    """
    messages = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': user_query}
    ]

    response: ChatResponse = chat(
        'llama3.2:3b',
        messages=messages,
        format=RESPONSE_SCHEMA,
    )

    try:
        structured_output = json.loads(response.message.content)
        if not isinstance(structured_output, dict):
            raise ValueError("Parsed output is not a dictionary.")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing model output: {e}")
        structured_output = {
            "year": None,
            "event": None,
            "session_type": None,
            "analysis_type": None,
            "drivers": [],
        }

    return structured_output


if __name__ == "__main__":
    intent = parse_user_query(" Do pace analysis for the 2026 Hungary Grand Prix race session between Verstappen and Piastri")
    print(intent)