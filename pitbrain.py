"""PitBrain entry point: one plain-English query in, a supervised, tool-calling analysis out.

Routing used to be done here by hand against a rigid, enum-constrained intent parser. It's now
a real LangGraph StateGraph (supervisor/graph.py) with its own LLM-driven routing agent (reasons
about the query instead of matching a fixed analysis_type string) and a reliability agent that
gates both analysis failures and narration-number accuracy before anything gets reported. See
supervisor/graph.py for the graph itself.

Usage:
    python pitbrain.py "give me a rundown of the monza race 2023"
"""
import json
import sys

from analysis_agent.analysis_agent import _json_default
from supervisor.graph import run_query

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python pitbrain.py "<your query>"')
        sys.exit(1)

    result = run_query(" ".join(sys.argv[1:]))
    print(json.dumps(result, indent=2, default=_json_default))

    if "report_path" in result:
        print(f"\nReport published: {result['report_path']}")
    elif "error" in result:
        print(f"\nBlocked: {result['error']}")
