import fastf1
import pandas as pd
from ollama import ChatResponse, chat
import os

def get_clean_laps(year: int, event: str, session_type: str):
    """
    Get clean laps from a FastF1 session.

    Parameters:
    year (int): The year of the session.
    event (str): The event name (e.g., 'Monza').
    session_type (str): The session type (e.g., 'R' for race).

    Returns:
    pd.DataFrame: A DataFrame containing clean laps with LapTime in seconds.
    """
    # Tool-call arguments come back as whatever the model actually sent, not
    # necessarily what the schema asked for (e.g. year as "2023" instead of
    # 2023) — coerce defensively rather than trust it.
    year = int(year)
    check_cache = True  # Default to True

    if check_cache:
       #create a cache directory for fastf1 if it doesn't exist
       os.makedirs('fastf1_cache', exist_ok=True)
       fastf1.Cache.enable_cache('fastf1_cache')
    session = fastf1.get_session(year, event, session_type)
    session.load()

    laps = session.laps
    clean_laps = laps.pick_track_status('1', how='equals')
    clean_laps = clean_laps[clean_laps['PitOutTime'].isna() & clean_laps['PitInTime'].isna()]
    clean_laps['LapTimeSeconds'] = clean_laps['LapTime'].dt.total_seconds()
    
    return clean_laps

def get_drivers_data_only(global_data: pd.DataFrame, driver_name: str):
    """
    Get driver data from clean laps.

    Parameters:
    global_data (pd.DataFrame): A DataFrame containing clean laps.
    driver_name (str): The name of the driver for whom to retrieve data.

    Returns:
    pd.DataFrame: A DataFrame containing driver data.
    """
    drivers_data = global_data[global_data['Driver'] == driver_name].copy()
    return drivers_data

# messages = [{'role': 'user', 'content': 'get me the clean laps for the 2026 Hungary Grand Prix race session'}]
# print('Prompt:', messages[0]['content'])

# response: ChatResponse = chat(
#   'llama3.2:3b',
#   messages=messages,
#   tools=[get_clean_laps],
# )
# available_functions = {
#   'get_clean_laps': get_clean_laps,
# }
# if response.message.tool_calls:
#   # There may be multiple tool calls in the response
#   # Add the assistant message with tool calls to the conversation
#   messages.append(response.message)

#   for tool in response.message.tool_calls:
#     # Ensure the function is available, and then call it
#     if function_to_call := available_functions.get(tool.function.name):
#       print('Calling function:', tool.function.name)
#       print('Arguments:', tool.function.arguments)
#       output = function_to_call(**tool.function.arguments)
#       print('Function output:', output)
#     else:
#       print('Function', tool.function.name, 'not found')
#       output = 'Function not found'

#     # Add each tool result as a separate message
#     messages.append({'role': 'tool', 'content': str(output), 'tool_name': tool.function.name})

#   # Get final response from model with all tool call results
#   final_response = chat('llama3.2:3b', messages=messages)
#   print('Final response:', final_response.message.content)

# else:
#   print('No tool calls returned from model')
