# tools/__init__.py
from agent.tools.math import tool_add, tool_multiply, tool_subtract, tool_divide, tool_square_root
from agent.tools.weather import tool_weather
from agent.tools.finance import tool_calculate_compound_interest
#from agent.tools.slack_notify import tool_slack_notify

TOOL_REGISTRY = {
    "tool_add": tool_add,
    "tool_subtract": tool_subtract,
    "tool_multiply": tool_multiply,
    "tool_divide": tool_divide,
    "tool_square_root": tool_square_root,
    "tool_weather": tool_weather,
    #"tool_slack_notify": tool_slack_notify,
    "tool_calculate_compound_interest": tool_calculate_compound_interest,
}
