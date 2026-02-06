# agent/safety/tool_governor.py
from __future__ import annotations

from typing import Any, Dict

from agent.tools import TOOL_REGISTRY

# Global allow-list based on least privilege
ALLOWED_TOOLS = {
    "tool_add": True,
    "tool_subtract": True,
    "tool_multiply": True,
    "tool_divide": True,
    "tool_square_root": True,
    "tool_weather": True,
    "tool_calculate_compound_interest": True,
    "tool_slack_notify": True,  # explicitly allowed
}


class ToolGovernor:
    """Central authority that validates all planner tool requests."""

    def is_allowed(self, tool_name: str) -> bool:
        """Check if tool exists and is allowed."""
        return bool(ALLOWED_TOOLS.get(tool_name, False))

    def validate_tool_args(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate arguments against the Python function signature.
        Reject unknown args to prevent injection.
        """
        if tool_name not in TOOL_REGISTRY:
            raise ValueError(f"Tool '{tool_name}' does not exist in registry")

        tool_fn = TOOL_REGISTRY[tool_name]

        sig = tool_fn.__signature__
        valid = sig.parameters.keys()

        # Remove attacker-injected arguments
        cleaned = {k: v for k, v in args.items() if k in valid}

        return cleaned

    def authorize(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Final authorization check. This is where real-world systems:
        - enforce permissions
        - enforce rate limits
        - enforce compliance policies
        """
        if not self.is_allowed(tool_name):
            raise PermissionError(f"Tool '{tool_name}' not permitted")

        return self.validate_tool_args(tool_name, args)
