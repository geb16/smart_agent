import pytest

from agent.tools.math import (
    tool_add,
    tool_divide,
    tool_multiply,
    tool_square_root,
    tool_subtract,
)


def test_math_tools_basic_operations():
    assert tool_add(12, 8) == 20
    assert tool_subtract(12, 8) == 4
    assert tool_multiply(7, 6) == 42
    assert tool_divide(12, 4) == 3
    assert tool_square_root(49) == 7


def test_divide_by_zero_raises():
    with pytest.raises(ValueError, match="Division by zero"):
        tool_divide(5, 0)


def test_square_root_negative_raises():
    with pytest.raises(ValueError, match="negative number"):
        tool_square_root(-1)
