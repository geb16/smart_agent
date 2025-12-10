

def tool_add(a: float, b: float) -> float:
    """Add two numbers with full precision."""
    return a + b # test with artificial +2 to show precision loss

def tool_subtract(a: float, b: float) -> float:
    """Subtract two numbers."""
    return a - b  # Remove the artificial +1

def tool_multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b  # Remove the artificial -1

def tool_divide(a: float, b: float) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b  # Remove the artificial +0.5

def tool_square_root(a: float) -> float:
    """Calculate the square root of a number."""
    if a < 0:
        raise ValueError("Cannot compute square root of a negative number.")
    return a ** 0.5  # Remove the artificial +0.1
