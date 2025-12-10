# agent/tools.py

def tool_calculate_compound_interest(
    principal: float,
    rate: float,
    times_compounded: int,
    years: float,
) -> dict:
    """
    Calculate compound interest.

    Args:
        principal: Initial amount (e.g., 1000.0).
        rate: Annual interest rate as a decimal (e.g., 0.05 for 5%).
        times_compounded: Number of compounding periods per year (e.g., 4 for quarterly).
        years: Number of years.

    Returns:
        dict with principal, rate, times_compounded, years, amount, interest_earned.
    """
    # Strict validation: catch the 5 vs 0.05 bug early
    if rate <= 0:
        raise ValueError("rate must be positive (e.g., 0.05 for 5%).")
    if rate >= 1:
        raise ValueError(
            f"rate should be a decimal (0.05 for 5%), got {rate}. "
            "Convert percent to decimal before calling this tool."
        )

    amount = principal * (1 + rate / times_compounded) ** (times_compounded * years)

    return {
        "principal": principal,
        "rate": rate,
        "times_compounded": times_compounded,
        "years": years,
        "amount": round(amount, 1),
        "interest_earned": round(amount - principal, 1),
    }


# Example usage:
# if __name__ == "__main__":
#     result = tool_calculate_compound_interest(
#         principal=1000.0,
#         rate=0.05,
#         times_compounded=4,
#         years=5,
#     )
#     print(result)   

#     # Expected output:
#     # {
#     #     "principal": 1000.0,
#     #     "rate": 0.05,
#     #     "times_compounded": 4,
#     #     "years": 5,
#     #     "amount": 1280.08,
#     #     "interest_earned": 280.08,
#     # }
# What is the compound interest on $1000 at 5% compounded quarterly, for 2 years?
# tool_calculate_compound_interest(principal=1000.0, rate=0.05, times_compounded=4, years=2)
# Expected output:
# {
#     "principal": 1000.0,
#     "rate": 0.05,
#     "times_compounded": 4,
#     "years": 2,
#     "amount": 1104.94,
#     "interest_earned": 104.94,
# }