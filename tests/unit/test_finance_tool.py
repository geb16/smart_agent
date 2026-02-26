import pytest

from agent.tools.finance import tool_calculate_compound_interest


def test_compound_interest_quarterly_two_years():
    result = tool_calculate_compound_interest(
        principal=1000.0,
        rate=0.05,
        times_compounded=4,
        years=2,
    )
    assert result["principal"] == 1000.0
    assert result["rate"] == 0.05
    assert result["times_compounded"] == 4
    assert result["years"] == 2
    assert result["amount"] == 1104.5
    assert result["interest_earned"] == 104.5


@pytest.mark.parametrize("bad_rate", [0, -0.5, 1, 5])
def test_compound_interest_rejects_invalid_rate(bad_rate):
    with pytest.raises(ValueError, match="rate"):
        tool_calculate_compound_interest(
            principal=1000.0,
            rate=bad_rate,
            times_compounded=4,
            years=2,
        )
