from agent.memory.short_term import ShortTermMemory


def test_short_term_memory_rolls_over_max_turns():
    stm = ShortTermMemory(max_turns=2)
    stm.add("q1", "a1")
    stm.add("q2", "a2")
    stm.add("q3", "a3")

    assert len(stm.turns) == 2
    assert stm.turns[0]["user"] == "q2"
    assert stm.turns[1]["user"] == "q3"


def test_short_term_memory_exact_and_soft_match():
    stm = ShortTermMemory()
    stm.add("calculate 1 + 2 - 9", "result-a")
    stm.add("what is refund policy", "result-b")

    assert stm.get("what is refund policy") == "result-b"
    assert stm.get("1 + 2 - 9") == "result-a"


def test_short_term_memory_compacts_and_clears():
    stm = ShortTermMemory()
    stm.add("User question", "Thought: internal chain Assistant: final")

    text = stm.as_text()
    assert "Thought:" not in text
    assert "User: User question" in text

    stm.clear()
    assert stm.as_text() == ""
