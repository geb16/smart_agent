from __future__ import annotations
from agent.config import client, OPENAI_MODEL

# 🔖🈁🔴 For later use

# --- Suggestion Agent ---
class SuggestionAgent:
    """Generates next-sentence suggestions based on STM, LTM, and episodic memory."""

    def __init__(self, stm, ltm, epi):
        self.stm = stm
        self.ltm = ltm
        self.epi = epi

    def suggest(self, user_input: str) -> str:
        """
        Returns a short suggestion based on similar past queries or preferences.
        """
        past = self.stm.as_text()

        similar = self.epi.retrieve_similar(user_input, k=1)
        ltm_facts = self.ltm.recall(user_input, k=1)

        prompt = f"""
        User typed an incomplete query: {user_input}

        Recent conversation:
        {past}

        Similar queries from episodic memory:
        {similar}

        Relevant long-term memory:
        {ltm_facts}

        Suggest the next likely thing the user wants to ask.
        Keep the suggestion short, not intrusive, and relevant.
        Do NOT complete the whole query, just offer a subtle suggestion.
        """

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,
            messages=[{"role": "system", "content": prompt}]
        )

        return resp.choices[0].message.content.strip()
