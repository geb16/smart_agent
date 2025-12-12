# agent/planners/planner_agent.py

from __future__ import annotations
import json
from typing import Optional, List, Dict


class PlannerAgent:
    """
    High-level planning agent.
    - Collects all memory sources (STM, LTM, Episodic, Preferences)
    - Prepares memory_text in deterministic format
    - Delegates to WorkflowPlanner (planner_m16)
    """

    def __init__(self, planner, stm, ltm, epi) -> None:
        self.planner = planner
        self.stm = stm
        self.ltm = ltm
        self.epi = epi

    # --------------------------------------------------------------
    #                           PLAN
    # --------------------------------------------------------------
    def plan(self, user_input: str, memory_text: Optional[str] = None) -> List[Dict]:
        
        # ---------- 1. Construct memory text ----------
        if memory_text is None:

            # STM (short term history)
            stm_text = self.stm.as_text() or ""

            # User preferences (always JSON encoded)
            prefs = self.ltm.all_prefs()
            prefs_text = json.dumps(prefs, ensure_ascii=False) if prefs else ""

            # LTM semantic recall (relevant vector matches)
            ltm_chunks = self.ltm.recall(user_input) or []
            ltm_text = "\n".join(ltm_chunks)

            # Episodic memory (3 closest matches)
            epi_chunks = self.epi.retrieve_similar(user_input, k=3) or []
            epi_text = "\n".join(epi_chunks)

            # Combine with clear boundaries for determinism
            memory_text = "\n".join(
                section for section in [
                    f"[STM]\n{stm_text}" if stm_text else "",
                    f"[PREFERENCES]\n{prefs_text}" if prefs_text else "",
                    f"[LTM]\n{ltm_text}" if ltm_text else "",
                    f"[EPISODIC]\n{epi_text}" if epi_text else "",
                ]
                if section.strip()
            )

        # ---------- 2. Delegate to WorkflowPlanner ----------
        try:
            validated_steps = self.planner.plan(user_input, memory_text)

        except ValueError:
            # Hard invalid JSON / invalid action from model
            validated_steps = [
                {
                    "action": "rag",
                    "tool_name": None,
                    "tool_args": {},
                    "rag_query": user_input,
                }
            ]

        except Exception as e:
            # Safety fallback: planner failure (e.g. network)
            print(f"[PlannerAgent Fail-Safe Triggered] {e}")
            validated_steps = [
                {
                    "action": "rag",
                    "tool_name": None,
                    "tool_args": {},
                    "rag_query": user_input,
                }
            ]

        return validated_steps
