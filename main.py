# main_multi_agent.py

from agent.multi_agent_V2 import MultiAgentOrchestrator

if __name__ == "__main__":

    print("\nMulti-Agent System Ready 🧠 (Planner → Executor → Verifier)")
    print("Type 'exit' to quit.\n")
    orchestrator = MultiAgentOrchestrator()

    while True:
        q = input("🙋 User: ").strip()

        if q.lower() in {"exit", "quit"}:
            print("\nShutting down multi-agent system. Goodbye! 👋\n")
            break

        if not q:
            continue  # skip empty input

        print("\n🤖 : ", end="", flush=True)
        try:
            final_answer = orchestrator.handle(
                q,
                stream=True,  # Set to True to enable streaming
                on_token=lambda token: print(token, end="", flush=True),  # Token callback for streaming
            )
        except Exception as e:
            print(f"\n[Error] {e}")
            continue
        print("\n")  # Newline after streaming answer is complete
        # print("\n🤖 Final Answer:", final_answer, "\n")

        # ------------------------------------------------------
        # DEBUGGING / INTROSPECTION OUTPUT (VERY USEFUL)
        # ------------------------------------------------------

        # print("🧩 Short-Term Memory (STM):")
        # stm_dump = orchestrator.stm.as_text()
        # print(stm_dump if stm_dump else "(empty)")
        # print()

        # # test if saved preferences work

        # print("📌 Long-Term Preferences (LTM):")
        # prefs = orchestrator.ltm.all_prefs()
        # print(prefs if prefs else "(no stored preferences)")
        # print()

        # test if episodic memory works
    # print("📚 Episodic Memory Samples:")

    # to test episodic memory, uncomment below
    # eps= orchestrator.epi.store_episode(
    #     user_input=q,
    #     episode={
    #         "steps": [{"action": "tool", "tool_name": "search", "output": "Found info"}],
    #         "final": "This is the final answer based on the workflow.",
    #         "verifier_status": "approved",
    #         "results": ["Step 1 result", "Step 2 result"]
    #     }
    # )
    # if not eps:
    #     print("(no episodic memories stored yet)")
    # else:
    #     print(f"Stored episodic memory with ID: {eps['id']}")
    # print()
