class AutonomousLoopManager:
    """
    Runs the orchestrator in autonomous mode until:
      - Goal achieved
      - Max iterations
      - Safety Supervisor halts
      - Verifier rejects too many times
    """

    def __init__(self, orchestrator, max_iters=10):
        self.orchestrator = orchestrator
        self.max_iters = max_iters

    def run(self, goal: str) -> str:
        history = []
        current_task = goal

        for i in range(self.max_iters):
            print(f"\n🔁 Iteration {i+1}/{self.max_iters}")

            output = self.orchestrator.handle(current_task)
            history.append(output)

            # Check for termination signals
            if "TASK_COMPLETED" in output or "STOP" in output:
                return f"🎉 Autonomous loop completed in {i+1} iterations:\n{output}"

            # Feed back into next step
            current_task = f"Continue toward the goal: {goal}. Previous output: {output}"

        return "⚠️ Max iterations reached. Loop ended."
