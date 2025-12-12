# main.py
#from agent.planner import WorkflowPlanner
from agent.planners.planner_m16 import WorkflowPlanner
from agent.agent_core import WorkflowAgent

# whe can now create different planners and agents as needed
planner = WorkflowPlanner()
agent = WorkflowAgent(planner)

print("🤖 Workflow Agent ready. Type 'exit' to quit.\n")

while True:
    user_in = input("You: ").strip()
    if user_in.lower() in {"exit", "quit"}:
        print("👋 Goodbye!")
        break

    out = agent.handle(user_in)
    print("\n🤖:", out, "\n")
