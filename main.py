import os
from crewai import Crew, Process
from agents import create_agents
from tasks import create_tasks

# ── Config ──────────────────────────────────────────────
CSV_PATH = "CSV_PATH"   # ← change this
TARGET_COLUMN = "pain_scale"        # ← change this
os.environ["GROQ_API_KEY"] = "YOUR GROQ_API_KEY"  # ← add your key
# ────────────────────────────────────────────────────────

def main():
    print("🚀 Starting AutoML Agent Pipeline...\n")

    agents = create_agents()
    tasks = create_tasks(agents, CSV_PATH, TARGET_COLUMN)

    crew = Crew(
        agents=list(agents),
        tasks=tasks,
        process=Process.sequential,
        verbose=2,
    )

    result = crew.kickoff()

    print("\n✅ AutoML Pipeline Complete!")
    print("=" * 50)
    print(result)

if __name__ == "__main__":
    main()