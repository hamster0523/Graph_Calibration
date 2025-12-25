from .terminate import Terminate

SELF_REFLECTION_TERMINATE_DESCRIPTION = """Terminate the interaction after summarizing the agent's memory and providing a final reflection on the task."""

class ReflectionTerminate(Terminate):
    name: str = "reflection_terminate"
    description: str = SELF_REFLECTION_TERMINATE_DESCRIPTION

    def execute(self, status) -> str:
        # summarize the agent memory and return the summary when call this tool
        return super().execute(status)