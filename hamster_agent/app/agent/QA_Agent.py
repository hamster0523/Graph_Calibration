from pydantic import Field
from app.tool import Terminate, ToolCollection
from app.agent.toolcall import ToolCallAgent
from app.tool.local_retriever_tool import LocalRetrieverSearch
from app.tool.create_chat_completion import CreateChatCompletion

class RetrievalQAAgent(ToolCallAgent):

    name: str = "Retrieval_QA_Agent"
    description: str = (
        "A question answering agent that retrieves relevant text and synthesizes answers by combining retrieved information with its own reasoning. "
        "After answering the question, the agent will call the terminate tool to end the process."
    )

    system_prompt: str = (
        "You are a helpful question answering agent. "
        "When you do not have enough information to answer a user question, use the retrieval tool to search for relevant text. "
        "When you have sufficient information, use the create_chat_completion tool to generate a clear and accurate answer. "
        "After you have answered the question, always call the terminate tool to end the process. "
    )
    next_step_prompt: str = (
        "If you need more information, call the retrieval tool to search for relevant text. "
        "If you have enough information, call create_chat_completion to answer the question."
    )

    max_observe: int = 15000
    max_steps: int = 10

    # Add general-purpose tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            CreateChatCompletion(),
            LocalRetrieverSearch(),
            Terminate(),
        )
    )
