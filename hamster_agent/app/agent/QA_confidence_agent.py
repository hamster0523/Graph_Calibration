from pydantic import Field
from app.tool import Terminate, ToolCollection
from app.agent.toolcall import ToolCallAgent
from app.tool.local_retriever_tool import LocalRetrieverSearch
from app.tool.create_chat_completion import CreateChatCompletion

class RetrievalQAConfidenceAgent(ToolCallAgent):

    name: str = "Retrieval_QA_Confidence_Agent"
    description: str = (
        "A question answering agent that retrieves relevant text and synthesizes answers by combining retrieved information with its own reasoning. "
        "After answering the question, the agent will call the terminate tool to end the process."
    )

    system_prompt: str = (
        "You are a helpful question answering agent. "
        "When you do not have enough information to answer a user question, use the retrieval tool to search for relevant text. "
        "When you have sufficient information, use the create_chat_completion tool to generate a clear and accurate answer. "
        "After you have answered the question, always call the terminate tool to end the process. "
        "\n\nIMPORTANT: Before making any tool call, you must provide your reasoning and include a confidence score. "
        "Format your thinking as follows:\n"
        "- Explain your reasoning for the tool call\n"
        "- Include <confidence>X.X</confidence> where X.X is a number between 0.0 and 1.0 representing your confidence in this tool call decision\n"
        "- 0.0 means very uncertain, 1.0 means very confident\n"
        "- Consider factors like: information completeness, query clarity, expected tool effectiveness"
    )
    next_step_prompt: str = (
        "If you need more information, call the retrieval tool to search for relevant text. "
        "If you have enough information, call create_chat_completion to answer the question. "
        "\n\nRemember to include your reasoning and confidence score before each tool call:\n"
        "- Explain why you're choosing this specific tool\n"
        "- Add <confidence>X.X</confidence> where X.X is between 0.0 and 1.0\n"
        "- Higher confidence (0.8-1.0) for clear, straightforward decisions\n"
        "- Medium confidence (0.5-0.7) for moderately uncertain situations\n"
        "- Lower confidence (0.0-0.4) for highly uncertain or exploratory actions"
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
