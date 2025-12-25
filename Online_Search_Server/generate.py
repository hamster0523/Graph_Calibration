import sys
from typing import Dict, Any
import os
import re

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from .utils import setup_colored_logger
logger = setup_colored_logger(__name__)
from hamster_tool.llm import LLM, LLMSettings 
from hamster_tool.schema import Message

GENERATE_PROMPT = """
You are an expert knowledge writer.

Given the question below, generate a coherent, factual, and informative paragraph
that contains background knowledge, key concepts, or contextual facts 
relevant to understanding and answering the question.

The text should:
- Be directly related to the question topic;
- Include important facts, definitions, or relationships;
- Avoid answering the question directly;
- Be around 150–250 words;
- Read naturally as an informative paragraph.

---

Question:
{question}

---

Output:
<related_context>
(Your generated passage)
</related_context>
"""


class QueryInformationGenerateAgent:
    """
    Agent for performing generate information tasks.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.llm = LLM(llm_config=LLMSettings(
                    model = config.get("llm_model", "gemini-pro"),
                    base_url = config.get("llm_base_url", os.getenv("GENAI_API_BASE_URL", "https://api.gen.ai/v1")),
                    api_key = config.get("llm_api_key", os.getenv("GENAI_API_KEY", "")),
                    max_tokens = config.get("llm_max_tokens", None),
                    max_input_tokens = config.get("llm_max_input_tokens", None),
                    temperature = config.get("llm_temperature", 1.0),
                    api_type = config.get("llm_api_type", "Openai"),
                    api_version = config.get("llm_api_version", ""),
                ))

    def _extract_related_context(self, text: str) -> str:
        match = re.search(r"<related_context>(.*?)</related_context>", text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _extract_all_related_contexts(self, text: str) -> str:
        parsed = [m.strip() for m in re.findall(r"<related_context>(.*?)</related_context>", text, re.DOTALL)]
        return "\n".join(parsed)

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (Exception, ValueError)
        ), 
    )
    async def generate(self, question : str) -> str:
        prompt = GENERATE_PROMPT.format(question=question)
        extra_1 = ""
        extra_2 = ""
        try:
            response = await self.llm.ask(
                [Message.user_message(prompt)]
            )
            extra_1 = self._extract_related_context(response)
            extra_2 = self._extract_all_related_contexts(response)
            logger.info(f"Generated related context for question: {question}, context: {extra_2[:50]}")
        except Exception as e:
            logger.error(f"Error occurred while generating response: {e}")
        return extra_2 if extra_1 is not None and len(extra_2) > len(extra_1) else extra_1