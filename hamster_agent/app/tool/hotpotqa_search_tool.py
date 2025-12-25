from typing import Any, Optional, Dict
import re
import unicodedata
from difflib import SequenceMatcher

from app.logger import logger
from app.tool.base import BaseTool, ToolResult
from app.tool.web_search import WebSearch


def _norm(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s).casefold()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _best_fuzzy_match(query: str, choices, threshold: float = 0.8):
    q = _norm(query)
    best_title, best_score = None, 0.0
    for t in choices:
        score = SequenceMatcher(None, q, _norm(t)).ratio()
        if score > best_score:
            best_title, best_score = t, score
    if best_score >= threshold:
        return best_title, best_score
    return None, 0.0


class Search(BaseTool):
    name: str = "Search_Tool"
    description: str = "在给定上下文或网络中根据标题关键字检索相关内容（支持模糊匹配）。"

    parameters: dict = {
        "type": "object",
        "properties": {
            "search_key": {
                "type": "string",
                "description": "要检索的标题关键字（支持模糊匹配）",
            },
            "fuzzy_threshold": {
                "type": "number",
                "description": "匹配阈值，取值 0~1，默认 0.8",
                "default": 0.8
            }
        },
        "required": ["search_key"]
    }

    title_sentence_dict: Optional[Dict[str, Any]] = None

    def update_context(self, context: Dict[str, Any]):
        titles = context.get("title", [])
        sentences = context.get("sentences", []) 
        new_dict = {}
        for title, sentence_list in zip(titles, sentences):
            sentence_str = " ".join(sentence_list) if isinstance(sentence_list, list) else str(sentence_list)
            new_dict[title] = sentence_str
        self.title_sentence_dict = new_dict

    async def execute(self, search_key: str, fuzzy_threshold: float = 0.5, **kwargs) -> Any:
        exsit_titles = self.title_sentence_dict.keys() if self.title_sentence_dict else []

        if search_key in exsit_titles:
            searched_content = self.title_sentence_dict[search_key] if self.title_sentence_dict else ""
            return ToolResult(
                output=f"search '{search_key}' found (exact). content: {searched_content}"
            )

        best_title, score = _best_fuzzy_match(search_key, exsit_titles, threshold=fuzzy_threshold)
        if best_title is not None:
            searched_content = self.title_sentence_dict[best_title] if self.title_sentence_dict else ""
            return ToolResult(
                output=f"search '{search_key}' matched '{best_title}' with score={score:.2f}. content: {searched_content}"
            )

        ws = WebSearch()
        web_result = await ws.execute(query=search_key)

        return ToolResult(output=f"search '{search_key}' found (exact). content: {web_result}")

        #return ToolResult(output=f"'{search_key}' not found in context (exact or fuzzy≥{fuzzy_threshold}).")
