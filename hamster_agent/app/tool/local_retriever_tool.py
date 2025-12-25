from typing import Any, Optional, Dict, List, Tuple, Union
import re
import unicodedata
import json
import aiohttp

from app.logger import logger
from app.tool.base import BaseTool, ToolResult

def _norm(s: Optional[str]) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).casefold()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _best_fuzzy_match(query: str, choices, threshold: float = 0.8):
    from difflib import SequenceMatcher
    q = _norm(query)
    best_title, best_score = None, 0.0
    for t in choices:
        score = SequenceMatcher(None, q, _norm(t)).ratio()
        if score > best_score:
            best_title, best_score = t, score
    if best_score >= threshold:
        return best_title, best_score
    return None, 0.0

def _doc_to_string(doc: Union[str, Dict[str, Any]]) -> str:
    """将 /retrieve 返回的 document 统一转为字符串，且不截断。"""
    if isinstance(doc, dict):
        if "document" in doc and isinstance(doc["document"], dict) and "contents" in doc["document"]:
            return str(doc["document"]["contents"])
        if "contents" in doc:
            return str(doc["contents"])
        return json.dumps(doc, ensure_ascii=False)
    return str(doc)

def _format_items_full(items: List[Dict[str, Any]], return_scores: bool) -> str:
    """完整输出每条结果（不截断正文）。"""
    lines = []
    for i, it in enumerate(items, 1):
        doc = it.get("document", it) if isinstance(it, dict) else it
        score = it.get("score") if isinstance(it, dict) else None
        doc_str = _doc_to_string(doc)
        if return_scores and score is not None:
            lines.append(f"{i}. (score={float(score):.4f})\n{doc_str}")
        else:
            lines.append(f"{i}.\n{doc_str}")
    return "\n\n".join(lines)

class LocalRetrieverSearch(BaseTool):
    name: str = "Local_Retriever_Search"
    description: str = "POST queries to the local retrieval service (/retrieve) to obtain retrieval and re-ranking results; if a match is found in the in-context titles (exact or fuzzy), that content is returned instead."

    parameters: dict = {
        "type": "object",
        "properties": {
            "search_key": {
                "type": "string",
                "description": "Query string to search for (performs exact/fuzzy match against in-context titles first)."
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

    async def _post_retrieve(
        self,
        server_url: str,
        queries: List[str],
        topk_retrieval: int,
        topk_rerank: int,
        return_scores: bool,
        timeout: float
    ) -> List[List[Dict[str, Any]]]:
        payload = {
            "queries": queries,
            "topk_retrieval": topk_retrieval,
            "topk_rerank": topk_rerank,
            "return_scores": return_scores
        }
        timeout_cfg = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
            async with session.post(server_url, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {text}")
                data = await resp.json(content_type=None)
                return data.get("result", [[]])

    async def execute(
        self,
        search_key: str,
        server_url: str = "http://localhost:8100/retrieve",
        topk_retrieval: int = 10,
        topk_rerank: int = 3,
        return_scores: bool = True,
        timeout: float = 300.0,
        fuzzy_threshold: float = 0.8,
        **kwargs
    ) -> Any:
        exsit_titles = self.title_sentence_dict.keys() if self.title_sentence_dict else []

        if search_key in exsit_titles:
            searched_content = self.title_sentence_dict[search_key] if self.title_sentence_dict else ""
            return ToolResult(
                output=f"search '{search_key}' found in context (exact). content:\n\n{searched_content}"
            )

        best_title, score = _best_fuzzy_match(search_key, exsit_titles, threshold=fuzzy_threshold)
        if best_title is not None:
            searched_content = self.title_sentence_dict[best_title] if self.title_sentence_dict else ""
            return ToolResult(
                output=(f"search '{search_key}' matched '{best_title}' in context "
                        f"with score={score:.2f}. content:\n\n{searched_content}")
            )

        try:
            results = await self._post_retrieve(
                server_url=server_url,
                queries=[search_key],
                topk_retrieval=topk_retrieval,
                topk_rerank=topk_rerank,
                return_scores=return_scores,
                timeout=timeout
            )
        except Exception as e:
            logger.exception(f"[Local_Retriever_Search] request failed: {e}")
            return ToolResult(output=f"search '{search_key}' failed: {e}")

        items = results[0] if results else []
        pretty = _format_items_full(items, return_scores)

        return ToolResult(
            output=f"retrieval '{search_key}' results:\n\n{pretty}"
        )
