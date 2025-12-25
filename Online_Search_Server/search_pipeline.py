import sys

from typing import Dict, Any, Optional, List, Union, Awaitable
import asyncio
from .search import WebSearchAgent
from .fetch import WebReadAgent
from .generate import QueryInformationGenerateAgent
from .cache_manager import CacheManager
from .utils import setup_colored_logger
from .config import PipelineConfig

logger = setup_colored_logger(__name__)

# Module-level singleton cache manager (long-lived). Use an async lock to
# ensure only one coroutine instantiates it.
_GLOBAL_CACHE_MANAGER: Optional[CacheManager] = None
_GLOBAL_CACHE_LOCK: Optional[asyncio.Lock] = None


async def _ensure_global_cache_manager(cache_cfg: Dict[str, Any]) -> Optional[CacheManager]:
	"""Ensure a module-level CacheManager exists and return it.

	This is safe to call from multiple concurrent coroutines; instantiation
	happens once and is guarded by an async Lock.
	"""
	global _GLOBAL_CACHE_MANAGER, _GLOBAL_CACHE_LOCK
	if _GLOBAL_CACHE_MANAGER is not None:
		return _GLOBAL_CACHE_MANAGER

	if _GLOBAL_CACHE_LOCK is None:
		_GLOBAL_CACHE_LOCK = asyncio.Lock()

	async with _GLOBAL_CACHE_LOCK:
		if _GLOBAL_CACHE_MANAGER is not None:
			return _GLOBAL_CACHE_MANAGER
		try:
			# Instantiate synchronously; CacheManager is thread-safe internally
			_GLOBAL_CACHE_MANAGER = CacheManager(**(cache_cfg or {}))
			logger.info("Initialized global CacheManager singleton")
		except Exception as e:
			try:
				logger.warning(f"Failed to instantiate global CacheManager: {e}")
			except Exception:
				pass
			_GLOBAL_CACHE_MANAGER = None

	return _GLOBAL_CACHE_MANAGER

async def online_search_pipeline(query: str, pipeline_config: Union[PipelineConfig, Dict[str, Any]]) -> Dict[str, Any]:
	"""
	Run an online search pipeline for a single query:
	  1. run web search (WebSearchAgent)
	  2. for top_k results, fetch page content and summarize (WebReadAgent)
	  3. optional cache integration via config['cache_manager'] supporting
		 `get_search`/`set_search` and `get_read`/`set_read`.

	Args:
		query: user query
		config: pipeline configuration. Supported keys:
			- top_k: int number of search results to fetch (default 3)
			- search_agent_cfg: dict passed to WebSearchAgent
			- read_agent_cfg: dict passed to WebReadAgent
			- cache_manager: optional cache manager instance
			- prefer_cached_reads: bool, whether to check cache for reads (default True)

	Returns:
		dict with keys: 'query', 'search', 'reads' where 'search' is search result dict
		and 'reads' is list of read result dicts in same order as returned URLs.
	"""

	# Accept either a PipelineConfig dataclass or a plain dict (back-compat).
	if isinstance(pipeline_config, dict):
		# convert dict to PipelineConfig-like pieces
		pc_dict = pipeline_config
		top_k = pc_dict.get("top_k")
		search_agent_cfg = pc_dict.get("search_agent_cfg", {})
		read_agent_cfg = pc_dict.get("read_agent_cfg", {})
		cache_mgr_instance = pc_dict.get("cache_manager")
		cache_cfg = pc_dict.get("cache_manager_cfg", {}) or {}
		prefer_cached_reads: bool = bool(pc_dict.get("prefer_cached_reads", True))
	else:
		pc: PipelineConfig = pipeline_config
		top_k = pc.top_k
		search_agent_cfg = pc.search.to_kwargs()
		read_agent_cfg = pc.read.to_kwargs()
		cache_mgr_instance = pc.cache_manager_instance
		cache_cfg = pc.cache.to_kwargs()
		prefer_cached_reads: bool = pc.prefer_cached_reads

	# determine final top_k
	final_top_k = int(top_k) if (top_k is not None) else int(search_agent_cfg.get("top_k", 3))

	# Cache manager handling: prefer an explicit instance, otherwise use module singleton
	if isinstance(cache_mgr_instance, CacheManager):
		cache_manager = cache_mgr_instance
	else:
		if (isinstance(pipeline_config, dict) and pipeline_config.get("enable_cache", True) is False):
			cache_manager = None
		else:
			cache_manager = await _ensure_global_cache_manager(cache_cfg)

	search_agent = WebSearchAgent(search_agent_cfg)
	read_agent = WebReadAgent(read_agent_cfg)
	generate_agent = QueryInformationGenerateAgent(read_agent_cfg)

	# Try cache for search if available
	search_result = None
	if cache_manager is not None and hasattr(cache_manager, "get_search"):
		try:
			cached = cache_manager.get_search(query)
			if cached:
				search_result = cached.get("result")
		except Exception:
			# cache errors must not break pipeline
			search_result = None

	if search_result is None:
		search_result = await search_agent.search(query)
		# write to cache (best-effort) - convert to plain dict when possible
		if cache_manager is not None and hasattr(cache_manager, "set_search"):
			try:
				to_cache = search_result.dict() if hasattr(search_result, "dict") else search_result
				cache_manager.set_search(query, to_cache)
			except Exception:
				pass

	# convert pydantic model to serializable dict if needed
	try:
		search_dict = search_result.dict() if hasattr(search_result, "dict") else search_result
	except Exception:
		search_dict = search_result

	# search_dict may be a dict or a pydantic model; handle both
	if isinstance(search_dict, dict):
		url_items = search_dict.get("url_items", [])[:final_top_k]
	else:
		url_items = getattr(search_dict, "url_items", [])[:final_top_k]

	# fetch pages concurrently (bounded by read_agent semaphore)
	async def _fetch_one(item: Dict[str, Any]) -> Dict[str, Any]:
		url = item.get("url") if isinstance(item, dict) else getattr(item, "url", None)
		title = item.get("title") if isinstance(item, dict) else getattr(item, "title", "")
		if not url:
			return {"url": url, "title": title, "success": False, "error": "empty url"}

		# check read cache
		if prefer_cached_reads and cache_manager is not None and hasattr(cache_manager, "get_read"):
			try:
				cached = cache_manager.get_read(url, query)
				if cached:
					return {"url": url, "title": title, "success": True, "cached": True, "result": cached.get("result", cached)}
			except Exception:
				pass

		# perform read
		try:
			read_res = await read_agent.read(query, url)
			out = read_res.dict() if hasattr(read_res, "dict") else read_res
			# write read cache
			if cache_manager is not None and hasattr(cache_manager, "set_read"):
				try:
					cache_manager.set_read(url, query, out)
				except Exception:
					pass
			return {"url": url, "title": title, "success": True, "cached": False, "result": out}
		except Exception as e:
			return {"url": url, "title": title, "success": False, "error": str(e)}

	# peform generate information
	generate_info = await generate_agent.generate(query)
	
	# prepare list of items as dicts
	items: List[Dict[str, Any]] = []
	for it in url_items:
		if isinstance(it, dict):
			items.append(it)
		else:
			# pydantic model
			try:
				items.append(it.dict())
			except Exception:
				items.append({"url": getattr(it, "url", None), "title": getattr(it, "title", "")})

	tasks = [_fetch_one(it) for it in items]
	read_results = await asyncio.gather(*tasks)

	return {"query": query, "search": search_dict, "reads": read_results, "generated_information": generate_info}


def run_online_search_pipeline_sync(query: str, config: Dict[str, Any]) -> Dict[str, Any]:
	"""Convenience sync wrapper that runs the async pipeline."""
	return asyncio.run(online_search_pipeline(query, config))

def extract_summaries(pipeline_output: Dict[str, Any]) -> List[Dict[str, str]]:
	"""
	从 pipeline 的输出中提取每个 read 的摘要信息，返回列表，每项为 dict:
	  { 'query': str, 'url': str, 'summary': str }

	兼容 pipeline 输出中各种可能的结构：read 项可能为 dict 或 model，
	read['result'] 也可能嵌套 summary 或含有 result 字段（缓存格式）。
	"""
	summaries: List[Dict[str, str]] = []

	if not pipeline_output:
		return summaries

	# 获取 query 字段（支持 search 为 model 或 dict 的情况）
	query = pipeline_output.get("query")
	if not query:
		search = pipeline_output.get("search")
		if isinstance(search, dict):
			query = search.get("query")
		else:
			query = getattr(search, "query", None)

	reads = pipeline_output.get("reads", []) or []

	for read in reads:
		if isinstance(read, dict):
			url = read.get("url")
			summary = read.get("summary")
			if not summary and "result" in read:
				res = read.get("result")
				if isinstance(res, dict):
					summary = res.get("summary") or (res.get("result") or {}).get("summary")
				else:
					summary = getattr(res, "summary", None)
		else:
			url = getattr(read, "url", None)
			summary = getattr(read, "summary", None)
			if not summary:
				res = getattr(read, "result", None)
				if res is not None:
					summary = getattr(res, "summary", None)

		if summary is None:
			continue
		if not isinstance(summary, str):
			try:
				summary = str(summary)
			except Exception:
				continue
		if summary.strip() == "":
			continue

		summaries.append({"query": query or "", "url": url or "", "summary": summary})

	# get generated information if any
	generated_info = pipeline_output.get("generated_information", "")
	summaries.append({"query": query or "", "url": "generated_information", "summary": generated_info})
 
	return summaries

async def run_batch_search_async(queries: List[str], pipeline_config: Union[PipelineConfig, Dict[str, Any]]) -> List[List[Dict[str, str]]]:
	"""
	Asynchronous batch runner for multiple queries.

	Returns a list (per query) of summary dicts (each dict: {query,url,summary}).
	Use this inside an existing async event loop.
	"""
	tasks = [online_search_pipeline(q, pipeline_config) for q in queries]
	results = await asyncio.gather(*tasks)
	return [extract_summaries(f) for f in results if f is not None]

# def run_batch_search(queries: Optional[List[str]], config: Dict[str, Any]) -> List[List[Dict[str, str]]]:
# 	"""
# Online_Search_Server	Synchronous wrapper around the async batch runner. Safe to call from
# 	regular synchronous code (it uses asyncio.run()). If you are already in
# 	an async context, call `await run_batch_search_async(...)` instead.
# 	"""
# 	if queries is None:
# 		return []
# 	return asyncio.run(run_batch_search_async(queries, config))

def run_batch_search(
	queries: Optional[List[str]],
	config: Union[PipelineConfig, Dict[str, Any]],
	timeout: Optional[float] = 300.0,
) -> Union[List[List[Dict[str, str]]], Awaitable[List[List[Dict[str, str]]]]]:
	"""
	Dual-mode batch runner: returns an awaitable when called from an
	active asyncio event loop, otherwise runs synchronously and returns
	the final result. Synchronous execution supports an optional timeout.

	Args:
		queries: list of queries to run (or None)
		config: pipeline configuration (PipelineConfig or dict)
		timeout: seconds to wait for synchronous execution (None = no timeout)

	Returns:
		Either a final list of per-query summaries (synchronous path) or
		an awaitable/coroutine (async path) that the caller can await.
	"""
	if queries is None:
		return []

	coro = run_batch_search_async(queries, config)

	# Detect running event loop: if present, return the coroutine so caller can await
	try:
		loop = asyncio.get_running_loop()
	except RuntimeError:
		loop = None

	if loop and loop.is_running():
		return coro

	# No running loop: run synchronously with optional timeout
	try:
		if timeout is not None:
			return asyncio.run(asyncio.wait_for(coro, timeout=timeout))
		else:
			return asyncio.run(coro)
	except Exception:
		logger.exception("run_batch_search failed")
		raise

def format_to_str(batch_search_result: List[List[Dict[str, str]]]) -> List[str]:
	"""
	Convert batch search results into a list of strings, one per query.

	Each string contains the concatenated <information> blocks for that query.
	If a query has no summaries, an empty string is returned for that position.
	"""
	results: List[str] = []
	for summaries in batch_search_result:
		if not summaries:
			results.append("")
			continue

		output_lines: List[str] = []
		for summary_dict in summaries:
			query = summary_dict.get("query", "")
			summary = summary_dict.get("summary", "")
			output_lines.append(f"<information>query: {query}\n summary: {summary}\n</information>")

		results.append("\n".join(output_lines))

	return results