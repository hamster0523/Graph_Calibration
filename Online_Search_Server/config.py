from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional


@dataclass
class CacheManagerConfig:
    """Configuration for CacheManager.

    Fields match CacheManager.__init__ parameters used in project.
    """
    cache_dir: str = "cache"
    search_config: Dict[str, Any] = field(default_factory=dict)
    read_config: Dict[str, Any] = field(default_factory=dict)
    save_frequency: int = 10
    stats_report_interval: int = 300

    def to_kwargs(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WebSearchAgentConfig:
    """Configuration for WebSearchAgent (search).

    Includes sensible defaults that match the current codebase.
    """
    timeout: int = 30
    top_k: int = 10
    max_concurrent_requests: int = 300
    # provider override (e.g. 'serper' or 'manus')
    provider: Optional[str] = None

    def to_kwargs(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WebReadAgentConfig:
    """Configuration for WebReadAgent (read + llm summary).

    Includes fields used to construct the LLM wrapper and read behavior.
    """
    # LLM related
    llm_model: str = "gemini-pro"
    llm_base_url : str = "https://api.agicto.cn/v1"
    llm_api_key: Optional[str] = None
    llm_max_tokens: Optional[int] = None
    llm_max_input_tokens: Optional[int] = None
    llm_temperature: float = 1.0
    llm_api_type: str = "Openai"
    llm_api_version: str = ""

    # Read behavior
    timeout: int = 30
    max_concurrent_requests: int = 500
    max_content_words: int = 10000
    max_summary_words: int = 2048
    max_summary_retries: int = 3

    def to_kwargs(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineConfig:
    """Top-level configuration container for the search+read pipeline.

    This class groups the sub-configs and allows passing a single object
    around the codebase. It also supports providing an explicit
    `cache_manager` instance (e.g. pre-initialized and shared by the app).
    """
    search: WebSearchAgentConfig = field(default_factory=WebSearchAgentConfig)
    read: WebReadAgentConfig = field(default_factory=WebReadAgentConfig)
    cache: CacheManagerConfig = field(default_factory=CacheManagerConfig)

    # If set, this instance will be used instead of instantiating from cache
    # config. Must be a CacheManager instance.
    cache_manager_instance: Optional[Any] = None

    # Whether to prefer cached reads
    prefer_cached_reads: bool = True

    # Global top_k override (can be None to use search.top_k)
    top_k: Optional[int] = None

    def to_dicts(self) -> Dict[str, Any]:
        return {
            "search_agent_cfg": self.search.to_kwargs(),
            "read_agent_cfg": self.read.to_kwargs(),
            "cache_manager_cfg": self.cache.to_kwargs(),
            "prefer_cached_reads": self.prefer_cached_reads,
            "cache_manager": self.cache_manager_instance,
            "top_k": self.top_k,
        }


__all__ = [
    "CacheManagerConfig",
    "WebSearchAgentConfig",
    "WebReadAgentConfig",
    "PipelineConfig",
]
