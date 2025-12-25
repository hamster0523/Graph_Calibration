# Copyright 2025 Pokee AI Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# modify by Hamster Team for verl-hamster project

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import aiohttp
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

from .utils import setup_colored_logger

logger = setup_colored_logger(__name__)

import sys

from hamster_agent.app.tool import WebSearch

class SearchURLItem(BaseModel):
    """
    A single URL item from search results.

    Attributes:
        url: The full URL of the search result
        title: The title/headline of the webpage
        description: A brief description or snippet from the webpage
    """

    url: str
    title: str
    description: str

class SearchResult(BaseModel):
    """
    Structured results from a search query.

    Attributes:
        query: The original search query
        url_items: List of search result items with URL, title, and description
        raw_response: Original API response (truncated to 500 chars)
        success: Whether the search operation completed successfully
        metadata: Additional metadata about the search (provider, query, etc.)
        error: Error message if the search failed, empty string otherwise
    """

    query: str
    url_items: List[SearchURLItem]
    raw_response: str = ""
    success: bool
    metadata: Dict[str, Any]
    error: str = ""

def _extract_organic_from_serper_response(data: Dict[str, Any]) -> List[SearchURLItem]:
    """
    Extract URLs from Serper API response.

    Args:
        data: The JSON response from Serper API

    Returns:
        List of SearchURLItem objects from organic results
    """
    organic_results = data.get("organic", [])
    return [
        SearchURLItem(
            url=item.get("link", ""),
            title=item.get("title", "No Title"),
            description=item.get("snippet", "No Description"),
        )
        for item in organic_results
    ]


async def serper_search(query: str, timeout: int = 30, top_k: int = 10) -> SearchResult:
    """
    Perform a search using Serper API.

    Args:
        query: The search query string
        timeout: Maximum time in seconds to wait (default: 30)
        top_k: Maximum number of results to return (default: 10)

    Returns:
        SearchResult containing search results and metadata

    Example:
        >>> result = await serper_search("Python programming")
        >>> if result.success:
        ...     for item in result.url_items:
        ...         print(f"{item.title}: {item.url}")
    """
    api_key = os.getenv("SERPER_API_KEY")

    if not api_key:
        return SearchResult(
            query=query,
            url_items=[],
            success=False,
            metadata={
                "provider": "serper",
                "status": 401,
                "execution_time": 0.0,
                "total_results": 0,
            },
            error="SERPER_API_KEY environment variable not found",
        )

    # url = "https://google.serper.dev/search"
    url = "https://serpapi.com/search"
    payload = {"q": query, "location": "United States", "num": top_k}
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }

    loop = asyncio.get_running_loop()
    start_time = loop.time()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                execution_time = loop.time() - start_time

                if response.status == 200:
                    data = await response.json()
                    raw_response = await response.text()
                    url_items = _extract_organic_from_serper_response(data)

                    logger.info(
                        f"Search successful for '{query}', found {len(url_items)} results"
                    )

                    return SearchResult(
                        query=query,
                        url_items=url_items,
                        raw_response=raw_response[:500],
                        success=True,
                        metadata={
                            "provider": "serper",
                            "status": 200,
                            "execution_time": execution_time,
                            "total_results": len(url_items),
                            "searchParameters": data.get("searchParameters", {}),
                            "credits": data.get("credits", 0),
                        },
                    )

                else:
                    error_text = await response.text()
                    logger.warning(
                        f"Search failed with HTTP {response.status}: {error_text[:100]}"
                    )
                    return SearchResult(
                        query=query,
                        url_items=[],
                        success=False,
                        metadata={
                            "provider": "serper",
                            "status": response.status,
                            "execution_time": execution_time,
                            "total_results": 0,
                        },
                        error=f"HTTP {response.status}: {error_text[:200]}",
                    )

    except asyncio.TimeoutError:
        logger.warning(f"Search request timed out after {timeout}s")
        return SearchResult(
            query=query,
            url_items=[],
            success=False,
            metadata={
                "provider": "serper",
                "status": 408,
                "execution_time": loop.time() - start_time,
                "total_results": 0,
            },
            error=f"Search request timed out after {timeout}s",
        )

    except aiohttp.ClientError as e:
        logger.warning(f"Client error during search: {str(e)}")
        return SearchResult(
            query=query,
            url_items=[],
            success=False,
            metadata={
                "provider": "serper",
                "status": 502,
                "execution_time": loop.time() - start_time,
                "total_results": 0,
            },
            error=f"Client error: {str(e)[:200]}",
        )

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON response: {e}")
        return SearchResult(
            query=query,
            url_items=[],
            success=False,
            metadata={
                "provider": "serper",
                "status": 502,
                "execution_time": loop.time() - start_time,
                "total_results": 0,
            },
            error=f"Failed to parse JSON response: {str(e)[:200]}",
        )

    except Exception as e:
        logger.error(f"Unexpected error during search: {str(e)}")
        return SearchResult(
            query=query,
            url_items=[],
            success=False,
            metadata={
                "provider": "serper",
                "status": 500,
                "execution_time": loop.time() - start_time,
                "total_results": 0,
            },
            error=f"Unexpected error: {str(e)[:200]}",
        )

from hamster_agent.app.tool.web_search import SearchResponse
from collections import defaultdict
def _convert_searchResponse_to_SearchResult(search_response: SearchResponse, success : bool) -> Optional[SearchResult]:
    if search_response is None:
        return None
    
    r_eturn_search_result = SearchResult(
        query=search_response.query,
        url_items=[
            SearchURLItem(
                url = item.url,
                title = item.title,
                description = item.description
            )
            for item in search_response.results
        ],
        success = success,
        raw_response = "",
        metadata = defaultdict(str),
    )
    return r_eturn_search_result

async def search_tool_with_manus(query : str, timeout: int = 30, top_k: int = 10) -> Optional[SearchResult]:
    search_tool = WebSearch()
    success = True
    search_result : Optional[SearchResult] = None
    try:
        search_result : SearchResponse = await search_tool.execute(
            query = query,
            num_results = top_k
        )
    except Exception as e:
        logger.error(f"Error occurred during search: {e}")
        success = False

    return _convert_searchResponse_to_SearchResult(search_result, success)

class WebSearchAgent:
    """
    Agent for performing web searches with concurrency control.

    This agent handles search queries with semaphore-based rate limiting
    to prevent overwhelming the search API.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the WebSearchAgent.

        Args:
            config: Configuration dictionary with optional keys:
                - max_concurrent_requests: Max concurrent searches (default: 300)
        """
        self._timeout = config.get("timeout", 30)
        self._top_k = config.get("top_k", 10)
        self._semaphore = asyncio.Semaphore(config.get("max_concurrent_requests", 300))

    async def search(self, query: str) -> SearchResult:
        """
        Perform a web search with concurrency control.

        Args:
            query: Search query string to execute

        Returns:
            SearchResult containing search results and metadata

        Example:
            >>> agent = WebSearchAgent(config={})
            >>> result = await agent.search("Python programming")
            >>> if result.success:
            ...     for item in result.url_items:
            ...         print(item.title)
        """
        logger.info(f"Searching for '{query}'")

        try:
            async with self._semaphore:
                result = await serper_search(
                    query, timeout=self._timeout, top_k=self._top_k
                )

                if not result.success:
                    logger.warning(
                        f"Search failed for '{query}' with status "
                        f"{result.metadata.get('status', 'unknown')}: {result.error}"
                        f"try with manus search tool"
                    )
                    result = await search_tool_with_manus(
                        query, timeout=self._timeout, top_k=self._top_k
                    )
                    if result is None or not result.success:
                        logger.error(
                            f"Manus search also failed for '{query}'"
                        )

                return result

        except Exception as e:
            logger.error(
                f"Unexpected error in WebSearchAgent for '{query}': {str(e)}",
                exc_info=True,
            )
            return SearchResult(
                query=query,
                url_items=[],
                success=False,
                metadata={
                    "provider": "serper",
                    "status": 500,
                    "execution_time": 0.0,
                    "total_results": 0,
                },
                error=f"WebSearchAgent error: {str(e)[:200]}",
            )
