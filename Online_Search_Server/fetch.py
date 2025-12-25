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


import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import aiohttp
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .utils import (
    setup_colored_logger,
    _is_valid_url,
    get_genai_client,
    get_retry_delay,
    llm_summary,
)

load_dotenv()

logger = setup_colored_logger(__name__)


class ReadURLItem(BaseModel):
    """
    A single URL extracted from a webpage with contextual information.

    Attributes:
        url: The complete URL found on the page
        title: Descriptive title explaining what the URL links to
    """

    url: str = Field(description="The full URL found on the page")
    title: str = Field(description="Title or description of the linked content")


class ReadResult(BaseModel):
    """
    Results from reading and analyzing a webpage.

    Attributes:
        success: Whether the read operation completed successfully
        content: Raw text content extracted from the webpage
        summary: LLM-generated summary based on the question, or truncated
                content if LLM summarization fails
        raw_response: Original API response (truncated to 500 chars)
        url_items: Relevant URLs discovered on the page
        metadata: API usage statistics and response metadata
        error: Error message if operation failed, empty string otherwise
    """

    success: bool
    content: str
    summary: str = ""
    raw_response: str = ""
    url_items: List[ReadURLItem] = []
    metadata: Dict[str, Any] = {}
    error: str = ""

async def jina_read(url: str, timeout: int = 30) -> ReadResult:
    """
    Read and extract content from a webpage using Jina Reader API.

    Args:
        url: The URL of the webpage to read
        timeout: Maximum time in seconds to wait (default: 30)

    Returns:
        ReadResult containing extracted content, links, and metadata

    Example:
        >>> result = await jina_read("https://example.com")
        >>> if result.success:
        ...     print(result.content)
        ...     for item in result.url_items:
        ...         print(f"{item.title}: {item.url}")
    """
    api_key = os.getenv("JINA_API_KEY")

    if not api_key:
        return ReadResult(
            success=False,
            content="",
            metadata={
                "source": "jina_reader",
                "url": url,
                "status": 401,
                "execution_time": 0.0,
                "links_found": 0,
                "relevant_links": 0,
            },
            error="JINA_API_KEY environment variable not found",
        )

    reader_url = f"https://r.jina.ai/{url}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "X-Return-Format": "text",
        "X-With-Links-Summary": "true",
    }

    loop = asyncio.get_running_loop()
    start_time = loop.time()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                reader_url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                execution_time = loop.time() - start_time

                if response.status == 200:
                    data = await response.json()
                    raw_response = await response.text()

                    content = data.get("data", {}).get("text", "")
                    links_data = data.get("data", {}).get("links", {})

                    # Process links into ReadURLItem objects
                    url_items = []
                    for link_title, link_url in links_data.items():
                        if not link_url or not isinstance(link_url, str):
                            continue

                        link_url = link_url.strip()
                        link_title = link_title.strip() if link_title else "No Title"

                        if link_url and _is_valid_url(link_url):
                            url_items.append(
                                ReadURLItem(url=link_url, title=link_title)
                            )

                    metadata = {
                        "source": "jina_reader",
                        "url": url,
                        "status": 200,
                        "title": data.get("data", {}).get("title", ""),
                        "description": data.get("data", {}).get("description", ""),
                        "links_found": len(links_data),
                        "relevant_links": len(url_items),
                        "execution_time": execution_time,
                    }

                    if "usage" in data.get("data", {}):
                        metadata["usage"] = data["data"]["usage"]
                    if "usage" in data.get("meta", {}):
                        metadata["meta_usage"] = data["meta"]["usage"]

                    logger.info(
                        f"Successfully read '{url}', found {len(url_items)} relevant links"
                    )

                    return ReadResult(
                        success=True,
                        content=content,
                        url_items=url_items,
                        raw_response=raw_response[:500],
                        metadata=metadata,
                    )

                else:
                    error_text = await response.text()
                    logger.warning(
                        f"Read failed with HTTP {response.status}: {error_text[:100]}"
                    )
                    return ReadResult(
                        success=False,
                        content="",
                        url_items=[],
                        raw_response=error_text[:500],
                        metadata={
                            "source": "jina_reader",
                            "url": url,
                            "status": response.status,
                            "execution_time": execution_time,
                            "links_found": 0,
                            "relevant_links": 0,
                        },
                        error=f"HTTP {response.status}: {error_text[:200]}",
                    )

    except asyncio.TimeoutError:
        logger.warning(f"Read request timed out after {timeout}s")
        return ReadResult(
            success=False,
            content="",
            url_items=[],
            raw_response="Request timed out",
            metadata={
                "source": "jina_reader",
                "url": url,
                "status": 408,
                "execution_time": loop.time() - start_time,
                "links_found": 0,
                "relevant_links": 0,
            },
            error=f"Request timed out after {timeout}s",
        )

    except aiohttp.ClientError as e:
        logger.warning(f"Client error during read: {str(e)}")
        return ReadResult(
            success=False,
            content="",
            url_items=[],
            raw_response=str(e)[:500],
            metadata={
                "source": "jina_reader",
                "url": url,
                "status": 502,
                "execution_time": loop.time() - start_time,
                "links_found": 0,
                "relevant_links": 0,
            },
            error=f"Client error: {str(e)[:200]}",
        )

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON response: {e}")
        return ReadResult(
            success=False,
            content="",
            url_items=[],
            raw_response="Invalid JSON response",
            metadata={
                "source": "jina_reader",
                "url": url,
                "status": 502,
                "execution_time": loop.time() - start_time,
                "links_found": 0,
                "relevant_links": 0,
            },
            error=f"Failed to parse JSON response: {str(e)[:200]}",
        )

    except Exception as e:
        logger.error(f"Unexpected error during read: {str(e)}")
        return ReadResult(
            success=False,
            content="",
            url_items=[],
            raw_response=str(e)[:500],
            metadata={
                "source": "jina_reader",
                "url": url,
                "status": 500,
                "execution_time": loop.time() - start_time,
                "links_found": 0,
                "relevant_links": 0,
            },
            error=f"Unexpected error: {str(e)[:200]}",
        )

from hamster_agent.app.tool.web_search import WebContentFetcher
async def fetch_web_content(url: str, timeout : int = 30) -> ReadResult:
    """Fetch web content from a URL using the WebContentFetcher."""
    fetcher = WebContentFetcher()
    
    try:
        content  = await fetcher.fetch_content(url, timeout)
        if content is not None:
            return ReadResult(
                success=True,
                content = content,
                url_items = [],
                raw_response = content,
                metadata={
                    "source": "web_content_fetcher",
                    "url": url,
                    "status": 200,
                    "execution_time": 0.0,
                    "links_found": 0,
                    "relevant_links": 0,
                }
            )
    except Exception as e:
        logger.error(f"Error fetching web content: {e}")
    finally:
        return ReadResult(
            success=False,
            content="",
            url_items=[],
            raw_response="",
            metadata={
                "source": "web_content_fetcher",
                "url": url,
                "status": 500,
                "execution_time": 0.0,
                "links_found": 0,
                "relevant_links": 0,
            },
            error=f"Error fetching web content",
        )

from hamster_tool.llm import LLM, LLMSettings
class WebReadAgent:
    """
    Agent for reading web content with LLM summarization and concurrency control.

    This agent reads webpages, extracts content, and generates summaries using
    an LLM. It includes retry logic for recoverable errors and falls back to
    truncated content if LLM summarization fails.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the WebReadAgent.

        Args:
            config: Configuration dictionary with optional keys:
                - max_concurrent_requests: Max concurrent reads (default: 500)
                - max_content_words: Max words to send to LLM (default: 10000)
                - max_summary_words: Max words for fallback summary (default: 2048)
                - max_summary_retries: Max retries for LLM (default: 3)
        """
        self.client = get_genai_client()
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
        self._timeout = config.get("timeout", 30)
        self._semaphore = asyncio.Semaphore(config.get("max_concurrent_requests", 500))
        self.max_content_words = config.get("max_content_words", 10000)
        self.max_summary_words = config.get("max_summary_words", 2048)
        self.max_summary_retries = config.get("max_summary_retries", 3)

    def _truncate_content_to_words(self, content: str, max_words: int) -> str:
        """
        Truncate content to max words, keeping beginning and end.

        Args:
            content: Content to truncate
            max_words: Maximum number of words to keep

        Returns:
            Truncated content with " ... " in the middle if exceeded
        """
        words = content.split()
        if len(words) <= max_words:
            return content

        half_words = max_words // 2
        return " ".join(words[:half_words]) + " ... " + " ".join(words[-half_words:])

    async def read(self, question: str, url: str) -> ReadResult:
        """
        Read a webpage and generate a summary based on the question.

        This method:
        1. Reads webpage content using Jina API
        2. Truncates content if too long
        3. Generates LLM summary with up to 3 retries for recoverable errors
        4. Falls back to truncated content if LLM fails

        Args:
            question: Question or context for summarization
            url: URL of the webpage to read

        Returns:
            ReadResult with content and summary (either LLM-generated or
            truncated content as fallback)

        Example:
            >>> agent = WebReadAgent(config={})
            >>> result = await agent.read("What is Python?", "https://python.org")
            >>> if result.success:
            ...     print(result.summary)
        """
        logger.info(f"Reading '{url}' with question: '{question[:100]}...'")

        try:
            async with self._semaphore:
                result = await jina_read(url.strip(), timeout=self._timeout)

            if not result.success:
                logger.warning(
                    f"Read failed for '{url}' with status "
                    f"{result.metadata.get('status', 'unknown')}: {result.error}"
                    f"try with additional fetcher."
                )
                result = await fetch_web_content(url.strip(), timeout=self._timeout)
                if not result.success:
                    logger.error(
                        f"Fallback Fetch failed for '{url}'"
                    )
                    return result

            logger.info(
                f"Read successful for '{url}', content: {len(result.content)} chars"
            )

            # Truncate content if too long
            original_content = result.content
            words = result.content.split()
            if len(words) > self.max_content_words:
                result.content = self._truncate_content_to_words(
                    result.content, self.max_content_words
                )
                logger.info(
                    f"Truncated content from {len(words)} to {self.max_content_words} words"
                )

            # Generate summary with retry logic
            for attempt in range(self.max_summary_retries):
                summary_result = await llm_summary(
                    user_prompt=f"<question>{question}</question><content>{result.content}</content>",
                    client=self.client,
                    llm = self.llm
                )

                if summary_result.success:
                    logger.info(
                        f"Summary generated for '{url}' on attempt {attempt + 1}, "
                        f"length: {len(summary_result.text)} chars"
                    )
                    result.summary = summary_result.text
                    return result

                if (
                    summary_result.recoverable
                    and attempt < self.max_summary_retries - 1
                ):
                    retry_delay = get_retry_delay(attempt, summary_result.error or "")
                    logger.warning(
                        f"Recoverable error for '{url}' "
                        f"(attempt {attempt + 1}/{self.max_summary_retries}): "
                        f"{summary_result.error}. Retrying in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.warning(
                        f"Summary failed for '{url}': {summary_result.error}"
                    )
                    break

            # Fallback to truncated content
            logger.info(f"Using truncated content as fallback for '{url}'")
            result.summary = self._truncate_content_to_words(
                original_content, self.max_summary_words
            )
            return result

        except Exception as e:
            logger.error(
                f"Unexpected error in WebReadAgent for '{url}': {str(e)}",
                exc_info=True,
            )
            return ReadResult(
                success=False,
                content="",
                url_items=[],
                raw_response="",
                metadata={
                    "source": "jina_reader",
                    "url": url,
                    "status": 500,
                    "execution_time": 0.0,
                    "links_found": 0,
                    "relevant_links": 0,
                },
                error=f"WebReadAgent error: {str(e)[:200]}",
            )
