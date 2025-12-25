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

import sys
import logging
from pathlib import Path
from typing import Optional
import re
import asyncio
import os

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel
import colorlog

def setup_colored_logger(
    name: str, level: int = logging.INFO, log_file: Optional[str | Path] = None
) -> logging.Logger:
    """
    Set up a colored logger with optional file output.

    Args:
        name: Logger name (usually __name__ from the calling module)
        level: Logging level (default: INFO)
        log_file: Optional path to log file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Disable propagation to root logger to prevent duplicate logs
    logger.propagate = False

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Color formatter for console
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "blue",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)

        # Plain formatter for file (no colors)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

logger = setup_colored_logger(__name__, level=logging.INFO)
_genai_client = None

def _is_valid_url(url: str) -> bool:
    """
    Check if a URL is valid and not a common non-informative link.

    Args:
        url (str): The URL to validate

    Returns:
        bool: True if the URL is valid and informative
    """
    if not url or not isinstance(url, str):
        return False

    url_lower = url.lower()

    # Skip common non-informative link patterns
    skip_patterns = [
        "javascript:",
        "mailto:",
        "#",
        "tel:",
        "data:",
        "blob:",
    ]

    # Skip social media and common non-content domains
    skip_domains = [
        "facebook.com",
        "twitter.com",
        "x.com",  # Twitter's new domain
        "instagram.com",
        "youtube.com",
        "tiktok.com",
        "pinterest.com",
        "snapchat.com",
        "discord.com",
        "telegram.org",
        "whatsapp.com",
        "wechat.com",
        "weibo.com",
        "douyin.com",
        "substack.com",  # Newsletter platform
        "patreon.com",
        "onlyfans.com",
        "twitch.tv",
        "vimeo.com",
        "dailymotion.com",
        "rumble.com",
        "bitchute.com",
    ]

    # Check skip patterns
    if any(pattern in url_lower for pattern in skip_patterns):
        return False

    # Check skip domains (including subdomains)
    for domain in skip_domains:
        if domain in url_lower:
            return False

    # Must start with http/https or be a relative URL starting with /
    if not (url.startswith(("http://", "https://")) or url.startswith("/")):
        return False

    return True

def get_genai_client():
    """Get or create the global GenAI client instance."""
    global _genai_client
    if _genai_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            _genai_client = genai.Client(api_key=api_key)
        else:
            _genai_client = genai.Client()
    assert _genai_client is not None, "GenAI client initialization failed"
    return _genai_client


def extract_retry_delay_from_error(error_str: str) -> Optional[float]:
    """
    Extract retry delay from Gemini API error response.
    Returns the delay in seconds if found, None otherwise.
    """
    try:
        if "RESOURCE_EXHAUSTED" in error_str and "retryDelay" in error_str:
            # check retry delay for gemini models
            # Look for retryDelay pattern in the error message
            retry_delay_match = re.search(
                r"'retryDelay': '(\d+(?:\.\d+)?)s'", error_str
            )
            if retry_delay_match:
                return float(retry_delay_match.group(1))

            # Alternative pattern matching
            retry_delay_match = re.search(
                r'retryDelay["\']?\s*:\s*["\']?(\d+(?:\.\d+)?)s', error_str
            )
            if retry_delay_match:
                return float(retry_delay_match.group(1))

    except Exception as e:
        logger.warning(f"Could not extract retry delay from error: {e}")

    return None


def get_retry_delay(try_cnt: int, error_str: str) -> float:
    """
    Calculate appropriate retry delay based on error type and attempt number.

    First tries to extract explicit retry delay from API error response.
    Falls back to exponential backoff if no delay is specified.

    Args:
        try_cnt: Current attempt number (0-indexed)
        error_str: Error message to check for retry delay hints

    Returns:
        Delay in seconds before next retry (always returns a value)
    """
    # Try to extract API-specified retry delay
    retry_delay = extract_retry_delay_from_error(error_str)
    if retry_delay is None:
        # Fallback to exponential backoff: 1s, 4s, 16s, 64s, etc.
        retry_delay = min(4**try_cnt, 60)  # Cap at 60 seconds
    return retry_delay


SYSTEM_INSTRUCTION = """You are a helpful deep research assistant. I will provide you:
* A complex question that requires a deep research to answer.
* The content of a webpage returned by our web reader.

Your task is to read the webpage content carefully and extract all information that could help answer the question. Provide detailed information including numbers, dates, facts, examples, and explanations when available. Remove the irrelevant parts to reduce noise. Note that there could be no useful information on the webpage.

Important note: Use the same language as the user's main question for the summary. For example, if the question is in Chinese, then the summary should also be in Chinese.

Now think and extract the information that could help answer the question."""

MODEL = "gemini-2.5-flash-lite"


class LLMSummaryResult(BaseModel):
    """Result from LLM summarization attempt."""

    success: bool
    text: str
    error: Optional[str] = None
    recoverable: bool = False  # Whether the error is recoverable by retrying

from hamster_tool.llm import LLM
from hamster_tool.schema import Message
async def llm_summary(
    user_prompt: str,
    client: genai.Client,
    llm : LLM,
    timeout: float = 30.0,
    model: str = MODEL,
) -> LLMSummaryResult:
    """
    Generate a summary using LLM with robust error handling.

    Args:
        user_prompt: The prompt to send to the LLM
        client: The GenAI client instance
        timeout: Request timeout in seconds (default: 30.0)
        model: Model to use for generation (default: MODEL constant)

    Returns:
        LLMSummaryResult with success status, text, and error information

    Recoverable errors (worth retrying):
        - RESOURCE_EXHAUSTED (rate limiting)
        - Timeout errors
        - Transient network errors (503, 502, connection errors)
        - Internal server errors (500)

    Non-recoverable errors (not worth retrying):
        - Content blocked by safety filters
        - Empty or invalid prompts
        - Authentication errors (401, 403)
        - Invalid request errors (400)
        - Empty responses (likely a consistent issue)
    """

    def _normalize_enum(value) -> str:
        """Normalize enum values to uppercase string names."""
        if value is None:
            return ""
        name = getattr(value, "name", None)
        if name:
            return name.upper()
        value_str = str(value)
        if "." in value_str:
            value_str = value_str.split(".")[-1]
        return value_str.upper()

    def _detect_block(response) -> Optional[str]:
        """
        Detect if response was blocked by safety filters.

        Returns:
            Block reason string if blocked, None otherwise
        """
        # Check prompt-level blocking first
        prompt_feedback = getattr(response, "prompt_feedback", None)
        if prompt_feedback:
            block_reason = _normalize_enum(
                getattr(prompt_feedback, "block_reason", None)
            )
            if block_reason and block_reason not in {
                "",
                "BLOCK_REASON_UNSPECIFIED",
                "UNSPECIFIED",
            }:
                logger.debug(f"Prompt blocked: {block_reason}")
                return f"PROMPT_{block_reason}"

            # Check prompt safety ratings
            for rating in getattr(prompt_feedback, "safety_ratings", []) or []:
                if getattr(rating, "blocked", False):
                    category = _normalize_enum(getattr(rating, "category", None))
                    logger.debug(f"Prompt safety block: {category}")
                    return f"PROMPT_{category or 'SAFETY_BLOCKED'}"

        # Check candidate-level blocking
        for candidate in getattr(response, "candidates", []) or []:
            finish_reason = _normalize_enum(getattr(candidate, "finish_reason", None))
            if finish_reason in {"SAFETY", "RECITATION", "OTHER"}:
                logger.debug(f"Candidate finish reason: {finish_reason}")
                return finish_reason

            # Check candidate safety ratings
            for rating in getattr(candidate, "safety_ratings", []) or []:
                if getattr(rating, "blocked", False):
                    category = _normalize_enum(getattr(rating, "category", None))
                    logger.debug(f"Candidate safety block: {category}")
                    return category or "SAFETY_BLOCKED"

        return None

    def _extract_text_from_candidate(candidate) -> str:
        """Extract text content from a candidate response."""
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None)
        if parts is None and isinstance(content, list):
            parts = content
        if not parts:
            return ""

        chunks: list[str] = []
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text is None and isinstance(part, dict):
                part_text = part.get("text")
            if part_text:
                chunks.append(part_text)
        return "\n".join(chunks).strip()

    def _is_recoverable_error(error_msg: str) -> bool:
        """Check if an error is recoverable by retrying."""
        error_lower = error_msg.lower()

        # Recoverable patterns
        recoverable_patterns = [
            "resource_exhausted",  # Rate limiting
            "429",  # Too Many Requests
            "503",  # Service Unavailable
            "502",  # Bad Gateway
            "500",  # Internal Server Error (may be transient)
            "timeout",
            "timed out",
            "connection",
            "network",
            "temporary",
            "transient",
            "unavailable",
            "overloaded",
            "retrydelay",
        ]

        # Non-recoverable patterns
        non_recoverable_patterns = [
            "401",  # Unauthorized
            "403",  # Forbidden
            "400",  # Bad Request
            "invalid",
            "authentication",
            "permission",
            "quota exceeded",  # Different from rate limit - permanent quota issue
            "blocked",
            "safety",
        ]

        # Check non-recoverable first (higher priority)
        if any(pattern in error_lower for pattern in non_recoverable_patterns):
            return False

        # Check recoverable patterns
        if any(pattern in error_lower for pattern in recoverable_patterns):
            return True

        # Default: not recoverable for unknown errors
        return False

    return_result = None
    # Input validation - NOT RECOVERABLE
    if not user_prompt or not user_prompt.strip():
        logger.warning("Empty or whitespace-only prompt provided to llm_summary")
        return_result = LLMSummaryResult(
            success=False,
            text="",
            error="Empty or whitespace-only prompt",
            recoverable=False,
        )

    original_length = len(user_prompt)
    MAX_PROMPT_LENGTH = 1_000_000  # 1MB limit

    if original_length > MAX_PROMPT_LENGTH:
        logger.warning(
            f"Prompt too long ({original_length:,} chars), truncating to {MAX_PROMPT_LENGTH:,} chars"
        )
        user_prompt = user_prompt[:MAX_PROMPT_LENGTH]

    try:
        # Make API request with timeout
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=model,
                contents=user_prompt,
                config=GenerateContentConfig(
                    response_modalities=["TEXT"],
                    response_mime_type="text/plain",
                    system_instruction=SYSTEM_INSTRUCTION,
                    max_output_tokens=2048,
                    temperature=0.1,
                ),
            ),
            timeout=timeout,
        )

        # Check for safety blocks - NOT RECOVERABLE
        blocked_reason = _detect_block(response)
        if blocked_reason:
            error_msg = f"Content blocked by safety filters: {blocked_reason}"
            logger.info(error_msg)
            return_result = LLMSummaryResult(
                success=False,
                text="",
                error=error_msg,
                recoverable=False,
            )

        # Try to extract text using standard method
        text = getattr(response, "text", None)

        # Fallback: manually extract from candidates
        if text is None:
            for candidate in getattr(response, "candidates", []) or []:
                text = _extract_text_from_candidate(candidate)
                if text:
                    break

        # Validate extracted text - NOT RECOVERABLE
        if text is None:
            logger.warning("No text in response")
            return_result = LLMSummaryResult(
                success=False,
                text="",
                error="No text in response",
                recoverable=False,
            )

        text = text.strip()
        if not text:
            logger.warning("LLM returned empty text response")
            return_result = LLMSummaryResult(
                success=False,
                text="",
                error="Empty text response",
                recoverable=False,
            )

        if len(text) < 10:
            logger.warning(
                f"LLM returned very short response ({len(text)} chars): '{text}'"
            )

        logger.info(
            f"LLM summary successful, "
            f"input: {original_length:,} chars, output: {len(text):,} chars"
        )
        return_result = LLMSummaryResult(success=True, text=text, recoverable=False)

    except asyncio.TimeoutError:
        # RECOVERABLE - might succeed with retry
        error_msg = f"Request timed out after {timeout}s"
        logger.warning(error_msg)
        return_result = LLMSummaryResult(
            success=False,
            text="",
            error=error_msg,
            recoverable=True,
        )

    except Exception as e:
        error_msg = str(e)
        is_recoverable = _is_recoverable_error(error_msg)

        if is_recoverable:
            logger.warning(f"Recoverable LLM error: {e}")
        else:
            logger.error(f"Non-recoverable LLM error: {e}", exc_info=True)

        return_result = LLMSummaryResult(
            success=False,
            text="",
            error=error_msg,
            recoverable=is_recoverable,
        )
    
    # fallback to use llm to generate summary
    try:
        logger.info("Attempting fallback LLM summary...")
        response = await llm.ask(
            messages=[
                Message.system_message(SYSTEM_INSTRUCTION),
                Message.user_message(user_prompt)
            ],
            temperature=0.1
        )
        text = response.strip()
        return_result = LLMSummaryResult(
            success=True,
            text=text,
            recoverable=False
        )
    except Exception as e:
        logger.error(f"Fallback LLM summary also failed: {e}", exc_info=True)
        return_result = LLMSummaryResult(
           success=False,
           text="",
           error=str(e),
           recoverable=False
        )
    return return_result
