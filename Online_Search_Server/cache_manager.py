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

"""
Cache Manager for Tool Server

Provides persistent caching for search and read operations.
"""

import hashlib
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import setup_colored_logger

logger = setup_colored_logger(__name__)


class CacheManager:
    """Thread-safe cache manager with persistent storage."""

    def __init__(
        self,
        cache_dir: str = "cache",
        search_config: Optional[Dict] = None,
        read_config: Optional[Dict] = None,
        save_frequency: int = 100,  # Save to disk every N operations
        stats_report_interval: int = 300,  # Report stats every N seconds (5 minutes)
    ):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory to store cache files
            search_config: Search agent configuration for cache key generation
            read_config: Read agent configuration for cache key generation
            save_frequency: Save to disk every N write operations (default: 100)
            stats_report_interval: Interval in seconds to report cache stats (default: 300)
        """
        self.cache_dir = Path(__file__).parent / Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Setup file logging
        self._setup_file_logging()

        # Store configurations for cache key generation
        self.search_config = search_config or {}
        self.read_config = read_config or {}

        # Save frequency configuration
        self.save_frequency = save_frequency
        self.stats_report_interval = stats_report_interval

        # Separate cache files for each operation type
        self.search_cache_file = self.cache_dir / "search_cache.json"
        self.read_cache_file = self.cache_dir / "read_cache.json"

        # In-memory caches
        self._search_cache: Dict[str, Any] = {}
        self._read_cache: Dict[str, Any] = {}

        # Write counters for periodic saving
        self._search_write_count = 0
        self._read_write_count = 0

        # Dirty flags to track if cache needs saving
        self._search_dirty = False
        self._read_dirty = False

        # Hit rate tracking
        self._search_hits = 0
        self._search_misses = 0
        self._read_hits = 0
        self._read_misses = 0

        # Thread locks for thread safety
        self._search_lock = threading.Lock()
        self._read_lock = threading.Lock()

        # Stats reporting thread
        self._stop_stats_thread = threading.Event()
        self._stats_thread = None

        # Load existing caches
        self._load_caches()

        logger.info(f"Cache manager initialized with directory: {self.cache_dir}")
        logger.info(f"Save frequency: every {self.save_frequency} write operations")
        logger.info(
            f"Stats report interval: every {self.stats_report_interval} seconds"
        )
        logger.info(
            f"Loaded {len(self._search_cache)} search, {len(self._read_cache)} read entries"
        )

        # Start periodic stats reporting
        self._start_stats_reporting()

    def _setup_file_logging(self):
        """Setup file logging for cache operations."""
        import logging

        # Create logs directory
        log_dir = Path(__file__).parent / Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Add file handler to logger
        file_handler = logging.FileHandler(log_dir / "cache.log")
        file_handler.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(file_handler)

        logger.info("File logging initialized at logs/cache.log")

    def _start_stats_reporting(self):
        """Start background thread for periodic stats reporting."""
        if self.stats_report_interval > 0:
            self._stats_thread = threading.Thread(
                target=self._stats_reporting_loop,
                daemon=True,
                name="CacheStatsReporter",
            )
            self._stats_thread.start()
            logger.debug("Stats reporting thread started")

    def _stats_reporting_loop(self):
        """Background thread loop for periodic stats reporting."""
        while not self._stop_stats_thread.wait(self.stats_report_interval):
            try:
                self._report_stats()
            except Exception as e:
                logger.error(f"Error reporting cache stats: {e}")

    def _report_stats(self):
        """Generate and log a stats report."""
        stats = self.get_stats()
        pending = stats["pending_writes"]  # Extract pending writes

        logger.info("=" * 60)
        logger.info("CACHE STATISTICS REPORT")
        logger.info("=" * 60)

        # Entry counts
        logger.info("Cache Entries:")
        logger.info(f"  Search: {stats['entries']['search']}")
        logger.info(f"  Read:   {stats['entries']['read']}")
        logger.info(f"  Total:  {stats['entries']['total']}")

        # Hit rates
        logger.info("")
        logger.info("Hit Rates:")
        for cache_type in ["search", "read", "overall"]:
            hr = stats["hit_rates"][cache_type]
            rate_pct = hr["rate"] * 100
            logger.info(
                f"  {cache_type.capitalize():8s}: "
                f"{hr['hits']:4d} hits, {hr['misses']:4d} misses, "
                f"{hr['total']:4d} total ({rate_pct:5.1f}%)"
            )

        # Pending writes
        logger.info("")
        logger.info("Pending Writes:")
        logger.info(f"  Search: {pending['search']}/{self.save_frequency}")
        logger.info(f"  Read:   {pending['read']}/{self.save_frequency}")

        logger.info("=" * 60)

    def _load_caches(self):
        """Load all cache files from disk."""
        self._search_cache = self._load_cache_file(self.search_cache_file)
        self._read_cache = self._load_cache_file(self.read_cache_file)

    def _load_cache_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Load a cache file from disk.

        Args:
            file_path: Path to cache file

        Returns:
            Dictionary containing cache data
        """
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading cache from {file_path}: {e}")
                return {}
        return {}

    # def _save_cache_file(self, cache_data: Dict[str, Any], file_path: Path):
    #     """
    #     Save cache data to disk.

    #     Args:
    #         cache_data: Cache dictionary to save
    #         file_path: Path to save cache file
    #     """
    #     try:
    #         with open(file_path, "w", encoding="utf-8") as f:
    #             json.dump(cache_data, f, indent=2, ensure_ascii=False)
    #     except Exception as e:
    #         logger.error(f"Error saving cache to {file_path}: {e}")
    def _save_cache_file(self, cache_data: Dict[str, Any], file_path: Path):
        """
        原子覆盖写：写入临时文件后替换，避免 JSON 被追加破坏或并发时损坏。
        """
        try:
            tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
                f.flush()
                import os
                os.fsync(f.fileno())  # 确保刷到磁盘
            tmp_path.replace(file_path)  # 原子替换
        except Exception as e:
            logger.error(f"Error saving cache to {file_path}: {e}")
            # 这里也可以选择 raise，视你希望的失败策略而定

    def _maybe_save_search_cache(self):
        """Save search cache if write count reaches frequency threshold."""
        if self._search_write_count >= self.save_frequency:
            self._save_cache_file(self._search_cache, self.search_cache_file)
            self._search_write_count = 0
            self._search_dirty = False
            logger.debug(f"Saved search cache ({len(self._search_cache)} entries)")

    def _maybe_save_read_cache(self):
        """Save read cache if write count reaches frequency threshold."""
        if self._read_write_count >= self.save_frequency:
            self._save_cache_file(self._read_cache, self.read_cache_file)
            self._read_write_count = 0
            self._read_dirty = False
            logger.debug(f"Saved read cache ({len(self._read_cache)} entries)")

    def _generate_cache_key(self, config: Dict[str, Any], **kwargs) -> str:
        """
        Generate a cache key from configuration and keyword arguments.
        Uses SHA-256 for better collision resistance than MD5.

        Args:
            config: Agent configuration to include in key
            **kwargs: Parameters to hash

        Returns:
            SHA-256 hash string (64 characters)
        """
        # Combine config and kwargs for hashing
        cache_params = {"config": config, "params": kwargs}
        # Sort keys for consistent hashing
        key_string = json.dumps(cache_params, sort_keys=True, ensure_ascii=False)
        # Use SHA-256 instead of MD5 for better collision resistance
        return hashlib.sha256(key_string.encode("utf-8")).hexdigest()

    def _validate_cache_entry(
        self,
        entry: Dict[str, Any],
        expected_config: Dict[str, Any],
        expected_params: Dict[str, Any],
    ) -> bool:
        """
        Validate that a cache entry matches expected configuration and parameters.
        This provides an extra safety check against hash collisions.

        Args:
            entry: Cache entry to validate
            expected_config: Expected configuration
            expected_params: Expected parameters

        Returns:
            True if entry is valid, False otherwise
        """
        stored_config = entry.get("config", {})

        # Check if configurations match
        if stored_config != expected_config:
            logger.warning("Cache entry config mismatch detected")
            return False

        # Check if key parameters match
        for key, value in expected_params.items():
            if entry.get(key) != value:
                logger.warning(f"Cache entry parameter mismatch: {key}")
                return False

        return True

    def get_search(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get cached search result.

        Args:
            query: Search query

        Returns:
            Cached result or None if not found
        """

        cache_key = self._generate_cache_key(config=self.search_config, query=query)

        with self._search_lock:
            entry = self._search_cache.get(cache_key)
            if entry:
                # Validate entry to protect against hash collisions
                if self._validate_cache_entry(
                    entry, self.search_config, {"query": query}
                ):
                    self._search_hits += 1
                    logger.debug(f"Cache hit for search: {query[:50]}...")
                    return entry.get("result")
                else:
                    logger.warning(
                        f"Cache entry validation failed for search: {query[:50]}..."
                    )
                    # Remove invalid entry
                    del self._search_cache[cache_key]
                    self._search_dirty = True
                    self._search_write_count += 1
                    self._maybe_save_search_cache()

            self._search_misses += 1
            logger.debug(f"Cache miss for search: {query[:50]}...")
            return None

    def set_search(self, query: str, result: Dict[str, Any]):
        """
        Cache a search result.

        Args:
            query: Search query
            result: Result to cache
        """
        cache_key = self._generate_cache_key(config=self.search_config, query=query)

        entry = {
            "result": result,
            "query": query,
            "config": self.search_config,
            "cached_at": datetime.now().isoformat(),
        }

        with self._search_lock:
            self._search_cache[cache_key] = entry
            self._search_write_count += 1
            self._search_dirty = True
            self._maybe_save_search_cache()

        logger.debug(f"Cached search result for: {query[:50]}...")

    def get_read(self, url: str, question: str) -> Optional[Dict[str, Any]]:
        """
        Get cached read result.

        Args:
            url: URL to read
            question: Question asked

        Returns:
            Cached result or None if not found
        """
        cache_key = self._generate_cache_key(
            config=self.read_config, url=url, question=question
        )

        with self._read_lock:
            entry = self._read_cache.get(cache_key)
            if entry:
                # Validate entry to protect against hash collisions
                if self._validate_cache_entry(
                    entry, self.read_config, {"url": url, "question": question}
                ):
                    self._read_hits += 1
                    logger.debug(f"Cache hit for read: {url}")
                    return entry.get("result")
                else:
                    logger.warning(f"Cache entry validation failed for read: {url}")
                    # Remove invalid entry
                    del self._read_cache[cache_key]
                    self._read_dirty = True
                    self._read_write_count += 1
                    self._maybe_save_read_cache()

            self._read_misses += 1
            logger.debug(f"Cache miss for read: {url}")
            return None

    def set_read(self, url: str, question: str, result: Dict[str, Any]):
        """
        Cache a read result.

        Args:
            url: URL that was read
            question: Question that was asked
            result: Result to cache
        """
        cache_key = self._generate_cache_key(
            config=self.read_config, url=url, question=question
        )

        entry = {
            "result": result,
            "url": url,
            "question": question,
            "config": self.read_config,
            "cached_at": datetime.now().isoformat(),
        }

        with self._read_lock:
            self._read_cache[cache_key] = entry
            self._read_write_count += 1
            self._read_dirty = True
            self._maybe_save_read_cache()

        logger.debug(f"Cached read result for: {url}")

    def flush_all(self):
        """Force save all dirty caches to disk."""
        with self._search_lock:
            if self._search_dirty:
                self._save_cache_file(self._search_cache, self.search_cache_file)
                self._search_write_count = 0
                self._search_dirty = False

        with self._read_lock:
            if self._read_dirty:
                self._save_cache_file(self._read_cache, self.read_cache_file)
                self._read_write_count = 0
                self._read_dirty = False

        logger.info("All caches flushed to disk")

    def clear_all(self):
        """Clear all caches (in-memory and on disk). Use with caution!"""
        with self._search_lock, self._read_lock:
            self._search_cache.clear()
            self._read_cache.clear()

            self._save_cache_file(self._search_cache, self.search_cache_file)
            self._save_cache_file(self._read_cache, self.read_cache_file)

            self._search_write_count = 0
            self._read_write_count = 0

            self._search_dirty = False
            self._read_dirty = False

            # Reset hit rate counters
            self._search_hits = 0
            self._search_misses = 0
            self._read_hits = 0
            self._read_misses = 0

        logger.info("All caches cleared (in-memory and disk)")

    def invalidate_old_entries(self, max_age_days: int = 30):
        """
        Remove cache entries older than specified age.

        Args:
            max_age_days: Maximum age in days for cache entries

        Returns:
            Number of entries removed
        """
        from datetime import timedelta

        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        removed_count = 0

        with self._search_lock:
            keys_to_remove = []
            for key, entry in self._search_cache.items():
                cached_at = entry.get("cached_at")
                if cached_at:
                    try:
                        if datetime.fromisoformat(cached_at) < cutoff_date:
                            keys_to_remove.append(key)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid cached_at value in search cache: {e}")

            for key in keys_to_remove:
                del self._search_cache[key]
            removed_count += len(keys_to_remove)
            if keys_to_remove:
                self._search_dirty = True
                # Force save by setting counter to threshold
                self._search_write_count = self.save_frequency
                self._maybe_save_search_cache()

        with self._read_lock:
            keys_to_remove = []
            for key, entry in self._read_cache.items():
                cached_at = entry.get("cached_at")
                if cached_at:
                    try:
                        if datetime.fromisoformat(cached_at) < cutoff_date:
                            keys_to_remove.append(key)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid cached_at value in read cache: {e}")

            for key in keys_to_remove:
                del self._read_cache[key]
            removed_count += len(keys_to_remove)
            if keys_to_remove:
                self._read_dirty = True
                # Force save by setting counter to threshold
                self._read_write_count = self.save_frequency
                self._maybe_save_read_cache()

        logger.info(
            f"Removed {removed_count} cache entries older than {max_age_days} days"
        )
        return removed_count

    def reset_stats(self):
        """Reset hit rate statistics without clearing cache entries."""
        with self._search_lock, self._read_lock:
            self._search_hits = 0
            self._search_misses = 0
            self._read_hits = 0
            self._read_misses = 0

        logger.info("Cache statistics reset")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics including hit rates.

        Returns:
            Dictionary with cache entry counts and hit rates
        """
        # Calculate hit rates
        search_total = self._search_hits + self._search_misses
        read_total = self._read_hits + self._read_misses

        search_hit_rate = self._search_hits / search_total if search_total > 0 else 0.0
        read_hit_rate = self._read_hits / read_total if read_total > 0 else 0.0

        overall_hits = self._search_hits + self._read_hits
        overall_total = search_total + read_total
        overall_hit_rate = overall_hits / overall_total if overall_total > 0 else 0.0

        return {
            "entries": {
                "search": len(self._search_cache),
                "read": len(self._read_cache),
                "total": len(self._search_cache) + len(self._read_cache),
            },
            "hit_rates": {
                "search": {
                    "hits": self._search_hits,
                    "misses": self._search_misses,
                    "total": search_total,
                    "rate": round(search_hit_rate, 3),
                },
                "read": {
                    "hits": self._read_hits,
                    "misses": self._read_misses,
                    "total": read_total,
                    "rate": round(read_hit_rate, 3),
                },
                "overall": {
                    "hits": overall_hits,
                    "misses": self._search_misses + self._read_misses,
                    "total": overall_total,
                    "rate": round(overall_hit_rate, 3),
                },
            },
            "pending_writes": {
                "search": self._search_write_count,
                "read": self._read_write_count,
            },
        }

    def shutdown(self):
        """Gracefully shutdown the cache manager."""
        try:
            logger.info("Shutting down cache manager...")

            # Stop stats reporting thread
            if self._stats_thread and self._stats_thread.is_alive():
                self._stop_stats_thread.set()
                self._stats_thread.join(timeout=5)
                logger.debug("Stats reporting thread stopped")

            # Generate final stats report
            self._report_stats()

            # Flush all caches
            self.flush_all()

            logger.info("Cache manager shutdown complete")
        except Exception:
            # Suppress errors during shutdown (e.g., closed log files)
            pass

    def __del__(self):
        """Cleanup on deletion."""
        try:
            # Check if attributes exist before accessing
            if hasattr(self, "_search_cache"):
                self.shutdown()
        except Exception:
            pass  # Suppress all errors during cleanup
