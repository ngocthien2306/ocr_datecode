"""
Redis connection and cache management
"""

import json
import hashlib
import logging
from typing import Optional, Any
from datetime import datetime, timedelta, timezone
from redis import Redis
from redis.exceptions import RedisError

from app.core.config import settings

logger = logging.getLogger(__name__)

# Global Redis client
redis_client: Optional[Redis] = None


def get_redis() -> Optional[Redis]:
    """
    Get Redis client instance

    Returns:
        Redis client or None if connection failed
    """
    global redis_client

    if redis_client is None:
        try:
            redis_client = Redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            redis_client.ping()
            logger.info("✅ Redis connected successfully")
        except RedisError as e:
            logger.warning(f"⚠️ Redis connection failed: {e}. Cache will be disabled.")
            redis_client = None

    return redis_client


def close_redis():
    """Close Redis connection"""
    global redis_client
    if redis_client:
        try:
            redis_client.close()
            logger.info("✅ Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis: {e}")
        finally:
            redis_client = None


def generate_cache_key(prefix: str, params: dict) -> str:
    """
    Generate cache key from prefix and parameters

    Args:
        prefix: Cache key prefix (e.g., "stats:summary")
        params: Parameters to hash

    Returns:
        Cache key string
    """
    # Sort params for consistent hashing
    sorted_params = json.dumps(params, sort_keys=True, default=str)
    params_hash = hashlib.md5(sorted_params.encode()).hexdigest()[:16]
    return f"{prefix}:{params_hash}"


def get_cache_ttl(end_date: Optional[datetime]) -> int:
    """
    Determine cache TTL based on date range

    Args:
        end_date: End date of query range

    Returns:
        TTL in seconds
    """
    if end_date is None:
        return settings.REDIS_CACHE_TTL_RECENT

    # Get current time as timezone-aware UTC
    now = datetime.now(timezone.utc)

    # Make end_date timezone-aware if it's naive
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    # If end date is within last 24 hours, use recent TTL
    if (now - end_date) < timedelta(days=1):
        return settings.REDIS_CACHE_TTL_RECENT

    # Otherwise use historical TTL
    return settings.REDIS_CACHE_TTL_HISTORICAL


async def get_cached_data(cache_key: str) -> Optional[dict]:
    """
    Get cached data from Redis

    Args:
        cache_key: Cache key

    Returns:
        Cached data or None if not found/error
    """
    client = get_redis()
    if not client:
        return None

    try:
        cached = client.get(cache_key)
        if cached:
            logger.debug(f"✅ Cache HIT: {cache_key}")
            return json.loads(cached)
        logger.debug(f"❌ Cache MISS: {cache_key}")
        return None
    except Exception as e:
        logger.warning(f"Error reading from cache: {e}")
        return None


async def set_cached_data(cache_key: str, data: dict, ttl: int) -> bool:
    """
    Set cached data in Redis

    Args:
        cache_key: Cache key
        data: Data to cache
        ttl: Time to live in seconds

    Returns:
        True if successful
    """
    client = get_redis()
    if not client:
        return False

    try:
        serialized = json.dumps(data, default=str)
        client.setex(cache_key, ttl, serialized)
        logger.debug(f"✅ Cache SET: {cache_key} (TTL: {ttl}s)")
        return True
    except Exception as e:
        logger.warning(f"Error writing to cache: {e}")
        return False


async def invalidate_cache_pattern(pattern: str) -> int:
    """
    Invalidate all cache keys matching pattern

    Args:
        pattern: Pattern to match (e.g., "stats:*")

    Returns:
        Number of keys deleted
    """
    client = get_redis()
    if not client:
        return 0

    try:
        keys = client.keys(pattern)
        if keys:
            deleted = client.delete(*keys)
            logger.info(f"🗑️ Invalidated {deleted} cache keys matching: {pattern}")
            return deleted
        return 0
    except Exception as e:
        logger.warning(f"Error invalidating cache: {e}")
        return 0
