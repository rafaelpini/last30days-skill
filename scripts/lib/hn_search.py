"""Hacker News search via Algolia API for last30days skill.

Uses the free Algolia HN Search API - no API key required.
https://hn.algolia.com/api
"""

import sys
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from . import http

# Algolia HN Search API endpoint
HN_API_URL = "https://hn.algolia.com/api/v1/search_by_date"

# Depth configuration: (min_results, max_results) per depth level
DEPTH_CONFIG = {
    "quick": (8, 12),
    "default": (20, 30),
    "deep": (50, 70),
}


def _log_error(msg: str):
    """Log error to stderr."""
    sys.stderr.write(f"[HN] ERROR: {msg}\n")
    sys.stderr.flush()


def _log_info(msg: str):
    """Log info to stderr."""
    sys.stderr.write(f"[HN] {msg}\n")
    sys.stderr.flush()


def _date_to_timestamp(date_str: str) -> int:
    """Convert YYYY-MM-DD to Unix timestamp."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp())


def _timestamp_to_date(ts: int) -> str:
    """Convert Unix timestamp to YYYY-MM-DD."""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def search_hn(
    topic: str,
    from_date: str,
    to_date: str,
    depth: str = "default",
    mock_response: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Search Hacker News via Algolia API.

    Args:
        topic: Search query
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        depth: 'quick', 'default', or 'deep'
        mock_response: Optional mock response for testing

    Returns:
        Raw API response dict
    """
    if mock_response is not None:
        return mock_response

    min_results, max_results = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["default"])

    # Convert dates to timestamps for numeric filter
    from_ts = _date_to_timestamp(from_date)
    to_ts = _date_to_timestamp(to_date) + 86400  # Include full end day

    # Build query params
    params = {
        "query": topic,
        "tags": "story",  # Only stories (not comments)
        "numericFilters": f"created_at_i>={from_ts},created_at_i<={to_ts}",
        "hitsPerPage": max_results,
    }

    url = f"{HN_API_URL}?{urlencode(params)}"

    try:
        response = http.get(url, timeout=30)
        return response
    except http.HTTPError as e:
        _log_error(f"API error: {e}")
        return {"error": str(e), "hits": []}
    except Exception as e:
        _log_error(f"{type(e).__name__}: {e}")
        return {"error": str(e), "hits": []}


def parse_hn_response(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse Algolia HN API response into normalized items.

    Args:
        response: Raw API response

    Returns:
        List of normalized HN item dicts
    """
    if "error" in response:
        return []

    hits = response.get("hits", [])
    items = []

    for i, hit in enumerate(hits):
        # Skip items without essential fields
        if not hit.get("title") and not hit.get("story_title"):
            continue

        # Get the URL - prefer the linked URL, fall back to HN discussion
        object_id = hit.get("objectID", "")
        story_url = hit.get("url") or f"https://news.ycombinator.com/item?id={object_id}"
        hn_url = f"https://news.ycombinator.com/item?id={object_id}"

        # Parse date
        created_at = hit.get("created_at_i")
        date_str = None
        if created_at:
            try:
                date_str = _timestamp_to_date(created_at)
            except (ValueError, OSError):
                pass

        # Build item
        item = {
            "id": f"HN{i + 1}",
            "title": hit.get("title") or hit.get("story_title") or "",
            "url": story_url,
            "hn_url": hn_url,  # Always include HN discussion link
            "author": hit.get("author") or "",
            "date": date_str,
            "engagement": {
                "score": hit.get("points"),
                "num_comments": hit.get("num_comments"),
            },
            "relevance": 0.7,  # Default relevance (HN results are pre-filtered by Algolia)
            "why_relevant": f"Hacker News discussion about {hit.get('title', 'topic')[:50]}",
        }

        items.append(item)

    return items
