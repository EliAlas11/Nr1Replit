"""
Utility to extract YouTube video ID from a URL.
"""

import re

def extract_video_id(url: str) -> str | None:
    """Extract the YouTube video ID from a URL string. Returns None if not found."""
    pattern = r"(?:v=|youtu\.be/|embed/|shorts/)([\w-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None
