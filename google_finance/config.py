"""
Configuration constants for Google Finance integration.
"""


class RateLimitConfig:
    """Rate limiting and anti-bot configuration."""

    # Request limits
    MIN_REQUEST_INTERVAL_MS = 3000      # Minimum 3s between requests
    MAX_REQUESTS_PER_MINUTE = 10        # Conservative limit
    DEEP_SEARCH_COOLDOWN_MS = 30000     # 30s between Deep Search queries

    # Backoff strategy
    INITIAL_BACKOFF_MS = 5000
    MAX_BACKOFF_MS = 300000             # 5 minutes max
    BACKOFF_MULTIPLIER = 2.0

    # Human-like delays
    TYPING_DELAY_MS = (50, 150)         # Random delay per character
    PRE_CLICK_DELAY_MS = (200, 500)     # Delay before clicking
    POST_ACTION_DELAY_MS = (500, 1500)  # Delay after actions


# Google Finance URLs
GOOGLE_FINANCE_BETA_URL = "https://www.google.com/finance/beta"
GOOGLE_ACCOUNTS_URL = "https://accounts.google.com"
