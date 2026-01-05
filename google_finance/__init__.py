"""
Google Finance Beta Integration Module

Provides tools for interacting with Google Finance Beta's AI features:
- Real-time stock data
- AI chatbot for quick questions
- Deep Search for complex financial research
"""

from .client import GoogleFinanceClient, StockData, AIResponse
from .auth import GoogleAuthManager
from .config import RateLimitConfig
from .selectors import GoogleFinanceSelectors

__all__ = [
    'GoogleFinanceClient',
    'StockData',
    'AIResponse',
    'GoogleAuthManager',
    'RateLimitConfig',
    'GoogleFinanceSelectors',
]
