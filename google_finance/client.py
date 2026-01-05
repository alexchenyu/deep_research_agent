"""
Google Finance Beta Client

Handles interaction with Google Finance's AI features:
- Real-time stock data fetching
- AI chatbot for quick questions
- Deep Search for complex financial research
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from playwright.async_api import async_playwright, Page, BrowserContext

try:
    from playwright_stealth import Stealth
    HAS_STEALTH = True
except ImportError:
    HAS_STEALTH = False
    logging.warning("playwright_stealth not installed. Run: pip install playwright-stealth")

from .auth import GoogleAuthManager
from .selectors import GoogleFinanceSelectors
from .config import RateLimitConfig, GOOGLE_FINANCE_BETA_URL

logger = logging.getLogger(__name__)


@dataclass
class StockData:
    """Real-time stock data from Google Finance."""
    symbol: str
    price: float
    change: float = 0.0
    change_percent: float = 0.0
    volume: Optional[int] = None
    market_cap: Optional[str] = None
    pe_ratio: Optional[float] = None
    timestamp: Optional[str] = None
    raw_data: Dict = field(default_factory=dict)


@dataclass
class AIResponse:
    """Response from Google Finance AI."""
    answer: str
    citations: List[Dict[str, str]] = field(default_factory=list)
    research_plan: Optional[str] = None
    query_time_seconds: float = 0.0
    success: bool = True
    error: Optional[str] = None


class GoogleFinanceClient:
    """Client for Google Finance Beta with AI features."""

    def __init__(self):
        self._playwright = None
        self._context: Optional[BrowserContext] = None
        self._selectors = GoogleFinanceSelectors()
        self._last_request_time = 0

    async def __aenter__(self):
        self._playwright = await async_playwright().start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close browser context and playwright."""
        if self._context:
            try:
                await self._context.close()
            except Exception as e:
                logger.warning(f"Error closing context: {e}")
            self._context = None

        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception as e:
                logger.warning(f"Error stopping playwright: {e}")
            self._playwright = None

    async def _ensure_context(self) -> BrowserContext:
        """Ensure authenticated browser context exists."""
        if not self._context:
            self._context = await GoogleAuthManager.get_authenticated_context(
                self._playwright
            )
        return self._context

    async def _new_page(self) -> Page:
        """Create new page with stealth measures."""
        context = await self._ensure_context()
        page = await context.new_page()

        if HAS_STEALTH:
            await Stealth().apply(page)

        return page

    async def _enforce_rate_limit(self):
        """Enforce minimum time between requests."""
        elapsed = time.time() - self._last_request_time
        min_interval = RateLimitConfig.MIN_REQUEST_INTERVAL_MS / 1000

        if elapsed < min_interval:
            wait_time = min_interval - elapsed
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)

        self._last_request_time = time.time()

    async def _human_type(self, page: Page, selector: str, text: str):
        """Type text with human-like delays."""
        element = await page.wait_for_selector(selector, timeout=10000)

        # Clear existing text
        await element.click()
        await page.keyboard.press('Control+A')
        await page.keyboard.press('Backspace')

        # Type with random delays
        for char in text:
            delay = random.randint(
                RateLimitConfig.TYPING_DELAY_MS[0],
                RateLimitConfig.TYPING_DELAY_MS[1]
            )
            await element.type(char, delay=delay)

    async def _wait_for_ai_response(
        self,
        page: Page,
        timeout_ms: int = 60000,
        poll_interval_ms: int = 2000
    ) -> str:
        """
        Wait for AI response with polling.

        Args:
            page: Playwright page
            timeout_ms: Maximum wait time in milliseconds
            poll_interval_ms: Polling interval

        Returns:
            Response text

        Raises:
            TimeoutError: If response not received within timeout
        """
        start = time.time()
        last_content_length = 0

        while (time.time() - start) * 1000 < timeout_ms:
            # Check if loading indicator is visible
            loading = await page.is_visible(self._selectors.AI_LOADING_INDICATOR)

            # Try to get response content
            response_elem = await page.query_selector(self._selectors.AI_RESPONSE_CONTAINER)

            if response_elem:
                content = await response_elem.text_content()

                if content:
                    content = content.strip()
                    current_length = len(content)

                    # If content is substantial and not changing, we're done
                    if current_length > 50:
                        if not loading and current_length == last_content_length:
                            # Wait a bit more to ensure response is complete
                            await page.wait_for_timeout(1000)

                            # Re-check
                            content = await response_elem.text_content()
                            if content and len(content.strip()) == current_length:
                                logger.info(f"AI response received ({current_length} chars)")
                                return content.strip()

                        last_content_length = current_length

            await page.wait_for_timeout(poll_interval_ms)

        raise TimeoutError(f"AI response not received within {timeout_ms/1000}s")

    async def _extract_citations(self, page: Page) -> List[Dict[str, str]]:
        """Extract citation links from response."""
        citations = []

        try:
            citation_elems = await page.query_selector_all(self._selectors.CITATIONS)

            for elem in citation_elems[:10]:  # Limit to 10 citations
                try:
                    href = await elem.get_attribute('href')
                    text = await elem.text_content()

                    if href:
                        citations.append({
                            'url': href,
                            'title': text.strip() if text else ''
                        })
                except Exception:
                    continue

        except Exception as e:
            logger.warning(f"Error extracting citations: {e}")

        return citations

    async def _capture_research_plan(self, page: Page) -> Optional[str]:
        """Capture research plan if shown during Deep Search."""
        try:
            await page.wait_for_selector(
                self._selectors.RESEARCH_PLAN,
                timeout=10000
            )
            plan_elem = await page.query_selector(self._selectors.RESEARCH_PLAN)

            if plan_elem:
                return await plan_elem.text_content()

        except Exception:
            pass

        return None

    # === Public API ===

    async def get_stock_data(self, symbol: str) -> StockData:
        """
        Fetch real-time stock data from Google Finance.

        Args:
            symbol: Stock ticker symbol (e.g., 'NVDA', 'AAPL')

        Returns:
            StockData with current price and metrics
        """
        await self._enforce_rate_limit()
        page = await self._new_page()

        try:
            # Try different URL formats
            urls_to_try = [
                f"{GOOGLE_FINANCE_BETA_URL}/quote/{symbol}:NASDAQ",
                f"{GOOGLE_FINANCE_BETA_URL}/quote/{symbol}:NYSE",
                f"{GOOGLE_FINANCE_BETA_URL}/quote/{symbol}",
            ]

            for url in urls_to_try:
                try:
                    await page.goto(url, wait_until='networkidle', timeout=15000)

                    # Check if page loaded successfully
                    price_elem = await page.query_selector(self._selectors.STOCK_PRICE)
                    if price_elem:
                        break

                except Exception:
                    continue

            # Extract price
            price = 0.0
            price_elem = await page.query_selector(self._selectors.STOCK_PRICE)

            if price_elem:
                price_text = await price_elem.text_content()
                if price_text:
                    # Clean up price text
                    price_text = price_text.strip().replace('$', '').replace(',', '')
                    try:
                        price = float(price_text)
                    except ValueError:
                        logger.warning(f"Could not parse price: {price_text}")

            # Extract change
            change = 0.0
            change_percent = 0.0
            change_elem = await page.query_selector(self._selectors.STOCK_CHANGE)

            if change_elem:
                change_text = await change_elem.text_content()
                if change_text:
                    # Parse change (format varies)
                    logger.debug(f"Change text: {change_text}")

            return StockData(
                symbol=symbol.upper(),
                price=price,
                change=change,
                change_percent=change_percent,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )

        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            return StockData(
                symbol=symbol.upper(),
                price=0.0,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                raw_data={'error': str(e)}
            )

        finally:
            await page.close()

    async def ask_ai(
        self,
        question: str,
        symbol_context: Optional[str] = None
    ) -> AIResponse:
        """
        Ask a question to the Google Finance AI chatbot.

        Args:
            question: Question to ask
            symbol_context: Optional stock symbol for context

        Returns:
            AIResponse with answer and citations
        """
        await self._enforce_rate_limit()
        page = await self._new_page()

        try:
            # Navigate to Finance Beta
            base_url = GOOGLE_FINANCE_BETA_URL
            if symbol_context:
                base_url = f"{base_url}/quote/{symbol_context}"

            await page.goto(base_url, wait_until='networkidle', timeout=20000)

            # Find and interact with search/chat input
            await page.wait_for_selector(self._selectors.SEARCH_INPUT, timeout=15000)

            # Human-like typing
            await self._human_type(page, self._selectors.SEARCH_INPUT, question)

            # Small delay before submit
            await page.wait_for_timeout(random.randint(200, 500))

            # Submit query
            await page.keyboard.press('Enter')

            # Wait for response
            start_time = time.time()
            response_text = await self._wait_for_ai_response(page, timeout_ms=60000)
            query_time = time.time() - start_time

            # Extract citations
            citations = await self._extract_citations(page)

            return AIResponse(
                answer=response_text,
                citations=citations,
                query_time_seconds=query_time,
                success=True
            )

        except TimeoutError as e:
            logger.error(f"Timeout waiting for AI response: {e}")
            return AIResponse(
                answer="",
                success=False,
                error=f"Response timeout: {str(e)}"
            )

        except Exception as e:
            logger.error(f"Error asking AI: {e}")
            return AIResponse(
                answer="",
                success=False,
                error=str(e)
            )

        finally:
            await page.close()

    async def deep_search(
        self,
        query: str,
        include_research_plan: bool = True,
        timeout_seconds: int = 180
    ) -> AIResponse:
        """
        Perform Deep Search using Gemini-powered analysis.

        This is for complex queries that require multi-source research.
        Can take 1-5 minutes to complete.

        Args:
            query: Complex financial research query
            include_research_plan: Whether to capture research plan
            timeout_seconds: Maximum wait time

        Returns:
            AIResponse with comprehensive answer and citations
        """
        await self._enforce_rate_limit()

        # Extra cooldown for Deep Search
        cooldown = RateLimitConfig.DEEP_SEARCH_COOLDOWN_MS / 1000
        logger.info(f"Deep Search cooldown: waiting {cooldown}s")
        await asyncio.sleep(cooldown)

        page = await self._new_page()

        try:
            await page.goto(GOOGLE_FINANCE_BETA_URL, wait_until='networkidle', timeout=20000)

            # Try to enable Deep Search if toggle exists
            try:
                deep_search_toggle = await page.query_selector(self._selectors.DEEP_SEARCH_TOGGLE)
                if deep_search_toggle:
                    await deep_search_toggle.click()
                    await page.wait_for_timeout(500)
                    logger.info("Deep Search mode enabled")
            except Exception:
                logger.debug("No Deep Search toggle found, proceeding with regular search")

            # Enter query
            await page.wait_for_selector(self._selectors.SEARCH_INPUT, timeout=15000)
            await self._human_type(page, self._selectors.SEARCH_INPUT, query)

            # Submit
            await page.wait_for_timeout(random.randint(200, 500))
            await page.keyboard.press('Enter')

            # Track research plan if requested
            research_plan = None
            if include_research_plan:
                research_plan = await self._capture_research_plan(page)
                if research_plan:
                    logger.info(f"Captured research plan: {research_plan[:100]}...")

            # Wait for complete response (longer timeout for Deep Search)
            start_time = time.time()
            response_text = await self._wait_for_ai_response(
                page,
                timeout_ms=timeout_seconds * 1000,
                poll_interval_ms=3000  # Longer poll interval for Deep Search
            )
            query_time = time.time() - start_time

            # Extract citations
            citations = await self._extract_citations(page)

            logger.info(f"Deep Search completed in {query_time:.1f}s")

            return AIResponse(
                answer=response_text,
                citations=citations,
                research_plan=research_plan,
                query_time_seconds=query_time,
                success=True
            )

        except TimeoutError as e:
            logger.error(f"Deep Search timeout: {e}")
            return AIResponse(
                answer="",
                success=False,
                error=f"Deep Search timeout after {timeout_seconds}s"
            )

        except Exception as e:
            logger.error(f"Deep Search error: {e}")
            return AIResponse(
                answer="",
                success=False,
                error=str(e)
            )

        finally:
            await page.close()


# === Convenience functions for synchronous usage ===

def get_stock_data_sync(symbol: str) -> StockData:
    """Synchronous wrapper for get_stock_data."""
    async def _run():
        async with GoogleFinanceClient() as client:
            return await client.get_stock_data(symbol)

    return asyncio.run(_run())


def ask_ai_sync(question: str, symbol_context: str = None) -> AIResponse:
    """Synchronous wrapper for ask_ai."""
    async def _run():
        async with GoogleFinanceClient() as client:
            return await client.ask_ai(question, symbol_context)

    return asyncio.run(_run())


def deep_search_sync(
    query: str,
    include_research_plan: bool = True,
    timeout_seconds: int = 180
) -> AIResponse:
    """Synchronous wrapper for deep_search."""
    async def _run():
        async with GoogleFinanceClient() as client:
            return await client.deep_search(query, include_research_plan, timeout_seconds)

    return asyncio.run(_run())
