"""
CSS selectors for Google Finance Beta UI elements.

Note: These selectors are estimates based on typical Google patterns.
They should be verified and updated by inspecting the actual page with DevTools.
"""


class GoogleFinanceSelectors:
    """CSS selectors for Google Finance Beta UI elements."""

    # Main search/chat input
    SEARCH_INPUT = 'input[aria-label*="Search"], textarea[aria-label*="Ask"], input[type="text"]'
    SEARCH_SUBMIT = 'button[aria-label*="Search"], button[type="submit"], button[jsname]'

    # Deep Search specific
    DEEP_SEARCH_TOGGLE = '[data-deep-search], button:has-text("Deep Search"), [aria-label*="Deep"]'
    DEEP_SEARCH_INPUT = 'textarea[placeholder*="Ask"], input[aria-label*="question"]'

    # AI Response containers
    AI_RESPONSE_CONTAINER = '[data-response], .response-content, [role="article"], .ai-response'
    AI_LOADING_INDICATOR = '.loading, [aria-busy="true"], .spinner, [data-loading]'
    AI_RESPONSE_COMPLETE = '[data-complete="true"], .response-complete'

    # Research plan (shown during Deep Search)
    RESEARCH_PLAN = '[data-research-plan], .research-steps, .search-plan'

    # Stock data elements (for quote pages)
    STOCK_PRICE = '[data-last-price], .YMlKec, [data-price]'
    STOCK_CHANGE = '[data-change], .P6K39c, [data-change-percent]'
    STOCK_VOLUME = '[data-volume], .volume'
    STOCK_CHART = '[data-chart], canvas.chart, svg.chart'

    # News and articles
    NEWS_CONTAINER = '[data-news], .news-feed, article'
    NEWS_HEADLINE = 'h3, .headline, [data-headline]'

    # Citation/source links
    CITATIONS = 'a[data-citation], .source-link, a[href*="source"]'

    # Follow-up questions
    FOLLOWUP_INPUT = 'input[aria-label*="follow-up"], textarea[placeholder*="follow"]'

    # Common Google UI patterns
    MATERIAL_BUTTON = '.VfPpkd-LgbsSe, button[jsname]'
    MATERIAL_INPUT = '.VfPpkd-fmcmS-wGMbrd, input[jsname]'
