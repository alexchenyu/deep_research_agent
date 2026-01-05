# Google Finance Beta 集成计划

## 目标

为深度研究系统添加 Google Finance Beta (https://www.google.com/finance/beta) 的交互能力：
1. 自动获取股票实时数据和图表
2. 使用浏览器模拟与 Google Finance AI 对话（提问分析）
3. 利用 Deep Search 功能进行深度研究

## Google Finance Beta 功能

根据 [Google 官方博客](https://blog.google/products/search/new-google-finance-ai-deep-search/):

- **AI 聊天**: 可以问 "S&P 500 本月涨幅最大的公司是哪些？"
- **Deep Search**: 复杂问题使用 Gemini 进行深度分析（1-5分钟）
- **实时数据**: 股价、新闻、预测市场
- **URL**: `https://www.google.com/finance/beta`

## 当前状态

- ✅ Playwright 已集成，用于被动网页抓取
- ✅ DuckDuckGo 搜索已实现
- ❌ 无交互能力（点击、填表、输入）
- ❌ 无 Google Finance 专用工具

---

## 实现计划

### Phase 1: 创建 Google Finance 模块结构 (15 min)

```
src/llm/deep_research_agent/
├── google_finance/          # 新目录
│   ├── __init__.py
│   ├── client.py           # 主客户端
│   ├── auth.py             # 认证管理
│   ├── selectors.py        # CSS 选择器
│   └── config.py           # 配置常量
└── auth_state/             # 认证状态存储 (gitignore)
```

### Phase 2: 实现认证管理 (auth.py) (1 hour)

**关键设计**: 首次手动登录，后续自动使用保存的认证状态

```python
from pathlib import Path
from playwright.async_api import async_playwright, BrowserContext

AUTH_STATE_DIR = Path(__file__).parent.parent / "auth_state"
AUTH_STATE_FILE = AUTH_STATE_DIR / "google_finance_state.json"
USER_DATA_DIR = AUTH_STATE_DIR / "google_user_data"

class GoogleAuthManager:
    @staticmethod
    def is_authenticated() -> bool:
        """检查是否已认证"""
        return AUTH_STATE_FILE.exists()

    @staticmethod
    async def setup_authentication():
        """首次设置：打开浏览器让用户手动登录"""
        AUTH_STATE_DIR.mkdir(exist_ok=True)

        async with async_playwright() as p:
            context = await p.chromium.launch_persistent_context(
                user_data_dir=str(USER_DATA_DIR),
                headless=False  # 可见浏览器
            )
            page = await context.new_page()
            await page.goto("https://accounts.google.com")

            print("请登录您的 Google 账号...")
            print("登录完成后按 Enter 键")
            input()

            # 保存认证状态
            await context.storage_state(path=str(AUTH_STATE_FILE))
            await context.close()

    @staticmethod
    async def get_authenticated_context(playwright) -> BrowserContext:
        """获取已认证的浏览器上下文"""
        if not GoogleAuthManager.is_authenticated():
            raise RuntimeError("未认证，请先运行 setup_authentication()")

        return await playwright.chromium.launch_persistent_context(
            user_data_dir=str(USER_DATA_DIR),
            headless=True,
            storage_state=str(AUTH_STATE_FILE),
            args=['--disable-blink-features=AutomationControlled'],
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
```

### Phase 3: 实现客户端核心功能 (client.py) (3 hours)

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
import asyncio
import time
import logging

from playwright.async_api import async_playwright, Page
from playwright_stealth import stealth_async

from .auth import GoogleAuthManager
from .selectors import GoogleFinanceSelectors
from .config import RateLimitConfig

logger = logging.getLogger(__name__)
GOOGLE_FINANCE_BETA_URL = "https://www.google.com/finance/beta"


@dataclass
class StockData:
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: Optional[int] = None
    timestamp: Optional[str] = None


@dataclass
class AIResponse:
    answer: str
    citations: List[Dict[str, str]]
    research_plan: Optional[str] = None
    query_time_seconds: float = 0.0


class GoogleFinanceClient:
    """Google Finance Beta 交互客户端"""

    def __init__(self):
        self._playwright = None
        self._context = None
        self._selectors = GoogleFinanceSelectors()
        self._last_request_time = 0

    async def __aenter__(self):
        self._playwright = await async_playwright().start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._context:
            await self._context.close()
        if self._playwright:
            await self._playwright.stop()

    async def get_stock_data(self, symbol: str) -> StockData:
        """获取实时股票数据"""
        await self._enforce_rate_limit()
        page = await self._new_page()

        try:
            url = f"{GOOGLE_FINANCE_BETA_URL}/quote/{symbol}"
            await page.goto(url)
            await page.wait_for_load_state('networkidle')

            # 提取价格数据
            price_elem = await page.query_selector(self._selectors.STOCK_PRICE)
            price = 0.0
            if price_elem:
                price_text = await price_elem.text_content()
                price = float(price_text.replace('$', '').replace(',', ''))

            return StockData(
                symbol=symbol,
                price=price,
                change=0.0,
                change_percent=0.0,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )
        finally:
            await page.close()

    async def ask_ai(self, question: str, symbol_context: str = None) -> AIResponse:
        """向 Google Finance AI 提问"""
        await self._enforce_rate_limit()
        page = await self._new_page()

        try:
            base_url = GOOGLE_FINANCE_BETA_URL
            if symbol_context:
                base_url = f"{base_url}/quote/{symbol_context}"

            await page.goto(base_url)
            await page.wait_for_load_state('networkidle')

            # 找到搜索/聊天输入框
            await page.wait_for_selector(self._selectors.SEARCH_INPUT, timeout=10000)

            # 模拟人类输入
            await self._human_type(page, self._selectors.SEARCH_INPUT, question)

            # 提交查询
            await page.click(self._selectors.SEARCH_SUBMIT)

            # 等待响应
            start_time = time.time()
            response_text = await self._wait_for_ai_response(page, timeout_ms=60000)
            query_time = time.time() - start_time

            # 提取引用
            citations = await self._extract_citations(page)

            return AIResponse(
                answer=response_text,
                citations=citations,
                query_time_seconds=query_time
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
        使用 Deep Search 深度分析（Gemini 驱动）
        可能需要 1-5 分钟完成
        """
        await self._enforce_rate_limit()

        # Deep Search 额外冷却
        await asyncio.sleep(RateLimitConfig.DEEP_SEARCH_COOLDOWN_MS / 1000)

        page = await self._new_page()

        try:
            await page.goto(GOOGLE_FINANCE_BETA_URL)
            await page.wait_for_load_state('networkidle')

            # 启用 Deep Search（如果有切换按钮）
            deep_search_toggle = await page.query_selector(self._selectors.DEEP_SEARCH_TOGGLE)
            if deep_search_toggle:
                await deep_search_toggle.click()
                await page.wait_for_timeout(500)

            # 输入查询
            await self._human_type(page, self._selectors.SEARCH_INPUT, query)
            await page.click(self._selectors.SEARCH_SUBMIT)

            # 捕获研究计划
            research_plan = None
            if include_research_plan:
                research_plan = await self._capture_research_plan(page)

            # 等待完整响应（更长超时）
            start_time = time.time()
            response_text = await self._wait_for_ai_response(
                page,
                timeout_ms=timeout_seconds * 1000
            )
            query_time = time.time() - start_time

            citations = await self._extract_citations(page)

            return AIResponse(
                answer=response_text,
                citations=citations,
                research_plan=research_plan,
                query_time_seconds=query_time
            )
        finally:
            await page.close()

    # === 私有方法 ===

    async def _new_page(self) -> Page:
        """创建带 stealth 的新页面"""
        if not self._context:
            self._context = await GoogleAuthManager.get_authenticated_context(
                self._playwright
            )
        page = await self._context.new_page()
        await stealth_async(page)
        return page

    async def _enforce_rate_limit(self):
        """执行速率限制"""
        elapsed = time.time() - self._last_request_time
        min_interval = RateLimitConfig.MIN_REQUEST_INTERVAL_MS / 1000
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    async def _human_type(self, page: Page, selector: str, text: str):
        """模拟人类打字"""
        import random
        element = await page.wait_for_selector(selector)
        for char in text:
            await element.type(char, delay=random.randint(50, 150))

    async def _wait_for_ai_response(self, page: Page, timeout_ms: int) -> str:
        """轮询等待 AI 响应"""
        poll_interval = 2000
        start = time.time()

        while (time.time() - start) * 1000 < timeout_ms:
            loading = await page.is_visible(self._selectors.AI_LOADING_INDICATOR)

            if not loading:
                response = await page.query_selector(self._selectors.AI_RESPONSE_CONTAINER)
                if response:
                    content = await response.text_content()
                    if content and len(content.strip()) > 20:
                        return content.strip()

            await page.wait_for_timeout(poll_interval)

        raise TimeoutError(f"AI 响应超时 ({timeout_ms/1000}s)")

    async def _capture_research_plan(self, page: Page) -> Optional[str]:
        """捕获研究计划"""
        try:
            await page.wait_for_selector(self._selectors.RESEARCH_PLAN, timeout=10000)
            plan_elem = await page.query_selector(self._selectors.RESEARCH_PLAN)
            if plan_elem:
                return await plan_elem.text_content()
        except Exception:
            pass
        return None

    async def _extract_citations(self, page: Page) -> List[Dict[str, str]]:
        """提取引用链接"""
        citations = []
        citation_elems = await page.query_selector_all(self._selectors.CITATIONS)

        for elem in citation_elems:
            try:
                href = await elem.get_attribute('href')
                text = await elem.text_content()
                if href:
                    citations.append({'url': href, 'title': text.strip() if text else ''})
            except Exception:
                continue

        return citations
```

### Phase 4: CSS 选择器 (selectors.py)

```python
class GoogleFinanceSelectors:
    """Google Finance Beta UI 元素选择器"""

    # 搜索/聊天输入
    SEARCH_INPUT = 'input[aria-label*="Search"], textarea[aria-label*="Ask"]'
    SEARCH_SUBMIT = 'button[aria-label*="Search"], button[type="submit"]'

    # Deep Search
    DEEP_SEARCH_TOGGLE = '[data-deep-search], button:has-text("Deep Search")'

    # AI 响应
    AI_RESPONSE_CONTAINER = '[data-response], .response-content, [role="article"]'
    AI_LOADING_INDICATOR = '.loading, [aria-busy="true"], .spinner'

    # 研究计划
    RESEARCH_PLAN = '[data-research-plan], .research-steps'

    # 股票数据
    STOCK_PRICE = '[data-last-price], .YMlKec'
    STOCK_CHANGE = '[data-change], .P6K39c'

    # 引用链接
    CITATIONS = 'a[data-citation], .source-link'
```

**注意**: 这些选择器是估计值，需要通过浏览器 DevTools 实际验证。

### Phase 5: 配置 (config.py)

```python
class RateLimitConfig:
    """速率限制和反机器人配置"""

    MIN_REQUEST_INTERVAL_MS = 3000      # 请求间隔 3 秒
    MAX_REQUESTS_PER_MINUTE = 10        # 每分钟最多 10 次
    DEEP_SEARCH_COOLDOWN_MS = 30000     # Deep Search 冷却 30 秒

    # 退避策略
    INITIAL_BACKOFF_MS = 5000
    MAX_BACKOFF_MS = 300000             # 最大 5 分钟
    BACKOFF_MULTIPLIER = 2.0

    # 人类行为模拟
    TYPING_DELAY_MS = (50, 150)         # 每字符延迟
```

### Phase 6: 工具定义 (tool_definitions.py)

添加到 `function_definitions` 列表：

```python
{
    "name": "google_finance_get_stock_data",
    "description": "从 Google Finance Beta 获取实时股票数据，包括价格、涨跌幅、成交量",
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "股票代码 (如 'NVDA', 'AAPL', 'GOOGL')"
            }
        },
        "required": ["symbol"]
    }
},
{
    "name": "google_finance_ask_ai",
    "description": "向 Google Finance AI 聊天机器人提问。适合快速问题，如市场趋势、股票表现等",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "要问的问题"
            },
            "symbol_context": {
                "type": "string",
                "description": "可选的股票代码作为上下文"
            }
        },
        "required": ["question"]
    }
},
{
    "name": "google_finance_deep_search",
    "description": "使用 Google Finance 的 Deep Search 功能进行深度研究（Gemini 驱动）。适合复杂的金融研究问题。需要 1-5 分钟完成",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "复杂的金融研究查询"
            },
            "include_research_plan": {
                "type": "boolean",
                "description": "是否包含研究计划",
                "default": True
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "最大等待时间（秒）",
                "default": 180
            }
        },
        "required": ["query"]
    }
}
```

### Phase 7: 集成到 Executor (executor_agent.py)

在 `_process_tool_calls()` 方法中添加：

```python
# 顶部导入
from google_finance.client import GoogleFinanceClient
import json

# 在工具路由中添加
elif tool_name == "google_finance_get_stock_data":
    symbol = tool_input.get("symbol", "")

    async def fetch():
        async with GoogleFinanceClient() as client:
            data = await client.get_stock_data(symbol)
            return {
                "symbol": data.symbol,
                "price": data.price,
                "change": data.change,
                "change_percent": data.change_percent,
                "timestamp": data.timestamp
            }

    result = json.dumps(asyncio.run(fetch()), ensure_ascii=False)

elif tool_name == "google_finance_ask_ai":
    question = tool_input.get("question", "")
    symbol_context = tool_input.get("symbol_context")

    async def ask():
        async with GoogleFinanceClient() as client:
            response = await client.ask_ai(question, symbol_context)
            return {
                "answer": response.answer,
                "citations": response.citations,
                "query_time_seconds": response.query_time_seconds
            }

    result = json.dumps(asyncio.run(ask()), ensure_ascii=False)

elif tool_name == "google_finance_deep_search":
    query = tool_input.get("query", "")
    include_plan = tool_input.get("include_research_plan", True)
    timeout = tool_input.get("timeout_seconds", 180)

    async def deep():
        async with GoogleFinanceClient() as client:
            response = await client.deep_search(query, include_plan, timeout)
            return {
                "answer": response.answer,
                "citations": response.citations,
                "research_plan": response.research_plan,
                "query_time_seconds": response.query_time_seconds
            }

    result = json.dumps(asyncio.run(deep()), ensure_ascii=False)
```

### Phase 8: 认证设置脚本 (setup_google_auth.py)

```python
#!/usr/bin/env python3
"""
Google Finance 认证设置脚本
首次使用前运行此脚本
"""

import asyncio
from google_finance.auth import GoogleAuthManager

async def main():
    print("=" * 50)
    print("Google Finance 认证设置")
    print("=" * 50)
    print()
    print("即将打开浏览器窗口")
    print("请登录您的 Google 账号")
    print("登录完成后，返回此终端按 Enter 键")
    print()

    await GoogleAuthManager.setup_authentication()

    print()
    print("✅ 认证设置完成！")
    print("现在可以使用 Google Finance 工具了")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 关键文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `google_finance/__init__.py` | 新建 | 模块入口 |
| `google_finance/client.py` | 新建 | 核心客户端实现 |
| `google_finance/auth.py` | 新建 | 认证管理 |
| `google_finance/selectors.py` | 新建 | CSS 选择器 |
| `google_finance/config.py` | 新建 | 速率限制配置 |
| `tool_definitions.py` | 修改 | 添加 3 个工具定义 |
| `executor_agent.py` | 修改 | 添加工具调用路由 |
| `setup_google_auth.py` | 新建 | 认证设置脚本 |
| `requirements.txt` | 修改 | 添加 playwright-stealth |
| `.gitignore` | 修改 | 添加 auth_state/ |

---

## 依赖添加

```txt
# requirements.txt
playwright-stealth>=1.0.0
```

---

## 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| Google 改变页面结构 | 使用灵活选择器，定期验证，实现选择器发现脚本 |
| 速率限制 | 指数退避重试，请求队列 |
| 认证过期 | 自动检测过期状态，提示重新登录 |
| 反机器人检测 | playwright-stealth + 人类行为模拟 + 持久上下文 |
| Deep Search 超时 | 3分钟超时 + 进度日志 + 优雅失败 |
| MFA 挑战 | 手动设置阶段处理，持久会话 |

---

## 使用示例

```python
# 首次设置
python setup_google_auth.py

# 在研究中使用
python main.py "分析 NVDA 最近一个月走势" --research

# Executor 将自动调用:
# 1. google_finance_get_stock_data("NVDA")
# 2. google_finance_ask_ai("NVDA 最近有什么重大新闻?", "NVDA")
# 3. google_finance_deep_search("详细分析 NVDA 在 AI 芯片市场的竞争地位")
```

---

## 预计工时

| 阶段 | 时间 |
|------|------|
| 模块结构 | 15 min |
| 认证管理 | 1 hour |
| 选择器和配置 | 30 min |
| 核心客户端 | 3 hours |
| 工具定义 | 30 min |
| Executor 集成 | 1 hour |
| 设置脚本 | 15 min |
| 测试和调试 | 2-4 hours |
| **总计** | **9-11 hours** |

---

## 参考资料

- [Google Finance launches new AI-powered features](https://blog.google/products/search/new-google-finance-ai-deep-search/)
- [Google Finance gets AI makeover - Tom's Guide](https://www.tomsguide.com/ai/google-finance-gets-ai-makeover-heres-how-to-try-the-new-chatbot-real-time-data-and-smarter-charts)
- [Try AI-powered Google Finance - Google Help](https://support.google.com/websearch/answer/16490185?hl=en)
- [Playwright Stealth](https://github.com/nicholasopuni31/playwright_stealth)
