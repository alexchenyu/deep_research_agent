"""
Tool implementations for interactive chat system.
Contains search, web scraping, file operations, and package management tools.
"""

import asyncio
import logging
import os
import subprocess
import time
import hashlib
import json
from multiprocessing import Pool
from typing import List, Optional, Union, Dict
from urllib.parse import urlparse

from playwright.async_api import async_playwright
import html5lib
from ddgs import DDGS
import openai
import requests
from bs4 import BeautifulSoup

from common import TokenTracker
from llm_adapter import get_adapter

logger = logging.getLogger(__name__)

class CachedChatCompletion:
    """Handles chat completions with token usage tracking.

    Modified to use our LLM adapter (Grok/GLM) instead of OpenAI directly.
    """

    def __init__(self):
        self.adapter = get_adapter()
        self.token_tracker = TokenTracker()

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o",
        temperature: float = 1.0,
        functions: Optional[List[Dict]] = None,
        function_call: Optional[Union[str, Dict]] = None,
        reasoning_effort: str = 'high',
        role: str = "default",
    ):
        """Get chat completion using our LLM adapter (Grok/GLM)."""
        return self.adapter.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            functions=functions,
            function_call=function_call,
            reasoning_effort=reasoning_effort,
            role=role
        )

    def get_token_usage(self) -> Dict[str, Union[int, float]]:
        """Get current token usage statistics."""
        return self.adapter.get_token_usage()

# Initialize global chat completion instance
chat_completion = CachedChatCompletion()

# Search Engine Implementation
def search_with_retry(query: str, max_results: int = 10, max_retries: int = 3) -> List[dict]:
    """
    Search using DuckDuckGo and return results with URLs and text snippets.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        max_retries: Maximum number of retry attempts

    Returns:
        List of dictionaries containing search results
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Searching for query: {query} (attempt {attempt + 1}/{max_retries})")
            
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                
            if not results:
                logger.info("No results found")
                return []
            
            logger.info(f"Found {len(results)} results")
            return results
                
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:  # If not the last attempt
                # Exponential backoff: wait longer for each retry
                wait_time = 10 * (attempt + 1)  # 10s, 20s, 30s...
                logger.info(f"Rate limit hit. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} attempts failed")
                raise

def format_search_results(results: List[dict]) -> str:
    """
    Format search results into a readable string.

    Args:
        results: List of search result dictionaries

    Returns:
        Formatted string containing search results
    """
    output = []
    for i, result in enumerate(results, 1):
        output.append(f"\n=== Result {i} ===")
        output.append(f"URL: {result.get('href', 'N/A')}")
        output.append(f"Title: {result.get('title', 'N/A')}")
        output.append(f"Snippet: {result.get('body', 'N/A')}")
    return "\n".join(output)

def search_with_grok(query: str, max_results: int = 5) -> List[dict]:
    """
    使用 Grok search_parameters 进行网络搜索
    
    Args:
        query: 搜索查询
        max_results: 最大结果数
    
    Returns:
        搜索结果列表
    """
    import os
    results = []
    
    try:
        api_key = os.getenv("GROK_API_KEY")
        if not api_key:
            logger.info("GROK_API_KEY not set, skipping Grok search")
            return results
        
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
        
        # 使用 search_parameters 启用搜索 (稳定 API)
        response = client.chat.completions.create(
            model="grok-3-fast",
            messages=[{
                "role": "user",
                "content": f"""Search the web for: {query}

Find and return the top {max_results} most relevant and recent results as a JSON array:
[{{"title": "article title", "href": "https://...", "body": "brief description"}}]

Return ONLY the JSON array, no explanation."""
            }],
            extra_body={"search_parameters": {"mode": "on"}},
            temperature=0,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content
        if content:
            import re
            import json
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                items = json.loads(json_match.group())
                for item in items[:max_results]:
                    results.append({
                        "title": item.get("title", ""),
                        "href": item.get("href", item.get("url", "")),
                        "body": item.get("body", item.get("snippet", "")),
                        "source": "grok"
                    })
        
        if results:
            logger.info(f"Grok search found {len(results)} results")
            
    except Exception as e:
        logger.warning(f"Grok search failed: {e}")
    
    return results


def perform_search(query: str, max_results: int = 10, max_retries: int = 3, use_grok: bool = True) -> str:
    """
    Perform a web search using DuckDuckGo + Grok (optional) and return formatted results.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        max_retries: Maximum number of retry attempts
        use_grok: Whether to also use Grok search (default True)

    Returns:
        Formatted string containing search results or error message
    """
    try:
        all_results = []
        seen_urls = set()
        
        # 1. DuckDuckGo 搜索
        ddg_results = search_with_retry(query, max_results, max_retries)
        for r in ddg_results:
            url = r.get('href', '')
            if url and url not in seen_urls:
                all_results.append(r)
                seen_urls.add(url)
        
        # 2. Grok 搜索 (补充)
        if use_grok and len(all_results) < max_results:
            grok_results = search_with_grok(query, max_results // 2)
            for r in grok_results:
                url = r.get('href', '')
                if url and url not in seen_urls:
                    all_results.append(r)
                    seen_urls.add(url)
        
        return format_search_results(all_results[:max_results])
    except Exception as e:
        return f"Error during search: {e}"

# Web Scraper Implementation
async def fetch_page(url: str, context, timeout_ms: int = 15000) -> Optional[str]:
    """
    Asynchronously fetch a webpage's content using Playwright.

    Args:
        url: URL to fetch
        context: Playwright browser context
        timeout_ms: Timeout in milliseconds (default 15 seconds)

    Returns:
        Page content as string if successful, None otherwise
    """
    page = await context.new_page()
    try:
        logger.info(f"Fetching {url}")
        # 使用较短超时，先尝试 domcontentloaded (更快)
        try:
            await page.goto(url, timeout=timeout_ms, wait_until='domcontentloaded')
        except Exception:
            # 如果失败，尝试不等待任何状态
            await page.goto(url, timeout=timeout_ms)
        
        # 尝试等待网络空闲，但不强制要求
        try:
            await page.wait_for_load_state('networkidle', timeout=5000)
        except Exception:
            pass  # 忽略超时，继续获取已加载的内容
        
        content = await page.content()
        logger.info(f"Successfully fetched {url}")
        return content
    except Exception as e:
        # 使用 warning 而不是 error，避免日志过于嘈杂
        logger.warning(f"跳过 {url}: {type(e).__name__}")
        print(f"      ⚠️ 跳过慢速网站: {url[:50]}...")
        return None
    finally:
        await page.close()

def parse_html(html_content: Optional[str]) -> str:
    """
    Parse HTML content and extract text with hyperlinks in markdown format.

    Args:
        html_content: HTML content to parse

    Returns:
        Extracted text in markdown format
    """
    if not html_content:
        return ""
    
    try:
        document = html5lib.parse(html_content)
        result = []
        seen_texts = set()
        
        def should_skip_element(elem) -> bool:
            """Check if the element should be skipped during parsing."""
            if elem.tag in ['{http://www.w3.org/1999/xhtml}script', 
                          '{http://www.w3.org/1999/xhtml}style']:
                return True
            if not any(text.strip() for text in elem.itertext()):
                return True
            return False
        
        def process_element(elem, depth: int = 0) -> None:
            """Process an HTML element and its children recursively."""
            if should_skip_element(elem):
                return
            
            if hasattr(elem, 'text') and elem.text:
                text = elem.text.strip()
                if text and text not in seen_texts:
                    if elem.tag == '{http://www.w3.org/1999/xhtml}a':
                        href = None
                        for attr, value in elem.items():
                            if attr.endswith('href'):
                                href = value
                                break
                        if href and not href.startswith(('#', 'javascript:')):
                            link_text = f"[{text}]({href})"
                            result.append("  " * depth + link_text)
                            seen_texts.add(text)
                    else:
                        result.append("  " * depth + text)
                        seen_texts.add(text)
            
            for child in elem:
                process_element(child, depth + 1)
            
            if hasattr(elem, 'tail') and elem.tail:
                tail = elem.tail.strip()
                if tail and tail not in seen_texts:
                    result.append("  " * depth + tail)
                    seen_texts.add(tail)
        
        body = document.find('.//{http://www.w3.org/1999/xhtml}body')
        if body is not None:
            process_element(body)
        else:
            process_element(document)
        
        filtered_result = []
        for line in result:
            if any(pattern in line.lower() for pattern in [
                'var ', 
                'function()', 
                '.js',
                '.css',
                'google-analytics',
                'disqus',
                '{',
                '}'
            ]):
                continue
            filtered_result.append(line)
        
        return '\n'.join(filtered_result)
    except Exception as e:
        logger.error(f"Error parsing HTML: {str(e)}")
        return ""

async def process_urls(urls: List[str], max_concurrent: int = 5) -> List[str]:
    """
    Process multiple URLs concurrently using Playwright.

    Args:
        urls: List of URLs to process
        max_concurrent: Maximum number of concurrent requests

    Returns:
        List of processed content strings
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        try:
            # Create browser contexts
            n_contexts = min(len(urls), max_concurrent)
            contexts = [await browser.new_context() for _ in range(n_contexts)]
            
            # Create tasks for each URL
            tasks = []
            for i, url in enumerate(urls):
                context = contexts[i % len(contexts)]
                task = fetch_page(url, context)
                tasks.append(task)
            
            # Gather results
            html_contents = await asyncio.gather(*tasks)
            
            # Parse HTML contents in parallel
            with Pool() as pool:
                results = pool.map(parse_html, html_contents)
                
            return results
            
        finally:
            # Cleanup
            for context in contexts:
                await context.close()
            await browser.close()

def validate_url(url: str) -> bool:
    """
    Validate if a string is a valid URL.

    Args:
        url: URL string to validate

    Returns:
        True if URL is valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

# Main Tool Functions
def fetch_web_content(urls: List[str], max_concurrent: int = 3) -> str:
    """
    Fetch and process web content from multiple URLs using Playwright.

    Args:
        urls: List of URLs to fetch and process
        max_concurrent: Maximum number of concurrent requests

    Returns:
        Formatted string containing processed content or error message
    """
    try:
        # Validate URLs
        valid_urls = [url for url in urls if validate_url(url)]
        if not valid_urls:
            return "No valid URLs provided"
        
        # Process URLs
        results = asyncio.run(process_urls(valid_urls, max_concurrent))
        
        # Format output
        output = []
        for url, content in zip(valid_urls, results):
            output.append(f"\n=== Content from {url} ===\n")
            output.append(content)
        
        return "\n".join(output)
    except Exception as e:
        return f"Error during web scraping: {e}"

def create_file(filename: str, content: str) -> str:
    """
    Create a file with the given content and return a success message.

    Args:
        filename: Name of the file to create
        content: Content to write to the file

    Returns:
        Success message or error message
    """
    try:
        # 检查是否有输出目录配置（通过环境变量）
        output_dir = os.environ.get('DEEP_RESEARCH_OUTPUT_DIR', '')
        
        # 所有文件（包括 scratchpad.md）都保存到输出目录
        if output_dir:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, filename)
        else:
            filepath = filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully created/updated file: {filepath}"
    except Exception as e:
        return f"Error creating file: {str(e)}"

def execute_python(filename: str) -> str:
    """
    Execute a Python script and return its stdout.

    Args:
        filename: Name of the Python file to execute

    Returns:
        Script output or error message
    """
    try:
        result = subprocess.run(
            ["python", filename],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing Python script: stdout={e.stdout}, stderr={e.stderr}"
    except Exception as e:
        return f"Error executing Python script: {str(e)}" 