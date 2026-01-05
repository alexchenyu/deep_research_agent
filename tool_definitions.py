"""
Function definitions and descriptions for the interactive chat system tools.
"""

function_definitions = [
    {
        "name": "perform_search",
        "description": "Perform a web search using our native search tool.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum search results",
                    "default": 10
                },
                "max_retries": {
                    "type": "integer",
                    "description": "Maximum number of retry attempts",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "fetch_web_content",
        "description": "Fetch the content of web pages using our web scraper tool.",
        "parameters": {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of URLs to scrape"
                },
                "max_concurrent": {
                    "type": "integer",
                    "description": "Maximum number of concurrent requests",
                    "default": 3
                }
            },
            "required": ["urls"]
        }
    },
    {
        "name": "create_file",
        "description": "Create or change a file with the given content and return a success message.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Name of the file to create"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["filename", "content"]
        }
    },
    {
        "name": "execute_command",
        "description": "Execute a terminal command and return its output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The terminal command to execute"
                },
                "explanation": {
                    "type": "string",
                    "description": "Explanation of what the command does"
                }
            },
            "required": ["command", "explanation"]
        }
    },
    # === Google Finance Beta Tools ===
    {
        "name": "google_finance_get_stock_data",
        "description": "Fetch real-time stock data from Google Finance Beta including current price, change, and volume. Use this for getting latest stock quotes.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., 'NVDA', 'AAPL', 'GOOGL', 'MSFT')"
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "google_finance_ask_ai",
        "description": "Ask a question to Google Finance AI chatbot. Good for quick questions about stocks, market trends, company analysis, or financial concepts. Response time: 10-60 seconds.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the AI chatbot"
                },
                "symbol_context": {
                    "type": "string",
                    "description": "Optional stock symbol to provide context (e.g., 'NVDA')"
                }
            },
            "required": ["question"]
        }
    },
    {
        "name": "google_finance_deep_search",
        "description": "Perform a Deep Search query using Google Finance's Gemini-powered analysis. Use for complex financial research questions that require multi-source analysis. Takes 1-5 minutes to complete.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Complex financial research query"
                },
                "include_research_plan": {
                    "type": "boolean",
                    "description": "Whether to include the research plan in response",
                    "default": True
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Maximum time to wait for response (seconds)",
                    "default": 180
                }
            },
            "required": ["query"]
        }
    }
] 