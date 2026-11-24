"""
Tool definitions the agent can call.
Kept deliberately simple so the LLM doesn't hallucinate too hard.
"""

from langchain_core.tools import tool
from typing import Optional
import httpx
import math
from datetime import datetime, timezone


@tool
def get_current_time(timezone_name: str = "UTC") -> str:
    """Return the current UTC time. Useful when the user asks about 'now' or deadlines."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M:%S UTC")


@tool
def calculate(expression: str) -> str:
    """
    Safely evaluate a simple math expression.
    Supports +, -, *, /, **, parentheses. No variables or functions beyond basic ops.
    Example: "2 * (15 + 7) / 3"
    """
    allowed = set("0123456789+-*/().% ")
    if not all(c in allowed for c in expression):
        return "Error: only basic arithmetic operators are allowed."

    try:
        # very restricted eval
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def lookup_exchange_rate(base: str = "USD", target: str = "EUR") -> str:
    """
    Fetch a live FX rate using the free Frankfurter API.
    Returns the rate for 1 unit of base currency in target currency.
    """
    base = base.upper().strip()
    target = target.upper().strip()
    url = f"https://api.frankfurter.app/latest?from={base}&to={target}"

    try:
        with httpx.Client(timeout=8.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
            rate = data["rates"].get(target)
            if rate is None:
                return f"Could not find rate for {base} → {target}"
            date = data.get("date", "unknown")
            return f"1 {base} = {rate} {target} (as of {date})"
    except Exception as e:
        return f"FX lookup failed: {str(e)}"


@tool
def search_company_policy(query: str) -> str:
    """
    Placeholder tool – in production this would hit an internal knowledge base.
    Right now it just acknowledges the query so the agent can fall back to RAG.
    """
    return (
        f"Policy search for '{query}' is best handled by the RAG retriever. "
        "If you have access to documents, prefer using retrieved context."
    )


# Export the list the agent will bind
AVAILABLE_TOOLS = [
    get_current_time,
    calculate,
    lookup_exchange_rate,
    search_company_policy,
]
