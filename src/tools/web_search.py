"""Web search tool using Tavily API.

Tavily is designed specifically for LLM agents:
- Returns structured results (title, content, url) not raw HTML
- Optimized for relevance over recency or popularity
- Free tier: 1000 searches/month

Reference: https://docs.tavily.com/
"""
from typing import Any

from tavily import TavilyClient

from src.config import settings
from src.tools.base import BaseTool


class WebSearchTool(BaseTool):
  """Search the web using Tavily and return relevant results."""

  name = "web_search"
  description = (
      "Searches the web for current, factual information. "
      "Use this when the user asks about recent events, current versions, "
      "pricing, latest features, or anything that may have changed recently. "
      "Returns top results with title, summary, and URL. "
      "Does NOT use this for general knowledge questions about well-known topics."
  )

  @property
  def input_schema(self) -> dict[str, Any]:
      return {
          "type": "object",
          "properties": {
              "query": {
                  "type": "string",
                  "description": (
                      "Search query. Be specific and use 3-6 words. "
                      "Examples: 'AWS Bedrock pricing 2026', "
                      "'Lambda cold start optimization', "
                      "'EKS vs Fargate comparison'."
                  ),
              },
              "max_results": {
                  "type": "integer",
                  "description": "Number of results to return (default 5, max 10).",
                  "default": 5,
                  "minimum": 1,
                  "maximum": 10,
              },
          },
          "required": ["query"],
      }

  def __init__(self):
      self._client = TavilyClient(api_key=settings.tavily_api_key)

  def execute(self, query: str, max_results: int = 5) -> str:
      """Run search and return formatted results."""
      # Cap max_results to avoid abuse
      max_results = min(max_results, 10)

      # Tavily returns a dict with 'results' key containing list of dicts
      response = self._client.search(
          query=query,
          max_results=max_results,
          search_depth="basic",  # 'advanced' is more expensive
          include_answer=False,  # We let Claude synthesize, not Tavily
      )

      results = response.get("results", [])

      if not results:
          return f"No results found for query: '{query}'"

      # Format results as readable string for Claude
      formatted = [f"Search results for: '{query}'\n"]
      for i, result in enumerate(results, 1):
          title = result.get("title", "No title")
          content = result.get("content", "No content")
          url = result.get("url", "No URL")

          formatted.append(
              f"\n[Result {i}]\n"
              f"Title: {title}\n"
              f"URL: {url}\n"
              f"Content: {content}\n"
          )

      return "\n".join(formatted)
