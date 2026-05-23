"""Web fetch tool: extract content from a specific URL.

Uses httpx for HTTP requests and BeautifulSoup for HTML parsing.
The goal: convert any web page into clean text that Claude can analyze.

Design decisions:
- Timeout: 15 seconds (websites can be slow)
- Max content length: 50,000 chars (~12K tokens) to stay within context window
- User-Agent: identifies the agent transparently (no stealth scraping)
- Removes scripts/styles/navigation: keeps only meaningful content
"""
from typing import Any

import httpx
from bs4 import BeautifulSoup

from src.tools.base import BaseTool

# Limit response size to stay within Claude context window
# 50,000 chars ≈ 12,500 tokens (one fetch shouldn't dominate context)
MAX_CONTENT_LENGTH = 8_000

# HTML elements that don't contain meaningful content
NOISE_TAGS = ["script", "style", "noscript", "header", "footer", "nav", "aside"]

# Identify ourselves transparently
USER_AGENT = (
  "AWS-Research-Agent/1.0 (educational portfolio project; "
  "https://github.com/HenryGlo/aws-research-agent)"
)


class WebFetchTool(BaseTool):
  """Fetch and extract text content from a specific URL."""

  name = "web_fetch"
  description = (
      "Fetches the full text content of a specific URL. Use this AFTER "
      "web_search has identified a promising URL and you need the complete "
      "article/documentation content (not just the snippet). "
      "Returns clean text with HTML stripped. "
      "Examples: fetch an AWS documentation page, a blog post, "
      "a release notes page. "
      "Does NOT work for PDFs, login-required pages, or pages that block scraping."
  )

  @property
  def input_schema(self) -> dict[str, Any]:
      return {
          "type": "object",
          "properties": {
              "url": {
                  "type": "string",
                  "description": (
                      "Full URL to fetch (must include https:// or http://). "
                      "Typically obtained from a previous web_search result."
                  ),
              },
          },
          "required": ["url"],
      }

  def execute(self, url: str) -> str:
      """Fetch URL and return cleaned text content."""
      # Validate URL has scheme
      if not url.startswith(("http://", "https://")):
          return f"Error: URL must start with http:// or https://. Got: {url}"

      try:
          response = httpx.get(
              url,
              timeout=15.0,
              follow_redirects=True,
              headers={"User-Agent": USER_AGENT},
          )
          response.raise_for_status()
      except httpx.TimeoutException:
          return f"Error: Request to {url} timed out after 15 seconds."
      except httpx.HTTPStatusError as e:
          return (
              f"Error: HTTP {e.response.status_code} when fetching {url}. "
              f"The page may not exist or require authentication."
          )
      except httpx.RequestError as e:
          return f"Error: Failed to fetch {url}: {type(e).__name__}"

      # Parse HTML and extract clean text
      content_type = response.headers.get("content-type", "").lower()
      if "html" not in content_type:
          # If it's not HTML (e.g., PDF, JSON), return first chunk of raw text
          text = response.text[:MAX_CONTENT_LENGTH]
          return f"[Non-HTML content from {url}]\n\n{text}"

      text = self._extract_clean_text(response.text)

      if not text.strip():
          return f"Error: Page at {url} contained no extractable text."

      # Truncate if too long
      if len(text) > MAX_CONTENT_LENGTH:
          text = text[:MAX_CONTENT_LENGTH] + "\n\n[... content truncated ...]"

      return f"[Content from {url}]\n\n{text}"

  @staticmethod
  def _extract_clean_text(html: str) -> str:
      """Parse HTML, remove noise, return clean text."""
      soup = BeautifulSoup(html, "html.parser")

      # Remove noise tags (scripts, styles, nav, etc.)
      for tag_name in NOISE_TAGS:
          for tag in soup.find_all(tag_name):
              tag.decompose()

      # Extract text with paragraph separators
      text = soup.get_text(separator="\n", strip=True)

      # Collapse multiple newlines into double newlines (paragraph breaks)
      lines = [line.strip() for line in text.split("\n") if line.strip()]
      return "\n\n".join(lines)
