"""Base class for all agent tools.

Every tool the agent can use inherits from BaseTool. This enforces a
consistent interface:
- Each tool has a unique name (must match what Claude calls it)
- Each tool has a JSON schema for its inputs
- Each tool implements an execute() method that does the actual work

The schema is what Claude sees. The execute() is what runs locally.
"""
from abc import ABC, abstractmethod
from typing import Any

from src.logger import logger


class BaseTool(ABC):
  """Abstract base class for agent tools."""

  name: str  # Must be set by subclass (e.g., "web_search")
  description: str  # Must be set by subclass (clear, action-oriented)

  @property
  @abstractmethod
  def input_schema(self) -> dict[str, Any]:
      """JSON Schema describing the tool's input parameters."""
      ...

  @abstractmethod
  def execute(self, **kwargs) -> str:
      """Execute the tool with given parameters.

      Returns a string that gets fed back to Claude.
      Always returns a string, even for errors (errors as strings).
      """
      ...

  def to_anthropic_schema(self) -> dict[str, Any]:
      """Convert this tool to the format Anthropic API expects."""
      return {
          "name": self.name,
          "description": self.description,
          "input_schema": self.input_schema,
      }

  def safe_execute(self, **kwargs) -> str:
      """Execute with error handling. Errors are returned as strings
      so the agent can see them and adjust its strategy."""
      try:
          logger.info(
              "tool_executing",
              tool_name=self.name,
              input_keys=list(kwargs.keys()),
          )
          result = self.execute(**kwargs)
          logger.info(
              "tool_succeeded",
              tool_name=self.name,
              result_preview=result[:100] if result else "(empty)",
          )
          return result
      except Exception as e:
          error_msg = f"Tool '{self.name}' failed: {type(e).__name__}: {str(e)}"
          logger.error(
              "tool_failed",
              tool_name=self.name,
              error=str(e),
              error_type=type(e).__name__,
          )
          return error_msg
