"""Centralized configuration using Pydantic Settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
  """Application settings loaded from environment variables."""

  model_config = SettingsConfigDict(
      env_file=".env",
      env_file_encoding="utf-8",
      case_sensitive=False,
  )

  # Anthropic
  anthropic_api_key: str
  claude_model: str = "claude-sonnet-4-6"
  max_tokens: int = 4096
  temperature: float = 0.0

  # Tavily (web search)
  tavily_api_key: str

  # Agent behavior
  max_agent_iterations: int = 10
  tool_timeout_seconds: int = 30

  # Logging
  log_level: str = "INFO"


settings = Settings()
