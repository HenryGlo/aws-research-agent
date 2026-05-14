"""Structured logging configuration."""
import logging
import sys

import structlog

from src.config import settings


def setup_logging() -> structlog.BoundLogger:
  """Configure structured logging."""
  logging.basicConfig(
      format="%(message)s",
      stream=sys.stdout,
      level=settings.log_level,
  )

  structlog.configure(
      processors=[
          structlog.contextvars.merge_contextvars,
          structlog.processors.add_log_level,
          structlog.processors.TimeStamper(fmt="iso"),
          structlog.dev.ConsoleRenderer(),
      ],
      wrapper_class=structlog.make_filtering_bound_logger(
          getattr(logging, settings.log_level.upper())
      ),
  )

  return structlog.get_logger()


logger = setup_logging()
