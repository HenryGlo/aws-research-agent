"""Cost and safety guardrails for the agent.

Agents can run away: looping on unsolvable queries, accumulating context
until they hit rate limits, or burning budget. This class enforces hard
limits independently of the model's own decisions.

Design philosophy: the model decides WHAT to do; guardrails decide WHEN
to stop it from doing too much. Separation of concerns.

Pricing constants are for Claude Sonnet 4.6 (approximate, USD per token).
"""
from dataclasses import dataclass, field

from src.logger import logger

# Claude Sonnet 4.6 pricing (USD per token, approximate)
PRICE_PER_INPUT_TOKEN = 3.0 / 1_000_000
PRICE_PER_OUTPUT_TOKEN = 15.0 / 1_000_000


@dataclass
class GuardrailLimits:
  """Configurable limits for agent execution."""

  max_total_input_tokens: int = 50_000
  max_cost_usd: float = 0.50
  max_iterations: int = 10
  max_duplicate_queries: int = 2


@dataclass
class GuardrailState:
  """Mutable state tracking usage during a single agent run."""

  total_input_tokens: int = 0
  total_output_tokens: int = 0
  iterations: int = 0
  estimated_cost_usd: float = 0.0
  recent_queries: list[str] = field(default_factory=list)
  stop_triggered: bool = False
  stop_reason: str = ""


class AgentGuardrails:
  """Enforces cost and safety limits during agent execution."""

  def __init__(self, limits: GuardrailLimits | None = None):
      self.limits = limits or GuardrailLimits()
      self.state = GuardrailState()

  def record_usage(self, input_tokens: int, output_tokens: int) -> None:
      """Record token usage from a Claude API call."""
      self.state.total_input_tokens += input_tokens
      self.state.total_output_tokens += output_tokens
      self.state.iterations += 1
      self.state.estimated_cost_usd = (
          self.state.total_input_tokens * PRICE_PER_INPUT_TOKEN
          + self.state.total_output_tokens * PRICE_PER_OUTPUT_TOKEN
      )

  def record_query(self, query: str) -> None:
      """Record a search query to detect repetitive looping."""
      self.state.recent_queries.append(query.lower().strip())

  def _is_looping(self) -> bool:
      """Detect if the agent is repeating near-identical queries."""
      if len(self.state.recent_queries) < 2:
          return False

      latest = set(self.state.recent_queries[-1].split())
      duplicate_count = 0

      for prev in self.state.recent_queries[:-1]:
          prev_words = set(prev.split())
          if not prev_words or not latest:
              continue
          overlap = len(latest & prev_words) / len(latest | prev_words)
          if overlap > 0.6:
              duplicate_count += 1

      return duplicate_count >= self.limits.max_duplicate_queries

  def should_stop(self) -> bool:
      """Check all limits. Returns True if agent should stop and synthesize."""
      if self.state.total_input_tokens >= self.limits.max_total_input_tokens:
          self._trigger_stop(
              f"Token budget reached "
              f"({self.state.total_input_tokens:,} input tokens)"
          )
          return True

      if self.state.estimated_cost_usd >= self.limits.max_cost_usd:
          self._trigger_stop(
              f"Cost ceiling reached (${self.state.estimated_cost_usd:.3f})"
          )
          return True

      if self.state.iterations >= self.limits.max_iterations:
          self._trigger_stop(
              f"Max iterations reached ({self.state.iterations})"
          )
          return True

      if self._is_looping():
          self._trigger_stop(
              "Loop detected (agent repeating near-identical queries)"
          )
          return True

      return False

  def _trigger_stop(self, reason: str) -> None:
      """Mark the stop and log it."""
      self.state.stop_triggered = True
      self.state.stop_reason = reason
      logger.warning(
          "guardrail_triggered",
          reason=reason,
          total_input_tokens=self.state.total_input_tokens,
          estimated_cost_usd=round(self.state.estimated_cost_usd, 4),
          iterations=self.state.iterations,
      )

  def summary(self) -> dict:
      """Return a summary of usage for observability."""
      return {
          "total_input_tokens": self.state.total_input_tokens,
          "total_output_tokens": self.state.total_output_tokens,
          "iterations": self.state.iterations,
          "estimated_cost_usd": round(self.state.estimated_cost_usd, 4),
          "stop_triggered": self.state.stop_triggered,
          "stop_reason": self.state.stop_reason,
      }