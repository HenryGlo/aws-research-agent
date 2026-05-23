"""Unit tests for AgentGuardrails.

These tests validate the guardrail logic WITHOUT calling any API.
Pure logic testing — free and fast.
"""
from src.agent.guardrails import AgentGuardrails, GuardrailLimits


def test_token_budget_triggers_stop():
  """Exceeding token budget should trigger a stop."""
  limits = GuardrailLimits(max_total_input_tokens=1000)
  guardrails = AgentGuardrails(limits=limits)

  # Under budget — should not stop
  guardrails.record_usage(input_tokens=500, output_tokens=100)
  assert guardrails.should_stop() is False

  # Over budget — should stop
  guardrails.record_usage(input_tokens=600, output_tokens=100)
  assert guardrails.should_stop() is True
  assert "Token budget" in guardrails.state.stop_reason


def test_cost_ceiling_triggers_stop():
  """Exceeding cost ceiling should trigger a stop."""
  limits = GuardrailLimits(
      max_total_input_tokens=10_000_000,  # high, won't trigger
      max_cost_usd=0.01,  # low, will trigger
  )
  guardrails = AgentGuardrails(limits=limits)

  # 5000 input + 1000 output tokens
  # cost = 5000 * 3/1M + 1000 * 15/1M = 0.015 + 0.015 = 0.03 > 0.01
  guardrails.record_usage(input_tokens=5000, output_tokens=1000)
  assert guardrails.should_stop() is True
  assert "Cost ceiling" in guardrails.state.stop_reason


def test_max_iterations_triggers_stop():
  """Reaching max iterations should trigger a stop."""
  limits = GuardrailLimits(
      max_total_input_tokens=10_000_000,
      max_cost_usd=1000.0,
      max_iterations=3,
  )
  guardrails = AgentGuardrails(limits=limits)

  guardrails.record_usage(10, 10)  # iter 1
  assert guardrails.should_stop() is False
  guardrails.record_usage(10, 10)  # iter 2
  assert guardrails.should_stop() is False
  guardrails.record_usage(10, 10)  # iter 3
  assert guardrails.should_stop() is True
  assert "Max iterations" in guardrails.state.stop_reason


def test_loop_detection_triggers_stop():
  """Repeating near-identical queries should trigger a stop."""
  limits = GuardrailLimits(
      max_total_input_tokens=10_000_000,
      max_cost_usd=1000.0,
      max_iterations=100,
      max_duplicate_queries=2,
  )
  guardrails = AgentGuardrails(limits=limits)

  # Three very similar queries
  guardrails.record_query("AWS Bedrock rate limits Claude Sonnet")
  guardrails.record_query("AWS Bedrock rate limits Claude Sonnet 4.6")
  guardrails.record_query("AWS Bedrock rate limits Claude Sonnet quota")

  assert guardrails.should_stop() is True
  assert "Loop detected" in guardrails.state.stop_reason


def test_distinct_queries_no_loop():
  """Genuinely different queries should NOT trigger loop detection."""
  limits = GuardrailLimits(
      max_total_input_tokens=10_000_000,
      max_cost_usd=1000.0,
      max_iterations=100,
      max_duplicate_queries=2,
  )
  guardrails = AgentGuardrails(limits=limits)

  guardrails.record_query("AWS Lambda cold start times")
  guardrails.record_query("DynamoDB partition key design")
  guardrails.record_query("S3 bucket encryption options")

  assert guardrails.should_stop() is False


def test_summary_returns_usage():
  """Summary should return accurate usage stats."""
  guardrails = AgentGuardrails()
  guardrails.record_usage(input_tokens=1000, output_tokens=500)

  summary = guardrails.summary()
  assert summary["total_input_tokens"] == 1000
  assert summary["total_output_tokens"] == 500
  assert summary["iterations"] == 1
  assert summary["estimated_cost_usd"] > 0


def test_normal_operation_no_stop():
  """Under all limits, should not stop."""
  guardrails = AgentGuardrails()  # default limits (generous)
  guardrails.record_usage(input_tokens=2000, output_tokens=300)
  guardrails.record_query("AWS Lambda overview")
  assert guardrails.should_stop() is False
