"""Reusable agent loop with tool use, cost guardrails, and observability.

This is the core of the agent. It coordinates three concerns, each in its
own module:
- Agent (this file): orchestration of the tool-use loop
- AgentGuardrails: cost/safety circuit breakers (token budget, cost, loops)
- AgentObserver: real-time observability of the reasoning process

The loop: send message → check guardrails → call Claude → check stop_reason
→ execute tools → repeat. When a guardrail triggers, the agent gracefully
synthesizes with what it has instead of crashing.

Usage:
  agent = Agent(tools=[WebSearchTool(), WebFetchTool()])
  response = agent.run("What is the latest version of AWS Lambda?")
"""
from anthropic import Anthropic

from src.agent.guardrails import AgentGuardrails, GuardrailLimits
from src.agent.observer import AgentObserver
from src.config import settings
from src.logger import logger
from src.tools.base import BaseTool


class Agent:
  """Tool-using agent powered by Claude."""

  def __init__(
      self,
      tools: list[BaseTool],
      system_prompt: str | None = None,
      max_iterations: int | None = None,
      guardrail_limits: GuardrailLimits | None = None,
      verbose: bool = True,
  ):
      self.client = Anthropic(api_key=settings.anthropic_api_key)
      self.tools_by_name = {tool.name: tool for tool in tools}
      self.tool_schemas = [tool.to_anthropic_schema() for tool in tools]
      self.system_prompt = system_prompt or self._default_system_prompt()
      self.max_iterations = max_iterations or settings.max_agent_iterations
      self.guardrail_limits = guardrail_limits
      self.observer = AgentObserver(verbose=verbose)

  def _default_system_prompt(self) -> str:
      return (
          "You are a helpful AI research assistant specializing in AWS "
          "and cloud computing topics. You have access to tools that let "
          "you search the web and fetch page contents.\n\n"
          "═══ EFFICIENCY GUIDELINES (critical) ═══\n"
          "- Use AT MOST 2-3 tool calls per question before synthesizing.\n"
          "- After web_search, fetch only the MOST RELEVANT URL "
          "(typically the first or second result).\n"
          "- Avoid fetching the same domain twice unless absolutely necessary.\n\n"
          "═══ GRACEFUL FAILURE (critical) ═══\n"
          "Some questions don't have exact public answers. When this happens:\n"
          "- After 2 failed attempts to find specific info (numbers, exact specs, "
          "limits), STOP searching and synthesize with partial info.\n"
          "- Explicitly acknowledge what you couldn't find and WHY "
          "(e.g., 'AWS doesn't publish exact rate limit numbers publicly; "
          "they vary by account tier and region').\n"
          "- Provide the closest available information and suggest where "
          "the user might find exact numbers (e.g., 'Check AWS Service Quotas "
          "in your console for your account-specific limits').\n"
          "- It is BETTER to give a partial-but-honest answer than to keep "
          "searching endlessly.\n\n"
          "═══ SYNTHESIS ═══\n"
          "When you have enough information (or have hit the failure threshold), "
          "provide a clear, well-structured answer with:\n"
          "- Clear sections and headers\n"
          "- Specific numbers when available, or honest acknowledgment when not\n"
          "- Citations to the sources you used\n"
          "- Suggestions for where to find more info if applicable"
      )

  def run(self, user_query: str) -> dict:
      """Run the agent loop with guardrails and observability."""
      messages = [{"role": "user", "content": user_query}]
      tool_calls_log = []

      guardrails = AgentGuardrails(limits=self.guardrail_limits)

      logger.info("agent_started", query=user_query[:100])
      self.observer.on_agent_start(user_query)

      while True:
          self.observer.on_iteration_start(guardrails.state.iterations + 1)

          # CHECK GUARDRAILS before each iteration
          if guardrails.should_stop():
              logger.warning(
                  "agent_stopped_by_guardrail",
                  reason=guardrails.state.stop_reason,
              )
              self.observer.on_synthesis(
                  iteration=guardrails.state.iterations,
                  cost_usd=guardrails.state.estimated_cost_usd,
                  total_tokens=guardrails.state.total_input_tokens,
                  forced=True,
              )
              final_answer = self._force_synthesis(
                  messages, guardrails.state.stop_reason
              )
              self.observer.on_agent_complete(
                  iterations=guardrails.state.iterations,
                  cost_usd=guardrails.state.estimated_cost_usd,
                  total_tokens=guardrails.state.total_input_tokens,
                  stopped_by_guardrail=True,
              )
              return {
                  "final_answer": final_answer,
                  "iterations": guardrails.state.iterations,
                  "tool_calls": tool_calls_log,
                  "stopped_by_guardrail": True,
                  "guardrail_summary": guardrails.summary(),
              }

          # Call Claude
          response = self.client.messages.create(
              model=settings.claude_model,
              max_tokens=settings.max_tokens,
              temperature=settings.temperature,
              system=self.system_prompt,
              tools=self.tool_schemas,
              messages=messages,
          )

          guardrails.record_usage(
              response.usage.input_tokens,
              response.usage.output_tokens,
          )

          logger.info(
              "claude_response",
              iteration=guardrails.state.iterations,
              stop_reason=response.stop_reason,
              cumulative_cost_usd=round(guardrails.state.estimated_cost_usd, 4),
          )

          # CASE 1: Claude finished
          if response.stop_reason == "end_turn":
              final_text = self._extract_text(response.content)
              self.observer.on_synthesis(
                  iteration=guardrails.state.iterations,
                  cost_usd=guardrails.state.estimated_cost_usd,
                  total_tokens=guardrails.state.total_input_tokens,
                  forced=False,
              )
              self.observer.on_agent_complete(
                  iterations=guardrails.state.iterations,
                  cost_usd=guardrails.state.estimated_cost_usd,
                  total_tokens=guardrails.state.total_input_tokens,
                  stopped_by_guardrail=False,
              )
              logger.info(
                  "agent_completed",
                  iterations=guardrails.state.iterations,
                  cost_usd=round(guardrails.state.estimated_cost_usd, 4),
              )
              return {
                  "final_answer": final_text,
                  "iterations": guardrails.state.iterations,
                  "tool_calls": tool_calls_log,
                  "stopped_by_guardrail": False,
                  "guardrail_summary": guardrails.summary(),
              }

          # CASE 2: Claude wants tools
          if response.stop_reason == "tool_use":
              messages.append({"role": "assistant", "content": response.content})

              tool_results = []
              for block in response.content:
                  if block.type != "tool_use":
                      continue

                  # Record search queries for loop detection
                  if block.name in ("web_search",) and "query" in block.input:
                      guardrails.record_query(block.input["query"])

                  logger.info(
                      "tool_use_requested",
                      tool_name=block.name,
                      tool_input=block.input,
                  )
                  self.observer.on_tool_decision(
                      iteration=guardrails.state.iterations,
                      tool_name=block.name,
                      tool_input=block.input,
                      cost_usd=guardrails.state.estimated_cost_usd,
                      total_tokens=guardrails.state.total_input_tokens,
                  )

                  tool = self.tools_by_name.get(block.name)
                  if tool is None:
                      result = f"Error: Tool '{block.name}' not registered."
                  else:
                      result = tool.safe_execute(**block.input)

                  tool_calls_log.append({
                      "iteration": guardrails.state.iterations,
                      "tool_name": block.name,
                      "tool_input": block.input,
                      "result_preview": result[:200],
                  })

                  tool_results.append({
                      "type": "tool_result",
                      "tool_use_id": block.id,
                      "content": result,
                  })

              messages.append({"role": "user", "content": tool_results})
              continue

          # CASE 3: Unexpected stop_reason
          logger.warning("unexpected_stop_reason", stop_reason=response.stop_reason)
          return {
              "final_answer": f"Agent stopped: {response.stop_reason}",
              "iterations": guardrails.state.iterations,
              "tool_calls": tool_calls_log,
              "stopped_by_guardrail": False,
              "guardrail_summary": guardrails.summary(),
          }

  def _force_synthesis(self, messages: list, stop_reason: str) -> str:
      """Force Claude to synthesize a final answer when a guardrail triggers.

      Instead of crashing or returning an error, we ask Claude one final
      time (WITHOUT tools) to synthesize what it has gathered so far.
      """
      logger.info("forcing_synthesis", reason=stop_reason)

      synthesis_prompt = (
          f"You've reached a processing limit ({stop_reason}). "
          f"Based on the information gathered so far in this conversation, "
          f"provide the best possible answer NOW. If the information is "
          f"incomplete, clearly state what you found, what's missing, and "
          f"suggest where the user could find the rest. Do not request more tools."
      )

      messages.append({"role": "user", "content": synthesis_prompt})

      # Call Claude WITHOUT tools to force a text-only response
      response = self.client.messages.create(
          model=settings.claude_model,
          max_tokens=settings.max_tokens,
          temperature=settings.temperature,
          system=self.system_prompt,
          messages=messages,
      )

      return self._extract_text(response.content)

  @staticmethod
  def _extract_text(content_blocks) -> str:
      """Extract all text from Claude's response blocks."""
      text_parts = []
      for block in content_blocks:
          if block.type == "text":
              text_parts.append(block.text)
      return "\n".join(text_parts)
