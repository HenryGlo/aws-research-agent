"""Reusable agent loop with tool use.

This is the core of the agent. It handles:
- Multiple tools registered as a dict {name: tool}
- The conversation loop: send message → check stop_reason → execute tools → repeat
- Iteration limits (prevents infinite loops)
- Error recovery (tool failures don't crash the agent)
- Logging of each step for observability

Usage:
  agent = Agent(tools=[WebSearchTool(), AnotherTool()])
  response = agent.run("What is the latest version of AWS Lambda?")
"""
from anthropic import Anthropic

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
  ):
      self.client = Anthropic(api_key=settings.anthropic_api_key)
      self.tools_by_name = {tool.name: tool for tool in tools}
      self.tool_schemas = [tool.to_anthropic_schema() for tool in tools]
      self.system_prompt = system_prompt or self._default_system_prompt()
      self.max_iterations = max_iterations or settings.max_agent_iterations

  def _default_system_prompt(self) -> str:
      return (
          "You are a helpful AI research assistant specializing in AWS "
          "and cloud computing topics. You have access to tools that let "
          "you search the web for current information. "
          "Use tools when you need recent or specific data. "
          "When you have enough information, synthesize a clear, "
          "well-structured answer with citations to sources you used."
      )

  def run(self, user_query: str) -> dict:
      """Run the agent loop until completion.

      Returns a dict with:
      - final_answer: the text response
      - iterations: how many loops it took
      - tool_calls: list of tools used (for observability)
      - total_input_tokens, total_output_tokens
      """
      messages = [{"role": "user", "content": user_query}]
      tool_calls_log = []
      total_input_tokens = 0
      total_output_tokens = 0

      logger.info("agent_started", query=user_query[:100])

      for iteration in range(self.max_iterations):
          logger.info("agent_iteration", iteration=iteration + 1)

          # Call Claude with current message history + available tools
          response = self.client.messages.create(
              model=settings.claude_model,
              max_tokens=settings.max_tokens,
              temperature=settings.temperature,
              system=self.system_prompt,
              tools=self.tool_schemas,
              messages=messages,
          )

          total_input_tokens += response.usage.input_tokens
          total_output_tokens += response.usage.output_tokens

          logger.info(
              "claude_response",
              iteration=iteration + 1,
              stop_reason=response.stop_reason,
              input_tokens=response.usage.input_tokens,
              output_tokens=response.usage.output_tokens,
          )

          # CASE 1: Claude finished without using more tools
          if response.stop_reason == "end_turn":
              final_text = self._extract_text(response.content)
              logger.info(
                  "agent_completed",
                  iterations=iteration + 1,
                  answer_preview=final_text[:200],
              )
              return {
                  "final_answer": final_text,
                  "iterations": iteration + 1,
                  "tool_calls": tool_calls_log,
                  "total_input_tokens": total_input_tokens,
                  "total_output_tokens": total_output_tokens,
              }

          # CASE 2: Claude wants to use tools
          if response.stop_reason == "tool_use":
              # Append Claude's response to history
              messages.append({"role": "assistant", "content": response.content})

              # Process each tool_use block in the response
              tool_results = []
              for block in response.content:
                  if block.type != "tool_use":
                      continue

                  tool_name = block.name
                  tool_input = block.input
                  tool_use_id = block.id

                  logger.info(
                      "tool_use_requested",
                      tool_name=tool_name,
                      tool_input=tool_input,
                  )

                  # Lookup tool and execute
                  tool = self.tools_by_name.get(tool_name)
                  if tool is None:
                      result = f"Error: Tool '{tool_name}' not registered."
                  else:
                      result = tool.safe_execute(**tool_input)

                  # Log this call for observability
                  tool_calls_log.append({
                      "iteration": iteration + 1,
                      "tool_name": tool_name,
                      "tool_input": tool_input,
                      "result_preview": result[:200],
                  })

                  tool_results.append({
                      "type": "tool_result",
                      "tool_use_id": tool_use_id,
                      "content": result,
                  })

              # Feed all tool results back to Claude in one message
              messages.append({"role": "user", "content": tool_results})
              continue

          # CASE 3: Unexpected stop_reason (max_tokens, error, etc.)
          logger.warning(
              "unexpected_stop_reason",
              stop_reason=response.stop_reason,
          )
          return {
              "final_answer": (
                  f"Agent stopped unexpectedly with reason: {response.stop_reason}"
              ),
              "iterations": iteration + 1,
              "tool_calls": tool_calls_log,
              "total_input_tokens": total_input_tokens,
              "total_output_tokens": total_output_tokens,
          }

      # CASE 4: Hit max_iterations without resolution
      logger.warning(
          "max_iterations_reached",
          max_iterations=self.max_iterations,
      )
      return {
          "final_answer": (
              f"Agent reached max iterations ({self.max_iterations}) "
              f"without producing a final answer."
          ),
          "iterations": self.max_iterations,
          "tool_calls": tool_calls_log,
          "total_input_tokens": total_input_tokens,
          "total_output_tokens": total_output_tokens,
      }

  @staticmethod
  def _extract_text(content_blocks) -> str:
      """Extract all text from Claude's response blocks."""
      text_parts = []
      for block in content_blocks:
          if block.type == "text":
              text_parts.append(block.text)
      return "\n".join(text_parts)
