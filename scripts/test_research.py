"""Interactive test of the full research agent (search + fetch).

Tests scenarios where Claude needs to:
1. Search the web to find relevant URLs
2. Fetch full content from the most promising URL
3. Synthesize a comprehensive answer

Usage:
  python -m scripts.test_research
"""
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.agent.loop import Agent
from src.tools.web_fetch import WebFetchTool
from src.tools.web_search import WebSearchTool

console = Console()


def main():
  console.print(
      "\n[bold cyan]🤖 AWS Research Agent — "
      "Full Research Test (search + fetch)[/bold cyan]\n"
  )

  # Agent now has BOTH tools available
  agent = Agent(tools=[WebSearchTool(), WebFetchTool()])

  test_questions = [
      # Question 1: Should use search + fetch for deeper context
      "What are the latest features announced for AWS Bedrock in 2026? "
      "Check the official AWS announcements page if available.",

      # Question 2: Specific technical question that benefits from full doc reading
      "What are the exact rate limits and quotas for Claude Sonnet 4.6 "
      "on AWS Bedrock? Provide specific numbers if available.",

      # Question 3: Comparison that benefits from reading multiple sources
      "Compare AWS Lambda cold start times for Python 3.11 vs Node.js 20 "
      "based on recent benchmarks from 2025 or 2026.",
  ]

  for i, question in enumerate(test_questions, 1):
      console.print(f"\n[bold yellow]━━━ Question {i} ━━━[/bold yellow]")
      console.print(f"[bold]🤔 User:[/bold] {question}\n")

      result = agent.run(question)

      # Show the final answer
      console.print(Panel(
          Markdown(result["final_answer"]),
          title=f"✅ Agent Answer (iterations: {result['iterations']})",
          border_style="green",
      ))

      # Show observability info
      console.print(
          f"\n[dim]📊 Stats: "
          f"{result['iterations']} iterations, "
          f"{len(result['tool_calls'])} tool calls, "
          f"{result['total_input_tokens']:,} input tokens, "
          f"{result['total_output_tokens']:,} output tokens[/dim]"
      )

      # Show which tools were used and in what order
      if result['tool_calls']:
          console.print("\n[dim]🔧 Tool sequence:[/dim]")
          for call in result['tool_calls']:
              tool_input_preview = str(call['tool_input'])[:80]
              console.print(
                  f"[dim]  → {call['tool_name']}({tool_input_preview})[/dim]"
              )


if __name__ == "__main__":
  main()
