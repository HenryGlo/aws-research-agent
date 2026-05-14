"""Interactive test of the web search agent.

Runs questions that REQUIRE web search to answer (current info
beyond Claude's training cutoff).

Usage:
  python -m scripts.test_search
"""
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.agent.loop import Agent
from src.tools.web_search import WebSearchTool

console = Console()


def main():
  console.print("\n[bold cyan]🤖 AWS Research Agent — Web Search Test[/bold cyan]\n")

  # Initialize agent with only web_search tool for this test
  agent = Agent(tools=[WebSearchTool()])

  # Test questions that need web search
  test_questions = [
      "What is AWS Lambda? (this might NOT need web search, see what agent does)",
      "What are the newest AWS services announced in 2026?",
      "What is the current pricing for AWS Bedrock Claude Sonnet 4.6 in us-east-1?",
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

      # Show what tools were called
      if result['tool_calls']:
          console.print("\n[dim]🔧 Tools used:[/dim]")
          for call in result['tool_calls']:
              console.print(
                  f"[dim]  → {call['tool_name']}"
                  f"({call['tool_input']})[/dim]"
              )


if __name__ == "__main__":
  main()
