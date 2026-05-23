"""Test the agent with guardrails on a known problematic query.

This query ('exact rate limits') previously caused the agent to loop
10+ times without finding an answer (the info isn't public). With
guardrails, the agent should stop early and synthesize gracefully.

Usage:
  python -m scripts.test_guardrails
"""
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.agent.guardrails import GuardrailLimits
from src.agent.loop import Agent
from src.tools.web_fetch import WebFetchTool
from src.tools.web_search import WebSearchTool

console = Console()


def main():
  console.print(
      "\n[bold cyan]🛡️  Agent Guardrails Test[/bold cyan]\n"
  )

  # Set conservative limits to demonstrate guardrails kicking in
  limits = GuardrailLimits(
      max_total_input_tokens=30_000,  # lower than the 116K it hit before
      max_cost_usd=0.30,
      max_iterations=8,
      max_duplicate_queries=2,
  )

  agent = Agent(
      tools=[WebSearchTool(), WebFetchTool()],
      guardrail_limits=limits,
  )

  # The query that previously caused a runaway loop
  question = (
      "What are the exact rate limits and quotas for Claude Sonnet 4.6 "
      "on AWS Bedrock? Provide specific numbers if available."
  )

  console.print(f"[bold]🤔 User:[/bold] {question}\n")

  result = agent.run(question)

  # Show the answer
  console.print(Panel(
      Markdown(result["final_answer"]),
      title=f"✅ Agent Answer (iterations: {result['iterations']})",
      border_style="green" if not result["stopped_by_guardrail"] else "yellow",
  ))

  # Show guardrail status
  summary = result["guardrail_summary"]
  console.print("\n[bold]🛡️  Guardrail Summary:[/bold]")
  console.print(f"  Stopped by guardrail: {result['stopped_by_guardrail']}")
  if summary["stop_reason"]:
      console.print(f"  Stop reason: [yellow]{summary['stop_reason']}[/yellow]")
  console.print(f"  Iterations: {summary['iterations']}")
  console.print(f"  Total input tokens: {summary['total_input_tokens']:,}")
  console.print(f"  Estimated cost: [green]${summary['estimated_cost_usd']}[/green]")

  # Show tool sequence
  if result["tool_calls"]:
      console.print("\n[dim]🔧 Tool sequence:[/dim]")
      for call in result["tool_calls"]:
          console.print(
              f"[dim]  → {call['tool_name']}"
              f"({str(call['tool_input'])[:60]})[/dim]"
          )


if __name__ == "__main__":
  main()