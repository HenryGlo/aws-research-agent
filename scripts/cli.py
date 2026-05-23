"""Interactive CLI for the AWS Research Agent.

A conversational loop where you can ask the agent questions and watch
its reasoning in real time. Type 'exit' or 'quit' to leave.

Usage:
  python -m scripts.cli
"""

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from src.agent.guardrails import GuardrailLimits
from src.agent.loop import Agent
from src.tools.web_fetch import WebFetchTool
from src.tools.web_search import WebSearchTool

console = Console()

BANNER = """
[bold cyan]╔═══════════════════════════════════════════════╗
║         🤖 AWS Research Agent (CLI)           ║
║   Powered by Claude Sonnet 4.6 + tool use     ║
╚═══════════════════════════════════════════════╝[/bold cyan]

Ask me anything about AWS and cloud computing.
I'll search the web, read docs, and synthesize answers.

[dim]Commands: 'exit' or 'quit' to leave · Ctrl+C to interrupt[/dim]
"""


def main():
  console.print(BANNER)

  # Build the agent once, reuse across questions
  limits = GuardrailLimits(
      max_total_input_tokens=40_000,
      max_cost_usd=0.40,
      max_iterations=8,
  )
  agent = Agent(
      tools=[WebSearchTool(), WebFetchTool()],
      guardrail_limits=limits,
      verbose=True,
  )

  session_cost = 0.0
  session_questions = 0

  while True:
      try:
          question = Prompt.ask("\n[bold green]❓ Your question[/bold green]")
      except (KeyboardInterrupt, EOFError):
          console.print("\n\n[dim]Interrupted. Goodbye![/dim]")
          break

      # Handle exit commands
      if question.strip().lower() in ("exit", "quit", "q"):
          break

      if not question.strip():
          console.print("[dim]Please enter a question.[/dim]")
          continue

      # Run the agent
      try:
          result = agent.run(question)
      except Exception as e:
          console.print(f"\n[red]❌ Error: {type(e).__name__}: {e}[/red]")
          console.print("[dim]Try rephrasing or ask something else.[/dim]")
          continue

      # Show the answer
      console.print(Panel(
          Markdown(result["final_answer"]),
          title="✅ Answer",
          border_style="green",
      ))

      # Track session stats
      cost = result["guardrail_summary"]["estimated_cost_usd"]
      session_cost += cost
      session_questions += 1

      # Show per-question footer
      guardrail_note = ""
      if result["stopped_by_guardrail"]:
          reason = result["guardrail_summary"]["stop_reason"]
          guardrail_note = f" · [yellow]⚠️  {reason}[/yellow]"
      console.print(
          f"[dim]This answer: {result['iterations']} iterations · "
          f"${cost:.4f}{guardrail_note}[/dim]"
      )

  # Session summary on exit
  console.print(Panel(
      f"Questions answered: {session_questions}\n"
      f"Total session cost: ${session_cost:.4f}",
      title="📊 Session Summary",
      border_style="cyan",
  ))
  console.print("[bold cyan]Thanks for using AWS Research Agent! 👋[/bold cyan]\n")


if __name__ == "__main__":
  main()
