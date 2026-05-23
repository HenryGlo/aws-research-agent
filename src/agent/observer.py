"""Real-time observability for the agent's reasoning process.

While the agent runs, this surfaces each decision visually:
- Which tool the agent chose and with what input
- How long each step took
- Cumulative cost and token usage
- When the agent synthesizes vs continues

Design: the observer is a passive presentation layer. The agent loop
emits events; the observer formats them. No business logic here —
just observability. Same separation-of-concerns philosophy as guardrails.

Uses `rich` for terminal rendering. Can be toggled off (verbose=False)
for production/API use where you don't want console output.
"""
import time

from rich.console import Console
from rich.panel import Panel


class AgentObserver:
  """Renders the agent's reasoning steps in real time."""

  def __init__(self, verbose: bool = True):
      self.verbose = verbose
      self.console = Console()
      self._iteration_start: float = 0.0

  def on_agent_start(self, query: str) -> None:
      """Called when the agent begins processing a query."""
      if not self.verbose:
          return
      self.console.print(
          f"\n[bold cyan]🤔 Query:[/bold cyan] {query}\n"
      )

  def on_iteration_start(self, iteration: int) -> None:
      """Called at the start of each agent loop iteration."""
      self._iteration_start = time.time()

  def on_tool_decision(
      self,
      iteration: int,
      tool_name: str,
      tool_input: dict,
      cost_usd: float,
      total_tokens: int,
  ) -> None:
      """Called when the agent decides to use a tool."""
      if not self.verbose:
          return

      elapsed = time.time() - self._iteration_start
      input_preview = self._format_input(tool_input)

      body = (
          f"[bold]🔧 Decision:[/bold] [magenta]{tool_name}[/magenta]\n"
          f"   {input_preview}\n"
          f"[dim]⏱️  {elapsed:.1f}s · "
          f"💰 ${cost_usd:.4f} · "
          f"📊 {total_tokens:,} tokens[/dim]"
      )
      self.console.print(Panel(
          body,
          title=f"Iteration {iteration}",
          border_style="blue",
          expand=False,
      ))

  def on_synthesis(
      self,
      iteration: int,
      cost_usd: float,
      total_tokens: int,
      forced: bool = False,
  ) -> None:
      """Called when the agent produces its final answer."""
      if not self.verbose:
          return

      elapsed = time.time() - self._iteration_start
      label = (
          "⚠️  Forced synthesis (guardrail)" if forced
          else "✅ Synthesizing final answer"
      )
      body = (
          f"[bold]{label}[/bold]\n"
          f"[dim]⏱️  {elapsed:.1f}s · "
          f"💰 ${cost_usd:.4f} · "
          f"📊 {total_tokens:,} tokens[/dim]"
      )
      self.console.print(Panel(
          body,
          title=f"Iteration {iteration}",
          border_style="yellow" if forced else "green",
          expand=False,
      ))

  def on_agent_complete(
      self,
      iterations: int,
      cost_usd: float,
      total_tokens: int,
      stopped_by_guardrail: bool,
  ) -> None:
      """Called when the agent fully completes."""
      if not self.verbose:
          return

      status = (
          "[yellow]stopped by guardrail[/yellow]" if stopped_by_guardrail
          else "[green]completed normally[/green]"
      )
      self.console.print(
          f"\n[bold]🏁 Agent {status}[/bold] · "
          f"{iterations} iterations · "
          f"${cost_usd:.4f} · "
          f"{total_tokens:,} tokens\n"
      )

  @staticmethod
  def _format_input(tool_input: dict) -> str:
      """Format tool input for display, truncating long values."""
      parts = []
      for key, value in tool_input.items():
          value_str = str(value)
          if len(value_str) > 70:
              value_str = value_str[:67] + "..."
          parts.append(f"[cyan]{key}[/cyan]: {value_str}")
      return " · ".join(parts)