"""
Command-line interface for verbalized-sampling.
"""

import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from typing import List, Optional

from verbalized_sampling.pipeline import run_quick_comparison
from verbalized_sampling.tasks import Task
from verbalized_sampling.methods import Method

app = typer.Typer(help="Run controlled experiments with LLMs using different sampling methods")
console = Console()

# Task descriptions
TASK_DESCRIPTIONS = {
    Task.RANDOM_NUM: "Random number generation - tests basic sampling uniformity",
    Task.CREATIVE_STORY: "Creative story generation - tests narrative coherence and creativity",
    Task.BOOK: "Book continuation - tests long-form narrative coherence",
    Task.POEM: "Poetry generation - tests creative expression and poetic forms",
    Task.SPEECH: "Speech generation - tests rhetorical effectiveness",
    Task.STATE_NAME: "State name generation - tests creative naming capabilities",
    Task.JOKE: "Joke generation - tests humor and creative wordplay",
}

# Method descriptions
METHOD_DESCRIPTIONS = {
    Method.DIRECT: "Direct sampling - baseline method using prompt as-is",
    Method.SEQUENCE: "Sequential sampling - generates multiple responses in list format",
    Method.STRUCTURE: "Structured sampling - uses JSON format with response field",
    Method.STRUCTURE_WITH_PROB: "Structured with probability - JSON with response and confidence",
    Method.MULTI_TURN: "Multi-turn conversation - conversational format with multiple turns",
}

@app.command()
def run(
    task: Task = typer.Option(..., help="Task to run (e.g., JOKE, POEM, STATE_NAME)"),
    model: str = typer.Option(..., help="Model to use (e.g., openai/gpt-4.1)"),
    methods: List[Method] = typer.Option([Method.DIRECT], help="Methods to compare"),
    num_responses: int = typer.Option(50, help="Number of responses to generate"),
    num_samples: int = typer.Option(5, help="Number of samples per prompt"),
    num_prompts: int = typer.Option(1, help="Number of prompts to use"),
    strict_json: bool = typer.Option(False, help="Use strict JSON mode"),
    metrics: List[str] = typer.Option(["diversity", "length", "ngram"], help="Metrics to evaluate"),
    output_dir: Path = typer.Option(Path("results"), help="Output directory"),
    temperature: float = typer.Option(0.7, help="Sampling temperature"),
    top_p: float = typer.Option(1.0, help="Top-p sampling parameter"),
    rerun: bool = typer.Option(False, help="Rerun existing results"),
):
    """Run a quick comparison experiment."""
    console.print(f"🔬 Running {task.value} experiment with {model}")
    
    results = run_quick_comparison(
        task=task,
        methods=methods,
        model_name=model,
        metrics=metrics,
        output_dir=output_dir,
        num_responses=num_responses,
        num_samples=num_samples,
        num_prompts=num_prompts,
        strict_json=strict_json,
        rerun=rerun,
        temperature=temperature,
        top_p=top_p,
    )
    
    # Display results summary
    table = Table(title="Results Summary")
    table.add_column("Method")
    for metric in metrics:
        table.add_column(metric)
    
    for method, method_results in results.items():
        row = [method.value]
        for metric in metrics:
            if metric in method_results:
                row.append(f"{method_results[metric]:.3f}")
            else:
                row.append("N/A")
        table.add_row(*row)
    
    console.print(table)
    console.print(f"✅ Done! Check {output_dir}/pipeline_report.html")

@app.command()
def list_tasks():
    """List available tasks."""
    table = Table(title="Available Tasks")
    table.add_column("Task", style="bold cyan")
    table.add_column("Description")
    
    for task in Task:
        description = TASK_DESCRIPTIONS.get(task, "No description available")
        table.add_row(task.value, description)
    
    console.print(table)

@app.command()
def list_methods():
    """List available sampling methods."""
    table = Table(title="Available Methods")
    table.add_column("Method", style="bold cyan")
    table.add_column("Description")
    
    for method in Method:
        description = METHOD_DESCRIPTIONS.get(method, "No description available")
        table.add_row(method.value, description)
    
    console.print(table)

if __name__ == "__main__":
    app()