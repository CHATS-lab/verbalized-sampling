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
    # Creative writing tasks (Section 5)
    Task.POEM: "Poetry generation - tests creative expression and poetic forms",
    # Task.CREATIVE_STORY: "Creative story generation - tests narrative coherence and creativity",
    Task.JOKE: "Joke generation - tests humor and creative wordplay",
    Task.BOOK: "Story continuation - tests long-form narrative coherence",
    # Task.SPEECH: "Speech generation - tests rhetorical effectiveness",

    # Bias mitigation tasks (Appendix)
    Task.RANDOM_NUM: "Random number generation - tests basic sampling uniformity",
    Task.STATE_NAME: "State name generation - tests creative naming capabilities",

    # Knowledge and QA tasks (Appendix)
    Task.SIMPLE_QA: "Simple QA - tests factual knowledge diversity while maintaining accuracy",

    # Synthetic data generation tasks (Section 7)
    Task.GSM8K: "GSM8K math problems - generate grade school math word problems",
    Task.AMCAndAIMEMathTask: "AMC/AIME math - generate competition-level mathematics problems",
    Task.LIVECODEBENCH: "LiveCodeBench - generate coding problems for evaluation",
    Task.SYNTHETIC_NEGATIVE: "Negative examples - generate incorrect solutions for robust training",

    # Math evaluation tasks
    Task.MATH: "MATH dataset problems - complex mathematical reasoning tasks",
    Task.AIME: "AIME problems - American Invitational Mathematics Examination",
    Task.AMC: "AMC problems - American Mathematics Competitions",
    Task.MINERVA: "Minerva math - mathematical reasoning evaluation dataset",
    Task.OLYMPIAD_BENCH: "Olympiad problems - international mathematics olympiad challenges",

    # Safety evaluation (Appendix)
    Task.SAFETY: "Safety evaluation - test refusal rates for harmful content using StrongReject",
}

# Method descriptions (VS = Verbalized Sampling)
METHOD_DESCRIPTIONS = {
    # Baseline methods
    Method.DIRECT: "Direct sampling - baseline method using prompt as-is",
    Method.DIRECT_BASE: "Direct base sampling - direct prompting for base models without chat template",
    Method.DIRECT_COT: "Chain-of-Thought - adds reasoning step to direct prompting",
    Method.SEQUENCE: "Sequential sampling - generates multiple responses in list format",
    Method.STRUCTURE: "Structured sampling - uses JSON format with response field only",
    Method.MULTI_TURN: "Multi-turn sampling - generates multiple responses in multi-turn format",

    # Verbalized Sampling methods (paper-aligned)
    Method.VS_STANDARD: "VS-Standard - verbalized sampling with responses and probabilities",
    Method.VS_COT: "VS-CoT - verbalized sampling with chain-of-thought reasoning",
    Method.VS_MULTI: "VS-Multi - verbalized sampling with multi-turn/combined approach",

    # Additional specialized methods
    # Method.STANDARD_ALL_POSSIBLE: "All-possible standard - generates all reasonable variations of a response",

    # Note: Legacy method names (STRUCTURE_WITH_PROB, CHAIN_OF_THOUGHT, COMBINED)
    # are aliases that resolve to the same enum objects as VS_STANDARD, VS_COT, VS_MULTI
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
        # Use paper name if available, otherwise use method value
        display_name = method.paper_name if hasattr(method, 'paper_name') else method.value
        table.add_row(display_name, description)
    
    console.print(table)

if __name__ == "__main__":
    app()