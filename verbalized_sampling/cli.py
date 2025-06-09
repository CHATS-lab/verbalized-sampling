import typer
from pathlib import Path
from typing import List
from rich.console import Console
from rich.progress import Progress
from verbalized_sampling.tasks import Task
from verbalized_sampling.prompts import Method
import json

app = typer.Typer(help="Verbalized Sampling Experiment CLI")
console = Console()


@app.command()
def run_experiment(
    task: Task = typer.Option(..., help="Task to run (rand_num or story)"),
    model_name: str = typer.Option("meta-llama/Llama-3.1-70B-Instruct", help="Model name to use"),
    method: Method = typer.Option(..., help="Sampling method (direct, seq, structure, structure_with_prob)"),
    temperature: float = typer.Option(0.7, help="Sampling temperature"),
    top_p: float = typer.Option(0.9, help="Top-p sampling parameter"),
    num_responses: int = typer.Option(3, help="Number of responses to generate"),
    num_samples: int = typer.Option(1, help="Number of samples per response"),
    num_workers: int = typer.Option(128, help="Number of parallel workers"),
    output_file: Path = typer.Option("responses.jsonl", help="Output file path"),
    use_vllm: bool = typer.Option(False, help="Whether to use vLLM"),
    sample_size: int = typer.Option(1, help="Number of samples to generate"),
    random_seed: int = typer.Option(42, help="Random seed"),
    all_possible: bool = typer.Option(False, help="Whether to use all possible responses"),
    strict_json: bool = typer.Option(False, help="Whether to use strict JSON mode"),
):
    """Run a verbalized sampling experiment."""
    from verbalized_sampling.tasks import get_task
    from verbalized_sampling.llms import get_model
    
    console.print(f"Running experiment for task: {task}")
    console.print(f"Using model: {model_name}")
    
    kwargs = {}
    if task in [Task.POEM, Task.SPEECH]:
        kwargs["sample_size"] = sample_size
        kwargs["random_seed"] = random_seed
    
    # Get task and model
    model = get_model(
        model_name=model_name,
        method=method,
        config={"temperature": temperature, "top_p": top_p},
        use_vllm=use_vllm,
        num_workers=num_workers,
        strict_json=strict_json
    )
    task_instance = get_task(
        task, 
        model=model,
        method=method,
        num_responses=num_responses,
        num_samples=num_samples,
        sample_size=sample_size,
        random_seed=random_seed,
        all_possible=all_possible,
        strict_json=strict_json
    )
    
    # Run experiment
    with Progress() as progress:
        task = progress.add_task("[cyan]Running experiment...", total=num_responses)
        results = task_instance.run(
            progress=progress,
            task_id=task
        )
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    task_instance.save_results(results, output_file)
    console.print(f"Results saved to {output_file}")

@app.command()
def evaluate(
    metric: str = typer.Option(..., help="Metric to evaluate"),
    task: Task = typer.Option(..., help="Task to evaluate"),
    input_file: Path = typer.Option(..., help="Input file path"),
    output_file: Path = typer.Option(..., help="Output file path"),
    num_workers: int = typer.Option(128, help="Number of parallel workers"),
):
    """Evaluate the results of an experiment."""
    from verbalized_sampling.evals import get_evaluator
    from verbalized_sampling.tasks import get_task
    
    console.print(f"Evaluating {metric} for task: {task}")
    console.print(f"Using output file: {output_file}")

    # Get task and evaluator
    task_instance = get_task(task)
    evaluator = get_evaluator(metric, num_workers=num_workers)

    with open(input_file) as f:
        responses = []
        for line in f:
            try:
                responses.append(json.loads(line))
            except json.JSONDecodeError:
                responses.append(line)

    results = evaluator.evaluate(
        responses,
        responses,
        {}
        )

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    evaluator.save_results(results, output_file)
    console.print(f"Results saved to {output_file}")

@app.command()
def analyze_results(
    target_dir: Path = typer.Option(..., help="Directory containing response files"),
    output_dir: Path = typer.Option("analysis", help="Output directory for analysis"),
    sizes: List[int] = typer.Option([1, 3, 5, 10, 50], help="Sampling sizes to analyze"),
):
    """Analyze experiment results."""
    from verbalized_sampling.analysis.plotting import plot_histograms
    from verbalized_sampling.evals.chi_square import evaluate_chi_square
    import json
    
    console.print(f"Analyzing results from {target_dir}")
    
    # Create output directories
    plots_dir = output_dir / "plots"
    metrics_dir = output_dir / "metrics"
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot histograms
    plot_histograms(target_dir, plots_dir / "histograms.png", sizes)
    console.print(f"Plots saved to {plots_dir}")
    
    # Calculate metrics
    metrics = {}
    for size in sizes:
        responses = []
        for file in target_dir.glob(f"*_size_{size}_*.jsonl"):
            with open(file) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if isinstance(data, list):
                            responses.extend(data)
                        else:
                            responses.append(data)
                    except json.JSONDecodeError:
                        continue
        
        if responses:
            metrics[size] = evaluate_chi_square(responses)
    
    # Save metrics
    with open(metrics_dir / "chi_square_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    console.print(f"Metrics saved to {metrics_dir}")

# Add to verbalized_sampling/cli.py
@app.command()
def compare_evaluations(
    results_dir: Path = typer.Option(..., help="Directory containing evaluation result files"),
    output_dir: Path = typer.Option("comparison_plots", help="Output directory for plots"),
    evaluator_type: str = typer.Option("auto", help="Type of evaluator"),
    pattern: str = typer.Option("*_results.json", help="File pattern to match")
):
    """Compare evaluation results across different formats."""
    from verbalized_sampling.evals import plot_evaluation_comparison
    
    # Find all result files
    result_files = list(results_dir.glob(pattern))
    
    if not result_files:
        console.print(f"No result files found matching pattern '{pattern}' in {results_dir}")
        return
    
    # Create results dict from files (extract format name from filename)
    results = {}
    for file_path in result_files:
        format_name = file_path.stem.replace("_results", "")
        results[format_name] = file_path
    
    console.print(f"Comparing {len(results)} evaluation results...")
    plot_evaluation_comparison(results, output_dir, evaluator_type)
    console.print(f"Comparison plots saved to {output_dir}")
    
@app.command()
def run_pipeline(
    config_file: Path = typer.Option(..., help="Pipeline configuration file (YAML/JSON)"),
    output_dir: Path = typer.Option("pipeline_output", help="Base output directory"),
    skip_existing: bool = typer.Option(True, help="Skip existing files"),
    num_workers: int = typer.Option(128, help="Number of workers"),
    rerun: bool = typer.Option(False, help="Rerun everything from scratch (ignores skip_existing)"),
    create_backup: bool = typer.Option(True, help="Create backup before cleaning (only with --rerun)"),
    force: bool = typer.Option(False, help="Force rerun without any prompts (implies --rerun)")
):
    """Run the complete generation → evaluation → plotting pipeline."""
    from verbalized_sampling.pipeline import run_pipeline_cli
    
    # Handle force flag
    if force:
        rerun = True
        create_backup = False  # Skip backup for force mode
        console.print("[bold red]🚨 FORCE MODE: Rerunning without backup![/bold red]")
    elif rerun:
        console.print("[bold yellow]🔄 RERUN MODE: All existing results will be overwritten![/bold yellow]")
        if create_backup:
            console.print("[dim]💾 Backup will be created automatically[/dim]")
        else:
            console.print("[dim]⚠️  No backup will be created[/dim]")
    
    # No confirmation needed - user explicitly requested rerun
    console.print("[bold blue]🚀 Starting Complete Pipeline[/bold blue]")
    results = run_pipeline_cli(config_file, output_dir, skip_existing, num_workers, rerun, create_backup)
    console.print("[bold green]✅ Pipeline completed successfully![/bold green]")

@app.command()
def quick_compare(
    task: Task = typer.Option(..., help="Task to compare"),
    model_name: str = typer.Option("openai/gpt-4", help="Model to use"),
    output_dir: Path = typer.Option("quick_comparison", help="Output directory"),
    num_responses: int = typer.Option(10, help="Number of responses per method"),
    metrics: List[str] = typer.Option(["diversity", "length"], help="Metrics to evaluate"),
    rerun: bool = typer.Option(False, help="Rerun everything from scratch"),
    create_backup: bool = typer.Option(True, help="Create backup before cleaning"),
    force: bool = typer.Option(False, help="Force rerun without prompts or backup")
):
    """Quick comparison of all available methods for a task."""
    from verbalized_sampling.pipeline import run_quick_comparison
    from verbalized_sampling.prompts import Method
    
    # Handle force flag
    if force:
        rerun = True
        create_backup = False
        console.print(f"[bold red]🚨 FORCE MODE: Fresh comparison without backup[/bold red]")
    elif rerun:
        console.print(f"[bold yellow]🔄 RERUN MODE: Starting fresh comparison[/bold yellow]")
    
    # Use all available methods
    methods = [Method.DIRECT, Method.SEQ, Method.STRUCTURE, Method.STRUCTURE_WITH_PROB]
    
    console.print(f"[bold blue]🏃‍♂️ Quick comparison: {task.value} with {model_name}[/bold blue]")
    results = run_quick_comparison(task, methods, model_name, metrics, output_dir, num_responses, rerun, create_backup)
    console.print("[bold green]✅ Quick comparison completed![/bold green]")

if __name__ == "__main__":
    app()