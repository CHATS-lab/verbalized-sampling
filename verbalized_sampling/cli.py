import typer
from pathlib import Path
from typing import List, Optional
from rich.console import Console
from rich.progress import Progress

app = typer.Typer(help="Verbalized Sampling Experiment CLI")
console = Console()

@app.command()
def run_experiment(
    task: str = typer.Option(..., help="Task to run (rand_num or story)"),
    model_name: str = typer.Option("meta-llama/Llama-3.1-70B-Instruct", help="Model name to use"),
    format: str = typer.Option(..., help="Sampling format (direct, seq, structure, structure_with_prob)"),
    temperature: float = typer.Option(0.7, help="Sampling temperature"),
    top_p: float = typer.Option(0.9, help="Top-p sampling parameter"),
    num_responses: int = typer.Option(3, help="Number of responses to generate"),
    num_samples: int = typer.Option(1, help="Number of samples per response"),
    num_workers: int = typer.Option(128, help="Number of parallel workers"),
    output_file: Path = typer.Option("responses.jsonl", help="Output file path"),
    use_vllm: bool = typer.Option(False, help="Whether to use vLLM"),
):
    """Run a verbalized sampling experiment."""
    from verbalized_sampling.tasks import get_task
    from verbalized_sampling.llms import get_model
    
    console.print(f"Running experiment for task: {task}")
    console.print(f"Using model: {model_name}")
    
    # Get task and model
    task_instance = get_task(task, format=format)
    model = get_model(
        model_name=model_name,
        sim_type=format,
        config={"temperature": temperature, "top_p": top_p},
        use_vllm=use_vllm
    )
    
    # Run experiment
    with Progress() as progress:
        task = progress.add_task("[cyan]Running experiment...", total=num_responses)
        results = task_instance.run(
            model=model,
            num_responses=num_responses,
            num_samples=num_samples,
            num_workers=num_workers,
            progress=progress,
            task_id=task
        )
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    task_instance.save_results(results, output_file)
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

if __name__ == "__main__":
    app() 