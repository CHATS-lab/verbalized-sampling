"""
Streamlined end-to-end pipeline for generation, evaluation, and plotting.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import typer
import shutil
import datetime

from verbalized_sampling.tasks import Task, get_task
from verbalized_sampling.prompts import Method
from verbalized_sampling.llms import get_model
from verbalized_sampling.evals import get_evaluator

console = Console()

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    task: Task
    method: Method
    model_name: str
    temperature: float = 0.7
    top_p: float = 0.9
    num_responses: int = 10
    num_samples: int = 1
    sample_size: int = 5
    random_seed: int = 42
    use_vllm: bool = False
    all_possible: bool = False # If True, the request would enable all possible responses
    strict_json: bool = False # If True, the request would enable JSON mode

@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    metrics: List[str]
    num_workers: int = 128

@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    experiments: List[ExperimentConfig]
    evaluation: EvaluationConfig
    output_base_dir: Path
    num_workers: int = 128
    skip_existing: bool = False
    rerun: bool = False
    create_backup: bool = False
    
    def _should_backup(self) -> bool:
        """Determine if backup should be created."""
        return self.create_backup

class Pipeline:
    """End-to-end pipeline for generation, evaluation, and plotting."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.validate_config()
        self.results = {}
    
    def validate_config(self) -> None:
        """Validate the configuration."""
        if self.config.rerun and self.config.skip_existing:
            raise ValueError("Rerun mode and skip_existing cannot be True at the same time.")
        
        if self.config.create_backup and not self.config.rerun:
            raise ValueError("Create backup is only allowed in rerun mode.")

    def _handle_rerun(self) -> None:
        """Handle rerun logic - clean up existing outputs."""
        if self.config.output_base_dir.exists():
            console.print(f"[bold yellow]🧹 Rerun mode: Cleaning up existing outputs in {self.config.output_base_dir}[/bold yellow]")
            
            # Create backup if enabled (but don't ask for confirmation)
            if self.config.create_backup:
                backup_dir = self.config.output_base_dir.parent / f"{self.config.output_base_dir.name}_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                try:
                    console.print(f"📦 Creating backup at: {backup_dir}")
                    shutil.copytree(self.config.output_base_dir, backup_dir)
                    console.print("✅ Backup created successfully")
                except Exception as e:
                    console.print(f"[yellow]⚠️  Backup failed: {str(e)} - continuing anyway[/yellow]")
            
            try:
                # Remove existing directory without confirmation
                console.print("🗑️  Removing existing output directory...")
                shutil.rmtree(self.config.output_base_dir)
                console.print("✅ Cleanup complete")
                
            except Exception as e:
                console.print(f"[bold red]❌ Error during cleanup: {str(e)}[/bold red]")
                console.print("Continuing with pipeline...")
        
        # Override skip_existing for rerun mode
        self.config.skip_existing = False
        
    def run_complete_pipeline(self) -> Dict[str, Any]:
        if self.config.rerun:
            self._handle_rerun()
            
        """Run the complete pipeline: generation → evaluation → plotting."""
        console.print("[bold blue]🚀 Starting Complete Pipeline[/bold blue]")
        
        # Step 1: Generate responses
        console.print("\n[bold green]Step 1: Generating Responses[/bold green]")
        generation_results = self.run_generation()
        
        # Step 2: Run evaluations
        console.print("\n[bold green]Step 2: Running Evaluations[/bold green]")
        evaluation_results = self.run_evaluation(generation_results)
        
        # Step 3: Create plots
        console.print("\n[bold green]Step 3: Creating Comparison Plots[/bold green]")
        plot_results = self.create_plots(evaluation_results)
        
        # Step 4: Generate summary report
        console.print("\n[bold green]Step 4: Generating Summary Report[/bold green]")
        report_path = self.generate_report(evaluation_results, plot_results)
        
        console.print(f"\n[bold blue]✅ Pipeline Complete![/bold blue]")
        console.print(f"📊 Summary report: {report_path}")
        console.print(f"📁 All outputs in: {self.config.output_base_dir}")
        
        return {
            "generation_results": generation_results,
            "evaluation_results": evaluation_results,
            "plot_results": plot_results,
            "report_path": report_path
        }
    
    def run_generation(self) -> Dict[str, Path]:
        """Run generation for all experiments."""
        generation_results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            overall_task = progress.add_task(
                "Generating responses...", 
                total=len(self.config.experiments)
            )
            
            for exp_config in self.config.experiments:
                # Setup output path
                exp_dir = self.config.output_base_dir / "generation" / exp_config.name
                exp_dir.mkdir(parents=True, exist_ok=True)
                output_file = exp_dir / "responses.jsonl"
                
                if self.config.skip_existing and output_file.exists():
                    console.print(f"⏭️  Skipping {exp_config.name} (already exists)")
                    generation_results[exp_config.name] = output_file
                    progress.advance(overall_task)
                    continue
                
                progress.console.print(f"🔄 Generating: {exp_config.name}")
                
                # Setup model and task
                model = get_model(
                    model_name=exp_config.model_name,
                    method=exp_config.method,
                    config={
                        "temperature": exp_config.temperature, 
                        "top_p": exp_config.top_p
                    },
                    use_vllm=exp_config.use_vllm,
                    num_workers=self.config.num_workers,
                    strict_json=exp_config.strict_json
                )
                
                task_kwargs = {}
                if exp_config.task in [Task.POEM, Task.SPEECH]:
                    task_kwargs.update({
                        "sample_size": exp_config.sample_size,
                        "random_seed": exp_config.random_seed
                    })
                
                num_samples = exp_config.num_samples if exp_config.method != Method.DIRECT else 1
                num_responses = exp_config.num_responses // num_samples
                task_instance = get_task(
                    exp_config.task,
                    model=model,
                    method=exp_config.method,
                    num_responses=num_responses,
                    num_samples=num_samples,
                    **task_kwargs
                )
                
                # Run generation
                gen_task = progress.add_task(
                    f"[cyan]{exp_config.name}[/cyan]", 
                    total=exp_config.num_responses
                )
                
                results = task_instance.run(progress=progress, task_id=gen_task)
                task_instance.save_results(results, output_file)
                
                generation_results[exp_config.name] = output_file
                progress.remove_task(gen_task)
                progress.advance(overall_task)
                
                console.print(f"✅ {exp_config.name}: {len(results)} responses saved")
        
        return generation_results
    
    def run_evaluation(self, generation_results: Dict[str, Path]) -> Dict[str, Dict[str, Path]]:
        """Run evaluations for all metrics on all experiments."""
        evaluation_results = {}
        
        total_evals = len(generation_results) * len(self.config.evaluation.metrics)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            overall_task = progress.add_task(
                "Running evaluations...", 
                total=total_evals
            )
            
            for exp_name, responses_file in generation_results.items():
                evaluation_results[exp_name] = {}
                
                # Load responses
                with open(responses_file, 'r') as f:
                    responses = []
                    prompts = []
                    for line in f:
                        try:
                            data = json.loads(line)
                            if isinstance(data, dict):
                                responses.append(data.get('text', str(data)))
                                prompts.append(data.get('prompt', ''))
                            else:
                                responses.append(str(data))
                                prompts.append('')
                        except json.JSONDecodeError:
                            responses.append(line.strip())
                            prompts.append('')
                
                # Run each metric
                for metric in self.config.evaluation.metrics:
                    eval_dir = self.config.output_base_dir / "evaluation" / exp_name
                    eval_dir.mkdir(parents=True, exist_ok=True)
                    eval_file = eval_dir / f"{metric}_results.json"
                    
                    if self.config.skip_existing and eval_file.exists():
                        console.print(f"⏭️  Skipping {exp_name}/{metric} (already exists)")
                        evaluation_results[exp_name][metric] = eval_file
                        progress.advance(overall_task)
                        continue
                    
                    progress.console.print(f"📊 Evaluating: {exp_name}/{metric}")
                    
                    try:
                        # Get evaluator and run evaluation
                        evaluator = get_evaluator(
                            metric, 
                            num_workers=self.config.evaluation.num_workers
                        )
                        
                        result = evaluator.evaluate(
                            prompts or responses,  # Use responses as prompts if no prompts available
                            responses,
                            metadata={"experiment": exp_name, "metric": metric}
                        )
                        
                        # Save results
                        evaluator.save_results(result, eval_file)
                        evaluation_results[exp_name][metric] = eval_file
                        
                        console.print(f"✅ {exp_name}/{metric}: Evaluation complete")
                        
                    except Exception as e:
                        console.print(f"❌ {exp_name}/{metric}: Error - {str(e)}")
                        evaluation_results[exp_name][metric] = None
                    
                    progress.advance(overall_task)
        
        return evaluation_results
    
    def create_plots(self, evaluation_results: Dict[str, Dict[str, Path]]) -> Dict[str, Path]:
        """Create comparison plots for each metric."""
        from verbalized_sampling.evals import plot_evaluation_comparison
        
        plot_results = {}
        plots_base_dir = self.config.output_base_dir / "plots"
        plots_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Group results by metric
        metric_results = {}
        for exp_name, exp_metrics in evaluation_results.items():
            for metric, result_file in exp_metrics.items():
                if result_file is None:
                    continue
                if metric not in metric_results:
                    metric_results[metric] = {}
                metric_results[metric][exp_name] = result_file
        
        # Create plots for each metric
        for metric, results in metric_results.items():
            if not results:
                continue
                
            console.print(f"📈 Creating plots for: {metric}")
            
            # try:
            plot_dir = plots_base_dir / metric
            plot_evaluation_comparison(
                results,
                output_dir=plot_dir,
                evaluator_type=metric
            )
            plot_results[metric] = plot_dir
            console.print(f"✅ {metric}: Plots saved to {plot_dir}")
                
            # except Exception as e:
            #     console.print(f"❌ {metric}: Plot error - {str(e)}")
            #     plot_results[metric] = None
        
        return plot_results
    
    def generate_report(self, evaluation_results: Dict[str, Dict[str, Path]], 
                       plot_results: Dict[str, Path]) -> Path:
        """Generate a comprehensive HTML report."""
        from verbalized_sampling.evals.base import EvalResult
        
        report_path = self.config.output_base_dir / "pipeline_report.html"
        
        # Load all evaluation results
        loaded_results = {}
        for exp_name, exp_metrics in evaluation_results.items():
            loaded_results[exp_name] = {}
            for metric, result_file in exp_metrics.items():
                if result_file and result_file.exists():
                    try:
                        with open(result_file, 'r') as f:
                            result_dict = json.load(f)
                            loaded_results[exp_name][metric] = EvalResult.from_dict(result_dict)
                    except Exception as e:
                        console.print(f"Warning: Could not load {result_file}: {e}")
        
        # Generate HTML report
        html_content = self._generate_html_report(loaded_results, plot_results)
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return report_path
    
    def _generate_html_report(self, results: Dict[str, Dict[str, Any]], 
                            plot_results: Dict[str, Path]) -> str:
        """Generate HTML report content."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pipeline Results Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 8px; }}
                .experiment {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ margin: 10px 0; }}
                .plots {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .number {{ text-align: right; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Pipeline Results Report</h1>
                <p><strong>Generated:</strong> {Path().absolute()}</p>
                <p><strong>Experiments:</strong> {len(self.config.experiments)}</p>
                <p><strong>Metrics:</strong> {', '.join(self.config.evaluation.metrics)}</p>
            </div>
        """
        
        # Experiment configurations
        html += "<h2>📋 Experiment Configurations</h2>"
        html += "<table><tr><th>Name</th><th>Task</th><th>Method</th><th>Model</th><th>Responses</th><th>Temperature</th></tr>"
        for exp in self.config.experiments:
            html += f"<tr><td>{exp.name}</td><td>{exp.task.value}</td><td>{exp.method.value}</td><td>{exp.model_name}</td><td class='number'>{exp.num_responses}</td><td class='number'>{exp.temperature}</td></tr>"
        html += "</table>"
        
        # Results summary
        html += "<h2>📊 Results Summary</h2>"
        for metric in self.config.evaluation.metrics:
            html += f"<h3>{metric.title()} Results</h3>"
            html += "<table><tr><th>Experiment</th>"
            
            # Get metric keys from first result
            first_result = None
            for exp_name in results:
                if metric in results[exp_name]:
                    first_result = results[exp_name][metric]
                    break
            
            if first_result:
                metric_keys = list(first_result.overall_metrics.keys())
                for key in metric_keys:
                    html += f"<th>{key.replace('_', ' ').title()}</th>"
                html += "</tr>"
                
                for exp_name, exp_results in results.items():
                    if metric in exp_results:
                        result = exp_results[metric]
                        html += f"<tr><td>{exp_name}</td>"
                        for key in metric_keys:
                            value = result.overall_metrics.get(key, 'N/A')
                            if isinstance(value, (int, float)):
                                html += f"<td class='number'>{value:.4f}</td>"
                            else:
                                html += f"<td>{value}</td>"
                        html += "</tr>"
            html += "</table>"
        
        # Plot links
        html += "<h2>📈 Visualization Links</h2>"
        for metric, plot_dir in plot_results.items():
            if plot_dir:
                html += f"<p><strong>{metric.title()}:</strong> <a href='{plot_dir.relative_to(self.config.output_base_dir)}'>{plot_dir.relative_to(self.config.output_base_dir)}</a></p>"
        
        html += "</body></html>"
        return html

# CLI Integration
def run_pipeline_cli(
    config_file: Path = typer.Option(..., help="Pipeline configuration file (YAML/JSON)"),
    output_dir: Path = typer.Option("pipeline_output", help="Base output directory"),
    skip_existing: bool = typer.Option(False, help="Skip existing files"),
    num_workers: int = typer.Option(128, help="Number of workers"),
    rerun: bool = typer.Option(False, help="Rerun everything from scratch"),
    create_backup: bool = typer.Option(True, help="Create backup before cleaning")
):
    """Run the complete pipeline from a configuration file."""
    
    # Load configuration
    import yaml
    
    with open(config_file, 'r') as f:
        if config_file.suffix.lower() in ['.yaml', '.yml']:
            config_data = yaml.safe_load(f)
        else:
            config_data = json.load(f)
    
    # Parse configuration
    experiments = []
    for exp_data in config_data['experiments']:
        experiments.append(ExperimentConfig(
            name=exp_data['name'],
            task=Task(exp_data['task']),
            method=Method(exp_data['method']),
            model_name=exp_data['model_name'],
            temperature=exp_data.get('temperature', 0.7),
            top_p=exp_data.get('top_p', 0.9),
            num_responses=exp_data.get('num_responses', 10),
            num_samples=exp_data.get('num_samples', 1),
            sample_size=exp_data.get('sample_size', 5),
            random_seed=exp_data.get('random_seed', 42),
            use_vllm=exp_data.get('use_vllm', False)
        ))
    
    evaluation_config = EvaluationConfig(
        metrics=config_data['evaluation']['metrics'],
        num_workers=config_data['evaluation'].get('num_workers', 128)
    )
    
    # Handle rerun override
    if rerun:
        skip_existing = False
        console.print("[bold yellow]🔄 Rerun mode enabled - will overwrite existing results[/bold yellow]")
    
    pipeline_config = PipelineConfig(
        experiments=experiments,
        evaluation=evaluation_config,
        output_base_dir=output_dir,
        num_workers=num_workers,
        skip_existing=skip_existing,
        rerun=rerun,
        create_backup=create_backup
    )
    
    # Run pipeline
    pipeline = Pipeline(pipeline_config)
    results = pipeline.run_complete_pipeline()
    
    return results

# Convenience function for programmatic use
def run_quick_comparison(
    task: Task,
    methods: List[Method],
    model_name: str,
    metrics: List[str],
    output_dir: Path,
    num_responses: int = 10,
    rerun: bool = False,
    create_backup: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """Quick comparison between different methods for a single task."""
    
    experiments = []
    for method in methods:
        experiments.append(ExperimentConfig(
            name=f"{task.value}_{method.value}",
            task=task,
            method=method,
            model_name=model_name,
            num_responses=num_responses,
            **kwargs
        ))
    
    evaluation_config = EvaluationConfig(metrics=metrics)
    
    pipeline_config = PipelineConfig(
        experiments=experiments,
        evaluation=evaluation_config,
        output_base_dir=output_dir,
        skip_existing=not rerun,
        rerun=rerun,
        create_backup=create_backup
    )
    
    pipeline = Pipeline(pipeline_config)
    return pipeline.run_complete_pipeline() 