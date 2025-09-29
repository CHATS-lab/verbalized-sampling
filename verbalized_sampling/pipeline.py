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
from verbalized_sampling.methods import Method
from verbalized_sampling.llms import get_model
from verbalized_sampling.analysis.evals import get_evaluator
from verbalized_sampling.methods.factory import PromptFactory

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
    min_p: float = 0.0
    num_responses: int = 10
    num_samples: int = 1
    num_prompts: int = 5
    num_samples_per_prompt: int = 2 # Number of samples per prompt for COMBINED method
    target_words: int = 200 # Minimum number of words in each response
    random_seed: int = 42
    use_vllm: bool = False
    all_possible: bool = False # If True, the request would enable all possible responses
    strict_json: bool = False # If True, the request would enable JSON mode
    probability_definition: str = "implicit" # Type of probability definition to use
    probability_tuning: float = -1 # Probability tuning for the probability definition
    custom_prompts: Optional[List[str]] = None # Optional override prompts

@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    metrics: List[str]
    num_workers: int = 128
    num_responses_per_prompt: int = 50

@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    experiments: List[ExperimentConfig]
    evaluation: EvaluationConfig
    output_base_dir: Path
    num_workers: int = 128
    skip_existing: bool = True
    rerun: bool = False
    create_backup: bool = False
    title: Optional[str] = None
    
    def _should_backup(self) -> bool:
        """Determine if backup should be created."""
        return self.create_backup
    
    @staticmethod
    def get_available_probability_definitions() -> List[str]:
        """Get list of available probability definition types."""
        return ["implicit", "explicit", "relative", "percentage", "confidence", "perplexity", "nll"]

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
        
        # Validate probability definitions for all experiments
        valid_probability_definitions = ["implicit", "explicit", "relative", "percentage", "confidence", "perplexity", "nll"]
        for exp in self.config.experiments:
            if exp.probability_definition not in valid_probability_definitions:
                raise ValueError(f"Invalid probability_definition '{exp.probability_definition}' for experiment '{exp.name}'. "
                               f"Valid options are: {valid_probability_definitions}")

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
        plot_results = self.create_plots(evaluation_results, title=self.config.title)
        
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
                model_config = {
                    "temperature": exp_config.temperature,
                    "top_p": exp_config.top_p,
                }

                # Only add min_p if use_vllm is True
                if exp_config.use_vllm:
                    model_config["min_p"] = exp_config.min_p

                model = get_model(
                    model_name=exp_config.model_name,
                    method=exp_config.method,
                    config=model_config,
                    use_vllm=exp_config.use_vllm,
                    num_workers=self.config.num_workers,
                    strict_json=exp_config.strict_json,
                )
                
                task_kwargs = {}
                # if exp_config.task in [Task.POEM, Task.SPEECH, Task.STATE_NAME, Task.SIMPLE_QA, Task.BOOK]:
                task_kwargs.update({
                    "num_prompts": exp_config.num_prompts,
                    "random_seed": exp_config.random_seed,
                    "all_possible": exp_config.all_possible,
                    "num_samples_per_prompt": exp_config.num_samples_per_prompt if exp_config.method == Method.VS_MULTI else None,
                })
                
                # probability_definition is handled by the prompt system, not the task
                # so we don't pass it to the task instance

                # The num_samples must be smaller than num_responses
                if exp_config.num_samples > exp_config.num_responses:
                    raise ValueError(f"Error: num_samples must be smaller than num_responses for {exp_config.name}")
                
                
                num_samples = exp_config.num_samples if exp_config.method != Method.DIRECT else 1
                num_responses = exp_config.num_responses // num_samples
                task_instance = get_task(
                    exp_config.task,
                    model=model,
                    method=exp_config.method,
                    num_responses=num_responses,
                    num_samples=num_samples,
                    target_words=exp_config.target_words,
                    probability_definition=exp_config.probability_definition,  # Pass to task for prompt generation
                    probability_tuning=exp_config.probability_tuning,
                    custom_prompts=exp_config.custom_prompts,
                    **task_kwargs
                )

                # Run generation
                gen_task = progress.add_task(
                    f"[cyan]{exp_config.name}[/cyan]", 
                    total=exp_config.num_responses
                )
                results = task_instance.run(progress=progress, task_id=gen_task)
                task_instance.save_results(results, output_file)

                # print("Generation results: ", results)
                
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
                        # Each line is expected to be a JSON object
                        try:
                            data = json.loads(line)
                        except Exception as e:
                            print(line)
                            raise e
                        
                        prompt = data["prompt"]
                        responses_list = data["responses"]
                        for i, response in enumerate(responses_list):
                            if isinstance(response, str):
                                response = {"text": response}
                            response['index'] = i
                            responses.append(response)
                            prompts.append(prompt)

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
                        if metric in ("response_count", "synthetic_data_quality", "diversity"):
                            num_prompts = len(set(prompts))
                            num_responses_per_prompt = self.config.evaluation.num_responses_per_prompt
                            print(f"Num prompts: {num_prompts}, Num responses per prompt: {num_responses_per_prompt}")
                            # Get evaluator and run evaluation
                            evaluator = get_evaluator(
                                metric, 
                                num_workers=self.config.evaluation.num_workers,
                                num_responses_per_prompt=num_responses_per_prompt
                            )
                        else:
                            evaluator = get_evaluator(
                                metric, 
                                num_workers=self.config.evaluation.num_workers,
                            )

                        

                        # For accuracy evaluation, we need to provide reference answers
                        evaluation_kwargs = {"metadata": {"experiment": exp_name, "metric": metric}}

                        if metric == "accuracy":
                            # Find the experiment config for this experiment
                            exp_config = None
                            for config in self.config.experiments:
                                if config.name == exp_name:
                                    exp_config = config
                                    break

                            if exp_config and hasattr(exp_config, 'task'):
                                # Load the task to get reference answers
                                from verbalized_sampling.tasks import get_task
                                task = get_task(
                                    exp_config.task.value,
                                    model=None,  # We don't need model for getting answers
                                    method="direct",  # Method doesn't matter for getting answers
                                    num_prompts=len(set(prompts)),  # Number of unique prompts
                                    num_responses=1,
                                    random_seed=exp_config.random_seed
                                )

                                # Extract reference answers corresponding to the prompts
                                reference_answers = []
                                unique_prompts = []
                                seen_prompts = set()

                                # Get unique prompts in order
                                for prompt in prompts:
                                    if prompt not in seen_prompts:
                                        unique_prompts.append(prompt)
                                        seen_prompts.add(prompt)

                                # Match prompts to task problems and extract answers
                                for prompt in unique_prompts:
                                    # Find the corresponding problem in the task
                                    matching_answer = None
                                    for problem in task.problems:
                                        # Extract the question from the formatted prompt
                                        if "Question:" in prompt:
                                            question_part = prompt.split("Question:")[1].split("Please reason")[0].strip()
                                        else:
                                            question_part = prompt.strip()

                                        if problem['problem'].strip() in question_part or question_part in problem['problem'].strip():
                                            matching_answer = problem['answer']
                                            break

                                    if matching_answer is None:
                                        matching_answer = "UNKNOWN"
                                    reference_answers.append(matching_answer)

                                # Expand reference answers to match all responses (including multiple responses per prompt)
                                expanded_answers = []
                                prompt_to_answer = dict(zip(unique_prompts, reference_answers))
                                for prompt in prompts:
                                    expanded_answers.append(prompt_to_answer.get(prompt, "UNKNOWN"))

                                evaluation_kwargs["reference_answers"] = expanded_answers

                        # print("Evaluation Prompts: ", prompts)
                        # print("Evaluation Responses: ", responses)

                        result = evaluator.evaluate(
                            prompts,
                            responses,
                            **evaluation_kwargs
                        )
                        
                        # Save results
                        evaluator.save_results(result, eval_file)
                        evaluation_results[exp_name][metric] = eval_file
                        
                        console.print(f"✅ {exp_name}/{metric}: Evaluation complete")
                        
                    except Exception as e:
                        console.print(f"❌ {exp_name}/{metric}: Error - {str(e)}")
                        raise e
                        evaluation_results[exp_name][metric] = None
                    
                    progress.advance(overall_task)
        
        return evaluation_results
    
    def create_plots(self, evaluation_results: Dict[str, Dict[str, Path]], title: Optional[str] = None) -> Dict[str, Path]:
        """Create comparison plots for each metric.
        
        Args:
            evaluation_results: Dict[str, Dict[str, Path]]
                The evaluation results to plot.
                Each key is the metric name, and the value is a dictionary with the following keys:
                - "exp_name": The name of the experiment.
                - "result_file": The path to the result file.

        Returns:
            Dict[str, Path]
        """
        from verbalized_sampling.analysis.evals import plot_evaluation_comparison, plot_comparison_chart
        
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
        
        # Create plots for each metric between methods
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
        
        plot_dir = plots_base_dir / "comparison_chart"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_comparison_chart(metric_results, plot_dir, title=title)
        plot_results["comparison_chart"] = plot_dir

        return plot_results
    
    def generate_report(self, evaluation_results: Dict[str, Dict[str, Path]], 
                       plot_results: Dict[str, Path]) -> Path:
        """Generate a comprehensive HTML report."""
        from verbalized_sampling.analysis.evals.base import EvalResult
        
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
        
        # Load sample generations for each experiment
        sample_generations = self._load_sample_generations()
        
        # Generate HTML report
        html_content = self._generate_html_report(loaded_results, plot_results, sample_generations)
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return report_path
    
    def _load_sample_generations(self, num_samples: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """Load sample generations for each experiment to include in the report."""
        sample_generations = {}
        
        for exp_config in self.config.experiments:
            exp_name = exp_config.name
            responses_file = self.config.output_base_dir / "generation" / exp_name / "responses.jsonl"
            
            if not responses_file.exists():
                console.print(f"Warning: No responses file found for {exp_name}")
                sample_generations[exp_name] = []
                continue
            
            try:
                samples = []
                with open(responses_file, 'r') as f:
                    lines = list(f)
                    
                    # Take samples from different prompts if possible
                    sample_lines = lines[:num_samples] if len(lines) >= num_samples else lines
                    
                    for line in sample_lines:
                        data = json.loads(line)
                        prompt = data["prompt"]
                        responses_list = data["responses"]
                        
                        # Take the first response for each sampled prompt
                        if responses_list:
                            sample_response = responses_list[0]
                            samples.append({
                                "prompt": prompt,
                                "response": sample_response,
                                "method": exp_config.method.value,
                                "task": exp_config.task.value
                            })
                
                sample_generations[exp_name] = samples
                
            except Exception as e:
                console.print(f"Warning: Could not load samples for {exp_name}: {e}")
                sample_generations[exp_name] = []
        
        return sample_generations
    
    def _generate_html_report(self, results: Dict[str, Dict[str, Any]], 
                            plot_results: Dict[str, Path],
                            sample_generations: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate HTML report content with embedded plots and sample generations."""
        import base64
        
        # Define metrics that should be excluded from the table due to their length
        EXCLUDED_METRICS = {
            'pairwise_diversities', 'pairwise_similarities', 'pairwise_rouge_l_scores',
            'detailed_results', 'raw_responses', 'embeddings', 'similarity_matrix'
        }
        
        # Define metrics that should show error statistics
        ERROR_STAT_METRICS = {
            'fluency', 'flexibility', 'originality', 'elaboration', 'overall', 'normalized_overall',
            'f1', 'accuracy_given_attempted', 'pass_at_k_accuracy', 'top_at_k_accuracy', 
            'first_response_accuracy', 'average_similarity', 'average_creativity_index',
            'mean_token_length', 'average_ngram_diversity'
        }
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pipeline Results Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
                .experiment {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ margin: 10px 0; }}
                .plots {{ margin: 30px 0; }}
                .plot-container {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; background: #fafafa; }}
                .plot-image {{ max-width: 100%; height: auto; display: block; margin: 20px auto; border: 1px solid #ccc; border-radius: 4px; }}
                .generation-examples {{ margin: 30px 0; }}
                .example-container {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; background: #f9f9f9; }}
                .example-prompt {{ background: #e3f2fd; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #2196f3; }}
                .example-response {{ background: #f3e5f5; padding: 15px; border-radius: 5px; border-left: 4px solid #9c27b0; }}
                .example-meta {{ font-size: 0.9em; color: #666; margin-bottom: 10px; }}
                .prompt-label {{ font-weight: bold; color: #1976d2; margin-bottom: 8px; }}
                .response-label {{ font-weight: bold; color: #7b1fa2; margin-bottom: 8px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .number {{ text-align: right; }}
                .excluded-note {{ font-style: italic; color: #666; margin-top: 10px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #444; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
                h3 {{ color: #555; }}
                h4 {{ color: #666; }}
                .metric-section {{ margin-bottom: 40px; }}
                .generation-section {{ margin-bottom: 40px; }}
                .method-tag {{ display: inline-block; background: #e0e0e0; padding: 3px 8px; border-radius: 3px; font-size: 0.8em; margin-right: 10px; }}
                .task-tag {{ display: inline-block; background: #c8e6c9; padding: 3px 8px; border-radius: 3px; font-size: 0.8em; }}
                pre {{ white-space: pre-wrap; word-wrap: break-word; }}
                .error-stats {{ background: #fff3cd; padding: 10px; border-radius: 5px; border-left: 4px solid #ffc107; margin: 10px 0; }}
                .error-stats h5 {{ margin: 0 0 10px 0; color: #856404; }}
                .error-stats table {{ margin: 0; font-size: 0.9em; }}
                .error-stats th, .error-stats td {{ padding: 4px 8px; border: 1px solid #ffeaa7; }}
                .error-stats th {{ background-color: #fff8dc; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Pipeline Results Report</h1>
                <p><strong>Generated:</strong> {Path().absolute()}</p>
                <p><strong>Experiments:</strong> {len(self.config.experiments)}</p>
                <p><strong>Metrics:</strong> {', '.join(self.config.evaluation.metrics)}</p>
                <p><strong>Available Probability Definitions:</strong> {', '.join(PromptFactory.get_available_probability_definitions().keys())}</p>
            </div>
        """
        
        # Experiment configurations
        html += "<h2>📋 Experiment Configurations</h2>"
        html += "<table><tr><th>Name</th><th>Task</th><th>Method</th><th>Model</th><th>Responses</th><th>Temperature</th><th>Probability Definition</th></tr>"
        for exp in self.config.experiments:
            html += f"<tr><td>{exp.name}</td><td>{exp.task.value}</td><td>{exp.method.value}</td><td>{exp.model_name}</td><td class='number'>{exp.num_responses}</td><td class='number'>{exp.temperature}</td><td>{exp.probability_definition}</td></tr>"
        html += "</table>"
        
        # Generation Examples Section
        html += "<h2>📝 Generation Examples</h2>"
        html += "<div class='generation-examples'>"
        
        for exp_name, samples in sample_generations.items():
            if samples:
                html += f"<div class='generation-section'>"
                html += f"<h3>{exp_name}</h3>"
                
                for i, sample in enumerate(samples, 1):
                    html += f"<div class='example-container'>"
                    html += f"<div class='example-meta'>"
                    html += f"<span class='method-tag'>Method: {sample['method']}</span>"
                    html += f"<span class='task-tag'>Task: {sample['task']}</span>"
                    html += f"</div>"
                    
                    html += f"<div class='example-prompt'>"
                    html += f"<div class='prompt-label'>Prompt:</div>"
                    html += f"<pre>{self._escape_html(sample['prompt'])}</pre>"
                    html += f"</div>"
                    
                    html += f"<div class='example-response'>"
                    html += f"<div class='response-label'>Response:</div>"
                    html += f"<pre>{self._escape_html(self._format_response(sample['response']))}</pre>"
                    html += f"</div>"
                    html += f"</div>"
                
                html += f"</div>"
            else:
                html += f"<div class='generation-section'>"
                html += f"<h3>{exp_name}</h3>"
                html += f"<p><em>No generation examples available</em></p>"
                html += f"</div>"
        
        html += "</div>"
        
        # Results Section with Error Statistics
        html += "<h2>📊 Evaluation Results</h2>"
        
        for exp_name, exp_metrics in results.items():
            html += f"<div class='experiment'>"
            html += f"<h3>{exp_name}</h3>"
            
            for metric_name, result in exp_metrics.items():
                if metric_name in EXCLUDED_METRICS:
                    continue
                    
                html += f"<div class='metric'>"
                html += f"<h4>{metric_name.replace('_', ' ').title()}</h4>"
                
                # Create results table
                html += "<table>"
                html += "<tr><th>Metric</th><th>Value</th></tr>"
                
                for key, value in result.overall_metrics.items():
                    if key in EXCLUDED_METRICS or key.endswith('_stats'):
                        continue
                    
                    if isinstance(value, (int, float)):
                        html += f"<tr><td>{key.replace('_', ' ').title()}</td><td class='number'>{value:.4f}</td></tr>"
                    else:
                        html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{str(value)}</td></tr>"
                
                html += "</table>"
                
                # Add error statistics for relevant metrics
                for key, value in result.overall_metrics.items():
                    if key.endswith('_stats') and key[:-6] in ERROR_STAT_METRICS:
                        metric_base = key[:-6]  # Remove '_stats' suffix
                        html += f"<div class='error-stats'>"
                        html += f"<h5>📈 Error Statistics for {metric_base.replace('_', ' ').title()}</h5>"
                        html += "<table>"
                        html += "<tr><th>Statistic</th><th>Value</th></tr>"
                        
                        if isinstance(value, dict):
                            for stat_name, stat_value in value.items():
                                if isinstance(stat_value, (int, float)):
                                    html += f"<tr><td>{stat_name.replace('_', ' ').title()}</td><td class='number'>{stat_value:.4f}</td></tr>"
                                else:
                                    html += f"<tr><td>{stat_name.replace('_', ' ').title()}</td><td>{str(stat_value)}</td></tr>"
                        
                        html += "</table>"
                        html += "</div>"
                
                html += "</div>"
            
            html += "</div>"
        
        # Overall comparison chart
        comparison_chart_dir = plot_results.get("comparison_chart")
        if comparison_chart_dir:
            html += "<h2>📈 Overall Comparison</h2>"
            html += self._embed_plots_for_metric("comparison_chart", comparison_chart_dir)
        
        html += "</body></html>"
        return html
    
    def _format_response(self, response: Any) -> str:
        """Format a response for display in HTML."""
        if isinstance(response, dict):
            # If it's a structured response, try to extract the main content
            if 'response' in response:
                return str(response['response'])
            elif 'text' in response:
                return str(response['text'])
            elif 'content' in response:
                return str(response['content'])
            else:
                # Format as JSON for structured responses
                return json.dumps(response, indent=2, ensure_ascii=False)
        else:
            return str(response)
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        import html
        return html.escape(text)
    
    def _embed_plots_for_metric(self, metric_name: str, plot_dir: Path) -> str:
        """Embed all plots for a given metric into HTML."""
        import base64
        
        html = f"<div class='plot-container'>"
        html += f"<h4>📈 {metric_name.replace('_', ' ').title()} Visualizations</h4>"
        
        # Find all PNG files in the plot directory
        if plot_dir.exists():
            png_files = list(plot_dir.glob("*.png"))
            
            if png_files:
                for png_file in sorted(png_files):
                    try:
                        # Read and encode the image
                        with open(png_file, "rb") as img_file:
                            img_data = img_file.read()
                            img_base64 = base64.b64encode(img_data).decode('utf-8')
                        
                        # Create a nice title from filename
                        plot_title = png_file.stem.replace('_', ' ').title()
                        
                        html += f"<div style='margin: 20px 0;'>"
                        html += f"<h5 style='text-align: center; color: #666; margin-bottom: 10px;'>{plot_title}</h5>"
                        html += f"<img src='data:image/png;base64,{img_base64}' class='plot-image' alt='{plot_title}' />"
                        html += f"</div>"
                        
                    except Exception as e:
                        console.print(f"Warning: Could not embed plot {png_file}: {e}")
                        # Fallback to link
                        html += f"<p><a href='{png_file.relative_to(self.config.output_base_dir)}'>View {png_file.name}</a></p>"
            else:
                html += f"<p><em>No plots found in {plot_dir}</em></p>"
        else:
            html += f"<p><em>Plot directory not found: {plot_dir}</em></p>"
        
        html += "</div>"
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
        exp_config = ExperimentConfig(
            name=exp_data['name'],
            task=Task(exp_data['task']),
            method=Method(exp_data['method']),
            model_name=exp_data['model_name'],
            temperature=exp_data.get('temperature', 0.7),
            top_p=exp_data.get('top_p', 0.9),
            num_responses=exp_data.get('num_responses', 10),
            num_samples=exp_data.get('num_samples', 1),
            num_prompts=exp_data.get('num_prompts', 5),
            random_seed=exp_data.get('random_seed', 42),
            use_vllm=exp_data.get('use_vllm', False),
            probability_definition=exp_data.get('probability_definition', "implicit")
        )

        # Only add min_p if use_vllm is True and min_p is provided
        if exp_config.use_vllm and 'min_p' in exp_data:
            exp_config.min_p = exp_data['min_p']

        experiments.append(exp_config)
    
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
    num_samples: int = 1,
    num_prompts: int = 5,
    num_samples_per_prompt: int = 2,
    prompt: Optional[str] = None,
    rerun: bool = False,
    create_backup: bool = False,
    probability_definition: str = "implicit",
    **kwargs
) -> Dict[str, Any]:
    """Quick comparison between different methods for a single task."""
    print("Running quick comparison with the following parameters:")
    print(f"Task: {task}")
    print(f"Methods: {methods}")
    print(f"Model: {model_name}")
    print(f"Metrics: {metrics}")
    print(f"Output dir: {output_dir}")
    print(f"Num responses: {num_responses}")
    print(f"Num samples: {num_samples}")
    print(f"Num prompts: {num_prompts}")
    print(f"Num samples per prompt: {num_samples_per_prompt}")
    print(f"Probability definition: {probability_definition}")
    print(f"Rerun: {rerun}")
    print(f"Create backup: {create_backup}")
    print(f"**kwargs: {kwargs}")
    
    experiments = []
    for method in methods:
        exp_kwargs = kwargs.copy()

        # Handle min_p parameter - only include if use_vllm is True
        use_vllm = exp_kwargs.get('use_vllm', False)
        if not use_vllm and 'min_p' in exp_kwargs:
            exp_kwargs.pop('min_p')

        experiments.append(ExperimentConfig(
            name=f"{task.value}_{method.value}",
            task=task,
            method=method,
            model_name=model_name,
            num_responses=num_responses,
            num_samples=num_samples,
            num_prompts=(1 if prompt else num_prompts),
            num_samples_per_prompt=num_samples_per_prompt,
            probability_definition=probability_definition,
            custom_prompts=([prompt] if prompt else None),
            **exp_kwargs
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
