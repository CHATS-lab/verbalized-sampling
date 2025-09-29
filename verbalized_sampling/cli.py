"""
Command-line interface for verbalized-sampling.
"""

import typer
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from typing import List, Optional

from verbalized_sampling.pipeline import run_quick_comparison
from verbalized_sampling.tasks import Task
from verbalized_sampling.methods import Method
from verbalized_sampling.tasks.dialogue.persuasion import PersuasionTask
from verbalized_sampling.llms import get_model
from verbalized_sampling.evals.dialogue import DonationEvaluator, DialogueLinguisticEvaluator
from datetime import datetime

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

    # Dialogue simulation (Section 6)
    Task.PERSUASION_DIALOGUE: "PersuasionForGood dialogue simulation - multi-turn persuasive conversations about charity donation",

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
    prompt: Optional[str] = typer.Option(None, help="Override dataset with a single user prompt"),
    model: str = typer.Option(..., help="Model to use (e.g., openai/gpt-4.1)"),
    methods: str = typer.Option(
        "DIRECT",
        "--methods",
        help="Methods to compare, space-separated (e.g., 'DIRECT VS_STANDARD')",
        show_default=True,
    ),
    num_responses: int = typer.Option(50, help="Number of responses to generate"),
    num_samples: int = typer.Option(5, help="Number of samples per prompt"),
    num_prompts: int = typer.Option(1, help="Number of prompts to use"),
    strict_json: bool = typer.Option(False, help="Use strict JSON mode"),
    metrics: str = typer.Option(
        "diversity length ngram",
        "--metrics",
        help="Metrics to evaluate, space-separated (e.g., 'diversity length ngram')",
        show_default=True,
    ),
    output_dir: Path = typer.Option(Path("results"), help="Output directory"),
    temperature: float = typer.Option(0.7, help="Sampling temperature"),
    top_p: float = typer.Option(1.0, help="Top-p sampling parameter"),
    rerun: bool = typer.Option(False, help="Rerun existing results"),
):
    """Run a quick comparison experiment."""
    # Parse space-separated strings into lists
    method_list = [Method(m.strip()) for m in methods.split()]
    metric_list = [m.strip() for m in metrics.split()]
    
    console.print(f"🔬 Running {task.value} experiment with {model}")
    
    results = run_quick_comparison(
        task=task,
        methods=method_list,
        model_name=model,
        metrics=metric_list,
        output_dir=output_dir,
        num_responses=num_responses,
        num_samples=num_samples,
        num_prompts=(1 if prompt else num_prompts),
        prompt=prompt,
        strict_json=strict_json,
        rerun=rerun,
        temperature=temperature,
        top_p=top_p,
    )
    
    # Display summary and point to the report
    report_path = results.get("report_path", output_dir / "pipeline_report.html")
    console.print(f"✅ Done! Check {report_path}")

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

@app.command()
def dialogue(
    persuader_model: str = typer.Option("openai/gpt-4.1-mini", help="Model name for persuader role"),
    persuadee_model: str = typer.Option("openai/gpt-4.1-mini", help="Model name for persuadee role"),
    method: Method = typer.Option(Method.VS_STANDARD, help="Sampling method for persuadee"),
    max_turns: int = typer.Option(10, help="Maximum conversation turns"),
    word_limit: int = typer.Option(160, help="Word limit per turn"),
    num_samplings: int = typer.Option(4, help="Number of samples for VS methods"),
    num_conversations: int = typer.Option(5, help="Number of conversations to simulate"),
    dataset_path: Optional[Path] = typer.Option(None, help="Path to PersuasionForGood dataset"),
    corpus_path: Optional[Path] = typer.Option(None, help="Path to persona corpus"),
    temperature: float = typer.Option(0.7, help="Sampling temperature"),
    top_p: float = typer.Option(0.9, help="Top-p sampling"),
    max_tokens: int = typer.Option(500, help="Max tokens per response"),
    response_selection: str = typer.Option("probability", help="Response selection strategy",
                                           rich_help_panel="Selection",
                                           case_sensitive=False),
    evaluate: bool = typer.Option(False, help="Compute donation and linguistic metrics"),
    ground_truth_path: Optional[Path] = typer.Option(None, help="Ground truth donation JSON for evaluation"),
    output_file: Optional[Path] = typer.Option(None, help="Output JSONL path for results"),
):
    """Run PersuasionForGood dialogue simulations with VS methods."""
    console.print("🎭 Running persuasion dialogue simulation")

    # Model configs
    config = {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}

    # Build models
    persuader = get_model(
        model_name=persuader_model,
        method=Method.DIRECT,
        config=config,
        num_workers=1,
        strict_json=False,
    )
    persuadee = get_model(
        model_name=persuadee_model,
        method=method,
        config=config,
        num_workers=1,
        strict_json=(method in [Method.VS_STANDARD, Method.VS_COT, Method.SEQUENCE]),
    )

    # Instantiate task
    task = PersuasionTask(
        persuader_model=persuader,
        persuadee_model=persuadee,
        method=method,
        max_turns=max_turns,
        word_limit=word_limit,
        num_samplings=num_samplings,
        sampling_method=method.value,
        dataset_path=str(dataset_path) if dataset_path else None,
        corpus_path=str(corpus_path) if corpus_path else None,
    )

    # Selection mapping
    if response_selection.lower().startswith("prob"):
        task.response_selection = "probability_weighted"
    else:
        task.response_selection = "random"

    # Run experiment
    conversations = task.run_experiment(num_conversations=num_conversations)

    # Save results
    if output_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("results") / "dialogue"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_file = out_dir / f"persuasion_{method.value}_{ts}.jsonl"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for conv in conversations:
            # Serialize ConversationState
            result = {
                "conversation_id": conv.conversation_id,
                "persuader_persona": conv.persuader_persona,
                "persuadee_persona": conv.persuadee_persona,
                "turns": [
                    {
                        "role": t.role.value,
                        "text": t.text,
                        "turn_number": t.turn_number,
                        "metadata": t.metadata,
                    }
                    for t in conv.turns
                ],
                "outcome": conv.outcome,
                "method": method.value,
                "persuader_model": persuader_model,
                "persuadee_model": persuadee_model,
                "config": config,
            }
            f.write(json.dumps(result) + "\n")
    console.print(f"✅ Results saved to {output_file}")

    # Optional evaluation
    if evaluate:
        donation_eval = DonationEvaluator(ground_truth_path=str(ground_truth_path) if ground_truth_path else None)
        ling_eval = DialogueLinguisticEvaluator()

        donation_metrics = [donation_eval.compute_instance_metric("", r) for r in _iter_results_jsonl(output_file)]
        linguistic_metrics = [ling_eval.compute_instance_metric("", r) for r in _iter_results_jsonl(output_file)]

        agg_donation = donation_eval.aggregate_metrics(donation_metrics)
        agg_linguistic = ling_eval.aggregate_metrics(linguistic_metrics)

        eval_file = output_file.parent / f"{output_file.stem}_evaluation.json"
        with open(eval_file, "w") as f:
            json.dump({"donation_metrics": agg_donation, "linguistic_metrics": agg_linguistic}, f, indent=2)
        console.print(f"📊 Evaluation saved to {eval_file}")

    # Summary
    total_turns = sum(len(c.turns) for c in conversations)
    donations = sum(1 for c in conversations if c.outcome and c.outcome.get("final_donation_amount", 0) > 0)
    console.print(f"📈 Conversations: {len(conversations)}, Avg turns: {total_turns/len(conversations):.1f}")
    console.print(f"💰 Donation rate: {donations/len(conversations):.1%}")


def _iter_results_jsonl(path: Path):
    with open(path, "r") as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception:
                continue

if __name__ == "__main__":
    app()
