# Reproducing Paper Results

This guide provides detailed instructions for reproducing all experiments from our paper: **"Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity"**.

Each experiment is mapped to its corresponding paper section with LaTeX labels for easy cross-referencing.

## Table of Contents

- [Main Experiments (§5-8)](#main-experiments-5-8)
    - [§5: Creative Writing](#5-creative-writing)
      - [5.1 Poem Continuation](#51-poem-continuation)
      - [5.2 Story Generation](#52-story-generation)
      - [5.3 Joke Writing](#53-joke-writing)
    - [§6: Dialogue Simulation](#6-dialogue-simulation)
      - [PersuasionForGood Task](#persuasionforgood-task)
    - [§7: Open-Ended QA](#7-open-ended-qa)
      - [CoverageQA and Bias Mitigation](#coverageqa-and-bias-mitigation)
    - [§8: Synthetic Data Generation](#8-synthetic-data-generation)
      - [8.1 Positive Synthetic Data Generation](#81-positive-synthetic-data-generation)
        - [GSM8K Dataset](#gsm8k-dataset)
        - [AMC/AIME Competition Math](#amcaime-competition-math)
        - [LiveCodeBench](#livecodebench)
      - [8.2 Negative Synthetic Data Generation](#82-negative-synthetic-data-generation)
  - [Appendix Experiments](#appendix-experiments)
    - [Random Number Generation](#random-number-generation)
    - [Geographic Bias (State Naming)](#geographic-bias-state-naming)
    - [Commonsense Reasoning](#commonsense-reasoning)
    - [Safety Evaluation](#safety-evaluation)
  - [Experimental Settings](#experimental-settings)
    - [Default Hyperparameters](#default-hyperparameters)
    - [Generation Settings](#generation-settings)
  - [Full Prompts](#full-prompts)
    - [VS-Standard Prompt Template](#vs-standard-prompt-template)
    - [VS-CoT Prompt Template](#vs-cot-prompt-template)
    - [VS-Multi Prompt Template](#vs-multi-prompt-template)
    - [Probability Tuning](#probability-tuning)
  - [Running All Experiments](#running-all-experiments)
  - [Citation](#citation)
  - [Need Help?](#need-help)

---

## Main Experiments (§5-8)

### §5: Creative Writing

**Paper Section**: Section 5 (Creative Writing)
**Key Results**: Figure 2 (a-i), Table 2
**Appendix Details**: Appendix §A.1 (Creative Writing)

We evaluate Verbalized Sampling on three creative writing tasks to measure diversity and quality improvements.

#### 5.1 Poem Continuation

**Script**: [`tasks/run_poem.py`](tasks/run_poem.py)
**Dataset**: PoemHunter.com (100 poems sampled)
**Metrics**: Semantic diversity, lexical diversity (ROUGE-L), quality scores

```bash
# Run with VS-Standard method
python scripts/tasks/run_poem.py --method vs_standard --model "gpt-4.1"

# Compare multiple methods
python scripts/tasks/run_poem.py \
    --methods "direct vs_standard vs_cot vs_multi" \
    --model "gpt-4.1"
```

**Expected Output**:
- Results: `results/poem/`
- Metrics: Diversity scores, quality evaluations
- Paper Results: VS achieves 30.4 diversity vs. 10.8 for direct prompting (Figure 2a)

#### 5.2 Story Generation

**Script**: [`tasks/run_story.py`](tasks/run_story.py)
**Dataset**: BookMIA dataset (100 stories sampled)
**Metrics**: Semantic diversity, lexical diversity, creativity scores

```bash
# Run story generation experiments
python scripts/tasks/run_story.py --method vs_standard --model "gpt-4.1"
```

**Expected Output**:
- Results: `results/story/`
- Paper Results: 2-3x diversity improvement (Figure 2b)

#### 5.3 Joke Writing

**Script**: [`tasks/run_jokes.py`](tasks/run_jokes.py)
**Dataset**: Reddit r/DadJokes (100 thematic prompts)
**Metrics**: Semantic diversity, humor quality

```bash
# Run joke writing experiments
python scripts/tasks/run_jokes.py --method vs_standard --model "gpt-4.1"
```

**Expected Output**:
- Results: `results/jokes/`
- Paper Results: Significant diversity gains while maintaining humor quality (Figure 2c)

**Human Evaluation**: See Appendix §A.1.2 (Human Study on Creative Writing) for human study details showing 25.7% improvement in human evaluation scores.

---

### §6: Dialogue Simulation

**Paper Section**: Section 6 (Dialogue Simulation)
**Key Results**: Figure 3 (a-b), Table 3
**Appendix Details**: Appendix §A.2 (Dialogue Simulation)

#### PersuasionForGood Task

**Script**: [`tasks/run_dialogue_simulation.py`](tasks/run_dialogue_simulation.py)
**Dataset**: PersuasionForGood (939 dialogues, 739 train / 200 test)
**Task**: Simulate multi-turn persuasive dialogues where one agent persuades another to donate to charity
**Metrics**: Donation distribution alignment (KL divergence), linguistic alignment (Distinct-N, semantic diversity)

```bash
# Run dialogue simulation with VS
python scripts/tasks/run_dialogue_simulation.py \
    --persuader-model "gpt-4.1" \
    --persuadee-model "gpt-4.1" \
    --method vs_standard \
    --num-conversations 50 \
    --num-samplings 5 \
    --max-turns 10 \
    --evaluate \
    --output-file results/dialogue/persuasion_vs_standard.jsonl

# Compare with baseline methods
python scripts/tasks/run_dialogue_simulation.py \
    --persuader-model "gpt-4.1" \
    --persuadee-model "gpt-4.1" \
    --method direct \
    --num-conversations 50 \
    --output-file results/dialogue/persuasion_direct.jsonl
```

**Expected Output**:
- Results: `results/dialogue/`
- Dialogue transcripts with donation amounts
- Evaluation metrics: KL divergence from human distribution, linguistic features
- Paper Results: VS achieves donation distributions closer to human behavior (Figure 3a), with improved linguistic alignment (Figure 3b)

**Note**: The persuadee is simulated with different methods, while the persuader uses GPT-4.1 with direct prompting. See Appendix §A.2 for fine-tuned baseline details.

---

### §7: Open-Ended QA

**Paper Section**: Section 7 (Open-Ended QA)
**Key Results**: Figure 4, Table 4
**Appendix Details**: Appendix §A.3 (Open-Ended QA)

#### CoverageQA and Bias Mitigation

**Scripts**:
- [`tasks/run_simple_qa.py`](tasks/run_simple_qa.py) - Main CoverageQA experiments
- [`tasks/run_bias_task_coverageqa.py`](tasks/run_bias_task_coverageqa.py) - CoverageQA specific
- [`tasks/run_bias_task_general.py`](tasks/run_bias_task_general.py) - General bias tasks

**Dataset**: CoverageQA (40 questions: 10 original + 30 newly created)
**Task**: Enumerative open-ended questions with multiple valid answers (e.g., "Name a US state")
**Metrics**: KL divergence from pretraining distribution, Coverage-N, Precision

```bash
# Run CoverageQA experiments
python scripts/tasks/run_simple_qa.py \
    --task coverageqa \
    --method vs_standard \
    --model "gpt-4.1" \
    --num-responses 100

# Run with different probability tuning
python scripts/tasks/run_bias_task_general.py \
    --method vs_standard \
    --probability-tuning 0.1 \
    --model "gpt-4.1"
```

**Expected Output**:
- Results: `results/bias/` or `results/openqa/`
- Answer distributions compared to RedPajama pretraining corpus
- Coverage and precision metrics
- Paper Results: VS achieves KL=0.12 vs human pretraining distribution, while direct prompting shows severe mode collapse (Figure 4, qualitative in Figure 1)

**Probing Pretraining Data**: See Appendix §A.8 (Comparing Pre-trained and VS-Elicited Distributions) for methodology on comparing VS-elicited distributions with pretraining data.

---

### §8: Synthetic Data Generation

**Paper Section**: Section 8 (Synthetic Data Generation)
**Key Results**: Table 5
**Appendix Details**: Appendix §A.4 (Synthetic Data Generation)

We evaluate VS on generating diverse synthetic training data for math reasoning tasks.

#### 8.1 Positive Synthetic Data Generation

Generate diverse, correct synthetic math problems for training.

##### GSM8K Dataset

**Script**: [`tasks/run_positive_gsm8k.py`](tasks/run_positive_gsm8k.py)
**Task**: Generate 1K diverse GSM8K-style grade school math problems
**Evaluation**: Fine-tune models on synthetic data, test on MATH500/OlympiadBench/Minerva Math

```bash
# Generate synthetic GSM8K questions
python scripts/tasks/run_positive_gsm8k.py \
    --method vs_standard \
    --model "gpt-4.1" \
    --num-samples 1000 \
    --output-file results/synthetic/gsm8k_vs_standard.jsonl

# For comparison with Gemini
python scripts/tasks/run_positive_gsm8k.py \
    --method vs_standard \
    --model "gemini/gemini-2.5-flash" \
    --num-samples 1000
```

**Expected Output**:
- Synthetic questions: `results/synthetic/gsm8k_*.jsonl`
- Diversity metrics computed automatically
- Paper Results: VS-generated data improves downstream accuracy by 13.8 points (Table 5, Appendix Table A4.1)

##### AMC/AIME Competition Math

**Script**: [`tasks/run_positive_amc_aime.py`](tasks/run_positive_amc_aime.py)
**Task**: Generate AMC/AIME competition-level math problems

```bash
# Generate AMC/AIME synthetic data
python scripts/tasks/run_positive_amc_aime.py \
    --method vs_standard \
    --model "gpt-4.1" \
    --num-samples 1000
```

##### LiveCodeBench

**Script**: [`tasks/run_positive_lcb.py`](tasks/run_positive_lcb.py)
**Task**: Generate coding problems for LiveCodeBench

```bash
# Generate LiveCodeBench synthetic data
python scripts/tasks/run_positive_lcb.py \
    --method vs_standard \
    --model "gpt-4.1" \
    --num-samples 1000
```

#### 8.2 Negative Synthetic Data Generation

**Script**: [`tasks/run_negative.py`](tasks/run_negative.py)
**Task**: Generate diverse, incorrect but plausible solutions for offline RL training
**Appendix**: Appendix §A.4.2, paragraph "Negative Synthetic Data Generation"

```bash
# Generate negative examples
python scripts/tasks/run_negative.py \
    --method vs_standard \
    --model "gpt-4.1" \
    --dataset gsm8k \
    --num-samples 50
```

**Expected Output**:
- Negative examples: `results/synthetic/negative_*.jsonl`
- Paper Results: Training with VS-generated negative data improves offline RL by 2.69 points (Appendix Table A4.2)

---

## Appendix Experiments

### Random Number Generation

**Paper Section**: Appendix §A.5 (Random Number Generation)
**Key Results**: Table 5 (Appendix), Figure 5 (Appendix)

**Script**: [`tasks/run_rng.py`](tasks/run_rng.py)
**Task**: Simulate rolling a fair 6-sided die to test uniform sampling ability
**Metrics**: KL divergence from uniform distribution

```bash
# Test random number generation
python scripts/tasks/run_rng.py \
    --method vs_standard \
    --model "gemini/gemini-2.5-pro" \
    --num-samples 600
```

**Expected Output**:
- Results: `results/rng/`
- Distribution of dice rolls (1-6)
- Paper Results: VS achieves KL=0.027, vs. 0.926 for direct prompting (Appendix Table 5)

---

### Geographic Bias (State Naming)

**Paper Section**: Appendix (Bias Mitigation experiments)
**Related to**: Open-Ended QA section

**Script**: [`tasks/run_state_name.py`](tasks/run_state_name.py)
**Task**: Test for geographic bias in naming US states
**Metrics**: Distribution uniformity, coverage of all 50 states

```bash
# Test state naming bias
python scripts/tasks/run_state_name.py \
    --method vs_standard \
    --model "gpt-4.1" \
    --num-samples 100
```

**Expected Output**:
- Results: `results/bias/state_name/`
- State frequency distributions
- Paper Results: VS reduces mode collapse on frequent states (CA, TX, NY)

---

### Commonsense Reasoning

**Paper Section**: Appendix §A.6 (Commonsense Reasoning)
**Key Results**: Appendix Table 6

**Script**: [`tasks/run_simple_qa.py`](tasks/run_simple_qa.py) (with SimpleQA dataset)
**Dataset**: SimpleQA (4,326 questions, 300 sampled across 10 domains)
**Task**: Verify that diversity gains don't sacrifice factual accuracy
**Metrics**: Top@1 accuracy, Pass@N accuracy

```bash
# Test factual accuracy with SimpleQA
python scripts/tasks/run_simple_qa.py \
    --task simpleqa \
    --method vs_standard \
    --model "gpt-4.1" \
    --num-samples 5 \
    --num-prompts 300
```

**Expected Output**:
- Results: `results/commonsense/`
- Accuracy metrics per method
- Paper Results: VS maintains comparable accuracy while increasing diversity (Appendix Table 6)

---

### Safety Evaluation

**Paper Section**: Appendix §A.7 (Safety Evaluation)
**Key Results**: Appendix Table 7

**Script**: [`tasks/run_safety.py`](tasks/run_safety.py)

**Dataset**: StrongReject benchmark (353 harmful prompts)
**Task**: Verify VS doesn't compromise safety alignment
**Metrics**: Refusal rate on harmful prompts

```bash
# Run safety evaluation
python scripts/tasks/run_safety.py \
    --method vs_standard \
    --model "gpt-4.1" \
    --dataset strongreject
```

**Expected Output**:
- Results: `results/safety/`
- Refusal rates per method
- Paper Results: VS maintains >97% refusal rate, only 0.3-0.8 points lower than baselines (Appendix Table 7)

---

## Experimental Settings

**Paper Reference**: Appendix §B (Experiment Settings)

### Default Hyperparameters

All experiments use consistent decoding parameters unless otherwise specified:

- **Temperature**: 0.7
- **Top-p**: 1.0
- **Max tokens**: 8,192
- **Random seed**: 42

**Exceptions**:
- Synthetic data generation (§8): temperature=0.6, top-p=0.95 (to match related work)
- Dialogue simulation (§6): temperature=0.7, top-p=0.9, max-tokens=500

### Generation Settings

- **Creative Writing**: N=30 responses, k=5 candidates per call, 100 prompts, ~200 words
- **Dialogue Simulation**: 50 conversations, max 10 turns, k=5 candidates
- **Open-Ended QA**: N=100 responses, k=20 candidates per call, 40 questions
- **Synthetic Data**: N=1000 samples, k=5 candidates per call
- **Random Number**: N=600 samples, k=5 candidates per call

---

## Full Prompts

**Paper Reference**: Appendix §C (Full Prompts)

All prompts used in our experiments are documented in the paper appendix. Key prompt templates:

### VS-Standard Prompt Template

```
Generate {num_samplings} responses to the input prompt. Each response should be approximately {target_words} words.

Return the responses in JSON format with the key: "responses" (list of dicts). Each dictionary must include:
- text: the response string only (no explanation or extra text).
- probability: the estimated probability from 0.0 to 1.0 of this response given the input prompt (relative to the full distribution).

Give ONLY the JSON object, with no explanations or extra text.
```

### VS-CoT Prompt Template

Adds chain-of-thought reasoning before generating responses (see Appendix §C for full prompt).

### VS-Multi Prompt Template

Generates responses across multiple turns for increased diversity (see Appendix §C for full prompt).

### Probability Tuning

For diversity tuning experiments (Figure 2g-i), add:
```
Randomly sample the responses from the distribution, with the probability of each response must be below {probability_threshold}.
```

Thresholds tested: 1.0 (no tuning), 0.9, 0.5, 0.1, 0.05, 0.01

---

## Running All Experiments

To reproduce all main paper results:

```bash
# Main experiments (Sections 5-8)
python scripts/tasks/run_poem.py --method vs_standard --model "gpt-4.1"
python scripts/tasks/run_story.py --method vs_standard --model "gpt-4.1"
python scripts/tasks/run_jokes.py --method vs_standard --model "gpt-4.1"
python scripts/tasks/run_dialogue_simulation.py --method vs_standard
python scripts/tasks/run_simple_qa.py --method vs_standard --model "gpt-4.1"
python scripts/tasks/run_positive_gsm8k.py --method vs_standard --model "gpt-4.1"
python scripts/tasks/run_positive_amc_aime.py --method vs_standard --model "gpt-4.1"

# Appendix experiments
python scripts/tasks/run_rng.py --method vs_standard --model "gpt-4.1"
python scripts/tasks/run_state_name.py --method vs_standard --model "gpt-4.1"
python scripts/tasks/run_safety.py --method vs_standard --model "gpt-4.1"
```

**Note**: These commands provide a starting point. See individual experiment sections above for complete parameter options.

---

## Citation

If you use these experiments in your research, please cite our paper:

```bibtex
@misc{zhang2025verbalizedsamplingmitigatemode,
  title={Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity},
  author={Jiayi Zhang and Simon Yu and Derek Chong and Anthony Sicilia and Michael R. Tomz and Christopher D. Manning and Weiyan Shi},
  year={2025},
  eprint={2510.01171},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2510.01171}
}
```

---

## Need Help?

- For issues or questions, please open an issue on our [GitHub repository](https://github.com/CHATS-lab/verbalize-sampling)
- See the main [README.md](../README.md) for installation and quick start instructions
- Refer to the paper and appendix for detailed methodology and results
