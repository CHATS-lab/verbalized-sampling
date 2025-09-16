import requests
import requests_cache
import random
import json
from datetime import timedelta
import time
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from datasets import Dataset
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import pandas as pd

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
session = requests_cache.CachedSession(
    'helpsteer_cache',
    expire_after=timedelta(days=14),
    allowable_methods=('GET', 'POST')
)

# HelpSteer contains 37,120 samples, each containing a prompt, a response as well as five human-annotated attributes of the response, each ranging between 0 and 4 where higher means better for each attribute.
# These attributes are:
#     Helpfulness: Overall helpfulness of the response to the prompt.
#     Correctness: Inclusion of all pertinent facts without errors.
#     Coherence: Consistency and clarity of expression.
#     Complexity: Intellectual depth required to write response (i.e. whether the response can be written by anyone with basic language competency or requires deep domain expertise).
#     Verbosity: Amount of detail included in the response, relative to what is asked for in the prompt.

# Dataset item example:
# {
#   "prompt": "What are the three most important things to consider when deciding what technology to use to build an assist device to help an elderly person with basic needs?",
#   "response": "To build an assistive device to help an elderly person with basic needs, one must consider three crucial things: safety, compatibility, and ease of use. Safety is paramount, as the device must not cause harm to the user. Compatibility with the user's environment and other devices is also essential. Finally, the device must be simple enough for the elderly person to operate.",
#   "helpfulness": 3,
#   "correctness": 4,
#   "coherence": 4,
#   "complexity": 2,
#   "verbosity": 2
# }

API_KEY = os.getenv("HYPERBOLIC_API_KEY", os.getenv("OPENAI_API_KEY", ""))
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.hyperbolic.xyz/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-405B")

def measure_logprobs(query: str, response: str,
                    api_base_url: str = None,
                    model: str = None,
                    api_key: str = None,
                    response_prefix: str = "Response") -> float:
    """
    Returns the average per-token logprob for a continuation of a prompt.

    Args:
        query: The input prompt
        response: The response to evaluate
        api_base_url: Base URL for the API (defaults to env var or Hyperbolic)
        model: Model name (defaults to env var or Llama-3.1-405B)
        api_key: API key (defaults to env vars)
        response_prefix: Prefix to look for in tokens (default: "Response")

    Returns:
        Average log probability of response tokens
    """

    # Use provided values or fall back to defaults
    url = f"{api_base_url or API_BASE_URL}/completions"
    model_name = model or MODEL_NAME
    key = api_key or API_KEY

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }

    full_prompt = f"""{query}

    {response_prefix}: {response}"""

    data = {
        "prompt": full_prompt,
        "model": model_name,
        "max_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "logprobs": 1,
        "echo": True,
    }

    backoff_secs = 1.0
    while True:
        llm_response = session.post(url, headers=headers, json=data)
        if llm_response.status_code == 200:
            break
        elif llm_response.status_code == 429:
            print(f"Rate limit exceeded, waiting {backoff_secs} seconds")
            time.sleep(backoff_secs)
            backoff_secs *= 2
        else:
            raise ValueError(f"Error: {llm_response.status_code}, Response: {llm_response.text}")

    llm_response_json = llm_response.json()

    # Find the token index of response prefix (e.g., "ĠResponse") followed by ":"
    response_token_index = None
    prefix_token = f"Ġ{response_prefix}"  # Most tokenizers add Ġ for space prefix

    for i, token in enumerate(llm_response_json["choices"][0]["logprobs"]["tokens"]):
        # Try both with and without the Ġ prefix
        if token in [prefix_token, response_prefix]:
            if i + 1 < len(llm_response_json["choices"][0]["logprobs"]["tokens"]):
                next_token = llm_response_json["choices"][0]["logprobs"]["tokens"][i + 1]
                if next_token == ":":
                    response_token_index = i
                    break

    if response_token_index is None:
        # Fallback: look for any occurrence of the prefix
        for i, token in enumerate(llm_response_json["choices"][0]["logprobs"]["tokens"]):
            if response_prefix.lower() in token.lower():
                response_token_index = i
                break

        if response_token_index is None:
            raise ValueError(f"Response token '{response_prefix}' not found in tokens: {llm_response_json['choices'][0]['logprobs']['tokens'][:10]}")

    # Grab the logprobs of everything beyond "Response:"
    logprobs = llm_response_json["choices"][0]["logprobs"]["token_logprobs"][
        response_token_index + 2 : -1
    ]

    if not logprobs:
        raise ValueError("No logprobs found for response tokens")

    # Average the logprobs
    average_logprob = sum(logprobs) / len(logprobs)

    return average_logprob


def measure_logprobs_batch(items, max_workers=10, **kwargs):
    """
    Compute logprobs for multiple query-response pairs in parallel.

    Args:
        items: List of (query, response) tuples or dict items with 'prompt'/'response' keys
        max_workers: Number of parallel threads
        **kwargs: Additional arguments passed to measure_logprobs

    Returns:
        List of logprobs in same order as input items
    """

    # Handle both tuple format and dict format
    if isinstance(items[0], dict):
        query_response_pairs = [(item["prompt"], item["response"]) for item in items]
    else:
        query_response_pairs = items

    # Create partial function with fixed kwargs
    measure_func = partial(measure_logprobs, **kwargs)

    logprobs = [None] * len(query_response_pairs)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(measure_func, query, response): i
            for i, (query, response) in enumerate(query_response_pairs)
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                logprobs[index] = future.result()
                completed += 1
                if completed % 50 == 0:
                    logger.info(f"Completed {completed}/{len(query_response_pairs)} logprob measurements")
            except Exception as exc:
                logger.error(f"Item {index} generated an exception: {exc}")
                logprobs[index] = None  # or some default value

    return logprobs




# Setup, load dataset, and select subset
random.seed(42)
ds = load_dataset("nvidia/HelpSteer")
train = ds["train"]  # len(train) = 35331 (95%)
val = ds["validation"]  # len(val) = 1789 (5%)
# n_subsample = 150
# train_ids = random.sample(range(len(train)), n_subsample)
ds_run = train.select(range(5000)) # Invariant: end with ds_run

# # Compute logprobs on subset
logger.info("Measuring logprobs...")
helpfulness = [item["helpfulness"] for item in ds_run]
correctness = [item["correctness"] for item in ds_run]

# Use parallel batch processing instead of sequential
logprob = measure_logprobs_batch(ds_run, max_workers=20)

# Save data for later rehydration
json_data = {
    "helpfulness": helpfulness,
    "correctness": correctness,
    "logprob": logprob
}
with open("helpsteer_data-train_5000.json", "w") as f:
    json.dump(json_data, f)

# Load data from dehydrated form
with open("helpsteer_data-train_5000.json", "r") as f:
    json_data = json.load(f)
    helpfulness = json_data["helpfulness"]
    correctness = json_data["correctness"]
    logprob = json_data["logprob"]

# Get counts of each correctness x helpfulness pair
# Just for interest for now, but can be used for weighting if needed
counts = np.zeros((5, 5))  # Correctness x Helpfulness
for i in range(len(json_data["helpfulness"])):
    counts[json_data["correctness"][i]][json_data["helpfulness"][i]] += 1
print(counts)

# Full model: Helpfulness vs Correctness and Logprob

# Y = Helpfulness (ordinal), X = [Correctness, logprob]
df = pd.DataFrame({
    "helpfulness": helpfulness,   # ordinal outcome: 1-5
    "correctness": correctness,   # control variable
    "logprob": logprob            # predictor of interest
})

# Ordered logistic regression
mod = OrderedModel(
    endog=df["helpfulness"],
    exog=df[["correctness", "logprob"]],   # NO constant
    distr="logit"
)
res = mod.fit(method="bfgs")
print(res.summary())

# Odds ratios for interpretation
print(np.exp(res.params))

# Reduced model: without logprob
mod_reduced = OrderedModel(
    endog=df["helpfulness"],
    exog=df[["correctness"]],
    distr="logit"
)
res_reduced = mod_reduced.fit(method="bfgs")

# Likelihood ratio test
import scipy.stats as st

llr_stat = -2 * (res_reduced.llf - res.llf)
pval = st.chi2.sf(llr_stat, df=1)

print("LLR statistic:", llr_stat)
print("p-value:", pval)


# If p < 0.05 → adding logprob significantly improves fit → evidence that logprob helps explain helpfulness even after controlling for correctness.
# If p ≥ 0.05 → no evidence that logprob adds explanatory power.


# ------------------------------------------------------------

# --- Estimate alpha (typicality weight) from within-prompt pairs ----------------
# Requires: ds = load_dataset("nvidia/HelpSteer"); val = ds["validation"]
#           helpfulness, correctness, logprob lists already loaded (same order as val)

import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import combinations
from collections import defaultdict

# 1) Build an item-level dataframe aligned with the validation set
ds_run_prompts = [item["prompt"] for item in ds_run]
ds_run_responses = [item["response"] for item in ds_run]  # optional, for length control

df_items = pd.DataFrame({
    "prompt": ds_run_prompts,
    "response": ds_run_responses,
    "helpfulness": helpfulness,
    "correctness": correctness,
    "logprob": logprob,
})
# Optional: crude length control if your logprob is NOT per-token:
# df_items["char_len"] = [len(r) for r in df_items["response"]]

# 2) Build within-prompt pairwise dataset
rows = []
prompt_to_indices = defaultdict(list)
for idx, p in enumerate(df_items["prompt"].values):
    prompt_to_indices[p].append(idx)

for p, idxs in prompt_to_indices.items():
    if len(idxs) < 2:
        continue
    # all unordered pairs i<j
    for i, j in combinations(idxs, 2):
        hi, hj = df_items.loc[i, "helpfulness"], df_items.loc[j, "helpfulness"]
        if hi == hj:
            continue  # drop helpfulness ties for a clean Bernoulli target
        yi_wins = 1 if hi > hj else 0

        d_lp = df_items.loc[i, "logprob"] - df_items.loc[j, "logprob"]
        d_corr = df_items.loc[i, "correctness"] - df_items.loc[j, "correctness"]

        rows.append({
            "prompt": p,
            "win": yi_wins,            # 1 if i preferred over j
            "d_logprob": d_lp,         # log p_i - log p_j (base model)
            "d_correctness": d_corr,   # correctness_i - correctness_j
        })

pairs = pd.DataFrame(rows)
print(f"Built {len(pairs):,} within-prompt pairs (non-tied helpfulness).")
print(pairs.head())

# 3) Helper to fit logistic with cluster-robust SEs and print interpretable stats
def fit_bt_and_report(df_pairs, extra_covs=None, label=""):
    covs = ["d_logprob"]
    if extra_covs:
        covs += extra_covs

    X = sm.add_constant(df_pairs[covs], has_constant="add")
    y = df_pairs["win"].astype(int).values

    # cluster by prompt (convert to integer codes to be safe)
    clusters = df_pairs["prompt"].astype("category").cat.codes.values

    glm = sm.GLM(y, X, family=sm.families.Binomial())
    res = glm.fit(cov_type="cluster", cov_kwds={"groups": clusters, "use_correction": True})

    alpha_hat = res.params["d_logprob"]
    alpha_se  = res.bse["d_logprob"]
    alpha_p   = res.pvalues["d_logprob"]

    sd_dlp = float(df_pairs["d_logprob"].std())
    or_per_sd = float(np.exp(alpha_hat * sd_dlp))

    # interpretability: win prob at ±1 SD of Δlogprob, holding other covs at 0
    def inv_logit(z): return 1.0 / (1.0 + np.exp(-z))
    z0   = res.params["const"]
    zneg = z0 + alpha_hat * (-sd_dlp)
    zpos = z0 + alpha_hat * ( sd_dlp)
    pneg, ppos = inv_logit(zneg), inv_logit(zpos)

    print("\n" + "="*80)
    print(f"Bradley–Terry (GLM-Binomial) for α {label}")
    print("="*80)
    print(res.summary())  # shows Covariance Type: cluster
    print("-"*80)
    print(f"alpha (coef on d_logprob): {alpha_hat:.4f}  (SE {alpha_se:.4f}, p={alpha_p:.3g})")
    print(f"SD(Δlogprob): {sd_dlp:.4f}  →  OR per 1 SD: {or_per_sd:.3f}")
    print(f"P(win) at Δlogprob -1 SD: {pneg:.3f} ; +1 SD: {ppos:.3f}  (Δ = {ppos - pneg:+.3f})")
    print("-"*80)

    return res


# 4A) Tie slice: equal correctness (isolates r_sem)
pairs_tie = pairs.loc[pairs["d_correctness"] == 0].copy()
print(f"Tie slice (equal correctness): {len(pairs_tie):,} pairs")
if len(pairs_tie) > 0:
    res_tie = fit_bt_and_report(pairs_tie, extra_covs=[], label="(equal correctness)")
else:
    print("No equal-correctness pairs with a non-tied helpfulness label.")

# 4B) All pairs with Δcorrectness as a covariate (adjusted estimate)
res_adj = fit_bt_and_report(pairs, extra_covs=["d_correctness"],
                            label="(adjusted for Δcorrectness)")


