# verbalized‑sampling (VS)

**Ask for a distribution, not a sample.**

*A tiny, batteries‑included Python library that makes **Verbalized Sampling (VS)** a one‑liner. Instead of asking an LLM for one answer, ask it for a small **distribution**—`k` candidates with **weights**—then deterministically **filter → normalize → order**, and choose via **`dist.argmax()`** or **`dist.sample()`**. VS restores diversity that post‑training suppresses while keeping quality and safety flat across tasks.*

---

## 0) Quick start

```python
pip install verbalized-sampling
```

```python
from verbalized_sampling import verbalize

# Learn a tiny distribution from the model
dist = verbalize(
    "Write an opening line for a mystery novel",
    k=5, tau=0.12, temperature=0.9,
)

print(dist.to_markdown())   # quick view of items & normalized masses

best   = dist.argmax()             # deterministic top item (returns Item)
choice = dist.sample(random_seed=7)       # seeded weighted sample (returns Item)

print(best.text)
print(choice.text)
```

**What you get:** a `DiscreteDist[Item]` that’s auditable, serializable, and easy to manipulate—plus ergonomic selection methods on the distribution itself.

---

## 1) Product principles

1. **One minute to “aha”.** One function (`verbalize()`), one composable object (`DiscreteDist`), and two intuitive methods (`.argmax()`, `.sample()`).
2. **Deterministic semantics.** Every call is an auditable sequence: **Elicit → Parse/Repair → Filter(τ) → Normalize → Order**.
3. **Distribution‑first mental model.** Elicited numbers are **sampling weights**, not calibrated probabilities. We preserve raw values and record any repairs.
4. **Provider‑agnostic posture, narrow v1.** First‑class OpenAI & Anthropic; optional adapters for local stacks later.
5. **Safety by default.** Final‑JSON‑only schema; no chain‑of‑thought unless explicitly enabled in future extensions.

---

## 2) Top‑level API (v1)

### 2.1 `verbalize()`

```python
from typing import Any, Dict, Optional, Sequence, Literal
from dataclasses import dataclass

def verbalize(
    prompt: Optional[str] = None, *,
    messages: Optional[Sequence[Dict[str, Any]]] = None,

    # Core knobs
    k: int = 5,
    tau: float = 0.12,
    temperature: float = 0.9,

    # Provider/model selection
    provider: Literal["auto", "openai", "anthropic"] = "auto",
    model: Optional[str] = None,
    # Extra kwargs passed to provider SDKs (e.g., timeouts)
    provider_kwargs: Optional[Dict[str, Any]] = None,

    # Robustness
    json_repair: bool = True,
    min_k_survivors: int = 3,
    retries: int = 2,

    # Weight handling
    weight_mode: Literal["elicited", "softmax", "uniform"] = "elicited",

    # Determinism
    seed: Optional[int] = None,

    # Introspection
    with_meta: bool = False,
) -> "DiscreteDist":
    """Elicit k weighted candidates → DiscreteDist (filtered, normalized, ordered)."""
```

**Behavioral defaults that feel good:** `k=5`, `tau=0.12`, `temperature=0.9`, `weight_mode="elicited"` (accept model’s numbers if valid; else repair/normalize or fallback to uniform if needed).

### 2.2 `DiscreteDist` (public methods)

```python
from collections.abc import Sequence

class DiscreteDist(Sequence["Item"]):
    # core
    def __getitem__(self, i: int) -> "Item": ...
    def __len__(self) -> int: ...
    @property
    def items(self) -> list["Item"]: ...   # stable, descending by p
    @property
    def p(self) -> list[float]: ...        # [item.p for item in items]

    # selection
    def argmax(self) -> "Item": ...
    def sample(self, seed: Optional[int] = None) -> "Item": ...

    # transforms (all return new DiscreteDists and preserve invariants)
    def map(self, fn) -> "DiscreteDist": ...
    def filter_items(self, pred) -> "DiscreteDist": ...
    def reweight(self, fn) -> "DiscreteDist": ...  # recompute masses then renorm

    # serialization & display
    def to_dict(self) -> Dict[str, Any]: ...
    def to_markdown(self, max_items: Optional[int] = None) -> str: ...

    # introspection
    @property
    def trace(self) -> Dict[str, Any]: ... # model, tokens, latency, mode_used, repairs, tau_final, seed, etc.
```

### 2.3 `Item`

```python
@dataclass(frozen=True)
class Item:
    text: str                 # the candidate
    p: float                  # normalized mass in [0,1] after filtering/order
    meta: Dict[str, Any]      # raw elicited weight(s) + repairs & provenance
```

`meta` includes:

* `p_raw`: as elicited (may be %, >1, or malformed)
* `p_clipped`: coerced to [0,1] pre‑normalize
* `repairs`: list[str] (e.g., ["percent_to_unit", "clipped>1"])
* `idx_orig`: original item index
* `tau_relaxed`: bool, `tau_final`: float
* `provider_meta`: tokens in/out (if available), latency, model, `mode_used` (`"json_schema"|"tool"|"freeform_extraction"`)

**Note:** In v1, `Item.text` is the payload. In v1.1 we may generalize to `Item.value` to support structured outputs.

### 2.4 Optional helper

```python
def select(
    dist: "DiscreteDist",
    strategy: Literal["argmax", "sample"] = "sample",
    seed: Optional[int] = None,
) -> "Item":
    """Neutral helper mirroring DiscreteDist methods."""
```

(Ergonomic alias `pick()` may be provided but is not required.)

---

## 3) Global invariants

After `verbalize()` finishes:

* **Filtering:** keep items whose *elicited* weight ≥ `tau`. If survivors `< min_k_survivors`, relax `tau` to the largest value that admits exactly `min_k_survivors` items (deterministically).
* **Normalization:** `Σ p == 1.0 ± 1e‑9`. If all masses are 0 after clipping, fall back to uniform masses and record a fallback reason.
* **Ordering:** stable, **descending by normalized `p`**, with deterministic tie‑breaks `(idx_orig, hash(text))`.
* **Determinism:** any stochastic step uses `seed` (recorded in `trace`).

---

## 4) Semantics: filter → normalize → order (→ select on the object)

### 4.1 Pseudocode

```python
def postprocess(items, tau, min_k_survivors, weight_mode, seed):
    # 1) Filter on elicited weights (before normalization)
    S = [it for it in items if it.p_raw >= tau]
    tau_relaxed = False
    if len(S) < min_k_survivors and items:
        sorted_by_raw = sorted(items, key=lambda it: it.p_raw, reverse=True)
        tau_final = sorted_by_raw[min_k_survivors-1].p_raw
        S = [it for it in items if it.p_raw >= tau_final]
        tau_relaxed = True
    else:
        tau_final = tau

    # 2) Clip/repair weights to [0,1] (e.g., "70%" -> 0.70)
    for it in S:
        it.p_clipped, repairs = repair_and_clip(it.p_raw) # returns [0,1] and repair tags
        it.meta["repairs"] = repairs

    # 3) Combine per weight_mode
    if weight_mode == "uniform":
        for it in S: it.p_norm = 1.0 / len(S)
    elif weight_mode == "softmax":
        # softmax over p_clipped (temperature 1.0)
        exps = [math.exp(x) for x in [it.p_clipped for it in S]]
        Z = sum(exps) or 1.0
        for it, e in zip(S, exps): it.p_norm = e / Z
    else:  # "elicited"
        Z = sum(it.p_clipped for it in S)
        if Z <= 1e-12:
            for it in S: it.p_norm = 1.0 / len(S)
        else:
            for it in S: it.p_norm = it.p_clipped / Z

    # 4) Stable order (desc p_norm, tie by idx_orig, then text hash)
    S_sorted = sorted(S, key=lambda it: (-it.p_norm, it.idx_orig, hash(it.text)))

    return DiscreteDist(S_sorted, meta={"tau_relaxed": tau_relaxed, "tau_final": tau_final, "seed": seed})
```

### 4.2 Selection

* `dist.argmax()` returns the first item (already top‑sorted), deterministic under ties.
* `dist.sample(seed=None)` draws from `p`. If `seed` is provided, the draw is deterministic without mutating the distribution.

---

## 5) Providers (v1 scope)

* **Supported:** `provider="auto"|"openai"|"anthropic"`

  * *Auto* selects the first provider with a valid API key in the environment; otherwise raises a friendly error that explains how to set keys or choose explicitly.
  * Prefer JSON Schema or tool function calls. If a model ignores structure, fall back to robust extraction.
* **Deferred (v1.1+):** adapters for HF/Transformers and vLLM via extras: `pip install verbalized-sampling[hf]`.

**Provider interface (internal, stable across adapters):**

```python
class _Provider:
    supports_json: bool
    supports_tools: bool
    def generate_json(self, prompt_or_messages, schema) -> Dict[str, Any]: ...
```

`trace["mode_used"]` records `"json_schema"`, `"tool"`, or `"freeform_extraction"`.

---

## 6) Prompt schema

**VS‑Standard (default, v1):**
“Return exactly `k` items as JSON `[{text, weight}]`. `weight` in `[0,1]`. Output JSON only.”

No chain‑of‑thought is requested or returned in v1.

---

## 7) Parsing & repair pipeline

1. **Structured mode first:** JSON Schema / tool call when supported by the provider & model.
2. **Fallback extraction:** first valid fenced JSON block (`json … `), else first balanced `{…}`/`[…]` region.
3. **Repairs (idempotent, recorded):**

   * Convert strings with `%` to unit interval; negatives → `0.0`; >1 → clipped to `1.0`.
   * Strip trailing commas; fix single vs double quotes when safe.
   * Coerce keys like `{"prob","score"}` → `weight` when unambiguous.
4. **Validation:** small Pydantic schema enforces exactly `k` entries with `text: str`, `weight: number`. If invalid, retry with a simplified prompt up to `retries`.
5. **Provenance:** every repair step is logged in `Item.meta["repairs"]` and surfaced via `dist.trace`.

---

## 8) Determinism & reproducibility

* **Seeds:** `seed` controls stochastic internals (retry jitter, any randomized backoff) and is recorded.
* **Selection seeds:** `dist.sample(seed=...)` accepts its own seed so generation seed ≠ selection seed.
* **Order invariants:** post‑normalized items are always in stable, descending order by `p`. Ties break deterministically.

---

## 9) Observability & UX

* **Trace:** `dist.trace` includes `provider`, `model`, `tokens_in/out` (when available), `latency_ms`, `mode_used`, `json_repair_applied` (bool), `tau_relaxed` (bool), `tau_final` (float), `seed`, fallback reasons.
* **`to_markdown()`** emphasizes auditability:

```
# verbalized-sampling
k=5  τ=0.12 (relaxed: False)  Σp=1.000  model=claude-3-x  json_schema=True

1. 0.37  "The clock struck thirteen in a house that swore it couldn’t count."  [percent_to_unit]
2. 0.21  "Rain wrote its alibi on the window."                                 []
3. 0.18  "The missing key sang in the wrong pocket."                           [clipped>1]
4. 0.13  "Every light in the village blinked once."                             []
5. 0.11  "The footprints ended at the river, then began again upstream."       []
```

---

## 10) Performance & cost

* **Token overhead:** ~2–3× vs single‑shot prompts (one call returns a small distribution).
* **Primary knobs:** `k` (breadth) and `τ` (filter strength) govern the diversity/latency trade‑off. Decoding knobs (temperature/top‑p) remain orthogonal.
* **Guardrail:** optionally warn if `k > 12` (can be disabled).
* **Presets (optional sugar):**

| preset        | k | τ    | retries |
| ------------- | - | ---- | ------- |
| cheap         | 3 | 0.20 | 1       |
| balanced      | 5 | 0.12 | 2       |
| max_diversity | 8 | 0.08 | 2       |

---

## 11) Integrations

### 11.1 LangChain / LCEL

```python
from verbalized_sampling.integrations.langchain import VSRunnable

# Returns a DiscreteDist
vs_chain = VSRunnable(k=5, tau=0.12, ?engine=VSRunnable?)
dist = vs_chain.invoke("Generate product headlines")

# If you want a string:
vs_pick = VSRunnable(k=5, tau=0.12, select="sample")   # or "argmax"
headline = vs_pick.invoke("Generate product headlines")
```

**Contract:** If `select` is set, the runnable returns `str`. Otherwise it returns a JSON‑serializable `DiscreteDist`.

---

## 12) CLI

```bash
# JSON to stdout; concise summary to stderr
vs run  "Tell 5 jokes with probabilities about coffee" \
  -k 5 -t 0.12 --provider openai --model gpt-4o

# Inline selection → returns text
vs pick "Five hero headlines with probabilities" \
  -k 5 -t 0.12 --select sample

# Batch NDJSON with retries & concurrency; write traces
vs batch prompts.txt -k 5 -t 0.12 --out ndjson --concurrency 8 --trace traces/
```

---

## 13) Recipes

### 13.1 Creative writing (diversity without quality loss)

```python
dist = verbalize("Write five first lines for a cozy mystery", k=5, tau=0.12)
print(dist.to_markdown())
best = dist.argmax()
```

### 13.2 Open‑ended QA (multi‑valid answers)

```python
dist = verbalize("Name a US state", k=20, tau=0.10)
coverage = len({it.text.lower() for it in dist})
print(f"Coverage: {coverage}/{len(dist)}; top-1 is: {dist.argmax().text}")
```

### 13.3 Negative synthetic data (plausible‑but‑wrong)

```python
prompt = "Give five plausible but incorrect solution sketches for this GSM8K problem:\n{problem}"
dist = verbalize(prompt, k=5, tau=0.12)
negatives = [it.text for it in dist]  # diverse, plausible mistakes for contrastive training
```

### 13.4 Distribution transforms (compose without breaking invariants)

```python
cleaned = (dist
           .map(lambda it: it.text.strip())
           .filter_items(lambda it: len(it.text) < 120)
           .reweight(lambda it: it.meta["p_raw"]))   # then renormalized

print(cleaned.to_markdown())
```

---

## 14) Testing & quality

* **Adapters:** golden tests for JSON/tool success and fallback extraction per provider.
* **Parsing:** property tests for fenced blocks, trailing commas, bracket balancing, and `%→[0,1]` conversions.
* **Semantics:** snapshot tests for **filter → normalize → order** invariants; deterministic tie‑breaks.
* **Determinism:** seeds recorded and honored; `dist.sample(seed=...)` is repeatable.
* **Reliability budgets:** CI tracks rates for `json_repair_applied`, τ‑relax events, and “zero‑mass→uniform fallback”.

---

## 15) Error taxonomy

```python
class VSError(Exception): ...
class ProviderNotConfigured(VSError): ...
class SchemaRejectedByModel(VSError): ...
class ParseError(VSError): ...
class NoJSONFound(ParseError): ...
class ValidationFailed(ParseError): ...
class InsufficientItems(VSError): ...       # after τ relaxation still < min_k_survivors
class ZeroMassAfterClipping(VSError): ...
```

All raised errors carry a `.trace` payload for debugging.

---

## 16) Security & privacy

* Keys never logged; prompts redacted by default in traces.
* **Final‑JSON‑only** default; no chain‑of‑thought by default.
* VS inherits provider safeguards; it does not encourage unsafe content.

---

## 17) Packaging & project layout

```
verbalized-sampling/
  ├─ verbalized_sampling/
  │   ├─ __init__.py          # exports: verbalize, select, Item, DiscreteDist
  │   ├─ api.py               # verbalize implementation
  │   ├─ providers/           # openai.py, anthropic.py, internal _Provider interface
  │   ├─ parsing.py           # extraction + repairs + schema validation
  │   ├─ dist.py              # Item/DiscreteDist (+ to_markdown / to_dict / transforms)
  │   ├─ integrations/
  │   │   └─ langchain.py
  │   └─ cli.py               # vs run / pick / batch
  ├─ tests/                   # golden + property + snapshot tests
  ├─ README.md                # mirrors the essentials of this plan
  ├─ pyproject.toml           # minimal deps; extras: [cli], [dev], later [hf]
  └─ LICENSE
```

**Dependencies (v1):**
runtime: provider SDK(s), `pydantic` (small), `click` (CLI)
dev: `pytest`, `hypothesis`, `mypy`

---

## 18) Documentation & messaging

* **Tagline:** *Ask for a distribution, not a sample.*
* **Quick start:** 30‑second snippet (see §0).
* **Honesty box:** *Masses are sampling weights, not calibrated probabilities.* Normalize, record repairs, expose raw values in `meta`.
* **Try‑it‑now:** One short notebook: call `verbalize()`, view `dist.to_markdown()`, tweak `τ` & `k`.
* **Budget knobs:** Explain cost trade‑offs and presets.

---

## 19) Roadmap

**v1.0 (this plan)**

* `verbalize()` + `DiscreteDist` with `.argmax()` & `.sample()`
* VS‑Standard schema only
* OpenAI/Anthropic providers
* LangChain runnable; small CLI
* Robust parsing/repair + determinism & trace
* Minimal transforms: `.map()`, `.filter_items()`, `.reweight()`

**v1.1**

* Optional adapters for HF/Transformers and vLLM (`pip install verbalized-sampling[hf]`)
* Async & high‑throughput batch APIs
* Pairwise/listwise judge plugins that return a reweighted distribution
* Auto‑τ utilities and diversity targets
* LlamaIndex adapter; LangSmith hooks
* Structured payloads via `Item.value` (back‑compat with `Item.text`)

---

## 20) Acceptance criteria (for v1 release)

* `pip install verbalized-sampling` yields a working `verbalize()` against OpenAI/Anthropic with a **one‑file quickstart**.
* The top invariants hold in tests: **(a)** Σp≈1, **(b)** stable ordering, **(c)** deterministic tie‑breaks.
* CLI `vs run` and `vs pick` operate end‑to‑end with structured JSON outputs and optional trace files.
* Docs show a 60‑second “aha,” frame weights as sampling masses, and demonstrate `τ` tuning.
* LangChain `VSRunnable` works in a README snippet.

---

## 21) Risks & mitigations

* **Provider idiosyncrasies:** Some models may emit extra text in JSON mode. → Robust extraction + repairs, visible `repairs` log, snapshot tests.
* **User confusion about probabilities:** Over‑communicate “sampling weights.” Show this language in `to_markdown()` and docstrings.
* **Scope creep:** Keep Multi/CoT/judging as contrib modules until v1.1+; v1 stays delightfully small.

---

## 22) Example session (end‑to‑end)

```python
from verbalized_sampling import verbalize, select

# 1) Elicit a tiny distribution
dist = verbalize(
    "Write five short product taglines for a minimalist desk lamp.",
    k=5, tau=0.12, temperature=0.9, seed=42
)

# 2) Inspect
print(dist.to_markdown())

# 3) Serialize (for audit or caching)
payload = dist.to_dict()

# 4) Deterministic top pick
print(dist.argmax().text)

# 5) Seeded weighted sampling (repeatable)
print(dist.sample(seed=42).text)

# 6) Optional neutral helper (same as methods)
print(select(dist, "argmax").text)
```

---

## 23) FAQs

**Q: Does VS replace decoding controls (temperature/top‑p)?**
**A:** No. VS is orthogonal. Use both to shift the diversity–quality frontier.

**Q: Are the weights calibrated probabilities?**
**A:** Treat them as **sampling weights**. We normalize, record repairs, and expose raw values in `meta`.

**Q: What happens if the model ignores JSON mode?**
**A:** We attempt tolerant extraction, log repairs, and retry with a simplified prompt if necessary. You get a valid `DiscreteDist` or a typed error with trace details.

**Q: Can I force equal weights?**
**A:** Yes: `weight_mode="uniform"` makes all survivors have equal mass after filtering.
