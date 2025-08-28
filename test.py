from datasets import load_dataset
import numpy as np

ds = load_dataset("livecodebench/code_generation_lite", version_tag="release_v6", split="test", trust_remote_code='True')
np.random.seed(42)
idxs = np.random.choice(range(len(ds)), 3, replace=False)
icl_examples = [ds[int(i)] for i in idxs]

print(icl_examples[0].keys())
# print(icl_examples[0]['public_test_cases'])
# print(icl_examples[0]['starter_code'])