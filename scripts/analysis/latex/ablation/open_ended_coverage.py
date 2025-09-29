# Measure percentage of time that the coverage of direct is include the coverage of sequence
import json
import os
from pathlib import Path


def read_model_method_coverage(file_path):
    method_coverage = {}

    with open(file_path, "r") as f:
        data = json.load(f)
    data = data["overall_metrics"]

    for prompt, meta_data in data["per_prompt_stats"].items():
        prompt = prompt.strip()
        coverage = meta_data["response_distribution"].keys()
        method_coverage[prompt] = coverage
    # print("Length of method coverage: ", len(method_coverage))
    return method_coverage


def calculate_vs_cover_sequence_rate(sequence_prompt_coverage, vs_standard_prompt_coverage, dataset):
    vs_cover_rate = 0
    sequence_cover_rate = 0
    both_cover_rate = 0
    no_cover_rate = 0
    vs_larger_rate = 0
    sequence_larger_rate = 0

    for prompt in dataset:
        prompt = prompt.strip()

        sequence_coverage = set(sequence_prompt_coverage[prompt])
        vs_standard_coverage = set(vs_standard_prompt_coverage[prompt])

        if sequence_coverage & vs_standard_coverage == sequence_coverage:
            vs_cover_rate += 1
        if sequence_coverage & vs_standard_coverage == vs_standard_coverage:
            sequence_cover_rate += 1
        if sequence_coverage & vs_standard_coverage == sequence_coverage and sequence_coverage & vs_standard_coverage == vs_standard_coverage:
            both_cover_rate += 1
        if sequence_coverage & vs_standard_coverage == set():
            no_cover_rate += 1
        if len(sequence_coverage) > len(vs_standard_coverage):
            sequence_larger_rate += 1
        if len(sequence_coverage) < len(vs_standard_coverage):
            vs_larger_rate += 1
    return vs_cover_rate / len(dataset), sequence_cover_rate / len(dataset), both_cover_rate / len(dataset), no_cover_rate / len(dataset), vs_larger_rate / len(dataset), sequence_larger_rate / len(dataset)



def main():
    folder = "openended_qa_general"
    dataset = "data/state_name.txt"

    with open(dataset, "r") as f:
        dataset = f.readlines()
    # print(len(dataset))

    method_name_list = {
        "direct": "direct",
        "direct_cot": "direct_cot",
        "sequence": "sequence",
        "multi_turn": "multi_turn",
        "vs_standard": "vs_standard",
        "vs_cot": "vs_cot",
        "vs_multi": "vs_multi",
    }

    from tabulate import tabulate

    results = []
    # headers = ["Model", "VS Cover Rate", "Sequence Cover Rate", "Both Cover Rate", "No Cover Rate", "VS Larger Rate", "Sequence Larger Rate"]
    headers = ["Model", "VS Larger Rate", "Sequence Larger Rate", "VS Cover Rate", "Sequence Cover Rate"]

    for model in os.listdir(folder):
        model_eval_dir = Path(folder) / model / "evaluation"

        sequence_prompt_coverage = {}
        vs_standard_prompt_coverage = {}
        for method in os.listdir(model_eval_dir):
            method_name = method_name_list[method.split(" ")[0]]

            if method_name == "sequence":
                sequence_prompt_coverage = read_model_method_coverage(model_eval_dir / method / "response_count_results.json")
            if method_name == "vs_standard":
                vs_standard_prompt_coverage = read_model_method_coverage(model_eval_dir / method / "response_count_results.json")

        vs_cover_rate, sequence_cover_rate, both_cover_rate, no_cover_rate, vs_larger_rate, sequence_larger_rate = calculate_vs_cover_sequence_rate(
            sequence_prompt_coverage, vs_standard_prompt_coverage, dataset
        )
        results.append([
            model,
            f"{vs_larger_rate:.3f}",
            f"{sequence_larger_rate:.3f}",
            f"{vs_cover_rate:.3f}",
            f"{sequence_cover_rate:.3f}",
        ])

    print(tabulate(results, headers=headers, tablefmt="grid"))



if __name__ == "__main__":
    main()