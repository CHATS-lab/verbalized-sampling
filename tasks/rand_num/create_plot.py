import os
import re
import json
import argparse
import matplotlib.pyplot as plt

TARGET_DIR = "output/meta-llama_Llama-3.3-70B-Instruct"


def extract_numbers_from_dicts(dict_list):
    # Assumes each dict has a 'text' field that is an integer (as string or int)
    return [int(d['text']) for d in dict_list]


def plot_histograms(sizes, numbers_by_size, output_path):
    n_files = len(sizes)
    fig, axs = plt.subplots(1, n_files, figsize=(5 * n_files, 4), sharey=True)
    if n_files == 1:
        axs = [axs]  # Ensure axs is iterable
    bins = 30
    for ax, size in zip(axs, sizes):
        numbers = numbers_by_size[size]
        ax.hist(numbers, bins=bins, color='skyblue', edgecolor='black')
        ax.set_title(f"size={size}")
        ax.set_xlabel("Number")
        if ax == axs[0]:
            ax.set_ylabel("Frequency")
    plt.suptitle("Histogram of Generated Numbers for Different Sampling Sizes (T=1.0)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.show()


def main(args=None):
    parser = argparse.ArgumentParser(description="Plot histograms for different sampling sizes (T=1.0)")
    parser.add_argument('--target_dir', type=str, default=TARGET_DIR, help='Directory containing the response files')
    parser.add_argument('--output', type=str, default="all_histograms_T1.0_subplots_row.png", help='Output image file')
    parser.add_argument('--sizes', type=int, nargs='+', default=[1, 3, 5, 10, 50], help='Sampling sizes to plot (in order)')
    parsed_args = parser.parse_args(args)

    pattern = re.compile(r"responses_(\d+)_1\.0_sampling\.jsonl$")
    numbers_by_size = {}
    for fname in os.listdir(parsed_args.target_dir):
        match = pattern.match(fname)
        if match:
            sampling_size = int(match.group(1))
            if sampling_size in parsed_args.sizes:
                filepath = os.path.join(parsed_args.target_dir, fname)
                numbers = []
                with open(filepath, 'r') as f:
                    for line in f:
                        try:
                            dict_list = json.loads(line)
                            numbers.extend(extract_numbers_from_dicts(dict_list))
                        except Exception as e:
                            print(f"Error parsing line in {fname}: {e}")
                            continue
                numbers_by_size[sampling_size] = numbers
    # Ensure order
    sizes = [size for size in parsed_args.sizes if size in numbers_by_size]
    plot_histograms(sizes, numbers_by_size, parsed_args.output)

if __name__ == "__main__":
    main()
