import json

input_file = "story_experiments_final/anthropic_claude-3.7-sonnet/anthropic_claude-3.7-sonnet_book/generation/vs_multi [strict] (samples=5)/responses.jsonl"
output_file = input_file.replace(".jsonl", "_filtered.jsonl")

def is_list_of_dicts(obj):
    return isinstance(obj, list) and all(isinstance(item, dict) for item in obj)

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        try:
            data = json.loads(line)
            if is_list_of_dicts(data.get("responses", [])):
                outfile.write(line)
        except Exception:
            continue  # skip lines that aren't valid JSON

print(f"Filtered lines written to {output_file}") 