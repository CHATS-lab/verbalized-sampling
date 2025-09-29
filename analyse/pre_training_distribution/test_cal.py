import json

def read_data(path):
    texts = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for response in record.get("responses", []):
                text = response.get("text")
                if not text:
                    continue
                text = text.strip()
                if text:
                    texts.append(text)
    return texts
    
def group_data(data):
    counts = {}
    for text in data:
        counts[text] = counts.get(text, 0) + 1
    total = len(data) if data is not None else 0
    grouped = {}
    for text, count in counts.items():
        grouped[text] = {"count": count, "percentage": (count / total) if total else 0.0}
    return grouped


def main():
    gpt_file = "pre_training_distribution/gpt-4.1/generation/direct (samples=50)/responses.jsonl"
    claude_file = "pre_training_distribution/claude-4-sonnet/generation/direct (samples=50)/responses.jsonl"

    gpt_data = read_data(gpt_file)
    claude_data = read_data(claude_file)

    grouped_gpt_data = group_data(gpt_data)
    grouped_claude_data = group_data(claude_data)

    print(grouped_gpt_data)
    print(grouped_claude_data)

    # Save grouped data as JSON files named after the model
    with open("pre_training_distribution/direct_gpt-4.1.json", "w") as f:
        json.dump(grouped_gpt_data, f, indent=2, ensure_ascii=False)
    with open("pre_training_distribution/direct_claude-4-sonnet.json", "w") as f:
        json.dump(grouped_claude_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()