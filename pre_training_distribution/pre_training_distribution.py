import requests
import json

def query_infini_gram(query):
    payload = {
        'index': 'v4_rpj_llama_s4',
        'query_type': 'count',
        'query': query,
    }
    result = requests.post('https://api.infini-gram.io/', json=payload).json()
    print("Query: ", query, "Count: ", result.get('count', 0))
    return result.get('count', 0)

def main():
    file_path = "data/state_name.json"

    key = "Name a US State. Only provide the answer without explanation or punctuation."

    answers = []
    with open(file_path, "r") as f:
        data = json.load(f)
    answers = data[key]["answers"]
    

    stats = {}
    total_count = 0
    # First, get all counts and sum them
    for answer in answers:
        state_name = answer[0]
        count = query_infini_gram(state_name)
        stats[state_name] = {"count": count}
        total_count += count

    # Now, compute percentages
    for state_name in stats:
        count = stats[state_name]["count"]
        percentage = count / total_count if total_count > 0 else 0
        stats[state_name]["percentage"] = percentage

    # INSERT_YOUR_CODE
    # Sort stats from large to small by count
    stats = dict(sorted(stats.items(), key=lambda x: x[1]["count"], reverse=True))
    output_file = "state_name_distribution.json"
    with open(output_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved state name distribution to {output_file}")

    # count = query_infini_gram(prompt)
    # print(count)


if __name__ == "__main__":
    main()