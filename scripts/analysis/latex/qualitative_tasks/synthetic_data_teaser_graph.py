import os
import json
from typing import Dict
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def compute_pairwise_cosine_similarities(responses, model_name="text-embedding-3-small"):
    # Use OpenAI's text-embedding-3-small model
    embeddings = []
    for response in tqdm(responses, desc="Computing embeddings"):
        response = extract_content(response)
        # response_embedding = get_embedding("Question: " + response['question'] + "\nTest Input: " + response['test_input'] + "\nAnswer: " + response['answer'], model_name)
        response_embedding = get_embedding("Question: " + response['question'] + "\nAnswer: " + response['answer'], model_name)
        embeddings.append(response_embedding)
    
    embeddings_array = np.array(embeddings)
    embeddings_normalized = normalize(embeddings_array, norm='l2', axis=1)
    similarity_matrix = cosine_similarity(embeddings_normalized)
    sims = []
    for i in range(len(similarity_matrix)):
        for j in range(i+1, len(similarity_matrix)):
            sims.append(similarity_matrix[i, j])
    return sims

def plot_similarity_histogram(sim_direct, sim_vs, bins=50, save_path=None):
    import seaborn as sns
    import numpy as np
    plt.figure(figsize=(8, 5))
    plt.style.use('default')  # Start with clean slate
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans']
    })
    data = [sim_direct, sim_vs]
    labels = ['Direct', 'VS-Standard']
    # Use the same color for both histogram and KDE for each method
    # bar_colors = ['#D5D1D1', '#F7A6AC', '#7FBDDA']
    # kde_colors = ['gray', 'deeppink', 'royalblue']
    bar_colors = ['#D5D1D1', '#7FBDDA']
    kde_colors = ['gray', 'royalblue']

    # Plot histograms
    plt.hist(data, bins=bins, alpha=0.5, color=bar_colors, label=labels, density=True, histtype='stepfilled', linewidth=2)
    # Plot KDEs
    for d, c in zip(data, kde_colors):
        try:
            sns.kdeplot(d, color=c, lw=2)
        except Exception:
            pass

    plt.xlabel("Cosine Similarity", fontsize=24)
    plt.ylabel("Density", fontsize=24)
    plt.title("", fontsize=18, fontweight='bold')
    plt.xticks(np.linspace(0, 1, 6), fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(0, 1)
    plt.ylim(bottom=0)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(fontsize=16, frameon=False, reverse=True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

# def extract_content(raw_response: str) -> Dict[str, str]:
#     # Only extract the first occurrence if there are multiple
#     if "Question:" not in raw_response:
#         raise ValueError("No 'Question:' found in response.")
#     first_question_split = raw_response.split("Question:", 1)[1]
    
#     if "Test Input:" not in first_question_split:
#         raise ValueError("No 'Test Input:' found after 'Question:'.")
#     question = first_question_split.split("Test Input:", 1)[0]
#     test_input_reasoning_answer = first_question_split.split("Test Input:", 1)[1]
    
#     if "Reasoning:" not in test_input_reasoning_answer:
#         raise ValueError("No 'Reasoning:' found after 'Test Input:'.")
#     test_input = test_input_reasoning_answer.split("Reasoning:", 1)[0]
#     reasoning_answer = test_input_reasoning_answer.split("Reasoning:", 1)[1]
    
#     if "Answer:" not in reasoning_answer:
#         raise ValueError("No 'Answer:' found after 'Reasoning:'.")
#     reasoning = reasoning_answer.split("Answer:", 1)[0]
#     answer = reasoning_answer.split("Answer:", 1)[1]

#     return {
#         "question": question.strip(),
#         "test_input": test_input.strip(),
#         "reasoning": reasoning.strip(),
#         "answer": answer.strip(),
#     }

def extract_content(raw_response: str) -> Dict[str, str]:
    parsed = raw_response.split("Question:")[1].split("Answer:")
    return {
        "question": parsed[0].strip(),
        "answer": parsed[1].strip(),
    }


def read_direct_response(response_file: str) -> Dict[str, str]:
    direct_responses = []
    with open(response_file, "r") as f:
        for line in f:
            textline = json.loads(line)
            content = textline['responses'][0]['text']
            direct_responses.append(content)
    print("Number of Direct responses: ", len(direct_responses))
    return direct_responses

def read_vs_response(response_file: str) -> Dict[str, str]:
    vs_responses = []
    with open(response_file, "r") as f:
        for line in f:
            textline = json.loads(line)
            for response in textline['responses']:
                vs_responses.append(response['text'])
    print("Number of Verbalized responses: ", len(vs_responses))
    return vs_responses

def read_sequence_response(response_file: str) -> Dict[str, str]:
    sequence_responses = []
    with open(response_file, "r") as f:
        for line in f:
            textline = json.loads(line)
            for response in textline['responses']:
                sequence_responses.append(response['text'])
    print("Number of Sequence responses: ", len(sequence_responses))
    return sequence_responses


def main():
    # direct_response_file = "method_results_lcb/gpt-4.1_livecodebench/generation/direct (samples=1)/responses.jsonl"
    # vs_response_file = "method_results_lcb/gpt-4.1_livecodebench/generation/vs_standard [strict] (samples=20)/responses.jsonl"
    direct_response_file = "method_results_gsm8k/gpt-4.1_gsm8k/generation/direct (samples=1)/responses.jsonl"
    sequence_response_file = "method_results_gsm8k/gpt-4.1_gsm8k/generation/sequence [strict] (samples=5)/responses.jsonl"
    vs_response_file = "method_results_gsm8k/gpt-4.1_gsm8k/generation/vs_standard [strict] (samples=5)/responses.jsonl"

    direct_responses = read_direct_response(direct_response_file)
    sequence_responses = read_sequence_response(sequence_response_file)
    vs_responses = read_vs_response(vs_response_file)

    # sim_direct = compute_pairwise_cosine_similarities(direct_responses)
    # sim_sequence = compute_pairwise_cosine_similarities(sequence_responses)
    # sim_vs = compute_pairwise_cosine_similarities(vs_responses)
    # # Save cosine similarities to JSON
    # similarity_dict = {
    #     "sim_direct": sim_direct,
    #     "sim_sequence": sim_sequence,
    #     "sim_vs": sim_vs
    # }
    # with open("latex/qualitative_tasks/synthetic_data_gsm8k_similarity.json", "w") as f:
    #     json.dump(similarity_dict, f, indent=2)
    # print("✓ Saved cosine similarities to latex/qualitative_tasks/synthetic_data_gsm8k_similarity.json")

    with open("latex/qualitative_tasks/synthetic_data_gsm8k_similarity.json", "r") as f:
        similarity_data = json.load(f)
    print("Loaded similarity data:", similarity_data.keys())

    # plot_similarity_histogram(similarity_data["sim_direct"], similarity_data["sim_sequence"], similarity_data["sim_vs"], save_path="latex/qualitative_tasks/synthetic_data_gsm8k_similarity_histogram.pdf")
    plot_similarity_histogram(similarity_data["sim_direct"], similarity_data["sim_vs"], save_path="latex/qualitative_tasks/synthetic_data_gsm8k_similarity_histogram.pdf")

if __name__ == "__main__":
    main()