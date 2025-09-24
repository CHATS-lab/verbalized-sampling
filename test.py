import os
import json


def parse_synthetic_postive_data(raw_response):
    # Handle case where raw_response is a dictionary instead of a string
    if isinstance(raw_response, dict):
        # If it's a dict, try to extract the actual response text
        if "response" in raw_response:
            raw_response = raw_response["response"]
        elif "text" in raw_response:
            raw_response = raw_response["text"]
        else:
            return None
    
    # Ensure raw_response is a string
    if not isinstance(raw_response, str):
        return None
    
    # Handle responses that contain both JSON reasoning and actual content
    if "```json" in raw_response and "Question:" in raw_response and "Difficulty:" in raw_response:
        # Find the last occurrence of "Question:" and "Difficulty:" in the response
        # This handles cases where the JSON is malformed but the actual content is at the end
        question_start = raw_response.rfind("Question:")
        difficulty_start = raw_response.rfind("Difficulty:")
        
        if question_start != -1 and difficulty_start != -1 and question_start < difficulty_start:
            question = raw_response[question_start + 9:difficulty_start].strip()
            difficulty = raw_response[difficulty_start + 10:].strip()
            return {
                "question": question,
                "difficulty": difficulty,
            }
    
    # Handle different formats of the response
    if "Question:" in raw_response and "Difficulty:" in raw_response:
        question = raw_response.split("Question:")[1].strip().split("Difficulty:")[0].strip()
        difficulty = raw_response.split("Difficulty:")[1].strip()
    elif "Question:\n" in raw_response and "Difficulty:\n" in raw_response:
        question = raw_response.split("Question:\n")[1].strip().split("Difficulty:\n")[0].strip()
        difficulty = raw_response.split("Difficulty:\n")[1].strip()
    else:
        # Fallback: try to extract question and difficulty from the response
        lines = raw_response.strip().split('\n')
        question = ""
        difficulty = ""
        
        for i, line in enumerate(lines):
            if line.strip().startswith("Question"):
                # Get the question content (everything after "Question" until "Difficulty")
                question_lines = []
                for j in range(i + 1, len(lines)):
                    if lines[j].strip().startswith("Difficulty"):
                        break
                    question_lines.append(lines[j])
                question = '\n'.join(question_lines).strip()
            elif line.strip().startswith("Difficulty"):
                difficulty = line.split("Difficulty")[1].strip().lstrip(":").strip()
                break
        
        if not question or not difficulty:
            return None
    
    return {
        "question": question,
        "difficulty": difficulty,
    }   


def main():
    # file_path = "method_results_amc_aime_1000_no_example/gemini-2.5-flash_amc_aime_math/generation/direct_cot (samples=1)/responses.jsonl"
    file_path = "method_results_amc_aime_1000_no_example/gemini-2.5-flash_amc_aime_math/generation/chain_of_thought [strict] (samples=5)/responses.jsonl"

    # Read the JSONL file
    responses = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                response = json.loads(line)
                prompt = response.get("prompt", None)
                if prompt is None:
                    continue
                for resp in response.get("responses", []):
                    parsed_data = parse_synthetic_postive_data(resp["text"])
                    if parsed_data is not None:
                        responses.append({
                            "prompt": prompt,
                            "responses": {
                                "text": "Question: " + parsed_data["question"] + "\n" + "Difficulty: " + parsed_data["difficulty"],
                            }
                        })
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {file_path}: {e}")
                
    print(len(responses))
    # with open("method_results_amc_aime_1000_no_example/gemini-2.5-flash_amc_aime_math/generation/direct_cot (samples=1)/responses_parsed.jsonl", "w", encoding="utf-8") as f:
    #     for response in responses:
    #         f.write(json.dumps(response) + "\n")
    with open("method_results_amc_aime_1000_no_example/gemini-2.5-flash_amc_aime_math/generation/chain_of_thought [strict] (samples=5)/responses_parsed.jsonl", "w", encoding="utf-8") as f:
        for response in responses:
            f.write(json.dumps(response) + "\n")

if __name__ == "__main__":
    main()