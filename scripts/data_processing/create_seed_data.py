from datasets import load_dataset
import json
import random
# Load the dataset
dataset = load_dataset("EleanorZzz/gsm8k_training_positive_1k")
seed = 42
random.seed(seed)
max_samples = 100

# Process the data
processed_data = []
for item in dataset['train']:
    instruction = item['instruction']
    output = item['output']
    
    # Remove the trailing instruction text
    trailing_text = "Think step by step then provide the numerical answer at the end after the delimiter '####', like '#### 24'."
    if instruction.endswith(trailing_text):
        instruction = instruction[:-len(trailing_text)].strip()
    # Remove "Question: " prefix if it exists
    if instruction.startswith("Question: "):
        instruction = instruction[len("Question: "):].strip()
    # Format the new prompt
    formatted_prompt = f"Here's the math problem: {instruction}\nGolden answer: {output}"
    processed_data.append(formatted_prompt)

subset_data = random.sample(processed_data, max_samples)
# Save to file
with open('data/synthetic_negative_new.jsonl', 'w') as f:
    for prompt in subset_data:
        f.write(json.dumps({'prompt': prompt}) + '\n')

print(f"Processed {len(subset_data)} items and saved to synthetic_negative_new.jsonl")
