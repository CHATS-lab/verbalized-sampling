# MODEL=meta-llama/Llama-3.3-70B-Instruct
MODEL=openai/gpt-4o
FORMAT=true

SIM_TYPE=sampling

python test.py --model_name $MODEL --format

# stop_server
