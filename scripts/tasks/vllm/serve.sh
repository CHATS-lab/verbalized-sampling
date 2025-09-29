# MODEL_NAME="meta-llama/Llama-3.1-70B-Instruct"
# MODEL_NAME="meta-llama/Llama-3.1-405B-Instruct-FP8"
MODEL_NAME="Qwen/Qwen3-235B-A22B-Instruct-2507"
# MODEL_NAME="meta-llama/Llama-3.1-70B"
TENSOR_PARALLEL_SIZE=4
DATA_PARALLEL_SIZE=2

vllm serve $MODEL_NAME \
    --data-parallel-size $DATA_PARALLEL_SIZE \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --port 8000 \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --max-model-len 16384


