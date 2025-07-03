MODEL_NAME="meta-llama/Llama-3.1-70B-Instruct"
TENSOR_PARALLEL_SIZE=8

vllm serve $MODEL_NAME \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --port 8000 \
    --host 0.0.0.0 \
    --dtype bfloat16 \