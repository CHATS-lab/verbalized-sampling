# MODEL=meta-llama/Llama-3.3-70B-Instruct

export CUDA_VISIBLE_DEVICES=1

MODEL=$1
QUANTIZATION=$2
FORMAT=true
SERVER_PORT=7000
SERVER_HOST=localhost
TENSOR_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=1
EVAL_ENGINE=vllm
SERVER_URL=http://${SERVER_HOST}:${SERVER_PORT}

SIM_TYPE=sampling

TOTAL=10000

SAMPLE_SIZES=(10 50)
MODEL_NAME=${MODEL/\//_}
mkdir -p output/${MODEL_NAME}

start_server() {
    echo "Starting $EVAL_ENGINE server..."
    
    if [ "$EVAL_ENGINE" == "vllm" ]; then
        TENSOR_PARALLEL_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
        TENSOR_PARALLEL_SIZE=$((TENSOR_PARALLEL_SIZE + 1))
        
        QUANT_ARG=""
        if [ "$QUANTIZATION" == "true" ]; then
            QUANT_ARG="--quantization=fp8"
        fi
        
        FORMAT_ARG=""
        if [ "$FORMAT" == "true" ]; then
            FORMAT_ARG="--guided-decoding-backend=auto"
        fi

        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES vllm serve $MODEL \
            --port $SERVER_PORT --tensor-parallel-size $TENSOR_PARALLEL_SIZE $QUANT_ARG $FORMAT_ARG > vllm.log 2>&1 &
        SERVER_PID=$!
    elif [ "$EVAL_ENGINE" == "sglang" ]; then
        DATA_PARALLEL_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
        DATA_PARALLEL_SIZE=$((DATA_PARALLEL_SIZE + 1))
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m sglang.launch_server --model-path $MODEL_NAME \
            --host $SERVER_HOST --dp $DATA_PARALLEL_SIZE --port $SERVER_PORT > sglang.log 2>&1 &
        SERVER_PID=$!
    else
        echo "Error: Unknown evaluation engine '$EVAL_ENGINE'"
        exit 1
    fi
    
    # Wait for server to start up
    echo "Waiting for server to start up..."
    while ! curl -s "$SERVER_URL/models" > /dev/null; do
        sleep 2
        echo "Still waiting for server..."
        # Check if server process is still running
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "Error: Server process died unexpectedly. Check logs."
            exit 1
        fi
    done
    echo "Server is up and running!"
}

# Function to stop the server
stop_server() {
    echo "Stopping $EVAL_ENGINE server..."
    if [ "$EVAL_ENGINE" == "vllm" ]; then
        pkill -f "serve $MODEL" || true
    elif [ "$EVAL_ENGINE" == "sglang" ]; then
        pkill -f "python -m sglang.launch_server --model-path $MODEL_NAME --host $SERVER_HOST --dp $DATA_PARALLEL_SIZE --port $SERVER_PORT" || true
    fi
    sleep 2
}

# Cleanup on script exit
cleanup() {
    stop_server
    echo "Cleanup complete"
}
trap cleanup EXIT

start_server
for SAMPLE_SIZE in ${SAMPLE_SIZES[@]}; do
    RESPONSE_SIZE=$((TOTAL / SAMPLE_SIZE))
    TEMPERATURES=(0.3 0.7 1.0 1.5)
    for TEMPERATURE in ${TEMPERATURES[@]}; do
        OUTPUT_FILE="output/${MODEL_NAME}/responses_${SAMPLE_SIZE}_${TEMPERATURE}_sampling.jsonl"
        echo "Running $OUTPUT_FILE"
        python test.py \
            --model_name $MODEL \
            --format \
            --is_sampling \
            --num_samples $SAMPLE_SIZE \
            --num_responses $RESPONSE_SIZE \
            --temperature $TEMPERATURE \
            --use_vllm \
            --output_file $OUTPUT_FILE
    done
done

# TEMPERATURES=(0.3 0.7 1.0 1.5)
# for TEMPERATURE in ${TEMPERATURES[@]}; do
#     OUTPUT_FILE="output/${MODEL_NAME}/responses_${TEMPERATURE}_greedy.jsonl"
#     echo "Running $OUTPUT_FILE"
#     RESPONSE_SIZE=$((TOTAL / 1))
#     python test.py \
#         --model_name $MODEL \
#         --num_responses $RESPONSE_SIZE \
#         --temperature $TEMPERATURE \
#         --use_vllm \
#         --output_file $OUTPUT_FILE
# done

stop_server