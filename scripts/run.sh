# python scripts/tasks/run_bias_task.py
MODELS=(
    allenai/Llama-3.1-Tulu-3-70B-SFT
    allenai/Llama-3.1-Tulu-3-70B-DPO
    # allenai/Llama-3.1-Tulu-3-70B
    # allenai/Llama-3.1-Tulu-3-8B-SFT
    # allenai/Llama-3.1-Tulu-3-8B-DPO
    # allenai/Llama-3.1-Tulu-3-8B
)
EVAL_ENGINE="vllm"
CUDA_VISIBLE_DEVICES=0,1,2,3
SERVER_PORT=8000
TENSOR_PARALLEL_SIZE=4
SERVER_HOST=0.0.0.0
SERVER_STARTUP_TIMEOUT=1200
SERVER_URL="http://0.0.0.0:8000/v1"

# Function to start the inference server
start_server() {
    echo "Starting $EVAL_ENGINE server..."
    
    if [ "$EVAL_ENGINE" == "vllm" ]; then
        TENSOR_PARALLEL_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
        TENSOR_PARALLEL_SIZE=$((TENSOR_PARALLEL_SIZE + 1))
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES vllm serve $MODEL_NAME \
            --port $SERVER_PORT --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
            --dtype bfloat16 > vllm.log 2>&1 &
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
    echo "Waiting for server to start up (timeout: ${SERVER_STARTUP_TIMEOUT}s)..."
    elapsed=0
    while ! curl -s "$SERVER_URL/models" > /dev/null; do
        sleep 2
        elapsed=$((elapsed + 2))
        echo "Still waiting for server... (${elapsed}s elapsed)"
        
        # Check timeout
        if [ $elapsed -ge $SERVER_STARTUP_TIMEOUT ]; then
            echo "Error: Server startup timeout after ${SERVER_STARTUP_TIMEOUT} seconds"
            echo "Check the server logs for details:"
            if [ "$EVAL_ENGINE" == "vllm" ]; then
                echo "vLLM log: $(pwd)/vllm.log"
            elif [ "$EVAL_ENGINE" == "sglang" ]; then
                echo "SGLang log: $(pwd)/sglang.log"
            fi
            exit 1
        fi
        
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
        pkill -f "vllm serve $MODEL_NAME --port $SERVER_PORT" || true
    elif [ "$EVAL_ENGINE" == "sglang" ]; then
        pkill -f "python -m sglang.launch_server --model-path $MODEL_NAME --host $SERVER_HOST --dp $DATA_PARALLEL_SIZE --port $SERVER_PORT" || true
    fi
    sleep 2
}

for MODEL_NAME in ${MODELS[@]}; do
    # start_server
    python scripts/tasks/run_poem_test.py --model $MODEL_NAME
    # stop_server
done