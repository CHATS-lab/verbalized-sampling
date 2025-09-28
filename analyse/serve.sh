vllm serve zai-org/GLM-4.5-Air-Base \
    --tensor-parallel-size 8 \
    --port 8000 \
    --host 0.0.0.0