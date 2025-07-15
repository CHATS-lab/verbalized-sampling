#!/usr/bin/env python3
"""
Script to test base model queries locally with custom prompts.
Handles the issue where base models don't stop properly.
"""

import json
from typing import Dict, Any
from verbalized_sampling.llms.vllm import VLLMOpenAI

def test_base_model_completion(
    prompt: str,
    model_name: str = "your-base-model-name",
    base_url: str = "http://localhost:8000/v1",
    max_tokens: int = 100,
    temperature: float = 0.7,
    stop_tokens: list = None
):
    """
    Test base model completion with custom prompt and stopping conditions.
    
    Args:
        prompt: The input prompt
        model_name: Name of the model
        base_url: vLLM server URL
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        stop_tokens: List of stop tokens/sequences
    """
    
    # Default stop tokens for base models
    if stop_tokens is None:
        stop_tokens = ["\n\n", "User:", "Human:", "Assistant:", "<|endoftext|>", "</s>"]
    
    # Configuration for the model
    config = {
        "base_url": base_url,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop_tokens,
        "stream": False
    }
    
    print(f"Testing base model: {model_name}")
    print(f"Server URL: {base_url}")
    print(f"Prompt: {prompt}")
    print(f"Config: {json.dumps(config, indent=2)}")
    print("-" * 50)
    
    try:
        # Initialize the vLLM client
        llm = VLLMOpenAI(model_name=model_name, config=config)
        
        # Generate completion
        response = llm._complete(prompt)
        
        print(f"Response: {response}")
        print(f"Response length: {len(response)} characters")
        
        return response
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def interactive_test():
    """Interactive mode for testing different prompts."""
    print("Interactive Base Model Tester")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    # Default configuration - modify these as needed
    model_name = input("Enter model name (or press Enter for default): ").strip()
    if not model_name:
        model_name = "your-base-model-name"
    
    base_url = input("Enter base URL (or press Enter for localhost:8000): ").strip()
    if not base_url:
        base_url = "http://localhost:8000/v1"
    
    while True:
        print("\n" + "-" * 30)
        prompt = input("Enter prompt (or 'quit' to exit): ").strip()
        
        if prompt.lower() == 'quit':
            break
            
        if not prompt:
            continue
            
        # Optional: customize generation parameters
        max_tokens = input("Max tokens (default 100): ").strip()
        max_tokens = int(max_tokens) if max_tokens else 100
        
        temperature = input("Temperature (default 0.7): ").strip()
        temperature = float(temperature) if temperature else 0.7
        
        # Custom stop tokens
        stop_input = input("Stop tokens (comma-separated, or press Enter for defaults): ").strip()
        stop_tokens = [s.strip() for s in stop_input.split(",")] if stop_input else None
        
        # Test the model
        response = test_base_model_completion(
            prompt=prompt,
            model_name=model_name,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_tokens=stop_tokens
        )

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line mode
        prompt = " ".join(sys.argv[1:])
        test_base_model_completion(prompt)
    else:
        # Interactive mode
        interactive_test()