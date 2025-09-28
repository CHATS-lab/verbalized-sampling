#!/usr/bin/env python3
"""
Simple test script for testing a single Qwen model on math tasks.
Run this for each model individually.
"""

import subprocess
import sys
from pathlib import Path

def run_test(model_name: str, datasets=None, num_samples=20, port=8000):
    """Run test for a single model."""
    if datasets is None:
        datasets = ["math", "amc"]  # Start with smaller datasets

    print(f"🚀 Testing {model_name}")
    print(f"📊 Datasets: {datasets}")
    print(f"🔢 Samples per dataset: {num_samples}")
    print(f"🌐 Expected vLLM server: http://localhost:{port}")
    print()

    # Check if vLLM server is running
    try:
        import requests
        response = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ vLLM server is running")
            if 'data' in models and models['data']:
                served_model = models['data'][0].get('id', 'unknown')
                print(f"📦 Currently serving: {served_model}")
                if model_name not in served_model:
                    print(f"⚠️  WARNING: Requested {model_name} but server has {served_model}")
            print()
        else:
            print(f"❌ vLLM server responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to vLLM server at localhost:{port}")
        print(f"   Error: {e}")
        print()
        print("To start vLLM server:")
        print(f"   vllm serve {model_name} --port {port}")
        return False

    # Run the test
    cmd = [
        "python", "test_math.py",
        "--models", model_name,
        "--datasets"] + datasets + [
        "--num_samples", str(num_samples),
        "--base_url", f"http://localhost:{port}",
        "--output", f"results_{model_name.replace('/', '_')}"
    ]

    print(f"🔄 Running: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ Test completed successfully for {model_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Test failed for {model_name}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Test interrupted for {model_name}")
        return False

def main():
    """Main function with example usage."""
    print("🧮 Math Model Testing - Single Model Runner")
    print("=" * 50)

    # Test configurations
    tests = [
        {
            "model": "Qwen/Qwen3-4B-Base",
            "datasets": ["math", "amc"],  # Start small
            "samples": 10,  # Small number for quick test
            "port": 8000
        },
        {
            "model": "Qwen/Qwen3-4B-Thinking-2507",
            "datasets": ["math", "amc"],
            "samples": 10,
            "port": 8000  # Same port - requires restart
        }
    ]

    print("Available test configurations:")
    for i, test in enumerate(tests):
        print(f"  {i+1}. {test['model']} on {test['datasets']} ({test['samples']} samples)")

    print("\nChoose option:")
    print("  1-{}: Run specific test".format(len(tests)))
    print("  0: Custom test")
    print("  q: Quit")

    choice = input("\nEnter choice: ").strip()

    if choice.lower() == 'q':
        return 0

    try:
        if choice == '0':
            # Custom test
            model = input("Model name: ").strip()
            if not model:
                model = "Qwen/Qwen3-4B-Base"

            datasets_input = input("Datasets (comma-separated, e.g., math,amc): ").strip()
            if datasets_input:
                datasets = [d.strip() for d in datasets_input.split(',')]
            else:
                datasets = ["math"]

            samples_input = input("Number of samples per dataset (default 10): ").strip()
            samples = int(samples_input) if samples_input else 10

            port_input = input("vLLM port (default 8000): ").strip()
            port = int(port_input) if port_input else 8000

        else:
            # Predefined test
            test_idx = int(choice) - 1
            if test_idx < 0 or test_idx >= len(tests):
                print("Invalid choice")
                return 1

            test_config = tests[test_idx]
            model = test_config["model"]
            datasets = test_config["datasets"]
            samples = test_config["samples"]
            port = test_config["port"]

        # Run the test
        success = run_test(model, datasets, samples, port)
        return 0 if success else 1

    except ValueError:
        print("Invalid input")
        return 1
    except KeyboardInterrupt:
        print("\nAborted by user")
        return 1

if __name__ == "__main__":
    exit(main())