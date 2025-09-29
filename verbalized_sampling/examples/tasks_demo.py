#!/usr/bin/env python3
"""
Demo script for the new tasks (book, poem, speech) in the verbalized sampling framework.

This script demonstrates:
1. How to use num_prompts and random_seed for reproducible sampling
2. How to iterate through multiple prompts from datasets
3. How to use different methods (DIRECT, SEQUENCE, STRUCTURE, etc.)
4. How to access task metadata
"""

from verbalized_sampling.tasks import get_task, Task
from verbalized_sampling.methods import Method

def demo_book_task():
    """Demo the BookTask with different configurations."""
    print("=" * 60)
    print("BOOK TASK DEMO")
    print("=" * 60)
    
    # Create task with specific sample size and seed
    book_task = get_task(Task.BOOK, num_prompts=3, random_seed=42)
    
    print(f"Task metadata: {book_task.get_metadata()}")
    print(f"Total prompts loaded: {len(book_task.get_prompts())}")
    print()
    
    # Demo different methods with the first prompt
    methods = [Method.DIRECT, Method.SEQUENCE, Method.STRUCTURE]
    
    for i, method in enumerate(methods):
        print(f"Method {i+1}: {method}")
        try:
            prompt = book_task.get_prompt(method, num_samples=3, prompt_index=0)
            print(f"Prompt preview: {prompt[:100]}...")
            print()
        except Exception as e:
            print(f"Error: {e}")
            print()
    
    # Show all sampled prompts (first 100 chars each)
    print("Sampled prompts:")
    for i, prompt in enumerate(book_task.get_prompts()):
        print(f"  {i+1}: {prompt[:80]}...")
    print()

def demo_poem_task():
    """Demo the PoemTask with different configurations."""
    print("=" * 60)
    print("POEM TASK DEMO")
    print("=" * 60)
    
    # Create task with different sample size and seed
    poem_task = get_task(Task.POEM, num_prompts=5, random_seed=123)
    
    print(f"Task metadata: {poem_task.get_metadata()}")
    print(f"Total prompts loaded: {len(poem_task.get_prompts())}")
    print()
    
    # Demo with structure with probability method
    try:
        prompt = poem_task.get_prompt(Method.VS_STANDARD, num_samples=4, prompt_index=2)
        print(f"STRUCTURE_WITH_PROB method prompt:")
        print(f"{prompt[:200]}...")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()
    
    # Show sample of prompts
    print("Sample prompts:")
    for i, prompt in enumerate(poem_task.get_prompts()[:3]):
        # Extract just the starting line for display
        if prompt.startswith("Please write a poem starting with the following line: "):
            starting_line = prompt[len("Please write a poem starting with the following line: "):]
            print(f"  {i+1}: {starting_line}")
        else:
            print(f"  {i+1}: {prompt[:60]}...")
    print()

def demo_speech_task():
    """Demo the SpeechTask with different configurations."""
    print("=" * 60)
    print("SPEECH TASK DEMO")
    print("=" * 60)
    
    # Create task with larger sample size
    speech_task = get_task(Task.SPEECH, num_prompts=7, random_seed=456)
    
    print(f"Task metadata: {speech_task.get_metadata()}")
    print(f"Total prompts loaded: {len(speech_task.get_prompts())}")
    print()
    
    # Demo iterating through multiple prompts
    print("Testing multiple prompts with DIRECT method:")
    for i in range(min(3, len(speech_task.get_prompts()))):
        try:
            prompt = speech_task.get_prompt(Method.DIRECT, prompt_index=i)
            # Extract just the starting sentence for display
            if prompt.startswith("Please write a speech starting with the following sentence: "):
                starting_sentence = prompt[len("Please write a speech starting with the following sentence: "):]
                print(f"  Prompt {i+1}: {starting_sentence}")
            else:
                print(f"  Prompt {i+1}: {prompt[:80]}...")
        except Exception as e:
            print(f"  Prompt {i+1}: Error - {e}")
    print()

def demo_reproducibility():
    """Demo the reproducibility feature with random seeds."""
    print("=" * 60)
    print("REPRODUCIBILITY DEMO")
    print("=" * 60)
    
    print("Creating two BookTasks with the same random seed:")
    
    # Create two tasks with same seed
    task1 = get_task(Task.BOOK, num_prompts=3, random_seed=999)
    task2 = get_task(Task.BOOK, num_prompts=3, random_seed=999)
    
    prompts1 = task1.get_prompts()
    prompts2 = task2.get_prompts()
    
    print(f"Task 1 prompts: {len(prompts1)}")
    print(f"Task 2 prompts: {len(prompts2)}")
    print(f"Prompts are identical: {prompts1 == prompts2}")
    print()
    
    print("Creating two BookTasks with different random seeds:")
    task3 = get_task(Task.BOOK, num_prompts=3, random_seed=111)
    task4 = get_task(Task.BOOK, num_prompts=3, random_seed=222)
    
    prompts3 = task3.get_prompts()
    prompts4 = task4.get_prompts()
    
    print(f"Task 3 prompts: {len(prompts3)}")
    print(f"Task 4 prompts: {len(prompts4)}")
    print(f"Prompts are identical: {prompts3 == prompts4}")
    print()

def demo_error_handling():
    """Demo error handling for edge cases."""
    print("=" * 60)
    print("ERROR HANDLING DEMO")
    print("=" * 60)
    
    book_task = get_task(Task.BOOK, num_prompts=2, random_seed=42)
    
    # Test index out of range
    try:
        prompt = book_task.get_prompt(Method.DIRECT, prompt_index=10)
    except ValueError as e:
        print(f"Expected error for out-of-range index: {e}")
    
    # Test sample size larger than available prompts
    print("\nTesting with num_prompts larger than available prompts:")
    large_task = get_task(Task.POEM, num_prompts=10000, random_seed=42)
    print(f"Requested 10000 prompts, got: {len(large_task.get_prompts())}")
    
    print()

def demo_data_statistics():
    """Show statistics about the data files."""
    print("=" * 60)
    print("DATA STATISTICS")
    print("=" * 60)
    
    tasks = [
        (Task.BOOK, "Book/Novel prompts"),
        (Task.POEM, "Poetry prompts"),
        (Task.SPEECH, "Speech prompts")
    ]
    
    for task_name, description in tasks:
        # Load all prompts to get total count
        task = get_task(task_name, num_prompts=0)  # num_prompts=0 loads all
        metadata = task.get_metadata()
        prompts = task.get_prompts()
        
        print(f"{description}:")
        print(f"  Total available: {len(prompts)}")
        print(f"  Sample prompt: {prompts[0][:80]}..." if prompts else "  No prompts available")
        print()

def main():
    """Run all demos."""
    print("VERBALIZED SAMPLING TASKS DEMO")
    print("=" * 80)
    
    try:
        demo_data_statistics()
    except Exception as e:
        print(f"Data statistics demo failed: {e}")
    
    try:
        demo_book_task()
    except Exception as e:
        print(f"Book task demo failed: {e}")
    
    try:
        demo_poem_task()
    except Exception as e:
        print(f"Poem task demo failed: {e}")
    
    try:
        demo_speech_task()
    except Exception as e:
        print(f"Speech task demo failed: {e}")
    
    try:
        demo_reproducibility()
    except Exception as e:
        print(f"Reproducibility demo failed: {e}")
    
    try:
        demo_error_handling()
    except Exception as e:
        print(f"Error handling demo failed: {e}")
    
    print("=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nUsage examples:")
    print("1. Basic usage:")
    print("   task = get_task(Task.BOOK, num_prompts=5, random_seed=42)")
    print("   prompt = task.get_prompt(Method.DIRECT, prompt_index=0)")
    print()
    print("2. With structured output:")
    print("   prompt = task.get_prompt(Method.VS_STANDARD, num_samples=3)")
    print()
    print("3. Iterate through all prompts:")
    print("   for i, _ in enumerate(task.get_prompts()):")
    print("       prompt = task.get_prompt(method, prompt_index=i)")

if __name__ == "__main__":
    main() 