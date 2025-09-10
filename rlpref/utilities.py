"""
Common utility functions shared across multiple modules.
"""
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import numpy as np

def format_prompt(prompt: str, response: str = None) -> str:
    """
    Format the prompt for the model.
    
    Args:
        prompt: The prompt text
        response: Optional response text to append
        
    Returns:
        A formatted prompt string
    """
    formatted = prompt
    if response:
        formatted += f" {response}"
    return formatted

def create_timestamp_dir(prefix: str = "results") -> str:
    """
    Create a timestamped directory for experiment results.
    
    Args:
        prefix: Prefix string for the directory name (default: 'results')
        
    Returns:
        Path to the created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = f"{prefix}_{timestamp}"
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def log_execution_time(start_time: float, description: str = "Operation") -> None:
    """
    Log the execution time of an operation.
    
    Args:
        start_time: Start time in seconds (from time.time())
        description: Description of the operation (default: 'Operation')
    """
    execution_time = time.time() - start_time
    print(f"{description} completed in {execution_time:.2f} seconds")

def prepare_for_serialization(data: Any) -> Any:
    """
    Prepare data for JSON serialization by converting non-serializable types.
    
    Args:
        data: Data to prepare for serialization
        
    Returns:
        Serializable version of the data
    """
    if isinstance(data, dict):
        return {k: prepare_for_serialization(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [prepare_for_serialization(item) for item in data]
    elif isinstance(data, tuple):
        return [prepare_for_serialization(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.number):
        return float(data) if isinstance(data, (np.floating, float)) else int(data)
    elif hasattr(data, 'item') and callable(data.item):
        # Handle PyTorch tensors and other objects with .item() method
        try:
            return data.item()
        except:
            return str(data)
    else:
        return data

def save_json_results(results: Dict[str, Any], filepath: str, indent: int = 2) -> None:
    """
    Save results to a JSON file, handling non-serializable types.
    
    Args:
        results: Dictionary of results to save
        filepath: Path to save the JSON file
        indent: JSON indentation level (default: 2)
    """
    try:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert non-serializable types
        serializable_results = prepare_for_serialization(results)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=indent)
        
        print(f"Results saved to {filepath}")
    except Exception as e:
        print(f"Error saving results to {filepath}: {e}")

def safe_file_operation(operation: Callable, filepath: str, *args, **kwargs) -> Optional[Any]:
    """
    Safely perform a file operation with proper error handling.
    
    Args:
        operation: Function to call (e.g., open, json.load)
        filepath: Path to the file
        *args: Additional positional arguments for the operation
        **kwargs: Additional keyword arguments for the operation
        
    Returns:
        Result of the operation, or None if it failed
    """
    try:
        if operation == open:
            with open(filepath, *args, **kwargs) as f:
                return f
        else:
            return operation(filepath, *args, **kwargs)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except PermissionError:
        print(f"Permission denied: {filepath}")
        return None
    except Exception as e:
        print(f"Error performing operation on {filepath}: {e}")
        return None