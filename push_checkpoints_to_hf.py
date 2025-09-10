#!/usr/bin/env python3
"""
Script to load model checkpoints from step directories and push them to Hugging Face Hub.
Usage: python push_checkpoints_to_hf.py --base_path /path/to/checkpoints --repo_prefix your-username/model-name
"""

import os
import argparse
from pathlib import Path
from typing import List, Optional
import logging
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import HfApi, login
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_checkpoint_dirs(base_path: str, pattern: str = "step_") -> List[Path]:
    """Find all checkpoint directories matching the pattern."""
    base_path = Path(base_path)
    checkpoint_dirs = []
    
    if not base_path.exists():
        logger.error(f"Base path {base_path} does not exist")
        return []
    
    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith(pattern):
            # Check if it contains model files
            if any(f.suffix in ['.bin', '.safetensors'] for f in item.glob('*')):
                checkpoint_dirs.append(item)
            elif (item / 'pytorch_model.bin').exists() or (item / 'model.safetensors').exists():
                checkpoint_dirs.append(item)
    
    # Sort by step number
    def extract_step_number(path: Path) -> int:
        try:
            return int(path.name.replace(pattern, ""))
        except ValueError:
            return 0
    
    checkpoint_dirs.sort(key=extract_step_number)
    return checkpoint_dirs

def load_model_from_checkpoint(checkpoint_path: Path, model_class=None):
    """Load model from checkpoint directory."""
    try:
        logger.info(f"Loading model from {checkpoint_path}")
        
        # Try to load config first
        config = None
        if (checkpoint_path / 'config.json').exists():
            config = AutoConfig.from_pretrained(checkpoint_path)
        
        # Load model
        if model_class:
            model = model_class.from_pretrained(checkpoint_path, config=config)
        else:
            model = AutoModel.from_pretrained(checkpoint_path, config=config)
        
        # Load tokenizer if available
        tokenizer = None
        if (checkpoint_path / 'tokenizer.json').exists() or (checkpoint_path / 'tokenizer_config.json').exists():
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        
        return model, tokenizer, config
    
    except Exception as e:
        logger.error(f"Failed to load model from {checkpoint_path}: {e}")
        return None, None, None

def push_to_huggingface(model, tokenizer, config, repo_name: str, step: int, 
                       commit_message: Optional[str] = None, private: bool = False):
    """Push model to Hugging Face Hub."""
    try:
        logger.info(f"Pushing model to {repo_name}")
        
        # Create repository if it doesn't exist
        api = HfApi()
        try:
            api.create_repo(repo_name, private=private, exist_ok=True)
        except Exception as e:
            logger.warning(f"Repository creation warning: {e}")
        
        # Push model
        if model:
            model.push_to_hub(
                repo_name,
                commit_message=commit_message or f"Add model checkpoint at step {step}",
                private=private
            )
        
        # Push tokenizer
        if tokenizer:
            tokenizer.push_to_hub(
                repo_name,
                commit_message=commit_message or f"Add tokenizer at step {step}",
                private=private
            )
        
        # Push config
        if config:
            config.push_to_hub(
                repo_name,
                commit_message=commit_message or f"Add config at step {step}",
                private=private
            )
        
        logger.info(f"Successfully pushed step {step} to {repo_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to push step {step} to {repo_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Push model checkpoints to Hugging Face Hub")
    parser.add_argument("--base_path", type=str, required=True,
                       help="Base path containing checkpoint directories")
    parser.add_argument("--repo_prefix", type=str, required=True,
                       help="Repository name prefix (e.g., 'username/model-name')")
    parser.add_argument("--pattern", type=str, default="step_",
                       help="Pattern to match checkpoint directories (default: 'step_')")
    parser.add_argument("--steps", type=str, default=None,
                       help="Comma-separated list of specific steps to process (e.g., '50,100,150')")
    parser.add_argument("--private", action="store_true",
                       help="Make repositories private")
    parser.add_argument("--model_class", type=str, default=None,
                       help="Specific model class to use (e.g., 'AutoModelForCausalLM')")
    parser.add_argument("--dry_run", action="store_true",
                       help="Perform a dry run without actually pushing to HF")
    
    args = parser.parse_args()
    
    # Login to Hugging Face
    if not args.dry_run:
        try:
            login()
            logger.info("Successfully logged in to Hugging Face")
        except Exception as e:
            logger.error(f"Failed to login to Hugging Face: {e}")
            return
    
    # Find checkpoint directories
    checkpoint_dirs = find_checkpoint_dirs(args.base_path, args.pattern)
    
    if not checkpoint_dirs:
        logger.error(f"No checkpoint directories found in {args.base_path} with pattern '{args.pattern}'")
        return
    
    logger.info(f"Found {len(checkpoint_dirs)} checkpoint directories")
    
    # Filter by specific steps if provided
    if args.steps:
        requested_steps = set(int(s.strip()) for s in args.steps.split(','))
        checkpoint_dirs = [
            d for d in checkpoint_dirs 
            if int(d.name.replace(args.pattern, "")) in requested_steps
        ]
        logger.info(f"Filtered to {len(checkpoint_dirs)} requested steps")
    
    # Determine model class
    model_class = None
    if args.model_class:
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
        model_classes = {
            'AutoModelForCausalLM': AutoModelForCausalLM,
            'AutoModelForSeq2SeqLM': AutoModelForSeq2SeqLM,
            'AutoModelForSequenceClassification': AutoModelForSequenceClassification,
        }
        model_class = model_classes.get(args.model_class, AutoModel)
    
    # Process each checkpoint
    successful_pushes = 0
    for checkpoint_dir in checkpoint_dirs:
        step_num = int(checkpoint_dir.name.replace(args.pattern, ""))
        repo_name = f"{args.repo_prefix}-step-{step_num}"
        
        logger.info(f"Processing {checkpoint_dir.name} -> {repo_name}")
        
        # Load model
        model, tokenizer, config = load_model_from_checkpoint(checkpoint_dir, model_class)
        
        if model is None:
            logger.warning(f"Skipping {checkpoint_dir.name} - failed to load model")
            continue
        
        if args.dry_run:
            logger.info(f"[DRY RUN] Would push {checkpoint_dir.name} to {repo_name}")
            successful_pushes += 1
        else:
            # Push to Hugging Face
            if push_to_huggingface(model, tokenizer, config, repo_name, step_num, private=args.private):
                successful_pushes += 1
        
        # Clean up memory
        del model
        if tokenizer:
            del tokenizer
        if config:
            del config
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    logger.info(f"Successfully processed {successful_pushes}/{len(checkpoint_dirs)} checkpoints")

if __name__ == "__main__":
    main() 