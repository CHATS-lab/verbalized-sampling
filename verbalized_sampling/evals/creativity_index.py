from typing import Dict, List, Any, Optional
import json
import time
import requests
import nltk
import numpy as np
import torch
import pickle
import os
from tqdm import tqdm
from dataclasses import dataclass
from unidecode import unidecode
from sacremoses import MosesDetokenizer
from nltk.corpus import stopwords
from string import punctuation
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base import BaseEvaluator, EvalResult
from verbalized_sampling.llms import get_model

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

md = MosesDetokenizer(lang='en')

@dataclass
class MatchedSpan:
    start_index: int
    end_index: int
    span_text: str
    ref_span_text: str = None  # For semantic matches
    score: float = 1.0  # 1.0 for exact matches, similarity score for semantic matches
    occurrence: int = 0  # For exact matches

class CreativityIndexEvaluator(BaseEvaluator):
    """Evaluator for measuring creativity by analyzing overlap with pretraining data."""
    
    def __init__(self, 
                 method: str = "exact",  # "exact" or "semantic"
                 corpus: str = "v4_dolma-v1_7_llama",  # Infini-gram corpus
                 min_ngram: int = 5,
                 threshold: float = 0.95,  # For semantic matching
                 embed_model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                 api_url: str = "https://api.infini-gram.io/",
                 embed_table_path: str = None,
                 num_workers: int = 128):
        super().__init__("creativity_index", num_workers)
        self.method = method
        self.corpus = corpus
        self.min_ngram = min_ngram
        self.threshold = threshold
        self.api_url = api_url
        self.embed_model = embed_model
        self.embed_table_path = embed_table_path
        
        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(embed_model, add_bos_token=False, add_eos_token=False)
        except:
            # Fallback to a public model if the specified one is not accessible
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", add_bos_token=False, add_eos_token=False)
        
        # For semantic matching
        if method == "semantic":
            if embed_table_path and os.path.exists(embed_table_path):
                with open(embed_table_path, 'rb') as f:
                    self.sim_table = pickle.load(f)
            else:
                print("Warning: No embedding table provided for semantic matching. Will use exact matching instead.")
                self.method = "exact"
        
        # Setup stop words and punctuation for semantic matching
        if method == "semantic":
            stop_words = stopwords.words('english') + ["'m", "'d", "'ll", "'o", "'re", "'ve", "'y"]
            self.stop_tokens = set([t for w in stop_words for t in self.tokenizer.tokenize(w)])
            self.punctuations = list(punctuation)
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text using casual tokenization."""
        return nltk.tokenize.casual.casual_tokenize(unidecode(text))
    
    def detokenize_text(self, tokens: List[str]) -> str:
        """Detokenize tokens back to text."""
        return md.detokenize(tokens)
    
    def query_infini_gram(self, query_text: str) -> int:
        """Query the Infini-gram API for exact matches."""
        request_data = {
            'corpus': self.corpus,
            'engine': 'c++',
            'query_type': 'count',
            'query': query_text,
        }
        
        try:
            time.sleep(0.1)  # Rate limiting
            response = requests.post(self.api_url, json=request_data, timeout=10)
            result = response.json()
            return result.get('count', 0)
        except Exception as e:
            print(f"Error querying Infini-gram: {e}")
            return 0
    
    def find_exact_matches(self, tokens: List[str]) -> List[MatchedSpan]:
        """Find exact n-gram matches using Infini-gram API."""
        matched_spans = []
        first_pointer, second_pointer = 0, self.min_ngram
        
        while second_pointer <= len(tokens):
            span_text = self.detokenize_text(tokens[first_pointer:second_pointer])
            occurrence = self.query_infini_gram(span_text)
            
            if occurrence > 0:
                matched_span = MatchedSpan(
                    start_index=first_pointer,
                    end_index=second_pointer,
                    span_text=span_text,
                    occurrence=occurrence
                )
                
                # Merge overlapping spans or extend existing ones
                if matched_spans and matched_span.start_index <= matched_spans[-1].start_index and matched_spans[-1].end_index <= matched_span.end_index:
                    matched_spans[-1] = matched_span
                else:
                    matched_spans.append(matched_span)
                
                second_pointer += 1
            else:
                if second_pointer - first_pointer > self.min_ngram:
                    first_pointer += 1
                elif second_pointer - first_pointer == self.min_ngram:
                    first_pointer += 1
                    second_pointer += 1
        
        return matched_spans
    
    def convert_to_content_tokens(self, text: str):
        """Convert text to content tokens (removing stop words and punctuation)."""
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        content_tokens, content_indices = [], []
        
        for i, token in enumerate(tokens):
            clean_token = token.replace('Ġ', '').lower()
            is_stopword = clean_token in self.stop_tokens
            is_punct = all(c in self.punctuations for c in clean_token)
            
            if not is_stopword and not is_punct:
                content_tokens.append(token_ids[i])
                content_indices.append(i)
        
        return content_tokens, content_indices, token_ids
    
    def compute_semantic_similarity(self, source_token_ids: List[int], target_token_ids: List[int]) -> float:
        """Compute semantic similarity between token sequences using embedding table."""
        if not hasattr(self, 'sim_table'):
            return 0.0
        
        similarities = []
        for token_id in source_token_ids:
            max_sim = max([self.sim_table[token_id][t] for t in target_token_ids])
            similarities.append(max_sim)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def find_semantic_matches(self, tokens: List[str], reference_texts: List[str] = None) -> List[MatchedSpan]:
        """Find semantic matches using Earth Mover Distance. This is a simplified version."""
        if not hasattr(self, 'sim_table'):
            return []
        
        # For now, return empty list since we need reference documents
        # In a full implementation, this would require retrieved reference documents
        return []
    
    def compute_coverage(self, tokens: List[str], matched_spans: List[MatchedSpan]) -> float:
        """Compute coverage score (percentage of tokens covered by matches)."""
        if not matched_spans:
            return 0.0
        
        covered_flags = [False] * len(tokens)
        for span in matched_spans:
            for i in range(span.start_index, min(span.end_index, len(tokens))):
                covered_flags[i] = True
        
        coverage = sum(covered_flags) / len(covered_flags)
        return coverage
    
    def compute_instance_metric(self, prompt: str, response: str) -> Dict[str, Any]:
        """Compute creativity index for a single response."""
        tokens = self.tokenize_text(response)
        
        if len(tokens) < self.min_ngram:
            return {
                "response": response,
                "creativity_index": 1.0,  # High creativity for very short responses
                "coverage": 0.0,
                "matched_spans": [],
                "num_tokens": len(tokens),
                "avg_span_length": 0.0,
                "error": "Response too short for analysis"
            }
        
        if self.method == "exact":
            matched_spans = self.find_exact_matches(tokens)
        else:  # semantic
            matched_spans = self.find_semantic_matches(tokens)
        
        coverage = self.compute_coverage(tokens, matched_spans)
        creativity_index = 1.0 - coverage  # Creativity is inverse of coverage
        
        avg_span_length = np.mean([span.end_index - span.start_index for span in matched_spans]) if matched_spans else 0.0
        
        return {
            "response": response,
            "creativity_index": float(creativity_index),
            "coverage": float(coverage),
            "matched_spans": [
                {
                    "start_index": span.start_index,
                    "end_index": span.end_index,
                    "span_text": span.span_text,
                    "ref_span_text": span.ref_span_text,
                    "score": float(span.score),
                    "occurrence": span.occurrence
                }
                for span in matched_spans
            ],
            "num_tokens": len(tokens),
            "avg_span_length": float(avg_span_length)
        }
    
    def aggregate_metrics(self, instance_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate creativity metrics across all responses."""
        if not instance_metrics:
            return {
                "average_creativity_index": 0.0,
                "std_creativity_index": 0.0,
                "average_coverage": 0.0,
                "std_coverage": 0.0,
                "total_responses": 0,
                "responses_with_matches": 0,
                "average_span_length": 0.0
            }
        
        creativity_scores = [m["creativity_index"] for m in instance_metrics]
        coverage_scores = [m["coverage"] for m in instance_metrics]
        span_lengths = [m["avg_span_length"] for m in instance_metrics if m["avg_span_length"] > 0]
        responses_with_matches = sum(1 for m in instance_metrics if m["matched_spans"])
        
        return {
            "average_creativity_index": float(np.mean(creativity_scores)),
            "std_creativity_index": float(np.std(creativity_scores)),
            "average_coverage": float(np.mean(coverage_scores)),
            "std_coverage": float(np.std(coverage_scores)),
            "min_creativity_index": float(np.min(creativity_scores)),
            "max_creativity_index": float(np.max(creativity_scores)),
            "total_responses": len(instance_metrics),
            "responses_with_matches": responses_with_matches,
            "match_rate": float(responses_with_matches / len(instance_metrics)),
            "average_span_length": float(np.mean(span_lengths)) if span_lengths else 0.0,
            "method": self.method,
            "corpus": self.corpus,
            "min_ngram": self.min_ngram
        }
    
    def evaluate(self, prompts: List[str], responses: List[str], 
                metadata: Optional[Dict[str, Any]] = None) -> EvalResult:
        """Evaluate creativity index for responses."""
        if metadata is None:
            metadata = {}
            
        metadata.update({
            "evaluation_method": "creativity_index",
            "matching_method": self.method,
            "corpus": self.corpus,
            "min_ngram": self.min_ngram,
            "num_responses": len(responses)
        })
        
        return super().evaluate(prompts, responses, metadata)

# Utility function to create embedding table (adapted from their code)
def create_embedding_table(model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct", 
                          save_path: str = 'data/embed_distance/creativity_index_embeddings.pkl'):
    """Create and save embedding similarity table for semantic matching."""
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        embed_table = model.get_input_embeddings().weight.to('cuda' if torch.cuda.is_available() else 'cpu')
        num_vocab = embed_table.shape[0]
        
        cos_sim = torch.nn.CosineSimilarity(dim=1)
        sim_table = torch.zeros((num_vocab, num_vocab))
        
        print(f"Computing similarity table for {num_vocab} vocabulary items...")
        
        with torch.no_grad():
            for i in tqdm(range(num_vocab)):
                word_embed = embed_table[i][None, :].expand(num_vocab, -1)
                sim_score = cos_sim(word_embed, embed_table).cpu()
                sim_table[i] = sim_score
        
        sim_table = sim_table.numpy()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(sim_table, f)
        
        print(f"Embedding table saved to {save_path}")
        return save_path
        
    except Exception as e:
        print(f"Error creating embedding table: {e}")
        return None
