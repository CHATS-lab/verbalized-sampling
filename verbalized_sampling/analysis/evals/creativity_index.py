from typing import Dict, List, Any, Optional
import json
import time
import requests
import nltk
import numpy as np
import torch
import pickle
import os
import signal
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

@dataclass
class RefDocument:
    """Reference document for semantic matching."""
    token_ids: List[int]
    content_token_ids: List[int]
    content_token_indices: List[int]
    text: str

def timeout_handler(signum, frame):
    """Timeout handler for Earth Mover computation."""
    raise TimeoutError("Earth Mover computation timed out")

class CreativityIndexEvaluator(BaseEvaluator):
    """Evaluator for measuring creativity by analyzing overlap with pretraining data."""
    instance_plot_metrics = [
        ("creativity_index", "violin")
    ]
    aggregate_plot_metrics = [
        "avg_creativity_index",
        "avg_coverage",
        "match_rate"
    ]
    key_plot_metrics = [
        ("avg_creativity_index", "Creativity (Infini-gram)"),
    ]

    def __init__(self, 
                 method: str = "exact",  # "exact" or "semantic"
                 corpus: str = "v4_dolma-v1_7_llama",  # Infini-gram corpus
                 min_ngram: int = 5,
                 min_content_word: int = 2,  # For earth mover
                 threshold: float = 0.95,  # For semantic matching
                 embed_model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                 api_url: str = "https://api.infini-gram.io/",
                 embed_table_path: str = None,
                 reference_corpus_path: str = None,  # For semantic matching
                 timeout: int = 30,  # Timeout for earth mover computation
                 num_workers: int = 24):
        super().__init__("creativity_index", num_workers)
        self.method = method
        self.corpus = corpus
        self.min_ngram = min_ngram
        self.min_content_word = min_content_word
        self.threshold = threshold
        self.api_url = api_url
        self.embed_model = embed_model
        self.embed_table_path = embed_table_path
        self.reference_corpus_path = reference_corpus_path
        self.timeout = timeout
        
        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(embed_model, add_bos_token=False, add_eos_token=False)
        except:
            # Fallback to a public model if the specified one is not accessible
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", add_bos_token=False, add_eos_token=False)
        
        # Setup stop words and punctuation for semantic matching
        stop_words = stopwords.words('english') + ["'m", "'d", "'ll", "'o", "'re", "'ve", "'y"]
        self.stop_tokens = set([t for w in stop_words for t in self.tokenizer.tokenize(w)])
        self.punctuations = list(punctuation)
        
        # Load embedding table for semantic methods
        if method in ["semantic", "earth_mover"]:
            if embed_table_path and os.path.exists(embed_table_path):
                with open(embed_table_path, 'rb') as f:
                    self.sim_table = pickle.load(f)
                print(f"Loaded embedding similarity table from {embed_table_path}")
            else:
                print(f"Warning: No embedding table found at {embed_table_path}. Creating new one...")
                self.sim_table = None
                if embed_table_path:
                    self._create_embedding_table(embed_table_path)
        
        # Load reference corpus for semantic matching
        if method in ["semantic", "earth_mover"] and reference_corpus_path:
            self.reference_docs = self._load_reference_corpus(reference_corpus_path)
        else:
            self.reference_docs = []
    
    def _create_embedding_table(self, save_path: str):
        """Create embedding similarity table."""
        try:
            print(f"Creating embedding table for {self.embed_model}...")
            model = AutoModelForCausalLM.from_pretrained(self.embed_model)
            embed_table = model.get_input_embeddings().weight.to('cuda' if torch.cuda.is_available() else 'cpu')
            num_vocab = embed_table.shape[0]
            
            cos_sim = torch.nn.CosineSimilarity(dim=1)
            sim_table = torch.zeros((num_vocab, num_vocab))
            
            with torch.no_grad():
                for i in tqdm(range(num_vocab), desc="Computing similarities"):
                    word_embed = embed_table[i][None, :].expand(num_vocab, -1)
                    sim_score = cos_sim(word_embed, embed_table).cpu()
                    sim_table[i] = sim_score
            
            sim_table = sim_table.numpy()
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(sim_table, f)
            
            self.sim_table = sim_table
            print(f"Embedding table saved to {save_path}")
            
        except Exception as e:
            print(f"Error creating embedding table: {e}")
            self.sim_table = None
    
    def _load_reference_corpus(self, corpus_path: str) -> List[RefDocument]:
        """Load reference corpus for semantic matching."""
        if not os.path.exists(corpus_path):
            print(f"Warning: Reference corpus not found at {corpus_path}")
            return []
        
        reference_docs = []
        try:
            with open(corpus_path, 'r') as f:
                corpus_data = json.load(f)
            
            for doc_data in tqdm(corpus_data[:100], desc="Loading reference corpus"):  # Limit for demo
                text = doc_data.get('text', '')
                content_token_ids, content_token_indices, token_ids = self._convert_to_content_tokens(text, return_index=True)
                
                reference_docs.append(RefDocument(
                    token_ids=token_ids,
                    content_token_ids=content_token_ids,
                    content_token_indices=content_token_indices,
                    text=text
                ))
            
            print(f"Loaded {len(reference_docs)} reference documents")
            
        except Exception as e:
            print(f"Error loading reference corpus: {e}")
        
        return reference_docs
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text using casual tokenization."""
        return nltk.tokenize.casual.casual_tokenize(unidecode(text))
    
    def detokenize_text(self, tokens: List[str]) -> str:
        """Detokenize tokens back to text."""
        return md.detokenize(tokens)
    
    def _convert_to_content_tokens(self, text: str, return_index: bool = False):
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
        
        if return_index:
            return content_tokens, content_indices, token_ids
        return content_tokens
    
    def _compute_similarity(self, source_token_ids: List[int], target_token_ids: List[int]) -> float:
        """Compute semantic similarity between token sequences using embedding table."""
        if self.sim_table is None:
            return 0.0
        
        similarities = []
        for token_id in source_token_ids:
            max_sim = max([self.sim_table[token_id][t] for t in target_token_ids])
            similarities.append(max_sim)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
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
    
    def _find_matched_span_earth_mover(self, tgt_token_ids: List[int], ref_doc: RefDocument) -> Dict[str, float]:
        """Find matched spans using Earth Mover Distance."""
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.timeout)
        
        try:
            ref_to_tgt_scores = []
            for token_id in ref_doc.content_token_ids:
                ref_to_tgt_scores.append(max([self.sim_table[token_id][t] for t in tgt_token_ids]))
            
            # Find sub-arrays with mean >= threshold using cumulative sum
            cumulative_sum = [0] * (len(ref_to_tgt_scores) + 1)
            for i in range(1, len(ref_to_tgt_scores) + 1):
                cumulative_sum[i] = cumulative_sum[i-1] + ref_to_tgt_scores[i-1]
            
            matched_spans = {}
            for start in range(len(ref_to_tgt_scores)):
                for end in range(start + self.min_content_word, len(ref_to_tgt_scores) + 1):
                    subarray_sum = cumulative_sum[end] - cumulative_sum[start]
                    subarray_length = end - start
                    subarray_mean = subarray_sum / subarray_length
                    
                    if subarray_mean >= self.threshold:
                        cand_token_ids = ref_doc.content_token_ids[start:end]
                        tgt_to_ref_score = self._compute_similarity(tgt_token_ids, cand_token_ids)
                        
                        if tgt_to_ref_score >= self.threshold:
                            final_score = min(subarray_mean, tgt_to_ref_score)
                            start_idx = ref_doc.content_token_indices[start]
                            end_idx = ref_doc.content_token_indices[end] if end < len(ref_doc.content_token_ids) else len(ref_doc.token_ids)
                            ref_span_text = self.tokenizer.decode(ref_doc.token_ids[start_idx:end_idx])
                            matched_spans[ref_span_text] = final_score
            
            signal.alarm(0)
            return matched_spans
            
        except TimeoutError:
            signal.alarm(0)
            return {}
        except Exception as e:
            signal.alarm(0)
            print(f"Error in Earth Mover computation: {e}")
            return {}
    
    def find_earth_mover_matches(self, tokens: List[str]) -> List[MatchedSpan]:
        """Find semantic matches using Earth Mover Distance."""
        if self.sim_table is None or not self.reference_docs:
            return []
        
        matched_spans = []
        first_pointer, second_pointer = 0, self.min_ngram
        
        while second_pointer <= len(tokens):
            span_text = self.detokenize_text(tokens[first_pointer:second_pointer])
            span_token_ids = self._convert_to_content_tokens(span_text)
            
            if len(span_token_ids) < self.min_content_word:
                second_pointer += 1
                continue
            
            # Search across all reference documents
            all_matched_spans = {}
            for ref_doc in self.reference_docs:
                matched_spans_dict = self._find_matched_span_earth_mover(span_token_ids, ref_doc)
                all_matched_spans.update(matched_spans_dict)
            
            if all_matched_spans:
                # Take the best match
                best_ref_span = max(all_matched_spans.keys(), key=lambda k: all_matched_spans[k])
                best_score = all_matched_spans[best_ref_span]
                
                matched_span = MatchedSpan(
                    start_index=first_pointer,
                    end_index=second_pointer,
                    span_text=span_text,
                    ref_span_text=best_ref_span,
                    score=best_score
                )
                matched_spans.append(matched_span)
                second_pointer += 1
            else:
                if second_pointer - first_pointer > self.min_ngram:
                    first_pointer += 1
                elif second_pointer - first_pointer == self.min_ngram:
                    first_pointer += 1
                    second_pointer += 1
        
        return matched_spans
    
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
    
    def compute_instance_metric(self, prompt: str, response: Dict) -> Dict[str, Any]:
        """Compute creativity index for a single response."""
        tokens = self.tokenize_text(response['text'])
        
        if len(tokens) < self.min_ngram:
            return {
                "response": response['text'],
                "creativity_index": 1.0,  # High creativity for very short responses
                "coverage": 0.0,
                "matched_spans": [],
                "num_tokens": len(tokens),
                "avg_span_length": 0.0,
                "error": "Response too short for analysis"
            }
        
        # Choose matching method
        if self.method == "exact":
            matched_spans = self.find_exact_matches(tokens)
        elif self.method == "earth_mover":
            matched_spans = self.find_earth_mover_matches(tokens)
        else:  # semantic fallback
            matched_spans = []  # Could implement basic semantic matching here
        
        coverage = self.compute_coverage(tokens, matched_spans)
        creativity_index = 1.0 - coverage  # Creativity is inverse of coverage
        
        avg_span_length = np.mean([span.end_index - span.start_index for span in matched_spans]) if matched_spans else 0.0
        
        return {
            "response": response['text']    ,
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
            "avg_span_length": float(avg_span_length),
            "method": self.method
        }
    
    def aggregate_metrics(self, instance_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate creativity metrics across all responses."""
        if not instance_metrics:
            return {
                "avg_creativity_index": 0.0,
                "std_creativity_index": 0.0,
                "avg_coverage": 0.0,
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
            "avg_creativity_index": float(np.mean(creativity_scores)),
            "std_creativity_index": float(np.std(creativity_scores)),
            "avg_coverage": float(np.mean(coverage_scores)),
            "std_coverage": float(np.std(coverage_scores)),
            "min_creativity_index": float(np.min(creativity_scores)),
            "max_creativity_index": float(np.max(creativity_scores)),
            "total_responses": len(instance_metrics),
            "responses_with_matches": responses_with_matches,
            "match_rate": float(responses_with_matches / len(instance_metrics)),
            "avg_span_length": float(np.mean(span_lengths)) if span_lengths else 0.0,
            "method": self.method,
            "corpus": self.corpus,
            "min_ngram": self.min_ngram,
            "threshold": self.threshold if self.method != "exact" else None
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
            "num_responses": len(responses),
            "threshold": self.threshold if self.method != "exact" else None
        })
        
        return super().evaluate(prompts, responses, metadata)

# Utility function to create embedding table
def create_embedding_table(model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct", 
                          save_path: str = 'data/embed_distance/creativity_index_embeddings.pkl'):
    """Create and save embedding similarity table for semantic matching."""
    evaluator = CreativityIndexEvaluator(embed_model=model_name)
    evaluator._create_embedding_table(save_path)
    return save_path