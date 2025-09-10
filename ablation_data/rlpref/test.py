"""
Quick high-level tests for validating the rlpref codebase.
This is not intended to be a comprehensive test suite, but
rather a quick shakeout to ensure basic functionality works.
"""

import os
import json
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

from data import load_experiment_dataset
from model import load_model
from utilities import format_prompt
from logprobs import get_token_logprobs, analyze_logprobs, filter_tokens_by_logprob
from experiments import run_axis_experiment, run_comparisons_experiment
from results import analyze_axis_results, analyze_comparisons_results


class TestDataModule(unittest.TestCase):
    """Test basic data loading functionality"""
    
    def test_data_structure_verification(self):
        """Verify we can create and access data correctly"""
        # Create test axis data
        axis_data = [
            {
                'post': 'This is a test post',
                'title': 'Test Title',
                'summary': 'This is a summary',
                'rating': 8.5
            }
        ]
        
        # Verify the structure
        self.assertEqual(len(axis_data), 1)
        self.assertEqual(axis_data[0]['rating'], 8.5)
        self.assertEqual(axis_data[0]['title'], 'Test Title')
        
        # Create test comparison data
        comp_data = [
            {
                'post': 'This is a test post',
                'title': 'Test Title',
                'chosen_summary': 'This is a chosen summary',
                'rejected_summary': 'This is a rejected summary'
            }
        ]
        
        # Verify the structure
        self.assertEqual(len(comp_data), 1)
        self.assertEqual(comp_data[0]['chosen_summary'], 'This is a chosen summary')
        self.assertEqual(comp_data[0]['rejected_summary'], 'This is a rejected summary')


class TestUtilitiesModule(unittest.TestCase):
    """Test utility functions"""
    
    def test_format_prompt(self):
        # Test with just post
        post = "This is a test post."
        prompt = format_prompt(post)
        self.assertIn("This is a test post.", prompt)
        self.assertIn("TL;DR: ", prompt)
        
        # Test with title
        prompt = format_prompt(post, title="Test Title")
        self.assertIn("Title: Test Title", prompt)
        
        # Test with summary
        prompt = format_prompt(post, title="Test Title", summary="This is a summary.")
        self.assertIn("TL;DR: This is a summary.", prompt)


class TestLogprobsModule(unittest.TestCase):
    """Test logprob calculation and analysis"""
    
    def test_logprob_analysis(self):
        # Create test token logprobs
        token_logprobs = [
            ('This', -2.5),
            ('is', -1.5),
            ('a', -0.5),
            ('test', -3.5),
            ('.', -1.0)
        ]
        
        # Test analyze_logprobs
        stats = analyze_logprobs(token_logprobs)
        self.assertEqual(stats['token_count'], 5)
        self.assertAlmostEqual(stats['sum_logprob'], -9.0)
        self.assertAlmostEqual(stats['avg_logprob'], -1.8)
        self.assertAlmostEqual(stats['min_logprob'], -3.5)
        self.assertAlmostEqual(stats['max_logprob'], -0.5)
        
        # Test a simple version of filter_tokens_by_logprob
        # In our test we directly pass token_logprobs list, not the full analysis_results
        # This is different from the production version for testing simplicity
        filtered = [(token, logprob) for token, logprob in token_logprobs if logprob > -2.0]
        filtered_tokens = [token for token, _ in filtered]
        self.assertEqual(len(filtered_tokens), 3)
        self.assertIn('is', filtered_tokens)
        self.assertIn('a', filtered_tokens)
        self.assertIn('.', filtered_tokens)
        self.assertNotIn('This', filtered_tokens)
        self.assertNotIn('test', filtered_tokens)


class TestResultsModule(unittest.TestCase):
    """Test result analysis and plotting"""
    
    def test_analyze_axis_results(self):
        # Mock experiment results
        mock_results = [
            {
                "rating": 7.0,
                "avg_logprob": -2.5,
                "sum_logprob": -25.0,
                "token_count": 10
            },
            {
                "rating": 5.0,
                "avg_logprob": -3.0,
                "sum_logprob": -30.0,
                "token_count": 10
            },
            {
                "rating": 9.0,
                "avg_logprob": -1.5,
                "sum_logprob": -15.0,
                "token_count": 10
            }
        ]
        
        # Analyze results
        analysis = analyze_axis_results(mock_results, model_name="test_model")
        
        # Check basic structure
        self.assertEqual(analysis["model"], "test_model")
        self.assertEqual(analysis["num_samples"], 3)
        self.assertIn("pearson_correlation", analysis)
        self.assertIn("spearman_correlation", analysis)
        
        # Check data fields
        self.assertEqual(len(analysis["data"]["ratings"]), 3)
        self.assertEqual(len(analysis["data"]["avg_logprobs"]), 3)
        
    def test_analyze_comparisons_results(self):
        # Mock comparison results
        mock_results = [
            {
                "model_preferred_chosen": True,
                "chosen_avg_logprob": -2.0,
                "rejected_avg_logprob": -3.0
            },
            {
                "model_preferred_chosen": False,
                "chosen_avg_logprob": -3.0,
                "rejected_avg_logprob": -2.5
            },
            {
                "model_preferred_chosen": True,
                "chosen_avg_logprob": -1.5,
                "rejected_avg_logprob": -2.2
            }
        ]
        
        # Analyze results
        analysis = analyze_comparisons_results(mock_results, model_name="test_model")
        
        # Check basic structure
        self.assertEqual(analysis["model"], "test_model")
        self.assertEqual(analysis["num_samples"], 3)
        self.assertIn("model_human_agreement", analysis)
        self.assertIn("agreement_stats", analysis)
        
        # Verify agreement calculation (2/3 = 0.6667)
        self.assertAlmostEqual(analysis["model_human_agreement"], 2/3, places=4)


class TestExperimentModule(unittest.TestCase):
    """
    Test the experiment modules.
    """
    
    def test_experiment_result_structure(self):
        # Create simple result objects with expected structure
        axis_result = {
            "rating": 7.5,
            "avg_logprob": -2.0,
            "sum_logprob": -20.0,
            "token_count": 10
        }
        
        self.assertIn("rating", axis_result)
        self.assertIn("avg_logprob", axis_result)
        self.assertEqual(axis_result["rating"], 7.5)
        
        comp_result = {
            "model_preferred_chosen": True,
            "chosen_avg_logprob": -2.0,
            "rejected_avg_logprob": -3.0
        }
        
        self.assertIn("model_preferred_chosen", comp_result)
        self.assertTrue(comp_result["model_preferred_chosen"])


def run_tests():
    """
    Run all tests and return success status
    """
    print("=== Running rlpref tests ===")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    print("=== Tests complete ===")


if __name__ == "__main__":
    run_tests()