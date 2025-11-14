# Copyright 2025 CHATS-Lab. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Response parsers for different sampling methods.
"""

import ast
import json
import re
from typing import Any, Dict, List, Union

from .factory import Method


def maybe_rename_response(response: Dict) -> Dict:
    if "response" in response:
        response["text"] = response["response"]
        del response["response"]
    if "confidence" in response:
        response["probability"] = response["confidence"]
        del response["confidence"]
    if "nll" in response:
        response["probability"] = response["nll"]
        del response["nll"]
    if "perplexity" in response:
        response["probability"] = response["perplexity"]
        del response["perplexity"]
    
    return response

class ParseError(Exception):
    """Exception raised when parsing fails."""

class ResponseParser:
    """Utility class for parsing responses from different sampling methods."""
    
    @staticmethod
    def parse_response(method: Method, response: str) -> List[Dict]:
        """Parse response based on the sampling method used."""

        match method:
            case Method.DIRECT:
                return ResponseParser.parse_direct(response)
            case Method.DIRECT_BASE:
                return ResponseParser.parse_direct(response)
            case Method.DIRECT_COT:
                return ResponseParser.parse_direct_cot(response)
            case Method.SEQUENCE:
                return ResponseParser.parse_sequence(response)
            case Method.STRUCTURE:
                return ResponseParser.parse_structure_response_only(response)
            case Method.VS_STANDARD:
                return ResponseParser.parse_structure_with_probability(response)
            case Method.MULTI_TURN:
                return ResponseParser.parse_multi_turn(response)
            case Method.VS_COT:
                return ResponseParser.parse_chain_of_thought(response)
            case Method.VS_MULTI:
                return ResponseParser.parse_combined(response)
            case Method.SEQUENCE_COT:
                return ResponseParser.parse_sequence_cot(response)
            case Method.SEQUENCE_MULTI:
                return ResponseParser.parse_sequence(response)
            case _:
                raise ValueError(f"Unknown parsing method: {method}")
    
    @staticmethod
    def parse_direct(response: str) -> List[Dict]:
        """Parse direct response - just return as-is."""
        return [{'text': response}]
    
    @staticmethod
    def parse_direct_cot(response: str) -> List[Dict]:
        """Parse direct response with chain-of-thought."""
        if isinstance(response, list):
            return [maybe_rename_response(item) for item in response]
        elif isinstance(response, dict):
            return [maybe_rename_response(response)]
        else:
           try:
                parsed = ResponseParser._extract_json(response)
                return [{'text': parsed["response"]}]
           except Exception:
                return [{'text': response}]
    
    @staticmethod
    def parse_sequence(response: str) -> List[Dict]:
        """Parse sequence response expecting JSON format."""
        # print('TYPE OF RESPONSE: ', type(response))
        # print('RESPONSE: ', response)

        if isinstance(response, list):
            return [{'text': item['response'] if 'response' in item else item} for item in response]
        elif isinstance(response, dict):
            response_list = response["responses"]
            return [{"text": item} for item in response_list]
            # return [{'text': item['text'] if 'text' in item else item} for item in response_list]
        else:
            try:
                # Try JSON parsing first (new format)
                print(f"Parsed: {parsed}")
                if isinstance(parsed, dict) and "responses" in parsed:
                    return [{'text': item} for item in parsed["responses"]]
                else:
                    return [{'text': response}]
            except Exception:
                # Fallback: try old Python list format for backward compatibility
                try:
                    parsed = ast.literal_eval(response)
                    if isinstance(parsed, list):
                        return [{'text': item} for item in parsed if item is not None]
                    else:
                        return [{'text': str(parsed)}]
                except Exception:
                    # Final fallback: treat as a single string
                    return [{'text': response}]
    
    @staticmethod
    def parse_structure_response_only(response: str) -> List[Dict]:
        """Parse structured response with response field only."""
        return ResponseParser.parse_structure_with_probability(response)

    @staticmethod
    def parse_chain_of_thought(response: str) -> List[Dict]:
        """Parse chain-of-thought response."""
        return ResponseParser.parse_structure_with_probability(response)

    @staticmethod
    def parse_combined(response: str) -> List[Dict]:
        """Parse VS-Multi (vs_multi) response."""
        return ResponseParser.parse_structure_with_probability(response)
    
    @staticmethod
    def parse_sequence_cot(response: str) -> List[Dict]:
        """Parse sequence with chain-of-thought response (reasoning + list of responses)."""
        # Extract responses from dict format (with reasoning field)
        if isinstance(response, dict):
            if "responses" in response:
                # Extract the responses list and parse it as a sequence
                responses_list = response["responses"]
                return ResponseParser.parse_sequence(responses_list)
            else:
                # Fallback: treat as single response
                return [{'text': str(response)}]
        
        # For string or other types, extract JSON first
        if not isinstance(response, (dict, list)):
            try:
                parsed = ResponseParser._extract_json(response)
                # Recursively parse the extracted JSON
                return ResponseParser.parse_sequence_cot(parsed)
            except Exception:
                # Fallback: try Python list format
                try:
                    parsed = ast.literal_eval(response)
                    return ResponseParser.parse_sequence_cot(parsed)
                except Exception:
                    return [{'text': response}]
        
        # If it's already a list, parse as sequence
        if isinstance(response, list):
            return ResponseParser.parse_sequence(response)
        
        return [{'text': str(response)}]
    
    @staticmethod
    def parse_structure_with_probability(response: str) -> List[Dict[str, Union[str, float]]]:
        """Parse structured response with response and probability fields."""
        if isinstance(response, list):
            return [maybe_rename_response(item) for item in response]
        elif isinstance(response, dict):
            response_list = response["responses"]
            return [maybe_rename_response(item) for item in response_list]
        else:
            # non-strict json
            try:
                response = ResponseParser._extract_json(response)
                return [maybe_rename_response(item) for item in response['responses']]
            except Exception:
                # Fallback: treat as a single string
                return [{'text': ""}]
    
    @staticmethod
    def parse_multi_turn(response: str) -> str:
        """Parse multi-turn response - return individual turn response."""
        return response.strip()
    
    @staticmethod
    def _extract_json(response: str) -> Dict[str, Any]:
        """Extract JSON from response, handling markdown code blocks."""
        response = response.strip()
        
        # Remove markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.rfind("```")
            if end != -1 and end > start:
                response = response[start:end].strip()
        
        # Find JSON object boundaries
        start_brace = response.find('{')
        end_brace = response.rfind('}')
        
        if start_brace != -1 and end_brace != -1:
            json_content = response[start_brace:end_brace + 1]
        else:
            json_content = response
        
        # Clean up common JSON issues
        json_content = re.sub(r',\s*}', '}', json_content)  # Remove trailing commas
        json_content = re.sub(r',\s*]', ']', json_content)  # Remove trailing commas in arrays
        
        return json.loads(json_content)