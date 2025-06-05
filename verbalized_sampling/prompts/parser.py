"""
Response parsers for different sampling methods.
"""

import json
import re
import ast
from typing import Any, Dict, List, Union, Optional
from enum import Enum

class ParseError(Exception):
    """Exception raised when parsing fails."""
    pass

class ResponseParser:
    """Utility class for parsing responses from different sampling methods."""
    
    @staticmethod
    def parse_response(method: str, response: str) -> Any:
        """Parse response based on the sampling method used."""
        method = method.lower()
        
        if method == "direct":
            return ResponseParser.parse_direct(response)
        elif method == "sequence":
            return ResponseParser.parse_sequence(response)
        elif method in ["structure", "structure_response_only"]:
            return ResponseParser.parse_structure_response_only(response)
        elif method in ["structure_with_prob", "structure_with_probability"]:
            return ResponseParser.parse_structure_with_probability(response)
        elif method == "multi_turn":
            return ResponseParser.parse_multi_turn(response)
        # elif method == "chain_of_thought":
        #     return ResponseParser.parse_chain_of_thought(response)
        # elif method == "self_reflection":
        #     return ResponseParser.parse_self_reflection(response)
        elif method == "temperature_sampling":
            return ResponseParser.parse_temperature_sampling(response)
        else:
            raise ValueError(f"Unknown parsing method: {method}")
    
    @staticmethod
    def parse_direct(response: str) -> str:
        """Parse direct response - just return as-is."""
        return response.strip()
    
    @staticmethod
    def parse_sequence(response: str) -> List[str]:
        """Parse sequence response expecting Python list format."""
        try:
            # Clean up the response
            response = response.strip()
            
            # Try to find list boundaries
            start_bracket = response.find('[')
            end_bracket = response.rfind(']')
            
            if start_bracket != -1 and end_bracket != -1:
                list_content = response[start_bracket:end_bracket + 1]
                # Use ast.literal_eval for safe evaluation
                parsed_list = ast.literal_eval(list_content)
                if isinstance(parsed_list, list):
                    return [str(item).strip() for item in parsed_list]
            
            # Fallback: split by common delimiters
            lines = response.split('\n')
            responses = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('//'):
                    # Remove numbering and common prefixes
                    cleaned = re.sub(r'^\d+[\.\)]\s*', '', line)
                    cleaned = re.sub(r'^[-*]\s*', '', cleaned)
                    if cleaned:
                        responses.append(cleaned)
            
            return responses if responses else [response]
            
        except Exception as e:
            # If all parsing fails, return the response as a single item
            return [response]
    
    @staticmethod
    def parse_structure_response_only(response: str) -> List[Dict[str, str]]:
        """Parse structured response with response field only."""
        try:
            parsed_json = ResponseParser._extract_json(response)
            
            if isinstance(parsed_json, dict) and "responses" in parsed_json:
                responses = parsed_json["responses"]
                if isinstance(responses, list):
                    return [
                        {"response": item.get("response", str(item)) if isinstance(item, dict) else str(item)}
                        for item in responses
                    ]
            
            # Fallback parsing
            return [{"response": response}]
            
        except Exception as e:
            return [{"response": response}]
    
    @staticmethod
    def parse_structure_with_probability(response: str) -> List[Dict[str, Union[str, float]]]:
        """Parse structured response with response and probability fields."""
        try:
            parsed_json = ResponseParser._extract_json(response)
            
            if isinstance(parsed_json, dict) and "responses" in parsed_json:
                responses = parsed_json["responses"]
                if isinstance(responses, list):
                    parsed_responses = []
                    total_prob = 0.0
                    
                    for item in responses:
                        if isinstance(item, dict):
                            prob = item.get("probability", 1.0 / len(responses))
                            parsed_responses.append({
                                "response": item.get("response", ""),
                                "probability": float(prob)
                            })
                            total_prob += float(prob)
                        else:
                            parsed_responses.append({
                                "response": str(item),
                                "probability": 1.0 / len(responses)
                            })
                    
                    # Normalize probabilities if they don't sum to 1
                    if abs(total_prob - 1.0) > 0.01 and total_prob > 0:
                        for item in parsed_responses:
                            item["probability"] = item["probability"] / total_prob
                    
                    return parsed_responses
            
            # Fallback parsing
            return [{"response": response, "probability": 1.0}]
            
        except Exception as e:
            return [{"response": response, "probability": 1.0}]
    
    @staticmethod
    def parse_multi_turn(response: str) -> str:
        """Parse multi-turn response - return individual turn response."""
        return response.strip()
    
    @staticmethod
    def parse_chain_of_thought(response: str) -> List[Dict[str, str]]:
        """Parse chain-of-thought response with reasoning and response fields."""
        try:
            parsed_json = ResponseParser._extract_json(response)
            
            if isinstance(parsed_json, dict) and "responses" in parsed_json:
                responses = parsed_json["responses"]
                if isinstance(responses, list):
                    return [
                        {
                            "reasoning": item.get("reasoning", ""),
                            "response": item.get("response", str(item) if not isinstance(item, dict) else "")
                        }
                        for item in responses
                    ]
            
            # Fallback parsing
            return [{"reasoning": "", "response": response}]
            
        except Exception as e:
            return [{"reasoning": "", "response": response}]
    
    @staticmethod
    def parse_self_reflection(response: str) -> List[Dict[str, Union[str, float]]]:
        """Parse self-reflection response with response, reflection, and confidence fields."""
        try:
            parsed_json = ResponseParser._extract_json(response)
            
            if isinstance(parsed_json, dict) and "responses" in parsed_json:
                responses = parsed_json["responses"]
                if isinstance(responses, list):
                    return [
                        {
                            "response": item.get("response", ""),
                            "reflection": item.get("reflection", ""),
                            "confidence": float(item.get("confidence", 0.5))
                        }
                        for item in responses if isinstance(item, dict)
                    ]
            
            # Fallback parsing
            return [{"response": response, "reflection": "", "confidence": 0.5}]
            
        except Exception as e:
            return [{"response": response, "reflection": "", "confidence": 0.5}]
    
    @staticmethod
    def parse_temperature_sampling(response: str) -> List[Dict[str, Union[str, float]]]:
        """Parse temperature sampling response with creativity levels."""
        try:
            parsed_json = ResponseParser._extract_json(response)
            
            if isinstance(parsed_json, dict) and "responses" in parsed_json:
                responses = parsed_json["responses"]
                if isinstance(responses, list):
                    return [
                        {
                            "response": item.get("response", ""),
                            "creativity_level": item.get("creativity_level", "moderate"),
                            "temperature": float(item.get("temperature", 0.7))
                        }
                        for item in responses if isinstance(item, dict)
                    ]
            
            # Fallback parsing
            return [{"response": response, "creativity_level": "moderate", "temperature": 0.7}]
            
        except Exception as e:
            return [{"response": response, "creativity_level": "moderate", "temperature": 0.7}]
    
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

# Helper function for backward compatibility
def parse_response_by_method(method: str, response: str) -> Any:
    """Backward compatibility function."""
    return ResponseParser.parse_response(method, response)