"""
Base class for dialogue simulation tasks.

This module provides the foundation for multi-turn dialogue simulation tasks
that leverage verbalized sampling methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import json
import random
import numpy as np
from dataclasses import dataclass
from enum import Enum

from verbalized_sampling.tasks.base import BaseTask
from verbalized_sampling.methods import Method
from verbalized_sampling.llms import BaseLLM


class DialogueRole(Enum):
    """Roles in a dialogue simulation."""
    PERSUADER = 0
    PERSUADEE = 1


@dataclass
class DialogueTurn:
    """Represents a single turn in a dialogue."""
    role: DialogueRole
    text: str
    turn_number: int
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ConversationState:
    """Manages the state of an ongoing conversation."""
    conversation_id: str
    turns: List[DialogueTurn]
    persuader_persona: Dict[str, Any]
    persuadee_persona: Dict[str, Any]
    max_turns: int
    current_turn: int = 0
    outcome: Optional[Dict[str, Any]] = None

    def add_turn(self, role: DialogueRole, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a new turn to the conversation."""
        turn = DialogueTurn(
            role=role,
            text=text,
            turn_number=self.current_turn,
            metadata=metadata
        )
        self.turns.append(turn)
        self.current_turn += 1

    def get_conversation_history(self) -> str:
        """Get formatted conversation history."""
        history = []
        for turn in self.turns:
            role_name = "Persuader" if turn.role == DialogueRole.PERSUADER else "Persuadee"
            history.append(f"{role_name}: {turn.text}")
        return "\n".join(history)

    def is_complete(self) -> bool:
        """Check if conversation has reached completion."""
        return self.current_turn >= self.max_turns or self.outcome is not None


class DialogueTask(BaseTask):
    """Base class for dialogue simulation tasks."""

    def __init__(self,
                 persuader_model: BaseLLM,
                 persuadee_model: BaseLLM,
                 method: Method,
                 max_turns: int = 10,
                 word_limit: int = 160,
                 num_samplings: int = 4,
                 sampling_method: str = "vs_standard",
                 **kwargs):
        """Initialize dialogue task.

        Args:
            persuader_model: LLM for persuader role
            persuadee_model: LLM for persuadee role
            method: Verbalized sampling method to use
            max_turns: Maximum number of conversation turns
            word_limit: Word limit per response
            num_samplings: Number of response samples for VS methods
            sampling_method: Specific sampling method variant
        """
        # Use persuadee model as primary model for base class
        super().__init__(model=persuadee_model, method=method, **kwargs)

        self.persuader_model = persuader_model
        self.persuadee_model = persuadee_model
        self.max_turns = max_turns
        self.word_limit = word_limit
        self.num_samplings = num_samplings
        self.sampling_method = sampling_method

        # Response selection strategy
        self.response_selection = "probability_weighted"  # or "random"

    @abstractmethod
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load dialogue dataset."""
        pass

    @abstractmethod
    def create_persuader_prompt(self,
                               conversation_state: ConversationState,
                               is_first_turn: bool = False) -> str:
        """Create prompt for persuader model."""
        pass

    @abstractmethod
    def create_persuadee_prompt(self,
                               conversation_state: ConversationState,
                               is_first_turn: bool = False) -> str:
        """Create prompt for persuadee model."""
        pass

    @abstractmethod
    def extract_outcome(self, conversation_state: ConversationState) -> Dict[str, Any]:
        """Extract final outcome from completed conversation."""
        pass

    def simulate_conversation(self,
                            persuader_persona: Dict[str, Any],
                            persuadee_persona: Dict[str, Any],
                            conversation_id: str) -> ConversationState:
        """Simulate a complete conversation between persuader and persuadee.

        Args:
            persuader_persona: Persona information for persuader
            persuadee_persona: Persona information for persuadee
            conversation_id: Unique identifier for this conversation

        Returns:
            ConversationState: Complete conversation with outcome
        """
        # Initialize conversation state
        state = ConversationState(
            conversation_id=conversation_id,
            turns=[],
            persuader_persona=persuader_persona,
            persuadee_persona=persuadee_persona,
            max_turns=self.max_turns
        )

        # Simulate conversation turns
        while not state.is_complete():
            # Determine whose turn it is (alternating, starting with persuader)
            is_persuader_turn = (state.current_turn % 2) == 0
            is_first_turn = state.current_turn == 0

            if is_persuader_turn:
                # Persuader turn - always use direct generation
                prompt = self.create_persuader_prompt(state, is_first_turn)
                # response = self.persuader_model.generate([prompt])[0]
                response = self.persuadee_model.chat([[{"role": "user", "content": prompt}]])[0] 
                state.add_turn(
                    role=DialogueRole.PERSUADER,
                    text=response,
                    metadata={"method": "direct"}
                )
            else:
                # Persuadee turn - use verbalized sampling
                responses = self._generate_persuadee_responses(state)
                selected_response = self._select_response(responses)

                state.add_turn(
                    role=DialogueRole.PERSUADEE,
                    text=selected_response["text"],
                    metadata={
                        "method": self.method.value,
                        "probability": selected_response.get("probability"),
                        "all_responses": responses
                    }
                )

        # Extract final outcome
        state.outcome = self.extract_outcome(state)
        return state

    def _generate_persuadee_responses(self,
                                    conversation_state: ConversationState) -> List[Dict[str, Any]]:
        """Generate multiple response options using verbalized sampling."""
        prompt = self.create_persuadee_prompt(conversation_state)

        if self.method in [Method.VS_STANDARD, Method.VS_COT, Method.VS_MULTI]:
            # Use verbalized sampling method
            responses = self.persuadee_model.chat([[{"role": "user", "content": prompt}]])[0]
            # Parse structured response (expecting JSON with responses and probabilities)
            try:
                if isinstance(responses, str):
                    parsed = json.loads(responses)
                else:
                    parsed = responses

                if "responses" in parsed:
                    return parsed["responses"]
                else:
                    # Fallback to single response
                    return [{"text": str(responses), "probability": 1.0}]
            except (json.JSONDecodeError, TypeError):
                # Fallback to treating as single response
                return [{"text": str(responses), "probability": 1.0}]
        else:
            # Direct or other methods - single response
            response = self.persuadee_model.chat([[{"role": "user", "content": prompt}]])[0]
            return [{"text": response, "probability": 1.0}]

    def _select_response(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select a response from multiple options."""
        if len(responses) == 1:
            return responses[0]

        if self.response_selection == "probability_weighted" and all("probability" in r for r in responses):
            # Probability-weighted selection
            probabilities = [r["probability"] for r in responses]
            # Normalize probabilities
            total_prob = sum(probabilities)
            if total_prob > 0:
                probabilities = [p / total_prob for p in probabilities]
                selected_idx = np.random.choice(len(responses), p=probabilities)
                return responses[selected_idx]

        # Random selection as fallback
        return random.choice(responses)

    def run_experiment(self,
                      dataset: Optional[List[Dict[str, Any]]] = None,
                      num_conversations: Optional[int] = None) -> List[ConversationState]:
        """Run dialogue simulation experiment.

        Args:
            dataset: Optional dataset to use (if None, loads default)
            num_conversations: Number of conversations to simulate

        Returns:
            List of completed conversations
        """
        if dataset is None:
            dataset = self.load_dataset()

        if num_conversations is not None:
            dataset = dataset[:num_conversations]

        conversations = []
        for i, data_point in enumerate(dataset):
            conversation_id = data_point.get("conversation_id", f"conv_{i}")
            persuader_persona = data_point.get("persuader_persona", {})
            persuadee_persona = data_point.get("persuadee_persona", {})

            # Simulate conversation
            conversation = self.simulate_conversation(
                persuader_persona=persuader_persona,
                persuadee_persona=persuadee_persona,
                conversation_id=conversation_id
            )
            conversations.append(conversation)

        return conversations

    def get_prompt(self,
                   method: Method,
                   prompt_index: int = 0,
                   num_samples: int = 5,
                   **kwargs) -> str:
        """Get prompt for a specific method (required by BaseTask interface)."""
        # For dialogue tasks, this is used primarily for testing individual prompts
        # Load a sample from dataset for prompt generation
        dataset = self.load_dataset()
        if dataset:
            sample = dataset[prompt_index % len(dataset)]
            # Create a mock conversation state
            state = ConversationState(
                conversation_id="test",
                turns=[],
                persuader_persona=sample.get("persuader_persona", {}),
                persuadee_persona=sample.get("persuadee_persona", {}),
                max_turns=self.max_turns
            )
            return self.create_persuadee_prompt(state, is_first_turn=True)
        return "No dataset available for prompt generation"

    def parse_response(self, method: Method, response: str) -> List[Dict[str, Any]]:
        """Parse response from model (required by BaseTask interface)."""
        if method in [Method.VS_STANDARD, Method.VS_COT, Method.VS_MULTI]:
            try:
                if isinstance(response, str):
                    parsed = json.loads(response)
                else:
                    parsed = response

                if "responses" in parsed:
                    return parsed["responses"]
                else:
                    return [{"text": str(response), "probability": 1.0}]
            except (json.JSONDecodeError, TypeError):
                return [{"text": str(response), "probability": 1.0}]
        else:
            return [{"text": response}]