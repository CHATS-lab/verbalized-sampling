"""
PersuasionForGood dialogue simulation task.

This task implements the donation persuasion dialogue simulation from the paper,
where a persuader tries to convince a persuadee to donate money to charity.
"""

import json
import re
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from textwrap import dedent

from .base import DialogueTask, ConversationState, DialogueRole
from verbalized_sampling.methods import Method


class PersuasionTask(DialogueTask):
    """PersuasionForGood donation dialogue simulation task."""

    def __init__(self,
                 dataset_path: Optional[str] = None,
                 corpus_path: Optional[str] = None,
                 **kwargs):
        """Initialize persuasion task.

        Args:
            dataset_path: Path to PersuasionForGood dataset
            corpus_path: Path to corpus for persona generation
        """
        super().__init__(**kwargs)

        # Set default paths relative to the persuasion_simulation directory
        base_path = Path(__file__).parent.parent.parent.parent / "persuasion_simulation"

        if dataset_path is None:
            self.dataset_path = base_path / "src/datasets/donation/persuasionforgood_test"
        else:
            self.dataset_path = Path(dataset_path)

        if corpus_path is None:
            self.corpus_path = base_path / "src/datasets/donation/persuasionforgood_corpus"
        else:
            self.corpus_path = Path(corpus_path)

        # Load persona utility if available
        self._persona_generator = None
        try:
            # Try to import persona generator from persuasion simulation
            import sys
            sys.path.append(str(base_path))
            from src.prompts.persona import DonationPersona
            self._persona_generator = DonationPersona(str(self.corpus_path), "gpt-4.1-mini")
        except ImportError:
            # Will use simplified persona generation
            pass

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load PersuasionForGood dataset."""
        dataset = []

        if not self.dataset_path.exists():
            # Return minimal test data if dataset not found
            return [{
                "conversation_id": "test_conv_1",
                "persuader_persona": {"name": "test_persuader"},
                "persuadee_persona": {"name": "test_persuadee"}
            }]

        # Load all .jsonl files from dataset directory
        for file_path in self.dataset_path.glob("*.jsonl"):
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            # Extract conversation metadata
                            conversation_data = {
                                "conversation_id": data.get("conversation_id", file_path.stem),
                                "persuader_persona": data.get("persuader", {}),
                                "persuadee_persona": data.get("persuadee", {}),
                                "original_dialogue": data.get("dialogue", [])
                            }
                            dataset.append(conversation_data)
                        except json.JSONDecodeError:
                            continue

        return dataset if dataset else self.load_dataset()  # Fallback to test data

    def create_persuader_prompt(self,
                               conversation_state: ConversationState,
                               is_first_turn: bool = False) -> str:
        """Create prompt for persuader model."""
        # Get persona description
        persona_desc = self._get_persona_description(conversation_state.persuader_persona)

        system_prompt = dedent(f"""
        You are an Amazon Mechanical Turk worker completing a $2 communication task.
        - You are motivated by this task payment — you value every cent you earn.
        - Act naturally as the person described below—think and respond as they would, including their quirks, beliefs, biases, and reasoning.
        - Your goal is to convince your chat partner to donate part of their task payment to charity.
        - Respond in a real-time chat interface. Keep each response under {self.word_limit} words, conversational, and authentic—avoid formal, robotic, or repetitive language.

        {persona_desc}

        Scenario: You will chat with another worker who also earned $2 from a task. Try to convince them to donate some of their payment to Save the Children, a charity that helps children in need around the world. You can donate between $0 to $2 from your task payment.
        """).strip()

        if is_first_turn:
            # First turn - introduce yourself and the donation opportunity
            user_prompt = "Start the conversation by greeting your chat partner and introducing the donation opportunity to Save the Children."
        else:
            # Subsequent turns - respond to the conversation
            history = conversation_state.get_conversation_history()
            user_prompt = f"Previous conversation:\n{history}\n\nRespond to your chat partner to continue persuading them to donate:"

        return f"{system_prompt}\n\n{user_prompt}"

    def create_persuadee_prompt(self,
                               conversation_state: ConversationState,
                               is_first_turn: bool = False) -> str:
        """Create prompt for persuadee model."""
        persona_desc = self._get_persona_description(conversation_state.persuadee_persona)

        # Base system prompt depends on sampling method
        if self.method in [Method.VS_STANDARD, Method.VS_COT, Method.VS_MULTI]:
            base_prompt = dedent(f"""
            You are an Amazon Mechanical Turk worker completing a $2 communication task.
            - You are motivated by this task payment — you value every cent you earn. Treat any spending as real, and weigh the value of money before making decisions.
            - Act naturally as the person described below—think and respond as they would, including their quirks, beliefs, biases, and reasoning.
            - Complete the communication task outlined in the scenario as this person would naturally respond.
            - Respond in a real-time chat interface. Keep each response under {self.word_limit} words, conversational, and authentic—avoid formal, robotic, or repetitive language.
            """).strip()
        else:
            base_prompt = dedent(f"""
            You are an Amazon Mechanical Turk worker completing a $2 communication task.
            - You are motivated by this task payment — you value every cent you earn.
            - Act naturally as the person described below—think and respond as they would, including their quirks, beliefs, biases, and reasoning.
            - Complete the communication task outlined in the scenario as this person would naturally respond.
            - Respond in a real-time chat interface. Keep each response under {self.word_limit} words, conversational, and authentic—avoid formal, robotic, or repetitive language.
            """).strip()

        # Add sampling-specific instructions
        additional_prompt = ""
        if self.method == Method.VS_STANDARD:
            additional_prompt = dedent(f"""
            Generate {self.num_samplings} plausible responses that you would naturally give to your chat partner based on the chat history and your persona.

            Return responses as a JSON object with the key "responses" (a list of dictionaries, each with 'text' and 'probability'):
            - 'text': the response string only (no explanation or extra text).
            - 'probability': the empirical probability representing how likely each response would be (from 0.0 to 1.0).
            Give ONLY the JSON object, with no explanations or extra text.
            """).strip()
        elif self.method == Method.VS_COT:
            additional_prompt = dedent(f"""
            Generate {self.num_samplings} plausible responses with reasoning that you would naturally give to your chat partner.

            Return responses as a JSON object with the key "responses" (a list of dictionaries, each with 'reasoning', 'text', and 'probability'):
            - 'reasoning': your thought process for this response.
            - 'text': the response string only (no explanation or extra text).
            - 'probability': the empirical probability representing how likely each response would be (from 0.0 to 1.0).
            Give ONLY the JSON object, with no explanations or extra text.
            """).strip()
        elif self.method == Method.SEQUENCE:
            additional_prompt = dedent(f"""
            Generate all plausible responses you would naturally give to your chat partner based on the chat history and your persona.

            Return the responses in JSON format with keys: "responses" (list of dicts with 'text'). Each dictionary must include:
            - 'text': the response string only (no explanation or extra text).

            Give ONLY the JSON object, with no explanations or extra text.
            """).strip()
        else:
            additional_prompt = "Only output your reply to your chat partner—do not explain your reasoning."

        system_prompt = f"{base_prompt}\n\n{persona_desc}\n\n{additional_prompt}"

        # Create user prompt based on conversation state
        if is_first_turn:
            # This shouldn't happen in normal flow, but handle it
            user_prompt = "Scenario: Another worker will try to convince you to donate part of your $2 task payment to Save the Children charity. You can choose to donate between $0 to $2. Wait for them to start the conversation."
        else:
            # Get the latest persuader message
            history = conversation_state.get_conversation_history()
            user_prompt = f"Conversation so far:\n{history}\n\nRespond to your chat partner:"

        return f"{system_prompt}\n\n{user_prompt}"

    def extract_outcome(self, conversation_state: ConversationState) -> Dict[str, Any]:
        """Extract donation outcome from completed conversation."""
        # Extract donation amount from persuadee messages
        donation_amounts = []
        donation_turns = []

        for turn in conversation_state.turns:
            if turn.role == DialogueRole.PERSUADEE:
                amount = self._extract_donation_amount(turn.text)
                if amount is not None:
                    donation_amounts.append(amount)
                    donation_turns.append(turn.turn_number)

        # Get final donation amount (last mentioned amount)
        final_donation = donation_amounts[-1] if donation_amounts else 0.0

        return {
            "final_donation_amount": final_donation,
            "donation_amounts": donation_amounts,
            "donation_turns": donation_turns,
            "total_turns": len(conversation_state.turns),
            "conversation_length": len(conversation_state.turns) // 2,  # Number of exchanges
            "persuader_turns": len([t for t in conversation_state.turns if t.role == DialogueRole.PERSUADER]),
            "persuadee_turns": len([t for t in conversation_state.turns if t.role == DialogueRole.PERSUADEE])
        }

    def _get_persona_description(self, persona_data: Dict[str, Any]) -> str:
        """Generate persona description for prompts."""
        if self._persona_generator and persona_data:
            try:
                # Use sophisticated persona generator if available
                persona_bullets = self._persona_generator.persona_bullet_points(persona_data)
                return self._persona_generator.persona_descriptive_prompt(persona_bullets)
            except Exception:
                # Fall back to simple persona description
                pass

        # Simple persona description
        if not persona_data:
            return "<persona>\nA typical Amazon Mechanical Turk worker.\n</persona>"

        persona_lines = ["<persona>"]
        for key, value in persona_data.items():
            if isinstance(value, str) and value.strip():
                persona_lines.append(f"- {key.replace('_', ' ').title()}: {value}")
        persona_lines.append("</persona>")

        return "\n".join(persona_lines)

    def _extract_donation_amount(self, text: str) -> Optional[float]:
        """Extract donation amount from text using regex patterns."""
        # Convert text to lowercase for matching
        text_lower = text.lower()

        # Pattern for cents
        pattern_cents = r'\b(?:donate|donating|give|giving|do|spare|contribute|contributing)\s*\$?(\d*\.\d{1,2}|\d+)\s*cents\b'
        match_cents = re.search(pattern_cents, text_lower)

        if match_cents:
            match = next((m for m in match_cents.groups() if m is not None), None)
            if match:
                amount = float(match.replace('$', '').replace(',', '').strip())
                if amount >= 1.0:
                    amount = amount / 100  # Convert cents to dollars
                return amount

        # Pattern for dollars
        pattern_dollars = r'\b(?:donate|donating|give|giving|do|spare|contribute|contributing)\s*(?:the whole|all|entire)?\s*\$?(\d*\.\d{1,2}|\d+)\b'
        match_dollars = re.search(pattern_dollars, text_lower)

        if match_dollars:
            match = next((m for m in match_dollars.groups() if m is not None), None)
            if match:
                amount = float(match.replace('$', '').replace(',', '').strip())
                # Cap at $2 (max possible donation)
                return min(amount, 2.0)

        return None

    def get_task_description(self) -> str:
        """Get description of the persuasion task."""
        return "PersuasionForGood donation dialogue simulation - persuader tries to convince persuadee to donate to charity"

    def get_evaluation_metrics(self) -> List[str]:
        """Get list of relevant evaluation metrics for this task."""
        return [
            "donation_amount",
            "dialogue_length",
            "linguistic_diversity",
            "persona_consistency",
            "persuasion_effectiveness"
        ]