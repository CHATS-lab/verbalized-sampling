"""
Math evaluation module with boxed answer extraction and verification.
"""

import logging
from math_verify import parse, verify

logging.getLogger("math_verify.parser").disabled = True
logging.getLogger("math_verify.grader").disabled = True


def last_boxed_only_string(string):
    """Find the last \\boxed{} or \\fbox{} in the string."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    """Remove \\boxed{} wrapper from string."""
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except:
        return None


def remove_text_boxed(s):
    """Remove \\text{} wrapper from string."""
    left = "\\text{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except:
        return s


def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    if not solution:
        return None

    solution = last_boxed_only_string(solution)
    if solution is None:
        return None

    solution = remove_boxed(solution)
    if solution is None:
        return None

    solution = remove_text_boxed(solution)
    return solution


def normalize_numeric_answer(answer):
    """Normalize numeric answers for comparison."""
    if answer is None:
        return None

    # Try to convert to float for numeric comparison
    try:
        return float(answer)
    except (ValueError, TypeError):
        # Return as string if not numeric
        return str(answer).strip()


def evaluate_math_answer(predicted: str, reference, dataset_type: str = "math") -> bool:
    """
    Evaluate a math answer against the reference.

    Args:
        predicted: The model's predicted answer
        reference: The reference/ground truth answer
        dataset_type: Type of dataset ("math", "aime", "amc", "minerva", "olympiad_bench")

    Returns:
        bool: True if correct, False otherwise
    """
    if isinstance(reference, list):
        for ref in reference:
            ref = ref.replace("\n", "")
            ref = f"${ref}$"
            if bool(verify(parse(ref, parsing_timeout=None), parse(predicted, parsing_timeout=None))):
                return True
        return False
    else:
        reference = reference.replace("\n", "")
        reference = f"${reference}$"
        return bool(verify(parse(reference, parsing_timeout=None), parse(predicted, parsing_timeout=None)))
    if not predicted:
        return False

    # Extract boxed answer from prediction if present
    extracted = extract_boxed_answer(predicted)
    if extracted is None:
        # Fallback: try to find answer at the end after common delimiters
        for delimiter in ["####", "Answer:", "answer:", "=", "Therefore"]:
            if delimiter in predicted:
                extracted = predicted.split(delimiter)[-1].strip()
                break
        if extracted is None:
            extracted = predicted.strip()

    # Handle different dataset answer formats
    if dataset_type == "amc":
        # AMC has float answers - do numeric comparison
        try:
            pred_num = float(extracted)
            ref_num = float(reference)
            return abs(pred_num - ref_num) < 1e-6
        except (ValueError, TypeError):
            return False

    elif dataset_type in ["minerva", "olympiad_bench"]:
        # These have list answers - compare against first valid answer
        if isinstance(reference, list) and len(reference) > 0:
            ref_answer = reference[0].strip()
        else:
            ref_answer = str(reference).strip()

        # Try math_verify first
        try:
            return bool(verify(parse(ref_answer), parse(extracted)))
        except:
            # Fallback to string/numeric comparison
            return normalize_numeric_answer(extracted) == normalize_numeric_answer(ref_answer)

    else:
        # MATH and AIME datasets - use math_verify for LaTeX comparison
        try:
            return bool(verify(parse(reference), parse(extracted)))
        except:
            # Fallback to string comparison (normalize whitespace and case)
            ref_normalized = str(reference).strip().lower()
            extracted_normalized = str(extracted).strip().lower()
            return ref_normalized == extracted_normalized


async def step(state, action, extra_info):
    """
    Evaluation step function for math tasks.

    Args:
        state: Current state (not used in math evaluation)
        action: The model's response/answer
        extra_info: Dictionary containing reference answer and dataset info

    Returns:
        Dict with reward, score, and completion status
    """
    reference_answer = extra_info.get("answer")
    dataset_type = extra_info.get("dataset_type", "math")

    # Evaluate the answer
    is_correct = evaluate_math_answer(action, reference_answer, dataset_type)

    reward = float(is_correct)

    return {
        "next_state": None,
        "reward": reward,
        "score": reward,
        "done": True,
        "extra_info": extra_info
    }