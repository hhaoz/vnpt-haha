import string
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data_processing.models import QuestionInput
    from src.state import GraphState


def choices_to_options(choices: list[str]) -> dict[str, str]:
    """Convert choices list to option dictionary (A, B, C, D, ...).
    
    Args:
        choices: List of choice strings
        
    Returns:
        Dictionary mapping option labels (A, B, C, ...) to choice strings
    """
    option_labels = string.ascii_uppercase
    options = {}
    for i, choice in enumerate(choices):
        if i < len(option_labels):
            options[option_labels[i]] = choice
    return options


def question_to_state(q: "QuestionInput") -> "GraphState":
    """Convert QuestionInput to GraphState for pipeline processing.
    
    Args:
        q: QuestionInput object
        
    Returns:
        GraphState dictionary
    """
    from src.state import GraphState
    
    options = choices_to_options(q.choices)

    state: GraphState = {
        "question_id": q.qid,
        "question": q.question,
        "option_a": options.get("A", ""),
        "option_b": options.get("B", ""),
        "option_c": options.get("C", ""),
        "option_d": options.get("D", ""),
        "all_choices": q.choices,
    }
    return state


def format_choices_display(choices: list[str]) -> str:
    """Format choices for display in console output.
    
    Args:
        choices: List of choice strings
        
    Returns:
        Formatted string with choices labeled A, B, C, D, etc.
    """
    option_labels = string.ascii_uppercase
    lines = []
    for i in range(0, len(choices), 2):
        line_parts = []
        for j in range(2):
            idx = i + j
            if idx < len(choices):
                label = option_labels[idx] if idx < len(option_labels) else str(idx)
                line_parts.append(f"{label}. {choices[idx]:<30}")
        if line_parts:
            lines.append("   " + " ".join(line_parts))
    return "\n".join(lines)