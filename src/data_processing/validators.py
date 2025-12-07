import string

from src.utils.logging import print_log


def validate_answer(answer: str, num_choices: int) -> tuple[bool, str]:
    """Validate if answer is within valid range and normalize it.
    
    Args:
        answer: Raw answer string from model
        num_choices: Number of choices available (A, B, C, D, ...)
        
    Returns:
        Tuple of (is_valid, normalized_answer)
    """
    option_labels = string.ascii_uppercase
    valid_answers = option_labels[:num_choices]
    
    # Check if answer is a valid option label
    if answer.upper() in valid_answers:
        return True, answer.upper()
    
    # Check for refusal responses
    refusal_patterns = ["TỪ CHỐI TRẢ LỜI", "TỪCHỐITRẢLỜI", "TỪ CHỐI", "REFUSE"]
    answer_upper = answer.upper().strip()
    if any(pattern in answer_upper for pattern in refusal_patterns):
        return True, "Từ chối trả lời"
    
    # Invalid answer
    return False, answer


def normalize_answer(
    answer: str,
    num_choices: int,
    question_id: str | None = None,
    default: str = "A",
) -> str:
    """Normalize and validate answer, with fallback to default.
    
    This function handles:
    - Validating answer is within valid range (A, B, C, D, ...)
    - Normalizing refusal responses
    - Providing fallback for invalid answers
    
    Args:
        answer: Raw answer string from model
        num_choices: Number of choices available
        question_id: Optional question ID for logging warnings
        default: Default answer to use if validation fails
        
    Returns:
        Normalized answer string
    """
    is_valid, normalized = validate_answer(answer, num_choices)
    
    if not is_valid:
        if question_id:
            print_log(
                f"        [Warning] Invalid answer '{answer}' for {question_id}, "
                f"defaulting to {default}"
            )
        return default
    
    return normalized