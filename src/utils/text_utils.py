"""Text processing utilities for answer extraction and formatting."""

import re


def extract_answer(response: str, max_choices: int = 26) -> str:
    """Robust extraction of answer from CoT response.
    
    Args:
        response: Response text from LLM
        max_choices: Maximum number of choices (A-Z)
        
    Returns:
        Answer letter (A, B, C, ..., Z)
    """
    clean_response = response.strip()
    valid_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:max_choices]

    match = re.search(r"(?:Đáp án|Answer|Lựa chọn)[:\s]+([A-Z])", clean_response, re.IGNORECASE)
    if match:
        answer = match.group(1).upper()
        if answer in valid_labels:
            return answer

    if clean_response.upper() in valid_labels:
        return clean_response.upper()
    
    for char in reversed(clean_response):
        if char.upper() in valid_labels:
            return char.upper()
    
    return "A"  # Default fallback

