import csv
import json
from pathlib import Path
from src.data_processing.models import QuestionInput


def load_test_data_from_json(file_path: Path) -> list[QuestionInput]:
    """Load test questions from JSON file.
    
    Expected format: List of dicts with qid, question, choices, answer (optional)
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of QuestionInput objects
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Test data file not found: {file_path}")

    if file_path.suffix.lower() != ".json":
        raise ValueError(f"Only JSON files are supported: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"JSON file must contain a list of questions: {file_path}")

    questions = []
    for item in data:
        if "choices" not in item or not isinstance(item["choices"], list):
            raise ValueError(f"Question {item.get('qid', 'unknown')} must have 'choices' as a list")

        questions.append(QuestionInput(
            qid=item["qid"],
            question=item["question"],
            choices=item["choices"],
            answer=item.get("answer"),
        ))

    return questions


def load_test_data_from_csv(file_path: Path) -> list[QuestionInput]:
    """Load test questions from CSV file.
    
    Expected CSV format:
    - Columns: qid, question, choice_a, choice_b, choice_c, choice_d
    - OR: qid, question, choices (where choices is JSON array string)
    - OR: qid, question, option_a, option_b, option_c, option_d
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        List of QuestionInput objects
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Test data file not found: {file_path}")
    
    questions = []
    with open(file_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row.get("qid", "").strip()
            question = row.get("question", "").strip()
            
            if not qid or not question:
                continue
            
            # Try to parse choices from separate columns
            choices = []
            if "choice_a" in row and row["choice_a"]:
                choices = [
                    row.get("choice_a", "").strip(),
                    row.get("choice_b", "").strip(),
                    row.get("choice_c", "").strip(),
                    row.get("choice_d", "").strip(),
                ]
                choices = [c for c in choices if c]  # Remove empty
            elif "option_a" in row and row["option_a"]:
                # Alternative column names
                choices = [
                    row.get("option_a", "").strip(),
                    row.get("option_b", "").strip(),
                    row.get("option_c", "").strip(),
                    row.get("option_d", "").strip(),
                ]
                choices = [c for c in choices if c]
            elif "choices" in row and row["choices"]:
                # Try to parse as JSON array
                try:
                    choices = json.loads(row["choices"])
                    if not isinstance(choices, list):
                        choices = []
                except:
                    # Fallback: split by comma or semicolon
                    choices = [c.strip() for c in row["choices"].replace(";", ",").split(",") if c.strip()]
            
            if not choices:
                # Default to empty choices if none found
                choices = ["", "", "", ""]
            
            questions.append(QuestionInput(
                qid=qid,
                question=question,
                choices=choices,
                answer=row.get("answer", "").strip() or None,
            ))
    
    return questions