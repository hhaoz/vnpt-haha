"""Entry point for running the RAG pipeline on test data with resume capability."""

import asyncio
import csv
import json
import string
import sys
import time
from pathlib import Path

from pydantic import BaseModel, Field

from src.config import BATCH_SIZE, DATA_INPUT_DIR, DATA_OUTPUT_DIR
from src.graph import get_graph
from src.state import GraphState
from src.utils.ingestion import ingest_all_data
from src.utils.llm import get_large_model, get_small_model
from src.utils.logging import (
    log_done,
    log_main,
    log_pipeline,
    log_stats,
    print_log,
)


class QuestionInput(BaseModel):
    """Input schema for a multiple-choice question."""

    qid: str = Field(description="Question identifier")
    question: str = Field(description="Question text in Vietnamese")
    choices: list[str] = Field(description="List of answer choices")
    answer: str | None = Field(default=None, description="Correct answer (A, B, C, ...)")


class PredictionOutput(BaseModel):
    """Output schema for a prediction."""

    qid: str = Field(description="Question identifier")
    answer: str = Field(description="Predicted answer: A, B, C, D, ...")


class InferenceLogEntry(BaseModel):
    """Schema for JSONL log entry."""

    qid: str
    question: str
    choices: list[str]
    final_answer: str
    raw_response: str
    route: str
    retrieved_context: str


# Global lock for thread-safe JSONL writes
_jsonl_lock = asyncio.Lock()


def _choices_to_options(choices: list[str]) -> dict[str, str]:
    """Convert choices list to option dictionary (A, B, C, D, ...)."""
    option_labels = string.ascii_uppercase
    options = {}
    for i, choice in enumerate(choices):
        if i < len(option_labels):
            options[option_labels[i]] = choice
    return options


def load_test_data(file_path: Path) -> list[QuestionInput]:
    """Load test questions from JSON file."""
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


def question_to_state(q: QuestionInput) -> GraphState:
    """Convert QuestionInput to GraphState."""
    options = _choices_to_options(q.choices)

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


def _format_choices_display(choices: list[str]) -> str:
    """Format choices for display."""
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


def load_processed_qids(log_path: Path) -> set[str]:
    """Load already processed question IDs from JSONL log."""
    processed = set()
    if not log_path.exists():
        return processed

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if "qid" in entry:
                    processed.add(entry["qid"])
            except json.JSONDecodeError:
                continue

    return processed


async def append_log_entry(log_path: Path, entry: InferenceLogEntry) -> None:
    """Append a single log entry to JSONL file (thread-safe)."""
    async with _jsonl_lock:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")


def generate_csv_from_log(log_path: Path, output_path: Path) -> int:
    """Generate submission CSV from JSONL log. Returns count of entries."""
    entries: dict[str, str] = {}

    if log_path.exists():
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entries[entry["qid"]] = entry["final_answer"]
                except (json.JSONDecodeError, KeyError):
                    continue

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "answer"])
        writer.writeheader()
        for qid, answer in entries.items():
            writer.writerow({"qid": qid, "answer": answer})

    return len(entries)


def is_rate_limit_error(error: Exception) -> bool:
    """Check if error is a VNPT API rate limit error."""
    error_str = str(error)
    return "401" in error_str and "Rate limit exceed" in error_str


async def run_pipeline_async(
    questions: list[QuestionInput],
    log_path: Path,
    force_reingest: bool = False,
    batch_size: int = BATCH_SIZE,
) -> int:
    """Run pipeline with checkpointing. Returns count of newly processed questions."""
    log_pipeline("Initializing knowledge base...")
    ingest_all_data(force=force_reingest)

    graph = get_graph()
    total = len(questions)
    start_time = time.perf_counter()
    processed_count = 0

    sem = asyncio.Semaphore(batch_size)

    async def process_single_question(q: QuestionInput) -> bool:
        nonlocal processed_count
        async with sem:
            print_log(f"\n[{q.qid}] {q.question}")
            print(_format_choices_display(q.choices))
            state = question_to_state(q)

            try:
                result = await graph.ainvoke(state)
            except Exception as e:
                print_log(f"        [Error] Failed to process {q.qid}: {e}")
                if is_rate_limit_error(e):
                    raise
                return False

            answer = result.get("answer", "A")
            route = result.get("route", "unknown")
            raw_response = result.get("raw_response", "")
            context = result.get("context", "")

            num_choices = len(q.choices)
            option_labels = string.ascii_uppercase
            valid_answers = option_labels[:num_choices]

            if answer not in valid_answers:
                print_log(f"        [Warning] Invalid answer '{answer}' for {q.qid}, defaulting to A")
                answer = "A"

            log_entry = InferenceLogEntry(
                qid=q.qid,
                question=q.question,
                choices=q.choices,
                final_answer=answer,
                raw_response=raw_response,
                route=route,
                retrieved_context=context,
            )
            await append_log_entry(log_path, log_entry)

            log_done(f"{q.qid}: {answer} (Route: {route})")
            processed_count += 1
            return True

    tasks = [process_single_question(q) for q in questions]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception) and is_rate_limit_error(result):
            log_pipeline("Rate limit exceeded. Saving progress and exiting...")
            output_file = DATA_OUTPUT_DIR / "submission.csv"
            total_entries = generate_csv_from_log(log_path, output_file)
            log_pipeline(f"Saved submission.csv with {total_entries} entries: {output_file}")
            sys.exit(0)

    elapsed = time.perf_counter() - start_time
    throughput = total / elapsed if elapsed > 0 else 0
    log_stats(f"Processed {processed_count}/{total} questions in {elapsed:.2f}s ({throughput:.2f} req/s)")

    return processed_count


def _find_test_file() -> Path | None:
    """Find the first available test JSON file in DATA_INPUT_DIR."""
    candidates = [
        "test.json",
        "val.json",
        "private_test.json",
        "public_test.json",
    ]
    for filename in candidates:
        path = DATA_INPUT_DIR / filename
        if path.exists():
            return path
    return None


async def async_main(batch_size: int = BATCH_SIZE) -> None:
    """Async main entry point with resume capability."""
    get_small_model()
    get_large_model()
    log_main("Models warmed up ready.")

    input_file = _find_test_file()

    if input_file is None:
        raise FileNotFoundError(
            f"No test JSON file found in {DATA_INPUT_DIR}. "
            "Expected files: test.json, val.json, private_test.json, or public_test.json"
        )

    log_main(f"Loading test data from: {input_file}")
    all_questions = load_test_data(input_file)
    log_main(f"Loaded {len(all_questions)} total questions")

    # Resume: check for existing progress
    log_path = DATA_OUTPUT_DIR / "inference_log.jsonl"
    processed_qids = load_processed_qids(log_path)

    if processed_qids:
        log_main(f"Resuming: Found {len(processed_qids)} already processed questions")
        remaining_questions = [q for q in all_questions if q.qid not in processed_qids]
        log_main(f"Remaining: {len(remaining_questions)} questions to process")
    else:
        log_main("Starting fresh run (no existing checkpoint found)")
        remaining_questions = all_questions

    if remaining_questions:
        await run_pipeline_async(
            remaining_questions,
            log_path,
            batch_size=batch_size,
        )
    else:
        log_main("All questions already processed. Skipping inference.")

    # Generate final CSV from JSONL log
    output_file = DATA_OUTPUT_DIR / "submission.csv"
    total_entries = generate_csv_from_log(log_path, output_file)
    log_pipeline(f"Generated submission.csv with {total_entries} entries: {output_file}")


def main(batch_size: int = BATCH_SIZE) -> None:
    """Main entry point that runs the async pipeline."""
    asyncio.run(async_main(batch_size=batch_size))


if __name__ == "__main__":
    main()