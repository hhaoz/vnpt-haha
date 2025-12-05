from pydantic import BaseModel, Field

class QuestionInput(BaseModel):
    """Input schema for a multiple-choice question."""

    qid: str = Field(description="Question identifier")
    question: str = Field(description="Question text in Vietnamese")
    choices: list[str] = Field(description="List of answer choices")
    answer: str | None = Field(default=None, description="Correct answer (A, B, C, ...)")


class PredictionOutput(BaseModel):
    """Output schema for a prediction."""

    qid: str = Field(description="Question identifier")
    answer: str = Field(description="Predicted answer: A, B, C, D, ... or 'Từ chối trả lời'")