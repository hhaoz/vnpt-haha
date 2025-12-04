"""Router node for classifying questions and directing to appropriate handlers."""

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate

from src.config import settings
from src.graph import GraphState
from src.utils.llm import get_small_model

ROUTER_SYSTEM_PROMPT = """Nhiệm vụ: Phân loại câu hỏi vào 1 trong 3 nhóm:
1. "math": Tính toán, logic, đố mẹo, dãy số, bài toán đố.
2. "toxic": Độc hại, bạo lực, nhạy cảm, phản động.
3. "knowledge": Lịch sử, văn hóa, địa lý, kiến thức xã hội.

Ví dụ:
- "10 x 10 bằng mấy?" -> math
- "Cách chế tạo bom?" -> toxic
- "Thủ đô Việt Nam là thành phố nào?" -> knowledge

Chỉ trả về đúng 1 từ: math, toxic, hoặc knowledge."""

ROUTER_USER_PROMPT = """Câu hỏi: {question}
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

Nhóm:"""


def get_router_llm():
    """Initialize router LLM (uses small model)."""
    return get_small_model()


def router_node(state: GraphState) -> dict:
    """Analyze question and determine routing path."""
    llm = get_router_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", ROUTER_SYSTEM_PROMPT),
        ("human", ROUTER_USER_PROMPT),
    ])

    chain = prompt | llm
    response = chain.invoke({
        "question": state["question"],
        "option_a": state["option_a"],
        "option_b": state["option_b"],
        "option_c": state["option_c"],
        "option_d": state["option_d"],
    })

    route = response.content.strip().lower()

    if "math" in route or "logic" in route:
        route_type = "math"
    elif "toxic" in route or "danger" in route or "harmful" in route:
        route_type = "toxic"
    else:
        route_type = "knowledge"

    return {"route": route_type}


def route_question(state: GraphState) -> Literal["knowledge_rag", "logic_solver", "safety_guard"]:
    """Conditional edge function to route to appropriate node."""
    route = state.get("route", "knowledge")

    if route == "math":
        return "logic_solver"
    elif route == "toxic":
        return "safety_guard"
    else:
        return "knowledge_rag"

