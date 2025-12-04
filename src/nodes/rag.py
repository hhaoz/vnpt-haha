"""RAG and Safety Guard nodes for knowledge-based question answering."""

import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore

from src.config import settings
from src.graph import GraphState
from src.utils.ingestion import get_embeddings, get_qdrant_client
from src.utils.llm import get_large_model

RAG_SYSTEM_PROMPT = """Bạn là trợ lý AI. Dựa vào văn bản cung cấp, hãy suy luận logic để chọn đáp án đúng nhất.

Văn bản:
{context}

Yêu cầu:
1. Suy luận ngắn gọn (1-2 câu) dựa trên văn bản.
2. Kết thúc bằng dòng: "Đáp án: X" (X là A, B, C, hoặc D)."""

RAG_USER_PROMPT = """Câu hỏi: {question}

A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}"""


def get_rag_llm():
    """Initialize RAG LLM (uses large model)."""
    return get_large_model()

_vector_store: QdrantVectorStore | None = None

def get_vector_store() -> QdrantVectorStore:
    """Get or initialize the Qdrant vector store from persistent storage."""
    global _vector_store
    if _vector_store is None:
        client = get_qdrant_client()
        embeddings = get_embeddings()
        
        _vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.qdrant_collection,
            embedding=embeddings,
        )
    return _vector_store

def set_vector_store(store: QdrantVectorStore) -> None:
    """Set the global vector store instance."""
    global _vector_store
    _vector_store = store


def knowledge_rag_node(state: GraphState) -> dict:
    """Retrieve relevant context and answer knowledge-based questions."""
    vector_store = get_vector_store()

    query = state["question"]
    print(f"    🔍 Retrieving context for: '{query}...'")
    
    docs = vector_store.similarity_search(query, k=settings.top_k_retrieval)
    context = "\n\n".join([doc.page_content for doc in docs])

    if docs:
        print(f"    📚 Found {len(docs)} docs. Top match: \"{docs[0].page_content[:150]}...\"")
    else:
        print("    ⚠️  No relevant documents found in Knowledge Base.")

    llm = get_rag_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", RAG_USER_PROMPT),
    ])

    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": state["question"],
        "option_a": state["option_a"],
        "option_b": state["option_b"],
        "option_c": state["option_c"],
        "option_d": state["option_d"],
    })
    content = response.content.strip()
    print(f"    🧠 Reasoning: {content}")
    answer = extract_answer(content)
    print(f"    ✅ RAG Answer: {answer}")
    return {"answer": answer, "context": context}


def safety_guard_node(state: GraphState) -> dict:
    """Handle toxic/sensitive questions with refusal response."""
    print("    🛡️  Safety Guard Triggered: Blocked toxic content.")
    return {
        "answer": "D",
        "context": "Nội dung không phù hợp. Hệ thống từ chối trả lời.",
    }


def extract_answer(response: str) -> str:
    """
    Robust extraction of answer from CoT response.
    Looks for pattern 'Đáp án: X' or creates a fallback.
    """
    clean_response = response.strip()
    
    # Priority 1: Find pattern "Đáp án: X" or "Answer: X" at the end
    match = re.search(r"(?:Đáp án|Answer|Lựa chọn)[:\s]+([ABCD])", clean_response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Priority 2: If response only has 1 letter A,B,C,D
    if clean_response.upper() in ["A", "B", "C", "D"]:
        return clean_response.upper()

    # Priority 3: Scan backwards from the end to find the nearest A,B,C,D
    for char in reversed(clean_response):
        if char.upper() in "ABCD":
            return char.upper()
            
    return "A" # Last fallback