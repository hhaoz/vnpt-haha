"""RAG and Safety Guard nodes for knowledge-based question answering."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore

from src.config import settings
from src.graph import GraphState
from src.utils.ingestion import get_embeddings, get_qdrant_client

RAG_SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên trả lời câu hỏi trắc nghiệm tiếng Việt.
Dựa vào ngữ cảnh được cung cấp, hãy chọn đáp án đúng nhất.

Ngữ cảnh:
{context}

QUAN TRỌNG: Chỉ trả lời MỘT chữ cái duy nhất: A, B, C, hoặc D."""

RAG_USER_PROMPT = """Câu hỏi: {question}

A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

Đáp án đúng là:"""


def get_rag_llm() -> ChatGoogleGenerativeAI:
    """Initialize RAG LLM."""
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key,
        temperature=0,
    )

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
    global _vector_store

    if _vector_store is None:
        _vector_store = get_vector_store()

    query = state["question"]
    print(f"    🔍 Retrieving context for: '{query}...'")
    
    docs = _vector_store.similarity_search(query, k=settings.top_k_retrieval)
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

    answer = extract_answer(response.content)
    print(f"    ✅ RAG Answer: {answer}")
    return {"answer": answer, "context": context}


def safety_guard_node(state: GraphState) -> dict:
    """Handle toxic/sensitive questions with refusal response."""
    print("    🛡️  Safety Guard Triggered: Blocked toxic content.")
    return {
        "answer": "D",
        "context": "Câu hỏi này liên quan đến nội dung không phù hợp. Hệ thống từ chối trả lời.",
    }


def extract_answer(response: str) -> str:
    """Extract single letter answer from LLM response."""
    response = response.strip().upper()
    if response in ["A", "B", "C", "D"]:
        return response
    for char in response:
        if char in "ABCD":
            return char
    return "A"