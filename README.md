# VNPT AI RAG Pipeline

**High-performance Agentic RAG Pipeline** designed for the VNPT AI Hackathon (Track 2).

This project implements a modular, model-agnostic workflow using **LangGraph** to intelligently route questions, execute Python code for math reasoning, and retrieve knowledge from a persistent vector store, optimizing for both accuracy and API quota efficiency.

## 🚀 Key Features

- **Agentic Workflow**: Uses a **Router Node** to classify questions (Math, Knowledge, or Toxic) and direct them to specialized solvers.
- **Quota Optimization**: 
  - **Tiered Modeling**: Uses "Small" models for routing (high volume) and "Large" models for reasoning/RAG (complex tasks).
  - **Persistent Embedding**: Implements local disk caching for Qdrant to prevent re-embedding and save quota.
- **Program-Aided Language Models (PAL)**: Solves math and logic problems by generating and executing Python code via a local REPL, eliminating LLM calculation errors.
- **Responsible AI**: Built-in safety guards to detect and refuse toxic or sensitive content.

## 🏗️ Architecture

The pipeline is orchestrated by a **LangGraph StateGraph**:

```mermaid
graph TD
    Start([Input Question]) --> RouterNode{Router Node<br/>Small Model}
    
    RouterNode -- "Math/Logic" --> LogicSolver[Logic Solver - Code Agent<br/>Large Model]
    RouterNode -- "History/Culture" --> KnowledgeRAG[Knowledge RAG - Retrieval<br/>Large Model]
    RouterNode -- "Toxic/Sensitive" --> SafetyGuard[Safety Guard - Refusal]
    
    subgraph "Knowledge Processing"
        KnowledgeRAG <--> VectorDB[(Qdrant Local Disk)]
        VectorDB <..- IngestionScript[Ingestion Logic<br/>Persistent Cache]
    end
    
    subgraph "Logic Processing"
        LogicSolver <--> PythonREPL[Python Interpreter<br/>Manual Code Execution]
    end
    
    LogicSolver --> End([Final Answer])
    KnowledgeRAG --> End
    SafetyGuard --> End
```

### Components

1. **Router Node**: Uses a lightweight small model to classify inputs into math, knowledge, or toxic categories.
2. **Logic Solver**: A Code Agent that writes Python code (extracted via regex) and executes it locally to solve math problems. Uses manual code execution patter.
3. **Knowledge RAG**: A Retrieval-Augmented Generation node using Qdrant vector store with persistent disk caching.
4. **Safety Guard**: A deterministic filter for harmful content.

## 🛠️ Tech Stack

| Component | Current Implementation |
| :--- | :--- |
| **Orchestration** | LangGraph, LangChain |
| **Router LLM** | HuggingFace Qwen (Small Model)  |
| **Reasoning/RAG LLM** | HuggingFace Qwen (Large Model)  |
| **Vector DB** | Qdrant (Local Persistence) |
| **Embedding** | BKAI Vietnamese Bi-encoder |
| **Code Execution** | PythonREPL (Local) |


## ⚡ Quick Start

### Prerequisites

- Python ≥3.10
- [uv](https://github.com/astral-sh/uv) (recommended for fast setup)
- CUDA-capable GPU (recommended for model inference)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/duongtruongbinh/vnpt-ai
cd vnpt-ai

# 2. Install dependencies
uv sync

# 3. Configure Environment (Optional)
# Create .env file to override default model paths
# LLM_MODEL_SMALL=/path/to/small/model
# LLM_MODEL_LARGE=/path/to/large/model
```

### Usage

**1. Generate Dummy Data (Optional)**
Creates sample questions and a knowledge base for testing.

```bash
uv run python data/generate_dummy_data.py
```

**2. Run the Pipeline**
The system automatically handles vector ingestion with smart caching.

- *First run:* Embeds data and saves to `data/qdrant_storage` (Consumes Embedding API quota).
- *Subsequent runs:* Loads from disk (Zero quota usage).

```bash
uv run python main.py
```

- **Input:** `data/public_test.csv` (or `data/private_test.csv`)
- **Output:** `data/pred.csv` (or `/output/pred.csv`)

## 📂 Project Structure

```
vnpt-ai/
├── data/                 
│   ├── qdrant_storage/   # Persistent Vector DB (Do not commit)
│   ├── knowledge_base.txt
│   ├── public_test.csv
│   └── generate_dummy_data.py
├── src/
│   ├── graph.py          # LangGraph workflow definition
│   ├── config.py         # Settings & Model configuration
│   ├── nodes/
│   │   ├── router.py     # Classification (Small Model)
│   │   ├── rag.py        # RAG Logic (Large Model)
│   │   └── logic.py      # Code Interpreter (Large Model)
│   └── utils/
│       ├── llm.py        # Model loading utilities
│       └── ingestion.py  # Smart ingestion with caching
├── main.py               # Entry point
└── pyproject.toml        # Dependencies
```
