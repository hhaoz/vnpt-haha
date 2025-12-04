"""
Logic solver node implementing a Manual Code Execution workflow.
Strategy: Regex Parsing + PythonREPL (ReAct Pattern without explicit Tool Binding).
"""

import re
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_experimental.utilities import PythonREPL

from src.config import settings
from src.graph import GraphState
from src.utils.llm import get_large_model

_python_repl = PythonREPL()

CODE_AGENT_PROMPT = """Nhiệm vụ của bạn là trả câu hỏi trắc nghiệm bằng cách VIẾT CODE PYTHON để tính toán.

QUY TRÌNH BẮT BUỘC:
1. Viết code Python đặt trong block markdown:
```python
# code tính toán
variable = ...
print(variable)
```

2.  Code sẽ được chạy và trả về kết quả cho bạn thông qua 'Kết quả chạy code: ...'.
3.  Dựa vào kết quả, xem xét tiếp tục viết code hoặc trả về đáp án cuối cùng bằng format 'Đáp án: X' (Trong đó X là A, B, C, hoặc D).

LƯU Ý:
- KHÔNG dùng lời văn mà chỉ dùng code để giải.
- Code phải có lệnh `print()` để thấy kết quả.
- Không trả lời trực tiếp đáp án mà chỉ trả lời khi có 'Kết quả chạy code: ...'"""


def extract_python_code(text: str) -> str | None:
    """Find and extract Python code from block ```python ...  ```"""
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def extract_final_answer(text: str) -> str | None:
    """Find the answer in the format 'Đáp án: X'"""
    match = re.search(r"Đáp án: ([A-D])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

def _indent_code(code: str) -> str:
    """Format code to make it easier to read in the terminal"""
    return "\n".join(f"        {line}" for line in code.splitlines())

def logic_solver_node(state: GraphState) -> dict:
    """
    Manual Code Agent Loop:
    LLM Gen Code -> Regex Extract -> PythonREPL -> LLM Output Final Answer
    """
    llm = get_large_model() 
    question_content = f"""

    Câu hỏi: {state["question"]}
    A. {state["option_a"]}
    B. {state["option_b"]}
    C. {state["option_c"]}
    D. {state["option_d"]}
    """

    messages: list[BaseMessage] = [
        SystemMessage(content=CODE_AGENT_PROMPT),
        HumanMessage(content=question_content)
    ]

    max_steps = 5 

    for step in range(max_steps):
        response = llm.invoke(messages)
        content = response.content
        messages.append(response) 

        final_ans = extract_final_answer(content)
        if final_ans:
            print(f"    ✅ Đã tìm thấy đáp án: {final_ans}")
            return {"answer": final_ans}

        code_block = extract_python_code(content)
        
        if code_block:
            print(f"    🐍 Step {step+1}: Found code Python. Running...")
            print(_indent_code(code_block))
            
            try:
                if "print" not in code_block:
                    lines = code_block.splitlines()
                    last_line = lines[-1]
                    if "=" in last_line:
                        var_name = last_line.split("=")[0].strip()
                    else:
                        var_name = last_line.strip()
                    code_block += f"\nprint({var_name})"

                output = _python_repl.run(code_block)
                output = output.strip() if output else "Code executed successfully but returned no output."
                print(f"    📄 Output: {output}")

                user_feedback = (f"Kết quả chạy code: {output}")
                messages.append(HumanMessage(content=user_feedback))
            
            except Exception as e:
                error_msg = f"Error running code: {str(e)}"
                print(f"    ❌ {error_msg}")
                messages.append(HumanMessage(content=f"{error_msg}. Hãy kiểm tra logic và viết lại code đúng."))
            
            continue 

        if step < max_steps - 1:
            print("    ⚠️ Model has not provided a specific action. Reminding model...")
            messages.append(HumanMessage(content="Lưu ý: Bạn vẫn chưa đưa ra đáp án cuối cùng. Hãy duyệt kết quả và quyết định tiếp tục viết code python hoặc chốt đáp án bằng 'Đáp án: X'"))

    print("    ⚠️ Max steps reached. Defaulting to A.")
    return {"answer": "A"}