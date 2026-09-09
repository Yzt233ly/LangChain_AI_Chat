# LangChain AI Chat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build five progressively teachable LangChain + DeepSeek chat examples, ending in a Streamlit app with memory and streaming output.

**Architecture:** Each file under `app/` is independently runnable and owns the small amount of model configuration it needs so learners can read examples in isolation. Pure helper functions separate message construction, history lookup, and chunk joining from network/UI code, allowing tests to verify behavior without calling DeepSeek.

**Tech Stack:** Python 3.10+, LangChain Core, LangChain OpenAI, Pydantic 2, python-dotenv, Streamlit, pytest

---

## File Map

- `app/basic_chat.py`: DeepSeek client creation and explicit System/Human/AI message example.
- `app/prompt_chat.py`: ChatPromptTemplate, LCEL, and StrOutputParser example.
- `app/structured_chat.py`: Pydantic schema and model structured-output example.
- `app/memory_chat.py`: session-scoped RunnableWithMessageHistory terminal chat.
- `app/streaming_chat.py`: Streamlit chat UI, browser-session memory, and streaming rendering.
- `tests/test_chat_examples.py`: offline unit tests for all pure behavior.
- `.env.example`: documented DeepSeek configuration template.
- `.gitignore`: excludes secrets and Python-generated files.
- `requirements.txt`: reproducible dependency ranges.
- `README.md`: setup, execution, concepts, and troubleshooting.

### Task 1: Project Configuration

**Files:**
- Create: `.env.example`
- Create: `.gitignore`
- Create: `requirements.txt`

- [ ] **Step 1: Create the configuration files**

```dotenv
# .env.example
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_TEMPERATURE=0.7
```

```gitignore
.env
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.streamlit/secrets.toml
```

```text
# requirements.txt
langchain-core>=0.3,<2.0
langchain-openai>=0.3,<2.0
pydantic>=2.10,<3.0
python-dotenv>=1.0,<2.0
streamlit>=1.40,<2.0
pytest>=8.0,<10.0
```

- [ ] **Step 2: Verify dependency declarations can be parsed**

Run: `python -m pip install --dry-run -r requirements.txt`
Expected: exit code 0 and a dependency resolution summary.

### Task 2: Basic Messages and DeepSeek Client

**Files:**
- Create: `tests/test_chat_examples.py`
- Create: `app/basic_chat.py`

- [ ] **Step 1: Write the failing message-construction tests**

```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from app.basic_chat import build_messages, append_reply

def test_basic_messages_use_all_three_chat_roles():
    messages = build_messages("你好")
    assert [type(item) for item in messages] == [SystemMessage, HumanMessage]
    completed = append_reply(messages, AIMessage(content="你好！"))
    assert [type(item) for item in completed] == [SystemMessage, HumanMessage, AIMessage]
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/test_chat_examples.py -v`
Expected: FAIL because `app.basic_chat` does not exist.

- [ ] **Step 3: Implement the basic example**

```python
def build_messages(question: str):
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]

def append_reply(messages, reply: AIMessage):
    return [*messages, reply]
```

Also implement `create_deepseek_model()` with `ChatOpenAI(api_key=..., base_url=..., model=..., temperature=...)`, validate the key, and add a guarded command-line `main()` that invokes the model.

- [ ] **Step 4: Run the test and verify GREEN**

Run: `python -m pytest tests/test_chat_examples.py -v`
Expected: 1 passed.

### Task 3: Prompt Template, LCEL, and Parser

**Files:**
- Modify: `tests/test_chat_examples.py`
- Create: `app/prompt_chat.py`

- [ ] **Step 1: Write the failing prompt-chain test**

```python
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from app.prompt_chat import create_prompt_chain

def test_prompt_chain_returns_parsed_string():
    chain = create_prompt_chain(FakeListChatModel(responses=["LCEL 把组件连接成管道。"]));
    assert chain.invoke({"topic": "LCEL"}) == "LCEL 把组件连接成管道。"
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/test_chat_examples.py::test_prompt_chain_returns_parsed_string -v`
Expected: FAIL because `app.prompt_chat` does not exist.

- [ ] **Step 3: Implement the LCEL chain**

```python
PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一位擅长用简洁中文讲解技术概念的老师。"),
    ("human", "请解释 {topic}，并给出一个小例子。"),
])

def create_prompt_chain(model):
    return PROMPT | model | StrOutputParser()
```

Add a guarded CLI that obtains a model from `basic_chat.create_deepseek_model()` and invokes the chain.

- [ ] **Step 4: Run and verify GREEN**

Run: `python -m pytest tests/test_chat_examples.py -v`
Expected: 2 passed.

### Task 4: Structured Output

**Files:**
- Modify: `tests/test_chat_examples.py`
- Create: `app/structured_chat.py`

- [ ] **Step 1: Write the failing schema test**

```python
import pytest
from pydantic import ValidationError
from app.structured_chat import StructuredAnswer

def test_structured_answer_rejects_confidence_outside_range():
    with pytest.raises(ValidationError):
        StructuredAnswer(answer="回答", key_points=["要点"], confidence=1.5)
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/test_chat_examples.py::test_structured_answer_rejects_confidence_outside_range -v`
Expected: FAIL because `app.structured_chat` does not exist.

- [ ] **Step 3: Implement the schema and structured chain**

```python
class StructuredAnswer(BaseModel):
    answer: str = Field(description="对用户问题的直接回答")
    key_points: list[str] = Field(description="回答中的关键要点")
    confidence: float = Field(ge=0, le=1, description="回答置信度，范围为 0 到 1")

def create_structured_chain(model):
    return STRUCTURED_PROMPT | model.with_structured_output(StructuredAnswer)
```

Add a CLI that prints `result.model_dump_json(indent=2)` and reports validation/API errors to stderr.

- [ ] **Step 4: Run and verify GREEN**

Run: `python -m pytest tests/test_chat_examples.py -v`
Expected: 3 passed.

### Task 5: Multi-Turn Message History

**Files:**
- Modify: `tests/test_chat_examples.py`
- Create: `app/memory_chat.py`

- [ ] **Step 1: Write the failing history-isolation test**

```python
from app.memory_chat import clear_histories, get_session_history

def test_memory_is_isolated_by_session_id():
    clear_histories()
    first = get_session_history("first")
    second = get_session_history("second")
    first.add_user_message("只属于 first")
    assert [message.content for message in first.messages] == ["只属于 first"]
    assert second.messages == []
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/test_chat_examples.py::test_memory_is_isolated_by_session_id -v`
Expected: FAIL because `app.memory_chat` does not exist.

- [ ] **Step 3: Implement session history and runnable wrapping**

```python
_HISTORIES: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str):
    return _HISTORIES.setdefault(session_id, InMemoryChatMessageHistory())

def clear_histories():
    _HISTORIES.clear()

def create_chat_with_history(model):
    chain = MEMORY_PROMPT | model | StrOutputParser()
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
```

Add a CLI loop using `config={"configurable": {"session_id": session_id}}` and `/exit` to stop.

- [ ] **Step 4: Run and verify GREEN**

Run: `python -m pytest tests/test_chat_examples.py -v`
Expected: 4 passed.

### Task 6: Streamlit Streaming Chat

**Files:**
- Modify: `tests/test_chat_examples.py`
- Create: `app/streaming_chat.py`

- [ ] **Step 1: Write failing streaming helper tests**

```python
import pytest
from langchain_core.messages import AIMessageChunk
from app.streaming_chat import collect_text_chunks

def test_collect_text_chunks_joins_model_chunks():
    chunks = [AIMessageChunk(content="你"), AIMessageChunk(content="好")]
    assert collect_text_chunks(chunks) == ["你", "好"]

def test_collect_text_chunks_propagates_stream_error():
    def broken_stream():
        yield AIMessageChunk(content="半句")
        raise RuntimeError("connection lost")
    with pytest.raises(RuntimeError, match="connection lost"):
        collect_text_chunks(broken_stream())
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/test_chat_examples.py::test_collect_text_chunks_joins_model_chunks tests/test_chat_examples.py::test_collect_text_chunks_propagates_stream_error -v`
Expected: FAIL because `app.streaming_chat` does not exist.

- [ ] **Step 3: Implement streaming helpers and Streamlit UI**

```python
def collect_text_chunks(chunks):
    return [chunk.content for chunk in chunks if isinstance(chunk.content, str) and chunk.content]

def stream_text(model, messages):
    for chunk in model.stream(messages):
        if isinstance(chunk.content, str) and chunk.content:
            yield chunk.content
```

Build the page with `st.chat_message`, `st.chat_input`, `st.session_state.messages`, `st.write_stream`, and a clear-history button. Convert stored dictionaries back to LangChain messages before calling `model.stream`; append the assistant response only after the stream completes successfully.

- [ ] **Step 4: Run and verify GREEN**

Run: `python -m pytest tests/test_chat_examples.py -v`
Expected: 6 passed.

### Task 7: Documentation and Final Verification

**Files:**
- Create: `README.md`
- Modify: all files if verification reveals an issue

- [ ] **Step 1: Write README**

Document Python prerequisites, virtual-environment setup, `pip install -r requirements.txt`, copying `.env.example` to `.env`, acquiring a DeepSeek key, individual commands for all five examples, `streamlit run app/streaming_chat.py`, a feature-to-file table, key concepts, security notes, and troubleshooting.

- [ ] **Step 2: Run the full offline test suite**

Run: `python -m pytest -v`
Expected: 6 passed, 0 failed.

- [ ] **Step 3: Compile every Python file**

Run: `python -m compileall -q app tests`
Expected: exit code 0 with no error output.

- [ ] **Step 4: Verify CLI help/import paths without an API call**

Run: `python -c "import app.basic_chat, app.prompt_chat, app.structured_chat, app.memory_chat, app.streaming_chat"`
Expected: exit code 0 and no network request.

- [ ] **Step 5: Check secrets and required concepts**

Run: `rg -n "sk-|api_key_here" app README.md .env.example`
Expected: only the documented placeholder in `.env.example`, with no real-looking key.

Run: `rg -n "ChatPromptTemplate|SystemMessage|HumanMessage|AIMessage|StrOutputParser|with_structured_output|RunnableWithMessageHistory|stream\(" app`
Expected: every required LangChain concept appears in its intended example.

