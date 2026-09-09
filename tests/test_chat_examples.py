"""离线测试：这些测试不会连接 DeepSeek，也不需要真实 API Key。"""

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from app.basic_chat import append_reply, build_messages, create_deepseek_model
from app.memory_chat import (
    clear_histories,
    create_chat_with_history,
    get_session_history,
)
from app.prompt_chat import create_prompt_chain
from app.structured_chat import StructuredAnswer, create_structured_chain
from app.streaming_chat import collect_text_chunks, stream_and_commit


def test_basic_messages_use_all_three_chat_roles() -> None:
    """基础示例应按 system -> human -> ai 的角色顺序组织消息。"""
    messages = build_messages("你好")

    assert [type(item) for item in messages] == [SystemMessage, HumanMessage]

    completed = append_reply(messages, AIMessage(content="你好！"))
    assert [type(item) for item in completed] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
    ]
    # append_reply 返回新列表，避免调用方保存的历史被意外原地修改。
    assert len(messages) == 2


def test_deepseek_client_has_a_finite_timeout(monkeypatch) -> None:
    """网络异常时请求必须最终返回控制权，不能无限挂起界面。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.delenv("DEEPSEEK_TIMEOUT", raising=False)

    model = create_deepseek_model()

    assert model.request_timeout == 60.0
    assert model.stream_chunk_timeout == 60.0


def test_prompt_chain_returns_parsed_string() -> None:
    """StrOutputParser 应把模型的 AIMessage 解包为易用的 str。"""
    fake_model = FakeListChatModel(responses=["LCEL 把组件连接成管道。"])

    chain = create_prompt_chain(fake_model)

    assert chain.invoke({"topic": "LCEL"}) == "LCEL 把组件连接成管道。"


def test_structured_answer_rejects_confidence_outside_range() -> None:
    """Pydantic 应在数据进入业务代码前拒绝越界置信度。"""
    with pytest.raises(ValidationError):
        StructuredAnswer(answer="回答", key_points=["要点"], confidence=1.5)


def test_structured_chain_binds_schema_and_json_mode() -> None:
    """结构化链必须把 Schema 和 DeepSeek 兼容的 JSON Mode 绑定给模型。"""

    class StructuredStub:
        schema = None
        method = None

        def with_structured_output(self, schema, *, method):
            self.schema = schema
            self.method = method
            return RunnableLambda(
                lambda _: StructuredAnswer(
                    answer="结构化回答", key_points=["要点"], confidence=0.9
                )
            )

    model = StructuredStub()
    chain = create_structured_chain(model)

    result = chain.invoke({"question": "什么是结构化输出？"})
    assert model.schema is StructuredAnswer
    assert model.method == "json_mode"
    assert result.answer == "结构化回答"


def test_memory_is_isolated_by_session_id() -> None:
    """不同 session_id 必须得到不同的消息历史，避免用户串话。"""
    clear_histories()
    first = get_session_history("first")
    second = get_session_history("second")

    first.add_user_message("只属于 first")

    assert [message.content for message in first.messages] == ["只属于 first"]
    assert second.messages == []


def test_memory_chain_writes_back_each_completed_turn() -> None:
    """历史包装器应在连续调用后自动保存每轮 Human/AI Message。"""
    clear_histories()
    # LangChain 1.6 仍保留此教学友好的 API，但推荐大型生产项目迁移到
    # LangGraph persistence。显式断言警告，既保持测试输出干净，也记录迁移信号。
    with pytest.warns(DeprecationWarning, match="RunnableWithMessageHistory"):
        chat = create_chat_with_history(
            FakeListChatModel(responses=["第一答", "第二答"])
        )
    config = {"configurable": {"session_id": "integration"}}

    chat.invoke({"question": "第一问"}, config=config)
    chat.invoke({"question": "第二问"}, config=config)

    history = get_session_history("integration").messages
    assert [(message.type, message.content) for message in history] == [
        ("human", "第一问"),
        ("ai", "第一答"),
        ("human", "第二问"),
        ("ai", "第二答"),
    ]


def test_collect_text_chunks_keeps_model_chunk_order() -> None:
    """流式片段必须按模型到达顺序交给界面。"""
    chunks = [AIMessageChunk(content="你"), AIMessageChunk(content="好")]

    assert collect_text_chunks(chunks) == ["你", "好"]


def test_collect_text_chunks_propagates_stream_error() -> None:
    """流式异常应上抛，让 UI 避免保存不完整的 assistant 消息。"""

    def broken_stream():
        yield AIMessageChunk(content="半句")
        raise RuntimeError("connection lost")

    with pytest.raises(RuntimeError, match="connection lost"):
        collect_text_chunks(broken_stream())


def test_streaming_turn_does_not_commit_partial_history() -> None:
    """底层流中断时，user/assistant 都不能进入已完成历史。"""

    class BrokenModel:
        def stream(self, _messages):
            yield AIMessageChunk(content="半句")
            raise RuntimeError("connection lost")

    history = [{"role": "user", "content": "旧问题"}]
    original = list(history)

    with pytest.raises(RuntimeError, match="connection lost"):
        stream_and_commit(BrokenModel(), history, "新问题", lambda parts: "".join(parts))

    assert history == original


def test_streaming_turn_commits_complete_user_and_ai_messages() -> None:
    """流完整结束后，应一次性提交完整的一问一答。"""

    class WorkingModel:
        def stream(self, _messages):
            yield AIMessageChunk(content="完整")
            yield AIMessageChunk(content="回答")

    history = []
    answer = stream_and_commit(
        WorkingModel(), history, "新问题", lambda parts: "".join(parts)
    )

    assert answer == "完整回答"
    assert history == [
        {"role": "user", "content": "新问题"},
        {"role": "assistant", "content": "完整回答"},
    ]
