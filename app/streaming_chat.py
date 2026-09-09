"""Streamlit 聊天应用：浏览器会话记忆 + DeepSeek 流式输出。"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import TypedDict

import streamlit as st
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

try:
    from app.basic_chat import SYSTEM_PROMPT, create_deepseek_model
except ModuleNotFoundError:
    from basic_chat import SYSTEM_PROMPT, create_deepseek_model


class StoredMessage(TypedDict):
    """Streamlit session_state 中使用的可序列化消息格式。"""

    role: str
    content: str


def collect_text_chunks(chunks: Iterable[AIMessageChunk]) -> list[str]:
    """收集非空文本片段；主要用于无网络测试和理解流式数据。

    这里故意不捕获迭代过程中的异常。若连接中断，异常会传给调用者，UI 就能
    判断本轮并未完整结束，从而不会把半句话当作完整 AIMessage 写进历史。
    """
    return [
        chunk.content
        for chunk in chunks
        if isinstance(chunk.content, str) and chunk.content
    ]


def stream_text(model, messages: list[BaseMessage]) -> Iterator[str]:
    """把 LangChain 的 AIMessageChunk 流转换为 Streamlit 需要的字符串流。"""
    # model.stream 会在 token/文本片段到达时立即 yield，而不是等待整段回答。
    for chunk in model.stream(messages):
        if isinstance(chunk.content, str) and chunk.content:
            yield chunk.content


def to_langchain_messages(history: list[StoredMessage]) -> list[BaseMessage]:
    """将页面历史恢复成带角色信息的 LangChain 消息对象。"""
    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    for item in history:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            messages.append(AIMessage(content=item["content"]))
    return messages


def stream_and_commit(
    model,
    history: list[StoredMessage],
    question: str,
    write_stream: Callable[[Iterator[str]], str],
) -> str:
    """执行一轮流式请求，并以“全部成功”为条件提交历史。

    ``write_stream`` 作为参数传入：页面使用 ``st.write_stream``，测试则可以传入
    一个普通函数。关键顺序是先完整消费流，再 extend 历史；若消费期间抛出
    异常，Python 会直接离开函数，因而不会执行历史提交。
    """
    request_messages = to_langchain_messages(history)
    request_messages.append(HumanMessage(content=question))
    full_answer = write_stream(stream_text(model, request_messages))
    answer_text = str(full_answer)
    history.extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer_text},
        ]
    )
    return answer_text


def main() -> None:
    st.set_page_config(page_title="LangChain AI Chat", page_icon="💬")
    st.title("💬 LangChain AI Chat")
    st.caption("DeepSeek · LangChain · 多轮记忆 · Streaming")

    # session_state 属于当前浏览器会话。Streamlit 每次交互都会从头执行脚本，
    # 因此不能把历史放在普通局部变量中，否则点击/输入后它就会丢失。
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.subheader("使用说明")
        st.write("在项目根目录创建 `.env` 并配置 `DEEPSEEK_API_KEY`。")
        if st.button("清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # 页面重跑时先恢复并绘制已经成功完成的历史消息。
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("给 DeepSeek 发消息")
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    # 先构造临时请求，不马上修改 session_state。只有流式响应完整结束后，
    # 才同时提交 user/assistant 两条消息，保证历史中不会出现残缺的一轮。
    try:
        model = create_deepseek_model(streaming=True)
        with st.chat_message("assistant"):
            stream_and_commit(
                model,
                st.session_state.messages,
                question,
                st.write_stream,
            )
    except Exception as exc:
        # 保留之前已经完成的历史，但不保存本轮不完整内容。
        st.error(f"本轮调用失败：{exc}")


if __name__ == "__main__":
    main()
