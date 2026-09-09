"""使用 RunnableWithMessageHistory 实现进程内多轮聊天。"""

from __future__ import annotations

import sys

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory

try:
    from app.basic_chat import create_deepseek_model
except ModuleNotFoundError:
    from basic_chat import create_deepseek_model


# MessagesPlaceholder 不生成新文本，而是把已有的 BaseMessage 列表原样插入模板。
# 这样历史中的 HumanMessage/AIMessage 角色得以保留，模型可以理解对话先后关系。
MEMORY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个能记住上下文的中文 AI 助手。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# 教学示例使用字典保存历史。键是 session_id，值是该会话的消息列表封装。
# 这是“进程内记忆”：关闭程序后数据会消失，生产环境通常替换成 Redis/数据库。
_HISTORIES: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """按会话 ID 获取历史；首次访问时创建一份空历史。"""
    return _HISTORIES.setdefault(session_id, InMemoryChatMessageHistory())


def clear_histories() -> None:
    """清空所有进程内历史，主要用于测试与演示。"""
    _HISTORIES.clear()


def create_chat_with_history(model: Runnable) -> RunnableWithMessageHistory:
    """用历史管理器包裹普通 LCEL 链。

    内层链只知道输入中有 ``question`` 和 ``history``；外层
    RunnableWithMessageHistory 会在调用前按 session_id 读取历史，在成功返回后
    自动把本轮 Human/AI 消息写回对应历史。
    """
    chain = MEMORY_PROMPT | model | StrOutputParser()
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )


def main() -> None:
    try:
        chat = create_chat_with_history(create_deepseek_model())
    except Exception as exc:
        print(f"初始化失败：{exc}", file=sys.stderr)
        return

    session_id = input("会话 ID（直接回车使用 demo）：").strip() or "demo"
    config = {"configurable": {"session_id": session_id}}
    print("输入 /exit 退出，输入 /clear 清空当前会话。")

    while True:
        question = input("你：").strip()
        if question == "/exit":
            break
        if question == "/clear":
            get_session_history(session_id).clear()
            print("AI：当前会话历史已清空。")
            continue
        if not question:
            continue

        try:
            answer = chat.invoke({"question": question}, config=config)
            print(f"AI：{answer}")
        except Exception as exc:
            print(f"调用失败：{exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
