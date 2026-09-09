"""最基础的 LangChain 聊天示例：显式创建 System/Human/AI Message。"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


SYSTEM_PROMPT = "你是一个耐心、准确的中文 AI 助手。回答要清晰、简洁。"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TIMEOUT = 60.0


def _read_temperature() -> float:
    """读取温度；配置写错时使用安全、易理解的默认值。"""
    raw_value = os.getenv("DEEPSEEK_TEMPERATURE", str(DEFAULT_TEMPERATURE))
    try:
        return float(raw_value)
    except ValueError:
        return DEFAULT_TEMPERATURE


def _read_timeout() -> float:
    """返回有限的请求超时，避免网络半开时 CLI 或网页无限等待。"""
    raw_value = os.getenv("DEEPSEEK_TIMEOUT", str(DEFAULT_TIMEOUT))
    try:
        timeout = float(raw_value)
    except ValueError:
        return DEFAULT_TIMEOUT
    return timeout if timeout > 0 else DEFAULT_TIMEOUT


def create_deepseek_model(*, streaming: bool = False) -> ChatOpenAI:
    """创建访问 DeepSeek 的 LangChain ChatModel。

    DeepSeek 暴露 OpenAI 兼容协议，所以这里使用 ``ChatOpenAI``。LangChain
    负责统一消息与流式接口，``base_url`` 则把请求发往 DeepSeek 而非 OpenAI。
    """
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key or api_key == "your_deepseek_api_key_here":
        raise ValueError(
            "未配置 DEEPSEEK_API_KEY。请复制 .env.example 为 .env，"
            "并填入 DeepSeek API Key。"
        )

    return ChatOpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
        model=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL),
        temperature=_read_temperature(),
        streaming=streaming,
        # request_timeout 约束整个 SDK 请求；stream_chunk_timeout 还会约束相邻
        # 流式片段的最长等待时间，使断线最终能进入上层异常处理。
        request_timeout=_read_timeout(),
        stream_chunk_timeout=_read_timeout(),
    )


def build_messages(question: str) -> list[BaseMessage]:
    """把一轮问题转换成 LangChain 的标准消息对象。

    SystemMessage 定义模型长期遵循的角色；HumanMessage 承载用户输入。
    模型返回的则是 AIMessage。使用对象而非手写字典，可让 LangChain 自动
    适配不同模型提供商的消息格式。
    """
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]


def append_reply(
    messages: Sequence[BaseMessage], reply: AIMessage
) -> list[BaseMessage]:
    """返回包含 AI 回复的新消息列表，不修改调用方传入的原列表。"""
    return [*messages, reply]


def main() -> None:
    question = input("你：").strip()
    if not question:
        print("请输入问题。")
        return

    try:
        model = create_deepseek_model()
        messages = build_messages(question)
        reply = model.invoke(messages)
        conversation = append_reply(messages, reply)
        # conversation[-1] 是 AIMessage；content 是模型回复的正文。
        print(f"AI：{conversation[-1].content}")
    except Exception as exc:  # 命令行示例需要把 API/网络错误转换成友好提示。
        print(f"调用失败：{exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
