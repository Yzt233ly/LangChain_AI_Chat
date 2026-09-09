"""ChatPromptTemplate + LCEL + Output Parser 教学示例。"""

from __future__ import annotations

import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

try:  # 同时兼容 `python -m app.prompt_chat` 与 `python app/prompt_chat.py`。
    from app.basic_chat import create_deepseek_model
except ModuleNotFoundError:
    from basic_chat import create_deepseek_model


# ChatPromptTemplate 会在真正调用前才把 topic 填入模板。与 f-string 相比，
# 它保留了 system/human 的消息角色，模型因而能正确理解每段文字的用途。
PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一位擅长用简洁中文讲解技术概念的老师。"),
        ("human", "请解释 {topic}，并给出一个小例子。"),
    ]
)


def create_prompt_chain(model: Runnable) -> Runnable:
    """使用 LCEL 的 ``|`` 把三个可运行组件串成数据管道。

    数据依次经历：
    dict 输入 -> ChatPromptValue -> AIMessage -> str。
    最后的 StrOutputParser 专门取出 AIMessage.content，让调用者无需了解消息对象。
    """
    return PROMPT | model | StrOutputParser()


def main() -> None:
    topic = input("想了解什么概念？").strip()
    if not topic:
        print("请输入一个主题。")
        return

    try:
        chain = create_prompt_chain(create_deepseek_model())
        print(chain.invoke({"topic": topic}))
    except Exception as exc:
        print(f"调用失败：{exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
