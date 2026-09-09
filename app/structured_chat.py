"""使用 Pydantic Schema 获取并校验 DeepSeek 的结构化输出。"""

from __future__ import annotations

import sys

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, ValidationError

try:
    from app.basic_chat import create_deepseek_model
except ModuleNotFoundError:
    from basic_chat import create_deepseek_model


class StructuredAnswer(BaseModel):
    """模型必须返回的业务数据形状。

    Field 不只是文档：``ge``/``le`` 会在运行时校验数值。即使大模型返回了
    语法正确的 JSON，只要字段缺失或类型、范围不正确，Pydantic 仍会拒绝它。
    """

    answer: str = Field(description="对用户问题的直接回答")
    key_points: list[str] = Field(description="回答中的关键要点，建议 2 到 4 项")
    confidence: float = Field(
        ge=0,
        le=1,
        description="回答置信度，必须位于 0 到 1 之间",
    )


STRUCTURED_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一位严谨的中文助手。请仅根据指定结构回答，"
            "必须返回包含 answer、key_points、confidence 的 JSON 对象。",
        ),
        ("human", "{question}"),
    ]
)


def create_structured_chain(model: Runnable) -> Runnable:
    """让模型输出 JSON，并自动反序列化为 ``StructuredAnswer``。

    ``with_structured_output`` 会把 Pydantic Schema 转成模型可理解的约束，
    同时在响应回来后执行解析和校验。这里使用 json_mode，以兼容 DeepSeek
    当前的 JSON 输出能力。
    """
    structured_model = model.with_structured_output(
        StructuredAnswer,
        method="json_mode",
    )
    return STRUCTURED_PROMPT | structured_model


def main() -> None:
    question = input("请输入需要结构化回答的问题：").strip()
    if not question:
        print("请输入问题。")
        return

    try:
        chain = create_structured_chain(create_deepseek_model())
        result: StructuredAnswer = chain.invoke({"question": question})
        # model_dump_json 证明 result 已经是 Pydantic 对象，而不是未校验的字符串。
        print(result.model_dump_json(indent=2))
    except ValidationError as exc:
        print(f"模型返回的数据未通过结构校验：{exc}", file=sys.stderr)
    except Exception as exc:
        print(f"调用或解析失败：{exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
