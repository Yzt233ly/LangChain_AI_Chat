# LangChain AI Chat 设计规格

## 目标

构建一个适合学习 LangChain 核心能力的小型聊天项目。项目默认通过 DeepSeek 的 OpenAI 兼容接口调用模型，并用五个由浅入深的 Python 示例分别展示基础消息、提示词模板、LCEL、输出解析、结构化输出、多轮记忆和流式输出。最终示例提供 Streamlit 聊天网页。

## 技术选型

- Python 3.10+
- `langchain-core`：消息、提示词模板、LCEL、输出解析器和运行历史封装
- `langchain-openai`：通过 OpenAI 兼容协议访问 DeepSeek
- `pydantic`：定义并校验结构化输出
- `python-dotenv`：从 `.env` 加载密钥及模型配置
- `streamlit`：提供交互式聊天界面
- `pytest`：执行不依赖真实 API 的单元测试

默认环境变量为：

- `DEEPSEEK_API_KEY`：必填，DeepSeek API 密钥
- `DEEPSEEK_BASE_URL`：默认 `https://api.deepseek.com`
- `DEEPSEEK_MODEL`：默认 `deepseek-chat`
- `DEEPSEEK_TEMPERATURE`：默认 `0.7`

## 项目结构

```text
langchain-ai-chat/
├── app/
│   ├── basic_chat.py
│   ├── prompt_chat.py
│   ├── structured_chat.py
│   ├── memory_chat.py
│   └── streaming_chat.py
├── tests/
│   └── test_chat_examples.py
├── docs/superpowers/specs/
│   └── 2026-09-09-langchain-ai-chat-design.md
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

除测试和设计文档外，用户指定的主结构保持不变。五个应用文件刻意保留少量初始化代码，使每个示例都能单独阅读和运行，不引入额外的共享模块。

## 示例职责

### `basic_chat.py`

加载 `.env`，创建 `ChatOpenAI` DeepSeek 客户端，并显式构造 `SystemMessage`、`HumanMessage` 和 `AIMessage`。脚本执行一次基础问答，并展示如何把模型回复追加到消息列表。

### `prompt_chat.py`

使用 `ChatPromptTemplate.from_messages` 创建 system/human 模板，通过 `prompt | model | StrOutputParser()` 组成 LCEL 链。脚本接受主题参数并输出纯字符串，解释每个管道组件的输入输出。

### `structured_chat.py`

定义一个 Pydantic 数据模型，包含回答、要点列表和置信度。通过模型的 `with_structured_output` 请求并校验结构化响应，同时捕获解析或调用异常并给出可理解的错误消息。

### `memory_chat.py`

使用 `ChatPromptTemplate`、`MessagesPlaceholder` 和 `RunnableWithMessageHistory` 实现按 `session_id` 隔离的多轮历史。内存存储仅在当前 Python 进程内有效，重启后清空；示例提供终端循环和退出命令。

### `streaming_chat.py`

作为 Streamlit 入口。使用 `st.session_state` 保存当前浏览器会话的消息；每次提问时，把 system 消息、历史消息和本轮 human 消息发送给 DeepSeek，并迭代模型的 `stream()` 结果，将文本片段实时渲染到页面。只有成功完成的回复才写入会话历史，并提供“清空对话”按钮。

## 数据流

1. `python-dotenv` 加载环境变量。
2. 客户端工厂读取 DeepSeek 配置；缺少密钥时立即返回明确提示。
3. 用户输入被转换为 `HumanMessage` 或注入 `ChatPromptTemplate`。
4. LCEL 链或模型客户端调用 DeepSeek。
5. 普通示例返回 `AIMessage`，文本链经 `StrOutputParser` 返回字符串，结构化链返回 Pydantic 对象。
6. 多轮示例在下一次调用时重新注入历史；Streamlit 示例同时逐片段渲染并在结束后保存完整回复。

## 错误处理

- 缺失 `DEEPSEEK_API_KEY`：在发起网络请求前抛出带配置方法的 `ValueError`；Streamlit 页面用 `st.error` 显示。
- 温度不是数字：回退到默认值 `0.7`，避免配置错误导致应用无法解释地崩溃。
- DeepSeek 网络错误、限流或鉴权失败：命令行示例输出简洁错误；Streamlit 保留已有历史并显示本轮失败。
- 流式调用中途失败：不把不完整的 assistant 回复写入历史。
- 结构化解析失败：报告校验失败原因，不伪造结构化结果。

## 测试策略

测试不调用真实 DeepSeek，也不需要 API Key。通过轻量 fake chat model 验证：

- 基础消息列表包含正确的 System/Human/AI 类型和顺序。
- LCEL 提示词能正确填充变量，`StrOutputParser` 返回字符串。
- 结构化 Pydantic 模型拒绝非法置信度。
- 两个 `session_id` 的历史互不污染，多轮调用能看到上一轮消息。
- 流式片段能够被正确拼接；异常时不提交不完整消息。

最终验证包括 `pytest`、Python 语法编译检查，以及 Streamlit 模块的无网络导入检查。真实 DeepSeek 调用需要用户自行提供有效密钥，因此不纳入自动测试。

## 文档与教学注释

关键代码旁使用详细中文注释，重点解释“为什么这样写”以及 LangChain 对象之间的数据形态变化，而不是给每一行添加重复注释。README 提供环境创建、依赖安装、`.env` 配置、五个示例的运行命令、概念映射和常见错误排查。

## 完成标准

- 用户列出的十项功能均能在代码或 Streamlit 示例中找到明确演示。
- 五个示例职责独立，关键 LangChain 代码有中文教学注释。
- 不写死 API Key，仓库忽略 `.env`。
- 自动测试和语法检查通过。
- README 中的安装及运行命令与实际文件一致。
