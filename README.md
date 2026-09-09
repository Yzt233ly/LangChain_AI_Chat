# LangChain AI Chat

一个面向初学者的 LangChain 小项目，使用 DeepSeek 构建支持多轮对话、流式输出和结构化输出的聊天应用。项目中的五个示例由浅入深，每个重要概念旁都有中文注释。

## 功能与技术

| 功能 | 主要文件 | 核心技术 |
| --- | --- | --- |
| 基础大模型调用 | `app/basic_chat.py` | `ChatOpenAI`、DeepSeek OpenAI 兼容接口 |
| 消息角色 | `app/basic_chat.py` | `SystemMessage`、`HumanMessage`、`AIMessage` |
| 提示词与 LCEL | `app/prompt_chat.py` | `ChatPromptTemplate`、`prompt \| model \| parser` |
| 输出解析 | `app/prompt_chat.py` | `StrOutputParser` |
| 结构化输出 | `app/structured_chat.py` | Pydantic、`with_structured_output`、JSON Mode |
| 多轮聊天 | `app/memory_chat.py` | `MessagesPlaceholder`、`RunnableWithMessageHistory` |
| 网页与流式输出 | `app/streaming_chat.py` | Streamlit、`model.stream()`、`st.write_stream()` |
| 密钥管理 | `.env.example` | `python-dotenv`、环境变量 |

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
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 1. 准备环境

需要 Python 3.10 或更高版本。以下命令均在项目根目录执行。

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

在 DeepSeek 开放平台创建 API Key，然后编辑 `.env`：

```dotenv
DEEPSEEK_API_KEY=替换为你的真实密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_TEMPERATURE=0.7
DEEPSEEK_TIMEOUT=60
```

`.env` 已加入 `.gitignore`。请勿把真实密钥写进 Python 文件、README 或提交到版本库。
`DEEPSEEK_TIMEOUT` 限制普通请求及流式片段的等待时间，网络异常时应用不会无限挂起。

## 2. 按顺序运行示例

### 基础消息

```powershell
python app/basic_chat.py
```

这个示例显式创建 System/Human Message，把它们传给模型，再将返回的 AI Message 加入消息列表。

### ChatPromptTemplate、LCEL 与 Output Parser

```powershell
python app/prompt_chat.py
```

关键表达式如下：

```python
chain = PROMPT | model | StrOutputParser()
```

`|` 是 LangChain Expression Language（LCEL）的组合操作符。数据依次从字典变成提示词、模型消息，最后由解析器变成普通字符串。

### 结构化输出

```powershell
python app/structured_chat.py
```

模型需要返回符合 `StructuredAnswer` 的 JSON。LangChain 负责请求结构化响应，Pydantic 负责字段类型和 `confidence` 范围校验。

### 命令行多轮对话

```powershell
python app/memory_chat.py
```

- 输入 `/clear` 清空当前会话。
- 输入 `/exit` 退出。
- 不同 `session_id` 使用不同的历史记录。
- 该示例只使用内存；关闭 Python 进程后历史会消失。

> 本项目用 `RunnableWithMessageHistory` 清晰展示 LangChain 多轮历史原理。
> LangChain 1.6 已建议大型生产项目改用 LangGraph persistence，因此运行本示例时
> 可能看到弃用提示；这不影响示例功能。

### Streamlit 聊天网页

```powershell
streamlit run app/streaming_chat.py
```

浏览器打开后即可聊天。页面通过 `st.session_state` 保存本次浏览器会话，通过 `model.stream()` 接收 DeepSeek 片段，再由 `st.write_stream()` 边接收边显示。点击侧边栏的“清空对话”可重置历史。

## 3. 理解关键概念

### System / Human / AI Message

- `SystemMessage`：规定助手角色、语气和长期规则。
- `HumanMessage`：表示用户输入。
- `AIMessage`：表示模型已经完成的回复。

保留消息角色比把所有内容拼成一个字符串更可靠，也方便 LangChain 适配不同模型 API。

### ChatPromptTemplate

提示词模板把固定指令和运行时变量分开。`{topic}`、`{question}` 等变量在调用 `invoke()` 时才填入，因此同一个模板可以重复使用。

### LCEL

LCEL 用统一的 Runnable 接口组合组件。管道内每一项接收前一项的输出，使普通调用、批处理、异步调用和流式调用可以共享相近的代码结构。

### Output Parser 与 Structured Output

`StrOutputParser` 只取出模型消息中的文本。结构化输出则进一步要求模型遵循 Schema，并将返回值转换为 Pydantic 对象；两者解决的是不同层级的问题。

### 多轮记忆

大模型 API 本身通常是无状态的。所谓“记忆”，其实是在下一次请求时把过去的 Human/AI Messages 一并发给模型。`RunnableWithMessageHistory` 自动完成读取和写回历史的工作。

### Streaming

`invoke()` 等到完整回复后一次返回，`stream()` 则不断产生 `AIMessageChunk`。网页可以立即显示这些片段，从而明显降低用户感受到的等待时间。

## 4. 运行测试

```powershell
python -m pytest -v
```

测试使用 LangChain 的 Fake Model 或纯消息对象，不会连接 DeepSeek，也不会消耗 Token。

还可以执行语法检查：

```powershell
python -m compileall -q app tests
```

## 常见问题

### 提示“未配置 DEEPSEEK_API_KEY”

确认项目根目录存在 `.env`，变量名拼写正确，并且没有继续使用示例占位值。运行命令时也应位于项目根目录。

### 返回 401 / Authentication Error

API Key 无效、已撤销或复制时带入了多余空格。请在 DeepSeek 开放平台重新确认密钥。

### 返回 429 / Rate Limit

请求频率过高或账户额度不足。降低调用频率，并检查 DeepSeek 账户余额及限额。

### Streamlit 页面没有逐步输出

确认运行的是 `streamlit run app/streaming_chat.py`，并保持 `.env` 中的模型为支持流式响应的聊天模型。代理或网络中间层也可能缓冲响应。

### 对话越来越慢或消耗越来越多 Token

每一轮都会把完整历史重新发送给模型，这是最简单的记忆实现。实际项目可增加历史条数限制、摘要记忆或数据库持久化。
