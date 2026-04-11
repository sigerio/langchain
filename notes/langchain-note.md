# LangChain 官方教程学习笔记

本笔记按照当前仓库中的官方教程例程整理，重点记录“概念理解”和“代码写法”类问题，便于后续回看和直接改写。
当前归档后的 LangChain 示例统一位于 `tutorials/langchain/` 目录下。

## 记录规则

本笔记用于记录我在学习 LangChain 官方教程过程中提出的“概念理解”和“代码写法”类问题。

记录时遵循以下规则：

- 只记录类似 `xxx 是什么意思`、`xxx 应该怎么写` 这类问题。
- 不记录类似 `下一个例子是什么`、`接下来讲什么` 这类教程顺序问题。
- 普通对话内容不被记录
- 每个问题后面应具有 `ACT` 或 `ACH` 标记，来指示当前问题是否活跃。
- 整篇文档仅能存在一个 `ACT` 活跃标记
- 每个问题都必须具有唯一 `问题ID`，格式为 `Q-章节缩写-序号-英文短标识`，例如 `Q-CHAT-05-trim-messages`。
- 后续新增或更新 `追加提问` 时，必须先根据当前 `ACT` 问题的 `问题ID` 定位，再写入该问题下的 `疑点` 部分。
- 明确说出 `追加提问` 的内容需要更新进入当前活跃问题内的 `疑点` 部分，按照“提问编号-提问内容、问题描述、问题解答”的格式进行阐述。
- 笔记应具有章节、标题；划分需要按照官方教程进行划分。
- 疑点内的追加提问需要按照规定格式，使之能在目录中快速找到
- 每个子章节都按固定结构整理：
  - 定义：解释 xxx 的详细定义。
  - 功能：解释 xxx 的具体功能。
  - 源码示例：提供渐进式的代码示例。
  - 疑点：关于官方子章节的疑点记录在此，并需要针对疑点提供可靠的解释；没有提出疑点的时候该部分留空。
- 源码示例应尽量使用当前工程和官方教程中的写法，保持可以直接参考和改写。
- 如果同一个知识点有多种常见写法，在 `源码示例` 下给出多个示例。
- 源码示例中的关键代码需要补充中文注释，便于直接阅读和理解。
- 在示例中，尽量打印 `AIMessage` 对象和解析后的文本结果，便于观察模型原始返回值。
- 在询问 `代码` 内的 `名词` 定义时，按照 `注释` 的格式记录在对应的代码后面。
- 大章节应具有对应的源码文件，并且在大章节描述部分指明

## 大章节：`llm_chain`

说明：本章对应当前仓库中的 [llm_chain_demo.py](/home/z/share/learn_pr/langchain/tutorials/langchain/basics/llm_chain_demo.py) 与 [langserve_server.py](/home/z/share/learn_pr/langchain/tutorials/langchain/serve/langserve_server.py)，内容基于 LangChain 官方入门教程里的“消息、输出解析器、提示词模板、LangServe 服务化”这条主线整理。

### 问题 1 [ACH]：`SystemMessage` 和 `HumanMessage` 是什么意思？

问题ID：`Q-LLM-01-system-human-message`

#### 定义

`SystemMessage` 和 `HumanMessage` 都是 LangChain 中的消息对象，用来描述一轮对话里不同角色发送的消息。

- `SystemMessage`：系统消息，用来给模型设定任务、规则或上下文。
- `HumanMessage`：用户消息，用来表示用户真正输入的内容。

它们共同组成发送给聊天模型的消息列表。

#### 功能

在官方教程的翻译示例里，这两个对象分别承担不同职责：

- `SystemMessage` 负责告诉模型“当前要做什么”，例如“把英文翻译成意大利语”。
- `HumanMessage` 负责提供本轮真正要处理的输入内容，例如 `hi!`。

模型执行 `invoke(...)` 时，会按消息顺序理解上下文，因此这两个对象是最基础的“对话输入单元”。

#### 源码示例

代码示例 1：直接构造消息列表调用模型

```python
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# 初始化聊天模型
model = ChatOpenAI(
    model="gpt-5.4",
    base_url="https://api.udcode.cn/v1",
    api_key=os.getenv("OPENAI_API_KEY", "你的API_KEY"),
)

# SystemMessage 负责描述任务
# HumanMessage 负责提供用户输入
messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"),
]

# 调用模型，返回值通常是 AIMessage 对象
response = model.invoke(messages)

# 输出完整的 AIMessage 对象，便于观察原始返回值
print(response)

# 输出模型回复中的文本内容
print(response.content)
```

#### 疑点


### 问题 2 [ACH]：`StrOutputParser` 是什么意思，应该怎么写？

问题ID：`Q-LLM-02-str-output-parser`

#### 定义

`StrOutputParser` 是 LangChain 中的输出解析器，用来把模型返回结果解析成普通字符串。

当你直接调用聊天模型时，返回值通常是 `AIMessage` 对象；而 `StrOutputParser` 的作用，就是把这个对象里的文本内容提取出来，得到更容易继续处理的字符串结果。

#### 功能

在官方教程里，`StrOutputParser` 的主要用途有两个：

- 把 `AIMessage` 解析为普通字符串，方便打印、拼接和后续处理。
- 作为链上的一个步骤，接在模型后面，让整条链直接返回文本结果。

它常见的两种写法是：

- 分步写法：先 `model.invoke(...)`，再 `parser.invoke(...)`。
- 链式写法：直接把 `model | parser` 串成一条链。

#### 源码示例

代码示例 1：分步写法

```python
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-5.4",
    base_url="https://api.udcode.cn/v1",
    api_key=os.getenv("OPENAI_API_KEY", "你的API_KEY"),
)

messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"),
]

# 创建字符串输出解析器
parser = StrOutputParser()

# 第一步：调用模型，得到 AIMessage 对象
ai_message = model.invoke(messages)
print(ai_message)

# 第二步：把 AIMessage 解析成普通字符串
text = parser.invoke(ai_message)
print(text)
```

代码示例 2：链式写法

```python
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-5.4",
    base_url="https://api.udcode.cn/v1",
    api_key=os.getenv("OPENAI_API_KEY", "你的API_KEY"),
)

messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"),
]

parser = StrOutputParser()

# 使用 | 把模型和输出解析器串成一条链
chain = model | parser

# 这时链的最终输出已经是字符串
result = chain.invoke(messages)
print(result)
```

#### 疑点

### 问题 3 [ACH]：`ChatPromptTemplate` 是什么意思，该怎么理解？

问题ID：`Q-LLM-03-chat-prompt-template`

#### 定义

`ChatPromptTemplate` 是 LangChain 中的聊天提示词模板，用来根据输入变量动态生成一组聊天消息。

它不是模型，也不是输出解析器，而是位于模型调用之前的一层“提示词构造器”。你可以先把消息结构定义好，等调用时再把变量填进去。

#### 功能

在官方教程里，`ChatPromptTemplate` 主要解决的是“把固定消息结构模板化”的问题。

它的作用包括：

- 把写死的消息列表变成可复用模板。
- 把“提示词结构”和“调用时传入的数据”拆开。
- 可以和模型、输出解析器用 `|` 连接成完整链路。

在翻译示例里，它会把 `language` 和 `text` 这些变量填入模板，最终生成真正发送给模型的消息列表。

#### 源码示例

代码示例 1：只用提示词模板生成消息

```python
from langchain_core.prompts import ChatPromptTemplate

# 定义系统提示词模板，其中 {language} 是待填充变量
system_template = "Translate the following into {language}:"

# 创建聊天提示词模板
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{text}"),
])

# 传入变量，生成真正的聊天消息
prompt_value = prompt_template.invoke(
    {
        "language": "italian",
        "text": "hi",
    }
)

# 查看模板展开后的消息列表
print(prompt_value.to_messages())
```

代码示例 2：把提示词模板接到模型和输出解析器后面

```python
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-5.4",
    base_url="https://api.udcode.cn/v1",
    api_key=os.getenv("OPENAI_API_KEY", "你的API_KEY"),
)

parser = StrOutputParser()

system_template = "Translate the following into {language}:"

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{text}"),
])

# 先构造“提示词模板 + 模型”链，便于观察 AIMessage
message_chain = prompt_template | model
ai_message = message_chain.invoke(
    {
        "language": "italian",
        "text": "hi",
    }
)
print(ai_message)

# 再把输出解析器接上，得到普通字符串结果
chain = prompt_template | model | parser
result = chain.invoke(
    {
        "language": "italian",
        "text": "hi",
    }
)
print(result)
```

#### 疑点


### 问题 4 [ACH]：使用 `LangServe` 提供服务是什么意思？

问题ID：`Q-LLM-04-langserve`

#### 定义

`LangServe` 是 LangChain 生态中的服务化工具，用来把 LangChain 的 `Runnable`、链或应用发布成一个可以通过 HTTP 调用的服务。

它的重点不是“帮助你写链”，而是“把已经写好的链暴露成 API”。

#### 功能

在官方教程里，使用 `LangServe` 的意义主要有这些：

- 把链封装成 REST API。
- 基于 `FastAPI` 暴露服务接口。
- 自动提供可视化 playground，方便测试。
- 可以在客户端使用 `RemoteRunnable` 像本地链一样远程调用服务。

简单理解就是：

- 不使用 `LangServe` 时，你只能在本地脚本里运行 `chain.invoke(...)`。
- 使用 `LangServe` 后，这条链可以被其他程序通过 HTTP 接口调用。

#### 源码示例

代码示例 1：服务端写法

```python
import os

from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes

system_template = "Translate the following into {language}:"

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{text}"),
])

model = ChatOpenAI(
    model="gpt-5.4",
    base_url="https://api.udcode.cn/v1",
    api_key=os.getenv("OPENAI_API_KEY", "你的API_KEY"),
)

parser = StrOutputParser()

# 把提示词模板、模型、输出解析器串成一条链
chain = prompt_template | model | parser

# 创建 FastAPI 应用
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

# 把链挂到 /chain 路径下
add_routes(app, chain, path="/chain")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8080)
```

代码示例 2：客户端远程调用写法

```python
from langserve import RemoteRunnable

# 连接到已经启动好的 LangServe 服务
remote_chain = RemoteRunnable("http://localhost:8080/chain/")

# 像调用本地链一样调用远程服务
result = remote_chain.invoke(
    {
        "language": "德语",
        "text": "hi",
    }
)

print(result)
```

#### 疑点


## 大章节：`chatbot`

说明：本章对应当前仓库中的 [chatbot_basic.py](/home/z/share/learn_pr/langchain/tutorials/langchain/history/chatbot_basic.py)、[chatbot_history.py](/home/z/share/learn_pr/langchain/tutorials/langchain/history/chatbot_history.py)、[chatbot_stream.py](/home/z/share/learn_pr/langchain/tutorials/langchain/history/chatbot_stream.py) 与 [chatbot_trim_history.py](/home/z/share/learn_pr/langchain/tutorials/langchain/history/chatbot_trim_history.py)，内容基于 LangChain 官方 chatbot 教程里的“提示词模板、历史消息、会话隔离、流式输出、消息裁剪”这条主线整理。

### 问题 1 [ACH]：消息历史（`Message History`）是什么，该怎么理解？

问题ID：`Q-CHAT-01-message-history`

#### 定义

消息历史是聊天机器人在多轮对话中保存“之前说过的话”的机制。

在 LangChain 里，模型本身不会自动长期记忆；所谓“记住上下文”，本质上是程序把历史消息保存下来，并在下一次调用时再一起传给模型。

#### 功能

在官方 chatbot 教程里，消息历史主要有这些作用：

- 让聊天机器人支持多轮对话，而不是每次只看当前一句话。
- 让模型在后续提问中利用之前的上下文，例如记住用户名字。
- 通过 `session_id` 区分不同会话。
- 为后续组合提示词模板、检索、工具调用等能力打基础。

教程里的关键做法是：

- 使用 `RunnableWithMessageHistory` 给原始链增加“自动读写历史”的能力。
- 使用 `get_session_history(session_id)` 返回当前会话对应的历史对象。
- 调用时通过 `config={"configurable": {"session_id": "xxx"}}` 指定当前会话。
- 当输入是字典时，用 `input_messages_key="messages"` 指明哪个字段存放消息列表。

#### 源码示例

代码示例 1：给模型增加消息历史能力

```python
import os

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-5.4",
    base_url="https://api.udcode.cn/v1",
    api_key=os.getenv("OPENAI_API_KEY", "你的API_KEY"),
)

# 使用字典保存不同 session_id 对应的聊天历史
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # 新会话会创建新的历史对象
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 给模型包装上消息历史能力
with_message_history = RunnableWithMessageHistory(
    model,
    get_session_history,
)

config = {"configurable": {"session_id": "abc2"}}

# 第一次对话：告诉模型名字
response = with_message_history.invoke(
    [HumanMessage(content="Hi! I'm Bob")],
    config=config,
)
print(response)
print(response.content)

# 第二次对话：继续追问名字
response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)
print(response)
print(response.content)
```

#### 疑点


### 问题 2 [ACH]：提示词模板（`Prompt Template`）是什么，该怎么理解？

问题ID：`Q-CHAT-02-prompt-template`

#### 定义

在 chatbot 场景里，提示词模板就是“生成最终聊天消息结构的模板”。

这里最常见的实现是 `ChatPromptTemplate`。它不是模型，也不是历史记录本身，而是用来规定“最终要发给模型的消息长什么样”。

#### 功能

在官方 chatbot 教程里，引入提示词模板主要是为了：

- 给 chatbot 增加稳定的 `system` 指令。
- 把“消息结构”和“调用数据”拆开。
- 配合 `MessagesPlaceholder` 把历史消息插入到指定位置。
- 同时接收历史消息和额外变量，例如 `language`。

可以把它理解为：

- `ChatPromptTemplate` 决定消息骨架。
- `MessagesPlaceholder` 决定历史消息插入的位置。
- `invoke({...})` 负责把变量真正填进去。

#### 源码示例

代码示例 1：不使用提示词模板，直接手写最终消息

```python
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-5.4",
    base_url="https://api.udcode.cn/v1",
    api_key=os.getenv("OPENAI_API_KEY", "你的API_KEY"),
)

# 这里是已经完全展开后的消息列表
messages = [
    SystemMessage(
        content="You are a helpful assistant. Answer all questions to the best of your ability in 中文."
    ),
    HumanMessage(content="hi! I'm Todd"),
]

response = model.invoke(messages)
print(response)
print(response.content)
```

代码示例 2：使用提示词模板和消息占位符

```python
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 这里只定义消息结构模板，还没有调用模型
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
    ),
    # 当前用户输入或历史消息会插入到这里
    MessagesPlaceholder(variable_name="messages"),
])

# 传入变量后，模板才会展开成真正的消息列表
prompt_value = prompt.invoke(
    {
        "messages": [HumanMessage(content="hi! I'm Todd")],
        "language": "中文",
    }
)

print(prompt_value.to_messages())
```

代码示例 3：把提示词模板接到带消息历史的 chatbot 链中

```python
import os

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-5.4",
    base_url="https://api.udcode.cn/v1",
    api_key=os.getenv("OPENAI_API_KEY", "你的API_KEY"),
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
    ),
    MessagesPlaceholder(variable_name="messages"),
])

# 先把提示词模板和模型串起来
chain = prompt | model

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 当前输入是字典，所以要显式指定消息字段名
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

config = {"configurable": {"session_id": "abc11"}}

response = with_message_history.invoke(
    {
        "messages": [HumanMessage(content="hi! I'm Todd")],
        "language": "中文",
    },
    config=config,
)
print(response)
print(response.content)
```

#### 疑点


### 问题 3 [ACH]：`variable_name` 和 `content` 有什么关系？

问题ID：`Q-CHAT-03-variable-name-and-content`

#### 定义

`variable_name` 和 `content` 不在同一层，它们处理的是两件不同的事：

- `variable_name`：模板变量名，用来告诉 LangChain 去哪里取值。
- `content`：消息对象里的实际内容，也就是最终发给模型的文本。

以 `MessagesPlaceholder(variable_name="messages")` 为例，这里的 `variable_name="messages"` 不是消息正文，而是在说“这个位置要从输入参数里的 `messages` 字段取一组消息插进来”。

而 `HumanMessage(content="hi! I'm Todd")` 里的 `content`，才是真正的消息文本。

#### 功能

在官方教程语境里，它们的分工可以概括成：

- `variable_name` 负责“找数据”。
- `content` 负责“装数据”。

所以两者的关系不是“同义字段”，而是：

- `variable_name` 决定模板去哪个键里取消息。
- `content` 决定每条消息里真正写了什么。

#### 源码示例

代码示例 1：先看两者分别出现在哪一层

```python
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    # variable_name 指定模板变量名
    MessagesPlaceholder(variable_name="messages"),
])

input_data = {
    # 这里的键名必须和 variable_name="messages" 对应
    "messages": [
        # content 才是这一条用户消息真正的文本内容
        HumanMessage(content="hi! I'm Todd"),
    ]
}

prompt_value = prompt.invoke(input_data)
print(prompt_value.to_messages())
```

代码示例 2：结合 chatbot 例子一起看

```python
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
    ),
    # 告诉模板：去 invoke(...) 的参数里找 messages 这个字段
    MessagesPlaceholder(variable_name="messages"),
])

prompt_value = prompt.invoke(
    {
        # 这个键名要和 variable_name="messages" 对上
        "messages": [
            # 这一条消息真正的正文写在 content 里
            HumanMessage(content="hi! I'm Todd"),
        ],
        "language": "中文",
    }
)

print(prompt_value.to_messages())
```

#### 疑点


### 问题 4 [ACH]：官方例程里，对话历史是如何管理的？

问题ID：`Q-CHAT-04-history-management`

#### 定义

在官方 LangChain chatbot 例程里，对话历史不是由模型自己保存，而是由程序在链外单独维护。

它的核心思路是：

- 用一个历史存储容器保存不同会话的消息记录。
- 用 `session_id` 区分不同会话。
- 每次调用前先取历史，再把当前输入和历史一起交给链。
- 调用结束后，当前轮的新输入和模型新输出会自动写回历史。

#### 功能

官方例程中的历史管理，大致分成 4 步：

1. 准备一个外部存储，例如 `store = {}`。
2. 定义 `get_session_history(session_id)`，按会话 ID 获取或创建历史对象。
3. 用 `RunnableWithMessageHistory` 包装原始链。
4. 调用时通过 `configurable.session_id` 指定当前会话。

可以把它压缩成一句话：

- 历史存储在链外。
- 会话依靠 `session_id` 隔离。
- `RunnableWithMessageHistory` 负责自动读写历史。

#### 源码示例

代码示例 1：最小化的历史管理结构

```python
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)

# 外部历史存储：按 session_id 保存不同会话的历史
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # 新会话创建新的历史对象
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
```

代码示例 2：在 chatbot 链中使用历史管理

```python
import os

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-5.4",
    base_url="https://api.udcode.cn/v1",
    api_key=os.getenv("OPENAI_API_KEY", "你的API_KEY"),
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
    ),
    MessagesPlaceholder(variable_name="messages"),
])

chain = prompt | model

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

# 会话 abc11：第一次提到名字
config = {"configurable": {"session_id": "abc11"}}
response = with_message_history.invoke(
    {
        "messages": [HumanMessage(content="hi! I'm Todd")],
        "language": "中文",
    },
    config=config,
)
print(response)
print(response.content)

# 会话 abc11：继续追问，会复用上一轮历史
response = with_message_history.invoke(
    {
        "messages": [HumanMessage(content="What's my name?")],
        "language": "中文",
    },
    config=config,
)
print(response)
print(response.content)

# 会话 abc01：新的 session_id，不会继承 abc11 的历史
config = {"configurable": {"session_id": "abc01"}}
response = with_message_history.invoke(
    {
        "messages": [HumanMessage(content="What's my name?")],
        "language": "中文",
    },
    config=config,
)
print(response)
print(response.content)
```

#### 疑点

##### 追加提问 1：官方例程中使用了“管理消息列表”，这是什么？

问题描述：

在官方 chatbot 例程里，代码里一直围绕 `messages` 这个字段组织输入，还配合 `MessagesPlaceholder(variable_name="messages")` 和 `RunnableWithMessageHistory(..., input_messages_key="messages")` 使用。这里的“管理消息列表”具体指什么，容易和“管理对话历史”混在一起，不容易一下看懂。

问题解答：

这里的“管理消息列表”并不是一个单独的新类，而是 LangChain 官方教程里组织对话上下文的一种方式。

可以拆成 3 层理解：

1. 在 LangChain 里，`messages` 本身就是聊天模型最核心的输入形式。

- 官方文档把 `messages` 视为模型上下文的基础单元。
- 一个消息列表里会按顺序放入 `SystemMessage`、`HumanMessage`、`AIMessage` 等对象。
- 模型拿到的其实不是“一个字符串”，而是“一个按顺序排列的消息列表”。

2. 在当前官方例程里，`messages` 字段是“当前要交给模板或链处理的消息列表入口”。

- `MessagesPlaceholder(variable_name="messages")` 的意思是：
  - 在提示词模板中预留一个位置。
  - 调用时把 `messages` 这个字段里的消息列表插进来。
- 所以这里的“管理消息列表”，首先是在管理“这一轮调用要送进去的消息结构”。

3. `RunnableWithMessageHistory` 会进一步帮你管理“历史消息如何并入这个消息列表”。

- 当你设置 `input_messages_key="messages"` 时，就是在告诉 LangChain：
  - 当前输入字典里，`messages` 这个键保存的是本轮输入消息。
- 再结合 `session_id`，LangChain 会在调用前读取历史消息，在调用后把新消息写回历史。
- 所以你看到的“管理消息列表”，本质上是：
  - 一部分在管理“当前输入消息”。
  - 一部分在管理“历史消息如何自动合并进来”。

如果用一句话压缩：

- “管理消息列表”就是把一轮对话要发送给模型的各条消息，统一用 `messages` 这个列表组织起来，并由模板和历史组件按规则把它们拼成最终上下文。

结合当前仓库里的例子：

- 在 [chatbot_basic.py](/home/z/share/learn_pr/langchain/tutorials/langchain/history/chatbot_basic.py) 里，`MessagesPlaceholder(variable_name="messages")` 负责告诉提示词模板“把消息列表插到这里”。
- 在 [chatbot_basic.py](/home/z/share/learn_pr/langchain/tutorials/langchain/history/chatbot_basic.py) 里，`RunnableWithMessageHistory(..., input_messages_key="messages")` 负责告诉历史管理器“当前输入消息放在 `messages` 这个字段里”。
- 两者配合后，最终送给模型的就不是单独一句话，而是一整组按顺序组织好的消息。

### 问题 5 [ACH]：`trim_messages` 是什么意思，该怎么理解？

问题ID：`Q-CHAT-05-trim-messages`

#### 定义

`trim_messages` 是 LangChain 提供的消息裁剪工具，用来把一组聊天消息裁剪到指定的 token 上限或消息数量上限以内。

它的核心作用不是“生成消息”，而是“在消息过长时，按规则删减消息列表”，让最终传给模型的上下文仍然满足长度限制。

#### 功能

按照 LangChain 官方参考文档，`trim_messages` 常用于聊天历史过长时的上下文控制。它在官方建议里的常见目标有这几个：

- 让裁剪后的消息总量不超过指定上限。
- 优先保留最近消息，而不是更早的旧消息。
- 在有 `SystemMessage` 时，尽量保留系统指令。
- 保证裁剪后的消息列表仍然是一个“对聊天模型合法”的结构，例如以 `HumanMessage` 开始，或者以 `SystemMessage + HumanMessage` 开始。

你这段代码里的参数可以这样理解：

- `max_tokens=65`：裁剪后的消息总 token 数不能超过 65。
- `strategy="last"`：优先保留后面的新消息，丢弃前面的旧消息。
- `token_counter=model`：使用当前模型自己的 tokenizer 来计算消息 token 数。官方参考文档说明，当这里传入 `BaseLanguageModel` 时，会调用 `get_num_tokens_from_messages()`。
- `include_system=True`：如果原始消息列表第 0 条是 `SystemMessage`，并且策略是 `last`，就尽量把它保留下来。
- `allow_partial=False`：不允许只保留某条消息的一部分内容；要保留就保留整条，要裁掉就整条裁掉。
- `start_on="human"`：裁剪后，消息列表应该从 `HumanMessage` 开始；如果第 0 条是系统消息，那么允许是 `SystemMessage` 后面接 `HumanMessage`。

所以这段代码的整体意思是：

- 保留系统提示词。
- 从后往前尽量保留最近的完整对话消息。
- 最终总长度不超过 65 tokens。
- 裁剪后的结果仍然要符合聊天模型常见的消息结构要求。

#### 源码示例

代码示例 1：官方思路下的消息裁剪写法

```python
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)

# 创建消息裁剪器
trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,  # 使用模型自己的 tokenizer 统计 token
    include_system=True,  # 尽量保留开头的系统消息
    allow_partial=False,  # 不截断半条消息，只保留完整消息
    start_on="human",     # 裁剪后从 human 开始，或 system + human 开始
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

# 裁剪消息列表
trimmed_messages = trimmer.invoke(messages)

# 输出裁剪后的消息对象列表
print(trimmed_messages)
```

代码示例 2：把这段逻辑翻译成更容易理解的话

```python
# 原始消息很多时，不一定能全部塞进模型上下文
# 所以先通过 trim_messages 把消息裁剪到 token 上限以内

trimmed_messages = trimmer.invoke(messages)

# 裁剪后的结果通常会满足这几个特征：
# 1. 开头的系统消息尽量保留
# 2. 更早的旧消息更容易被删掉
# 3. 最近几轮对话更容易被保留
# 4. 每条消息要么完整保留，要么完整删除

for message in trimmed_messages:
    print(type(message).__name__, message.content)
```

#### 疑点

##### 追加提问 1：官方例程中，为什么要先对 `messages` 运行修剪器，再传给提示模板，最后再包装进消息历史？

问题描述：

我想看一个完整的学习例子，理解官方教程里这条链路是怎么串起来的：

1. 先把 `messages` 输入交给 `trim_messages` 裁剪。
2. 再把裁剪后的 `messages` 传给 `ChatPromptTemplate`。
3. 最后再用 `RunnableWithMessageHistory` 包装整条链。

尤其是想看清楚：历史消息、当前输入消息、修剪器、提示模板，这几层在代码里分别处于什么位置。

问题解答：

按照 LangChain 官方教程和参考文档，这种组合方式可以理解成“先控制上下文长度，再生成最终提示，再自动接入历史消息”。

它的顺序是有原因的：

1. `RunnableWithMessageHistory` 先负责把历史消息和当前输入消息合并起来。

- 也就是把“本会话以前的消息”和“这次新传入的消息”统一放进 `messages` 字段。
- 这样后面的步骤拿到的，就是一份完整的上下文消息列表。

2. `trim_messages` 再负责把这份完整消息列表裁剪到可接受长度。

- 如果不先裁剪，历史越来越长时，后面的 prompt 和模型调用就可能超出上下文限制。
- 所以修剪器的位置应该放在“消息真正进入 prompt 之前”。

3. `ChatPromptTemplate` 最后把裁剪后的 `messages` 插入提示模板。

- 例如前面保留 `system` 指令，后面插入已经裁好的 `messages`。
- 这样最终交给模型的消息结构既完整，又不会太长。

如果压缩成一句话：

- `RunnableWithMessageHistory` 负责“拿到完整历史”。
- `trim_messages` 负责“把完整历史裁短”。
- `ChatPromptTemplate` 负责“把裁短后的消息装进最终提示结构”。

当前仓库里已经新增了一个完整学习脚本：

- [chatbot_trim_history.py](/home/z/share/learn_pr/langchain/tutorials/langchain/history/chatbot_trim_history.py)

这个脚本专门演示了下面这条链路：

- 先用 `RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)` 在进入 prompt 前修剪 `messages`。
- 再用 `ChatPromptTemplate(..., MessagesPlaceholder("messages"))` 生成最终提示。
- 最后用 `RunnableWithMessageHistory(..., input_messages_key="messages")` 包装整条链。

你可以重点看这几处：

- [chatbot_trim_history.py](/home/z/share/learn_pr/langchain/tutorials/langchain/history/chatbot_trim_history.py#L24) 到 [chatbot_trim_history.py](/home/z/share/learn_pr/langchain/tutorials/langchain/history/chatbot_trim_history.py#L55)：定义 prompt、trimmer，以及“先 trim 再进 prompt”的链。
- [chatbot_trim_history.py](/home/z/share/learn_pr/langchain/tutorials/langchain/history/chatbot_trim_history.py#L57) 到 [chatbot_trim_history.py](/home/z/share/learn_pr/langchain/tutorials/langchain/history/chatbot_trim_history.py#L76)：用 `RunnableWithMessageHistory` 包装整条链。
- [chatbot_trim_history.py](/home/z/share/learn_pr/langchain/tutorials/langchain/history/chatbot_trim_history.py#L79) 到 [chatbot_trim_history.py](/home/z/share/learn_pr/langchain/tutorials/langchain/history/chatbot_trim_history.py#L129)：在真正调用前，打印“合并后的 messages”“裁剪后的 messages”“prompt 展开后的最终消息”，便于直接观察整条链路。

补充说明：

如果你观察到“最后一轮 `invoke_with_preview(...)` 的裁剪输出里几乎看不到历史信息”，通常不是因为历史没有传进来，而是因为这份示例里历史已经在裁剪阶段被删掉了。

在当前脚本中，关键逻辑是：

- [chatbot_trim_history.py](/home/z/share/learn_pr/langchain/tutorials/langchain/history/chatbot_trim_history.py#L91)：先把 `history.messages + current_messages` 合并起来。
- [chatbot_trim_history.py](/home/z/share/learn_pr/langchain/tutorials/langchain/history/chatbot_trim_history.py#L92)：再立刻把这份合并结果交给 `trimmer.invoke(...)` 裁剪。

所以需要区分两件事：

1. 历史有没有进入链路。

- 进入了，因为 `merged_messages = history.messages + current_messages` 已经把历史消息和当前输入拼在一起了。
- 你真正应该先看的是打印出来的 `=== 合并后的 messages ===`，这里才能证明历史是否存在。

2. 为什么 `=== 裁剪后的 messages ===` 里可能看不到历史。

- 因为 `trim_messages` 配的是 `max_tokens=65` 和 `strategy="last"`。
- 官方文档对 `strategy="last"` 的含义就是优先保留最后面的新消息。
- 这意味着越早的历史消息越容易先被裁掉。
- 再加上 `allow_partial=False`，一条消息装不下时不会截半条，而是整条删除。

所以第三轮看起来“没有历史”的常见原因是：

- 历史确实先被合并进来了。
- 但是在 token 预算只有 65 的前提下，后面更新、更长的消息把前面的历史都挤掉了。
- 最后打印 `trimmed_messages` 时，你看到的就只剩下最后一部分消息，甚至可能只剩当前输入附近的几条。

还有一个容易误解的点：

- [chatbot_trim_history.py](/home/z/share/learn_pr/langchain/tutorials/langchain/history/chatbot_trim_history.py#L111) 到 [chatbot_trim_history.py](/home/z/share/learn_pr/langchain/tutorials/langchain/history/chatbot_trim_history.py#L123) 里，`preview_pipeline(...)` 是在真正 `with_message_history.invoke(...)` 之前执行的。
- 这意味着预览阶段看到的历史，只包含“前几轮已经写回 store 的历史”，不会包含“当前这一轮调用产生的新 AIMessage”。

所以更准确的一句话是：

- 最后一轮不是“没有历史”，而是“历史先被合并，再因为 `trim_messages(strategy=\"last\", max_tokens=65)` 的规则被裁掉了，而且当前轮的新回复在预览阶段还没写回历史”。

进一步压缩成一句更直观的话：

- 是的，关键原因就是这里把 `max_tokens` 限制成了 `65`。当总消息 token 超过 65 时，`strategy="last"` 会优先保留靠后的新消息，因此更早的旧历史会先被裁掉。

### 问题 6 [ACH]：流式处理和 `.stream()` 应该怎么用？

问题ID：`Q-CHAT-06-stream`

#### 定义

在 LangChain 官方教程里，流式处理指的是：模型在生成结果时，不等整段回答完全结束才一次性返回，而是边生成边把结果分块产出。

`.stream()` 就是 LangChain 为这种调用方式暴露出的同步流式接口。它不会像 `.invoke()` 那样一次性返回最终完整结果，而是返回一个可以迭代的流，调用方可以一边接收、一边打印、一边处理。

#### 功能

按照 LangChain 官方文档，`.stream()` 的核心用途有这些：

- 在模型还没生成完全部内容时，就先拿到前面的输出。
- 适合做终端实时打印、聊天界面逐字显示、长回答的增量处理。
- 既可以直接对聊天模型使用，也可以对由 `prompt | model` 组成的 Runnable 链使用。

可以把 `.invoke()` 和 `.stream()` 的区别理解成：

- `.invoke()`：等模型全部生成完，再一次性把完整结果返回。
- `.stream()`：模型每生成一部分，就先返回一个 chunk。

在聊天模型场景里，`.stream()` 常见地会产出多个 `AIMessageChunk`。如果你既想实时显示，又想最后拿到完整结果，可以把这些 chunk 用 `+` 逐步聚合起来。

#### 源码示例

代码示例 1：直接对聊天模型使用 `.stream()`

```python
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-5.4",
    base_url="https://api.udcode.cn/v1",
    api_key=os.getenv("OPENAI_API_KEY", "你的API_KEY"),
)

messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"),
]

# stream() 会持续产出多个 AIMessageChunk
full = None
for chunk in model.stream(messages):
    # 实时打印流式输出
    print(chunk.text, end="", flush=True)

    # 同时把 chunk 累加回完整结果
    full = chunk if full is None else full + chunk

print()
print(full)
print(full.text if full is not None else "")
```

代码示例 2：对提示词模板 + 模型组成的链使用 `.stream()`

```python
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-5.4",
    base_url="https://api.udcode.cn/v1",
    api_key=os.getenv("OPENAI_API_KEY", "你的API_KEY"),
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Translate the following into {language}."),
    ("user", "{text}"),
])

chain = prompt | model

full = None
for chunk in chain.stream(
    {
        "language": "Italian",
        "text": "I love building with LangChain.",
    }
):
    print(chunk.text, end="", flush=True)
    full = chunk if full is None else full + chunk

print()
print(full)
print(full.text if full is not None else "")
```

代码示例 3：当前工程里的独立学习脚本

```python
# 运行当前仓库中的流式示例脚本
python tutorials/langchain/history/chatbot_stream.py
```

当前脚本位置：

- [chatbot_stream.py](/home/z/share/learn_pr/langchain/tutorials/langchain/history/chatbot_stream.py)

代码示例 4：加上 `StrOutputParser` 后，直接流式拿字符串

```python
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-5.4",
    base_url="https://api.udcode.cn/v1",
    api_key=os.getenv("OPENAI_API_KEY", "你的API_KEY"),
)

parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Translate the following into {language}."),
    ("user", "{text}"),
])

chain = prompt | model | parser

full_text = ""
for chunk in chain.stream(
    {
        "language": "Italian",
        "text": "I love building with LangChain.",
    }
):
    # 这里拿到的已经是字符串 chunk，而不是 AIMessageChunk
    print(chunk, end="", flush=True)
    full_text += chunk

print()
print(full_text)
```

当前脚本包含三部分：

- `model.stream(messages)`：直接对聊天模型流式输出。
- `chain.stream({...})`：对 `ChatPromptTemplate | ChatOpenAI` 这条链进行流式输出。
- `chain.stream({...}) + StrOutputParser`：直接流式拿解析后的字符串。

#### 疑点

##### 追加提问 1：我是不是可以理解成，`response` 具有 `.invoke()` 和 `.stream()` 两种输出方式？

问题描述：

我现在容易把“调用对象”和“调用结果”混在一起。看到教程里有 `.invoke()` 和 `.stream()`，我会下意识觉得是不是 `response` 这个结果对象本身有两种输出方式：一种是 `.invoke()`，一种是 `.stream()`。

问题解答：

更准确的理解不是“`response` 有两种输出方式”，而是：

- 具有 `.invoke()` 和 `.stream()` 这两种调用方式的，通常是 `model`、`prompt | model` 得到的 `chain`，或者更一般的 `Runnable` 对象。
- `response` 是这些对象执行之后返回的结果，不是再拿来调用 `.invoke()` / `.stream()` 的主体。

也就是说，关系应该这样理解：

- `model` / `chain`：负责“怎么调用”
- `response`：负责“调用之后拿到什么结果”

你可以把它和当前脚本对照起来看：

- 在 [chatbot_stream.py](/home/z/share/learn_pr/langchain/tutorials/langchain/history/chatbot_stream.py#L31) 里，调用的是 `model.stream(messages)`，说明 `stream()` 挂在 `model` 上。
- 在 [chatbot_stream.py](/home/z/share/learn_pr/langchain/tutorials/langchain/history/chatbot_stream.py#L59) 里，调用的是 `chain.stream({...})`，说明 `stream()` 也可以挂在 `chain` 上。
- 当流式结束后，代码里把多个 chunk 聚合到 `full` 或 `full_text`，这些才是“返回结果”。

所以更准确的一句话是：

- 不是 `response` 同时支持 `.invoke()` 和 `.stream()`。
- 而是 `model` 或 `chain` 这样的 Runnable，可以选择用 `.invoke()` 一次性调用，或者用 `.stream()` 流式调用。

如果再压缩成一个最短公式，可以记成：

- `response = model.invoke(...)`
- `for chunk in model.stream(...): ...`

这里的方法都属于 `model`，不属于 `response`。

##### 追加提问 3：`Runnable` 又是什么？

问题描述：

在前面的解释里一直提到 `model`、`chain` 都属于 `Runnable`，但我还没有真正建立这个概念。想知道在 LangChain 官方教程和参考文档语境里，`Runnable` 到底是什么，它和 `model`、`chain` 有什么关系。

问题解答：

按照 LangChain 官方参考文档，`Runnable` 可以理解成：

- 一个“可执行的工作单元”。
- 这个工作单元可以被调用、批量执行、流式执行、转换、以及和其他工作单元组合起来。

官方原话的核心意思是：

- `Runnable` 是 LangChain 里统一调用接口的抽象层。
- 只要某个对象是 `Runnable`，它通常就会有一套统一的调用方法，例如 `.invoke()`、`.batch()`、`.stream()`、`.ainvoke()`、`.astream()`。

所以你可以把 `Runnable` 理解成“LangChain 里所有可调用步骤的共同协议”。

这时候再看你前面接触到的对象，就清楚了：

1. `model` 为什么能 `.invoke()` 和 `.stream()`

- 因为聊天模型对象本身就是一种 `Runnable`。
- 所以它天然支持 LangChain 统一的调用接口。

2. `chain` 为什么也能 `.invoke()` 和 `.stream()`

- 因为 `prompt | model` 组合出来的不是普通 Python 值，而是 `RunnableSequence`。
- `RunnableSequence` 也是 `Runnable` 的一种。
- 所以整条链在组合完成后，仍然保留统一的调用能力。

3. 为什么 LangChain 里很多不同对象看起来调用方式都差不多

- 因为它们底层都遵循同一套 `Runnable` 接口。
- 这也是为什么 `prompt`、`model`、`parser`、`chain` 可以用同样的 LCEL 语法拼接起来。

可以把它压缩成一句话：

- `Runnable` 不是某个具体业务对象，而是 LangChain 对“可执行步骤”的统一抽象。

如果再往下拆一层关系：

- `ChatOpenAI`：具体的聊天模型对象，同时也是 `Runnable`
- `ChatPromptTemplate`：提示词模板对象，也可以参与 Runnable 组合
- `prompt | model`：组合后得到 `RunnableSequence`
- `prompt | model | parser`：仍然是 `RunnableSequence`

所以前面那几个名词之间的关系可以整理成：

- `Runnable`：抽象层，表示“可以被 LangChain 统一调用的对象”
- `model`：某个具体的 Runnable
- `chain`：多个 Runnable 组合后的新 Runnable
- `chunk`：Runnable 在 `.stream()` 过程中逐步产出的片段
- `response`：Runnable 在 `.invoke()` 结束后的一次性最终结果

如果你只记一个最短版本，可以记成：

- `Runnable` = “LangChain 里的可调用对象接口”
- `model` 和 `chain` 都是 `Runnable`
- 所以它们都能用 `.invoke()` / `.stream()`

##### 追加提问 2：`model`、`chain`、`chunk`、`response` 的概念分别是什么？

问题描述：

我现在对流式调用里的几个名词还有点混乱，尤其是：

- `model`
- `chain`
- `chunk`
- `response`

我想知道它们在 LangChain 官方教程语境里分别是什么，以及它们之间是什么关系。

问题解答：

可以把这 4 个概念按“调用前 / 调用中 / 调用后”来理解，这样最清楚。

1. `model`：最基础的调用对象

- 在官方教程里，`model` 通常指聊天模型实例，例如 `ChatOpenAI(...)`。
- 它本身是一个可调用的 Runnable，可以执行 `.invoke()`、`.stream()` 等方法。
- 你可以把它理解成“真正负责向模型提供商发请求的对象”。

例如：

- [chatbot_stream.py](/home/z/share/learn_pr/langchain/tutorials/langchain/history/chatbot_stream.py#L19) 里的 `model = build_model()`
- [chatbot_stream.py](/home/z/share/learn_pr/langchain/tutorials/langchain/history/chatbot_stream.py#L31) 里的 `model.stream(messages)`

2. `chain`：把多个步骤串起来后的调用对象

- 在 LangChain 官方教程里，`chain` 通常是通过 `prompt | model` 或 `prompt | model | parser` 这种 LCEL 写法组合出来的。
- 它仍然是 Runnable，所以也可以执行 `.invoke()`、`.stream()`。
- 你可以把它理解成“比 model 更高一层的流水线对象”。

例如：

- [chatbot_stream.py](/home/z/share/learn_pr/langchain/tutorials/langchain/history/chatbot_stream.py#L54) 的 `chain = prompt | model`
- [chatbot_stream.py](/home/z/share/learn_pr/langchain/tutorials/langchain/history/chatbot_stream.py#L87) 的 `chain = prompt | model | parser`

所以 `model` 和 `chain` 的关系是：

- `model` 是单一步骤。
- `chain` 是多个步骤组合后的整体调用对象。

3. `chunk`：流式调用过程中产出的“中间小块”

- 官方文档说明，在聊天模型 streaming 时，会持续收到 `AIMessageChunk`。
- 如果链最后接了 `StrOutputParser`，那么流出来的可能就不再是 `AIMessageChunk`，而是字符串 chunk。
- 你可以把 `chunk` 理解成“流式调用过程中，一次迭代拿到的一小段结果”。

例如：

- [chatbot_stream.py](/home/z/share/learn_pr/langchain/tutorials/langchain/history/chatbot_stream.py#L31) 到 [chatbot_stream.py](/home/z/share/learn_pr/langchain/tutorials/langchain/history/chatbot_stream.py#L37)：这里的 `chunk` 是 `AIMessageChunk`
- [chatbot_stream.py](/home/z/share/learn_pr/langchain/tutorials/langchain/history/chatbot_stream.py#L92) 到 [chatbot_stream.py](/home/z/share/learn_pr/langchain/tutorials/langchain/history/chatbot_stream.py#L99)：这里的 `chunk` 是字符串

4. `response`：一次性调用后拿到的最终结果

- 在官方教程语境里，`response` 往往是 `.invoke()` 执行后的返回值。
- 如果直接对聊天模型调用 `.invoke()`，通常拿到的是 `AIMessage`。
- 如果对 `prompt | model | StrOutputParser()` 调用 `.invoke()`，最终拿到的就可能是字符串。
- 所以 `response` 更像是“最终结果变量名”，不是一种固定类名。

例如：

- `response = model.invoke(messages)` 时，`response` 通常是 `AIMessage`
- `response = chain.invoke({...})` 时，`response` 的类型取决于这条链最后输出什么

可以把这四者压缩成一张关系图：

- `model`：单个可调用步骤
- `chain`：多个步骤组合后的可调用流水线
- `chunk`：`.stream()` 过程中每次产出的一小块结果
- `response`：`.invoke()` 结束后一次性拿到的最终结果

如果再压缩成最短记忆法：

- `model` / `chain` 是“调用者”
- `chunk` 是“流式中的片段”
- `response` 是“最终返回值”


## 大章节：`agents`

说明：本章对应当前仓库中的 [agents_demo.py](/home/z/share/learn_pr/langchain/tutorials/langchain/agents/agents_demo.py)，内容基于 LangChain 官方教程中的“构建一个代理（Build an Agent）”一节，围绕“工具定义、模型绑定工具、`create_react_agent`、流式输出、记忆”这条主线整理。

### 问题 1 [ACH]：该章节内提到的代理是什么？

问题ID：`Q-AGENT-01-what-is-agent`

#### 定义

在这一章里，`代理（agent）` 不是单纯指“一个会聊天的模型”，而是指：

“使用大型语言模型作为推理引擎的系统”，它会根据当前问题判断是否需要采取行动、应该调用哪个工具、给工具传什么输入，并在拿到工具结果后决定是否继续下一步，或者直接结束并回复用户。

按照官方教程的语境，可以把它理解成：

- `大模型` 负责思考和决策。
- `工具` 负责执行外部动作，例如搜索天气。
- `代理` 负责把“思考”和“动作”串起来，形成完整执行流程。

所以，代理不是某一个单独的 API 名字，而是一种“让模型可以按需调用工具完成任务”的运行机制。

#### 功能

在本章官方教程里，代理主要承担这些功能：

- 接收用户消息，并判断当前问题是否需要调用工具。
- 如果需要工具，生成对应的 `tool_calls` 请求。
- 执行工具后，把工具结果再交回给模型继续推理。
- 当信息足够时，输出最终回答。
- 配合 `checkpointer` 和 `thread_id` 实现多轮对话记忆。

这也是为什么教程里会先演示：

- `model.bind_tools(tools)`：让模型具备“知道有哪些工具可用”的能力。
- `create_react_agent(model, tools)`：把模型和工具组装成真正可执行的代理。

可以直接这样理解：

- `bind_tools` 只是让模型“知道可以调用工具”。
- `agent` 才是真正会“判断 -> 调工具 -> 看结果 -> 再回答”的执行体。

#### 源码示例

代码示例 1：只绑定工具，还不是真正的代理

```python
import os

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4",
    api_key=os.getenv("OPENAI_API_KEY", "你的API_KEY"),
)

# 定义一个搜索工具
search = TavilySearchResults(max_results=2)
tools = [search]

# 这里只是把工具描述绑定给模型
# 此时模型可以“决定要不要调用工具”，但不会帮你自动执行工具
model_with_tools = model.bind_tools(tools)

response = model_with_tools.invoke(
    [HumanMessage(content="What's the weather in SF?")]
)

# 观察模型返回的原始 AIMessage
print(response)

# 如果模型认为需要工具，这里通常会出现 tool_calls
print(response.tool_calls)
```

代码示例 2：使用官方教程里的方式创建真正的代理

```python
import os

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

model = ChatOpenAI(
    model="gpt-4",
    api_key=os.getenv("OPENAI_API_KEY", "你的API_KEY"),
)

search = TavilySearchResults(max_results=2)
tools = [search]

# create_react_agent 会把“模型 + 工具”组装成一个真正可执行的代理
agent_executor = create_react_agent(model, tools)

# 这里传入的是消息列表，代理会自行判断是否需要调用工具
response = agent_executor.invoke(
    {
        "messages": [HumanMessage(content="What's the weather in SF?")]
    }
)

# 打印完整消息列表，便于观察代理执行后的最终状态
print(response["messages"])
```

代码示例 3：给代理增加记忆

```python
import os

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

model = ChatOpenAI(
    model="gpt-4",
    api_key=os.getenv("OPENAI_API_KEY", "你的API_KEY"),
)

search = TavilySearchResults(max_results=2)
tools = [search]

# MemorySaver 用来保存同一个线程下的历史状态
memory = MemorySaver()

agent_executor = create_react_agent(
    model,
    tools,
    checkpointer=memory,
)

# thread_id 用来标识当前会话
config = {"configurable": {"thread_id": "abc123"}}

result = agent_executor.invoke(
    {"messages": [HumanMessage(content="hi im bob!")]},
    config=config,
)
print(result["messages"])

result = agent_executor.invoke(
    {"messages": [HumanMessage(content="whats the weather where I live?")]},
    config=config,
)
print(result["messages"])
```

#### 疑点

##### 追加提问 1：章节中提到的流式令牌是什么？

问题描述：

在 `agents` 这一章里，官方教程除了演示“流式消息（stream messages）”，还单独提到了“流式令牌（stream tokens）”。这里容易困惑的是：消息和令牌看起来都像是在“边生成边返回”，但它们到底有什么区别，`stream` 和 `astream_events` 又分别在流什么。

问题解答：

按照 LangChain 官方教程在本章里的语境，`流式令牌（stream tokens）` 指的是：

- 模型在生成最终回复时，不等整段内容全部完成，而是把生成过程中的一个个小文本片段持续往外发送。
- 这些小片段通常可以理解成“逐步生成的 token 内容”或“极小粒度的输出块”。

它和“流式消息”的区别，关键在粒度不同：

1. `流式消息` 看的是“代理执行过程中的阶段性消息”。

- 官方教程里用的是 `agent_executor.stream(...)`。
- 你看到的通常是代理节点、工具节点返回的阶段性结果。
- 例如先看到模型发起 `tool_calls`，再看到工具执行结果，最后看到代理产出的最终 `AIMessage`。
- 所以它更像是在流“执行步骤”或“阶段结果”。

2. `流式令牌` 看的是“模型生成文本时的细粒度输出”。

- 官方教程里这里改用 `agent_executor.astream_events(..., version="v1")`。
- 然后监听 `on_chat_model_stream` 事件，从事件数据里不断取出 `chunk`。
- 每个 `chunk` 都是模型当前新吐出来的一小段内容。
- 所以它更像是在流“生成中的字词片段”，而不是完整一步。

可以直接压缩成一句话：

- `流式消息` 是按“步骤/消息”往外推。
- `流式令牌` 是按“模型生成的细小文本片段”往外推。

为什么官方教程要单独讲“流式令牌”？

- 因为代理执行通常比单次模型调用更慢，中间可能还要经过工具调用。
- 如果只等最终结果，界面会显得卡住。
- 流式令牌可以让你更早看到模型正在生成什么，适合做终端实时打印、聊天界面逐字显示、调试模型输出过程。

本章官方教程里的关键对应关系是：

- `agent_executor.stream(...)`：更偏向看代理步骤和中间消息。
- `agent_executor.astream_events(...)`：更偏向看事件流，其中 `on_chat_model_stream` 能拿到流式令牌。

如果再结合你前面学过的概念，这里的“流式令牌”可以理解成：

- 它和 `chatbot` 章节里 `.stream()` 的“chunk”是同一类思路。
- 只是到了 `agents` 章节，官方为了同时暴露“代理步骤事件 + 模型流式输出”，改成用事件流接口来观察。
- 所以这里本质上仍然是在看模型输出的增量片段，只是包在 agent 的事件体系里。

##### 追加提问 2：`agents` 教程里“添加内存”应该怎么理解？

问题描述：

在 `构建一个代理` 这节官方教程里，前面已经可以正常运行 agent 了，后面又单独加入了 `MemorySaver` 和 `thread_id`。这里容易混淆的是：所谓“添加内存”到底是在给模型加记忆，还是在给代理保存状态；`checkpointer`、`MemorySaver`、`thread_id` 三者又分别起什么作用。

问题解答：

按照官方教程“添加内存”这一小节的原意，这里的“内存”更准确地说是：

- 给代理增加“同一线程下的历史状态保存能力”。
- 它不是让模型永久记住所有事情，而是让代理在同一个会话线程里，能把前面对话和执行状态继续接上。

可以把这三个关键点分开理解：

1. `agent` 默认是无状态的

- 官方教程明确说，代理默认 `does not remember previous interactions`。
- 也就是说，如果你每次调用都不额外保存状态，那么第二次提问时，代理并不知道你前面说过什么。

2. `checkpointer` 是“状态存档机制”

- 官方教程里说，要添加内存，就需要传入一个 `checkpointer`。
- 它的职责是把代理运行过程中的状态保存下来，并在下次同线程调用时再取回来。
- 所以这里保存的核心不是“模型参数里的记忆”，而是“这条会话线程的消息和运行状态”。

3. `thread_id` 是“会话主键”

- 一旦用了 `checkpointer`，调用时就必须提供 `thread_id`。
- 因为代理要靠它判断：这次请求应该接到哪一段历史后面。
- 同一个 `thread_id` 表示继续同一段对话；换一个 `thread_id`，就等于开启新的对话线程。

教程里的运行逻辑可以压缩成一句话：

- `MemorySaver` 负责存。
- `thread_id` 负责找。
- `agent_executor.invoke(...)` 或 `.stream(...)` 负责在“取出旧状态 -> 接上本轮输入 -> 产生新结果 -> 再保存回去”这条链路上执行。

为什么官方示例里先说 `hi im bob!`，后面再问 `whats my name?`？

- 因为这正是在验证“同一个线程下的短期记忆”是否生效。
- 如果两次调用使用同一个 `thread_id`，代理就能从已保存的历史里知道用户前面说过自己叫 Bob。
- 如果改成新的 `thread_id`，代理就会把它当成新会话，因此答不出名字。

因此，这一小节最重要的理解不是“模型突然会记忆了”，而是：

- 代理借助 `checkpointer` 获得了线程级的会话持久化能力。
- `MemorySaver` 是教程里采用的内存型保存方式。
- `thread_id` 决定这次调用到底接着哪段历史继续运行。

##### 追加提问 3：这里的内存到底是存成 `json`、`markdown`，还是数据库？

问题描述：

在 `agents` 教程的“添加内存”示例里，官方代码直接写的是 `memory = MemorySaver()`。这时容易误解为：它是不是把聊天记录写成了某个 `json` 文件、`markdown` 文件，或者默认已经落到数据库里了。

问题解答：

按照官方教程和 LangGraph 官方内存文档，这里要分“教程示例”和“生产环境”两种情况理解：

1. 教程示例里的 `MemorySaver` / `InMemorySaver`

- 它不是 `json` 文件存储。
- 它不是 `markdown` 文件存储。
- 它默认也不是数据库存储。
- 它本质上是“保存在当前 Python 进程内存里的 checkpoint 数据”。

这意味着：

- 只要当前程序还活着，同一个 `thread_id` 的历史状态就还能继续取到。
- 一旦进程重启、脚本结束，内存里的这份状态通常就会丢失。

2. 生产环境里的官方推荐方式

- LangGraph 官方文档明确建议：生产环境使用“由数据库支持的 checkpointer”。
- 官方示例里给出的就是 `PostgresSaver`。
- 也就是说，真正需要持久化到磁盘、跨进程、跨重启保留时，应该把 checkpoint 存到数据库，而不是继续用内存版。

所以，最准确的结论是：

- 教程这段“添加内存”默认存的是“进程内存中的状态”，不是 `json`、不是 `markdown`。
- 如果你想让它变成真正持久化的存储，官方路径是换成数据库型 `checkpointer`，例如 `PostgresSaver`。

## 大章节：`rag`

说明：本章对应当前仓库中的 [rag_demo.py](/home/z/share/learn_pr/langchain/tutorials/langchain/rag/rag_demo.py)，内容基于 LangChain 官方教程中的“构建一个检索增强生成（RAG）应用”一节整理。笔记会保留官方教程里的“向量检索主线”，同时补充当前仓库改写为使用 Hugging Face `BAAI/bge-m3` 的可运行版本，并把 BM25 作为对比方案保留下来。

### 问题 1 [ACH]：`RAG` 教程里，加载、分割、存储、检索和生成阶段分别做了哪些事情？

问题ID：`Q-RAG-01-rag-stages`

#### 定义

按照官方教程，这五个阶段可以先分成两大段理解：

- `加载 / 分割 / 存储` 属于 `索引（indexing）` 阶段。
- `检索 / 生成` 属于 `检索与生成（retrieval and generation）` 阶段。

它们的总体目标是：

- 先把原始资料整理成“可以被搜索”的知识库。
- 再在用户提问时，从知识库里找出最相关的片段交给模型回答。

如果压缩成一句话：

- 前三步是在“准备知识库”。
- 后两步是在“用知识库回答问题”。

#### 功能

1. `加载（Load）`

- 官方教程先用 `WebBaseLoader` 从网页加载原始内容。
- 加载后的结果不是普通字符串，而是 `Document` 对象列表。
- 每个 `Document` 通常包含两部分：
  - `page_content`：正文文本
  - `metadata`：来源、位置等附加信息

这一阶段做的事情，本质上是：

- 把“网页、PDF、Markdown”这类原始数据源，转成 LangChain 后续组件都能处理的 `Document` 格式。

2. `分割（Split）`

- 官方教程加载到的网页正文非常长，超过 42k 字符。
- 这么长的文本既不适合直接做相似度搜索，也不适合一次性塞进模型上下文。
- 所以教程使用 `RecursiveCharacterTextSplitter` 把文档切成多个较小文本块。
- 示例参数是：
  - `chunk_size=1000`
  - `chunk_overlap=200`
  - `add_start_index=True`

这一阶段做的事情，本质上是：

- 把一整篇长文拆成很多“可检索、可引用、可放进提示词”的小块。
- 其中 `chunk_overlap=200` 的作用，是减少一句话刚好被切断后丢失上下文的风险。
- `add_start_index=True` 会把原文中的起始位置保存在元数据里，便于回溯来源。

3. `存储（Store）`

- 官方教程接着用 `OpenAIEmbeddings` 为每个文本块生成向量表示。
- 然后把这些向量和对应文本块一起写入 `Chroma` 向量存储。
- 这样做完之后，知识库就不再只是“一堆文本块”，而是“可按语义搜索的索引”。

这一阶段做的事情，本质上是：

- 先把文本块变成向量。
- 再把“向量 + 原文块 + metadata”存起来。
- 以后用户提问时，也会先把问题转成向量，再和这些已存向量做相似度匹配。

4. `检索（Retrieve）`

- 到运行时，官方教程把 `vectorstore` 转成 `retriever`。
- 示例里使用的是 `vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})`。
- 这表示：给定用户问题后，系统会找出语义最接近的 6 个文本块。

这一阶段做的事情，本质上是：

- 不让模型直接面对整个知识库。
- 而是先从知识库里挑出“最可能有答案”的少量片段。
- 这样既降低上下文长度，也能提高回答和原始资料的相关性。

5. `生成（Generate）`

- 官方教程最后把“用户问题 + 检索到的上下文”一起塞进提示词。
- 然后把提示词交给聊天模型生成答案。
- 在 LCEL 写法里，这一步通常表现为：
  - `retriever | format_docs` 先把检索结果整理成上下文字符串
  - `RunnablePassthrough()` 原样传递用户问题
  - `prompt | llm | StrOutputParser()` 负责提示构造、模型推理和文本解析

这一阶段做的事情，本质上是：

- 让模型不是“凭训练记忆瞎猜”，而是“基于刚检索到的资料回答”。
- 所以生成阶段的质量，依赖于前面的检索是否真的拿回了相关上下文。

#### 源码示例

代码示例 1：官方教程里的完整主线写法

```python
import bs4
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 第 1 步：加载网页，得到 Document 列表
bs4_strainer = bs4.SoupStrainer(
    class_=("post-title", "post-header", "post-content")
)
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

# 第 2 步：把长文档切成小块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)

# 第 3 步：把文本块嵌入并写入向量存储
vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=OpenAIEmbeddings(
        # 显式指定 embedding 模型，避免兼容 OpenAI 的渠道回退到不可用的旧默认模型
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        base_url="https://api.udcode.cn/v1",
        api_key=os.getenv("OPENAI_API_KEY", "你的API_KEY"),
        # 对 OpenAI-compatible 提供方，官方文档建议关闭长度安全分词预处理
        check_embedding_ctx_length=False,
    ),
)

# 第 4 步：把向量存储转换成检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6},
)

model = ChatOpenAI(
    model="gpt-5.4",
    base_url="https://api.udcode.cn/v1",
    api_key=os.getenv("OPENAI_API_KEY", "你的API_KEY"),
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是问答助手。请使用检索到的上下文回答问题。"
        "如果上下文中没有答案，就明确说不知道。"
        "回答尽量简洁，不超过三句话。\n\n"
        "问题：{question}\n"
        "上下文：{context}",
    ),
])


def format_docs(docs):
    # 把多个 Document 的正文拼成一个字符串，供提示词使用
    return "\n\n".join(doc.page_content for doc in docs)


# 第 5 步：生成
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

answer = rag_chain.invoke("What is Task Decomposition?")
print(answer)
```

代码示例 2：当前仓库使用 Hugging Face `BAAI/bge-m3` 的可运行改写版

```python
import bs4
import os

from huggingface_hub import InferenceClient
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


class HuggingFaceBgeM3Embeddings(Embeddings):
    def __init__(self, api_key: str, model_name: str = "BAAI/bge-m3") -> None:
        self.model_name = model_name
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=api_key,
        )

    def _embed_text(self, text: str) -> list[float]:
        result = self.client.feature_extraction(
            text,
            model=self.model_name,
        )
        if hasattr(result, "tolist"):
            result = result.tolist()
        return list(result)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)


loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(
            class_=("post-title", "post-header", "post-content")
        )
    },
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)

embeddings = HuggingFaceBgeM3Embeddings(
    api_key=os.getenv("HUGGING_EMBEDDING_API_KEY", "你的HF_TOKEN"),
    model_name="BAAI/bge-m3",
)

# 当前仓库不再依赖 OpenAI 的 embedding 接口
# 改为用 Hugging Face 的 BAAI/bge-m3 生成向量并写入 Chroma
vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6},
)

model = ChatOpenAI(
    model="gpt-5.4",
    base_url="https://api.udcode.cn/v1",
    api_key=os.getenv("OPENAI_API_KEY", "你的API_KEY"),
    use_responses_api=False,
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是问答助手。请使用检索到的上下文回答问题。"
        "如果上下文中没有答案，就明确说不知道。"
        "回答尽量简洁，不超过三句话。\n\n"
        "上下文：{context}",
    ),
    ("human", "{question}"),
])


def format_docs(docs):
    # 把检索到的多个 Document 正文拼成上下文字符串
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

answer = rag_chain.invoke("What is Task Decomposition?")
print(answer)
```

代码示例 3：只观察检索阶段，不做生成

```python
query = "What are the approaches to Task Decomposition?"

retrieved_docs = retriever.invoke(query)

print(len(retrieved_docs))
print(retrieved_docs[0].metadata)
print(retrieved_docs[0].page_content[:500])
```

这两段代码特别适合用来区分：

- `检索` 的输出是 `Document` 列表
- `生成` 的输出才是最终答案文本

#### 疑点

##### 追加提问 1：LangChain 官方自己提供 embedding 模型和调用渠道吗？

问题描述：

在学习 `RAG` 教程时，官方示例会写 `OpenAIEmbeddings`、`ChatOpenAI` 这类类名。这时很容易误解成：既然这些类是 LangChain 提供的，那 embedding 模型和接口渠道是不是也是 LangChain 官方自己提供的。

问题解答：

按照 LangChain 官方文档当前的表述，答案应理解为：

- `LangChain` 主要提供的是统一抽象接口和集成层。
- 它通常不自己提供底层的 embedding 模型。
- 它通常也不自己提供模型调用渠道。

更准确地说，LangChain 做的是“适配器”这层工作：

1. 它提供统一接口

- 官方文档明确说，LangChain 通过 `Embeddings` 接口来统一不同提供方的 embedding 模型调用方式。
- 例如统一成：
  - `embed_documents(...)`
  - `embed_query(...)`

这意味着：

- 你换 OpenAI、Cohere、Mistral、Ollama 等不同提供方时，代码结构可以尽量保持一致。

2. 它提供集成包

- 官方文档列出了很多 embedding 集成，例如：
  - `OpenAIEmbeddings`
  - `CohereEmbeddings`
  - `MistralAIEmbeddings`
  - `OllamaEmbeddings`
  - `GoogleGenerativeAIEmbeddings`
- 官方还专门说明，provider 是“LangChain 集成进去的第三方服务或平台”。

所以这里的关键边界是：

- `langchain-openai` 不是在提供 OpenAI 自己的模型。
- 它是在提供“如何用 LangChain 统一方式去调用 OpenAI 模型”的封装。

3. 真正提供模型和渠道的是第三方 provider

- 例如你使用 `OpenAIEmbeddings` 时，真正的 embedding 模型来自 OpenAI 或兼容 OpenAI API 的第三方渠道。
- 例如你使用 `OllamaEmbeddings` 时，真正的模型和服务端来自本地或远程的 Ollama 实例。

因此，最短记忆法可以写成：

- `LangChain` 提供的是“接口 + 适配 + 编排”。
- `provider` 提供的是“模型 + API 渠道”。

如果再结合你当前 `tutorials/langchain/rag/rag_demo.py` 里遇到的问题，这个区分尤其重要：

- `ChatOpenAI` 和 `OpenAIEmbeddings` 都是 LangChain 的集成封装类。
- 但它们实际请求的模型是否存在、渠道是否兼容、接口是否可用，取决于你背后的 provider，而不是 LangChain 本身。

##### 追加提问 2：向量检索和 BM25 检索有什么区别，当前仓库为什么先后使用过两种方案？

问题描述：

LangChain 官方 `RAG` 教程默认演示的是“`embedding + vector store`”这条主线。当前仓库在学习过程中，曾因为没有可用 embedding URL 暂时改成 `BM25Retriever`；而现在又切回了使用 Hugging Face `BAAI/bge-m3` 的向量检索版本。这里容易混淆的是：这两种方式到底差在哪里，为什么当前仓库会先后用过两种方案。

问题解答：

可以先把两种方式压缩成一句话：

- `向量检索`：先把文本和问题都变成向量，再按语义相似度找相关片段。
- `BM25 检索`：不做向量化，直接按关键词相关性给文档块打分。

它们的核心区别有四个：

1. 依赖不同

- `向量检索` 依赖：
  - embedding 模型
  - 向量存储，例如 `Chroma`
- `BM25 检索` 依赖：
  - 已切分好的 `Document` 列表
  - `BM25Retriever`
  - 本地 Python 依赖 `rank_bm25`

所以在工程依赖上：

- 向量检索更重，但能力更完整。
- BM25 更轻，更容易在本地或受限环境里直接跑通。

2. 检索原理不同

- `向量检索` 的重点是“语义接近”，即使提问和原文用词不完全一致，也可能检索到正确片段。
- `BM25 检索` 的重点是“关键词匹配与统计权重”，更依赖问题里的词和文档块里的词是否重合。

因此通常可以这样理解：

- 向量检索更擅长“意思相近但措辞不同”的问题。
- BM25 更擅长“关键词明确”的问题。

3. 存储阶段是否存在

- 在官方教程主线里，`存储（Store）` 是一个明确阶段，因为要先生成 embedding 并写入向量数据库。
- 在当前仓库的 BM25 改写版里，这个阶段实际上被弱化了：
  - 仍然会有 `加载`
  - 仍然会有 `分割`
  - 但不会再做“embedding -> vector store”这一步
  - 而是直接从 `all_splits` 构造检索器

所以如果你对照当前代码来理解：

- 官方教程的 `Store` 更像是在“建立语义索引”。
- BM25 版本没有这个“向量化存储”步骤。

4. 适用场景不同

- `向量检索` 适合：
  - 有可用 embedding provider
  - 需要更强语义召回
  - 愿意接受更高的依赖与成本
- `BM25 检索` 适合：
  - 没有 embedding URL
  - 先把学习示例跑通
  - 问题以关键词检索为主

当前仓库先改成 BM25、后切回向量检索的直接原因是：

- 在没有可用 embedding 模型调用 URL 时，如果继续沿用官方教程中的 `OpenAIEmbeddings + Chroma` 路线，示例无法正常跑通。
- 因此当时先采用 LangChain 官方同样支持的 `BM25Retriever`，把教程临时改成“无 embedding 的关键词检索版”。
- 现在已经确认可以通过 Hugging Face token 访问 `BAAI/bge-m3`，所以当前仓库又切回了“向量检索 RAG”。
- 也就是说，BM25 是当前环境受限时的替代方案，而不是对官方主线的永久替换。

所以最准确的结论是：

- 官方教程主线仍然是“向量检索 RAG”。
- 当前仓库当前版本已经切回“基于 Hugging Face `BAAI/bge-m3` 的向量检索 RAG”。
- BM25 仍然是一个有效的替代方案，但更适合没有可用 embedding 服务时使用。
- 两者都属于 RAG，但底层检索机制不同。

## 大章节：`qa_chat_history`

说明：本章对应当前仓库中的 [qa_chat_history.py](/home/z/share/learn_pr/langchain/tutorials/langchain/agents/qa_chat_history.py)，内容基于 LangChain 官方教程中的“对话式RAG（qa_chat_history）”一节，围绕“历史感知检索、状态管理、检索工具、对话式 agent”这条主线整理。

### 问题 1 [ACT]：`qa_chat_history` 章节里完成的可对话 agent 是怎么工作的？

问题ID：`Q-QA-01-conversational-agent`

#### 定义

这章里完成的“可对话 agent”，不是一个会永久记住所有事情的通用智能体，而是一个：

- 以大型语言模型作为决策核心，
- 以检索器工具作为外部知识入口，
- 以 LangGraph 的 `MemorySaver` 作为会话状态存储，
- 能在多轮对话中决定“要不要检索、怎么检索、何时回答”的对话式 RAG agent。

如果压缩成一句话：

- 它本质上是“带检索工具和会话状态的 ReAct agent”。

和前面 `rag` 教程里的普通问答链相比，这里最大的变化不是“能检索”，而是：

- 检索步骤不再总是固定执行；
- agent 可以自己决定是否调用检索工具；
- 它还能借助历史状态处理后续追问。

#### 功能

按照官方教程，这个可对话 agent 主要由四部分组成：

1. `retriever`

- 先像普通 RAG 一样，加载文档、分割文本、建立向量检索器。
- 这一层仍然负责“从外部知识源里找相关片段”。
- 官方教程示例通常使用 `OpenAIEmbeddings` 生成向量；当前仓库为了适配现有可用渠道，改成通过 Hugging Face Inference Providers 调用 `BAAI/bge-m3`。
- 不过这并没有改变教程主线，本质上仍然是“embedding + 向量库 + retriever”的向量检索流程。

2. `retriever tool`

- 官方教程用 `create_retriever_tool(...)` 把检索器包装成工具。
- 包装后的工具会有：
  - 工具名字，例如 `blog_post_retriever`
  - 工具描述，告诉 agent 这个工具是干什么的
- 这样模型才知道：如果它需要查资料，可以调用这个工具。

这里很关键的一点是：

- 在链方案里，检索器是固定流程中的一步。
- 在 agent 方案里，检索器变成“可被调用的工具”。

3. `agent executor`

- 官方教程使用 `create_react_agent(llm, tools)` 创建 agent。
- 这个 agent 会根据用户输入决定：
  - 是否调用工具
  - 调哪个工具
  - 给工具传什么参数
  - 是继续下一步还是直接回答

所以它和固定链最大的区别是：

- 链的路径是提前写死的；
- agent 的路径是运行时由模型决定的。

4. `memory / checkpointer`

- 官方教程没有给 agent 再包 `RunnableWithMessageHistory`。
- 它直接使用 LangGraph 的 `MemorySaver()`，然后作为 `checkpointer` 传给 `create_react_agent(...)`。
- 之后再通过 `thread_id` 标识当前会话线程。

这意味着：

- 同一个 `thread_id` 下，多轮消息和中间状态会被持续保存；
- 追问时 agent 可以接着前面的上下文继续工作。

所以，这个“可对话”不是凭空出现的，而是来自：

- `tool` 让它能查资料；
- `MemorySaver + thread_id` 让它能延续对话。

#### 源码示例

代码示例 1：把检索器包装成工具

```python
from langchain_core.tools import create_retriever_tool

# retriever 是前面构造好的向量检索器
tool = create_retriever_tool(
    retriever,
    "blog_post_retriever",
    "Searches and returns excerpts from the Autonomous Agents blog post.",
)

tools = [tool]

# 工具本身也可以单独调用，输入通常是查询字符串
result = tool.invoke("task decomposition")
print(result)
```

代码示例 2：创建带内存的对话式 agent（当前仓库的 Hugging Face embedding 改写版）

```python
import os

import bs4
from huggingface_hub import InferenceClient
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.embeddings import Embeddings
from langchain_core.tools import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

HF_EMBEDDING_MODEL = "BAAI/bge-m3"


class HuggingFaceBgeM3Embeddings(Embeddings):
    """通过 Hugging Face Inference Providers 调用 BAAI/bge-m3。"""

    def __init__(self, api_key: str, model_name: str = HF_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=api_key,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 批量生成向量，避免逐条请求
        result = self.client.feature_extraction(
            texts,
            model=self.model_name,
        )
        if hasattr(result, "tolist"):
            result = result.tolist()
        return [list(item) for item in result]

    def embed_query(self, text: str) -> list[float]:
        result = self.client.feature_extraction(
            text,
            model=self.model_name,
        )
        if hasattr(result, "tolist"):
            result = result.tolist()
        return list(result)


model = ChatOpenAI(
    model="gpt-5.4",
    base_url="https://api.udcode.cn/v1",
    api_key=os.getenv("OPENAI_API_KEY", "你的API_KEY"),
    use_responses_api=False,
    temperature=0,
)

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    },
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=HuggingFaceBgeM3Embeddings(
        api_key=os.getenv("HUGGING_EMBEDDING_API_KEY", ""),
        model_name=HF_EMBEDDING_MODEL,
    ),
)
retriever = vectorstore.as_retriever()

tool = create_retriever_tool(
    retriever,
    "blog_post_retriever",
    "Searches and returns excerpts from the Autonomous Agents blog post.",
)

# LangGraph 直接用 checkpointer 保存会话状态
memory = MemorySaver()

agent_executor = create_react_agent(
    model,
    [tool],
    checkpointer=memory,
)

config = {"configurable": {"thread_id": "abc123"}}

# 第一轮：普通寒暄，不一定需要检索
for step in agent_executor.stream(
    {"messages": [{"role": "user", "content": "Hi! I'm Bob"}]},
    config=config,
):
    print(step)

# 第二轮：知识问题，agent 可能决定调用检索工具
for step in agent_executor.stream(
    {"messages": [{"role": "user", "content": "What is Task Decomposition?"}]},
    config=config,
):
    print(step)

# 第三轮：继续追问，agent 在同一 thread_id 下保留会话状态
for step in agent_executor.stream(
    {"messages": [{"role": "user", "content": "What are common ways of doing it?"}]},
    config=config,
):
    print(step)
```

#### 疑点

##### 追加提问 1：这章里的链方案和 agent 方案，本质区别是什么？

问题描述：

在 `qa_chat_history` 章节里，官方教程先讲了链，再讲 agent。两者最后都能做“带聊天历史的问答”，容易让人误以为只是写法不同。但实际上，这两种方案在“谁来决定检索步骤”这件事上有本质区别。

问题解答：

最核心的区别是“控制权”不同：

1. 链方案：控制权在程序员手里

- 官方教程在链方案里明确构造了：
  - `create_history_aware_retriever(...)`
  - `create_stuff_documents_chain(...)`
  - `create_retrieval_chain(...)`
- 也就是说，执行路径是提前写死的：
  - 先看聊天历史
  - 再把问题重写成独立查询
  - 然后执行检索
  - 最后根据上下文回答

可以把它压缩成：

- `(input, chat_history)` -> `重写问题` -> `检索` -> `回答`

所以它的特点是：

- 路径稳定、可预测；
- 每次都会走检索；
- 更适合流程明确的问答应用。

2. agent 方案：控制权更多交给模型

- agent 方案里，检索器先被包装成工具。
- 再通过 `create_react_agent(...)` 交给模型在运行时决定要不要调用。

这意味着：

- 遇到普通寒暄，例如 `"Hi! I'm Bob"`，agent 可能直接回复，不走检索。
- 遇到知识问题，例如 `"What is Task Decomposition?"`，agent 才可能调用检索工具。
- 遇到复杂问题时，它甚至可能多次调用工具。

所以它的特点是：

- 灵活性更高；
- 运行路径不完全固定；
- 更像“会决策的问答系统”。

3. 历史的整合方式也不同

- 链方案里，官方教程先演示手动维护 `chat_history`，后面再用 `RunnableWithMessageHistory` 自动管理。
- agent 方案里，官方教程直接利用 LangGraph 的 `MemorySaver` 和 `thread_id` 来管理状态。

所以在历史管理上：

- 链更像“把历史消息当作链输入的一部分”。
- agent 更像“把历史消息当作图执行状态的一部分”。

因此，这两种方案不是单纯“不同写法”，而是两种不同控制模式：

- 链：程序预先规定步骤。
- agent：模型在步骤之间做决策。

##### 追加提问 2：当前仓库为什么把这章里的 embedding 接入改成 Hugging Face，和官方示例有什么区别？

问题描述：

官方 `qa_chat_history` 教程在向量检索部分通常沿用 `OpenAIEmbeddings`。而当前仓库现在改成了通过 Hugging Face Inference Providers 调用 `BAAI/bge-m3`。这里容易混淆的是：这到底是不是换了一套 RAG 方案，还是只替换了 embedding 的提供方。

问题解答：

最准确的理解是：

- 官方教程主线没有变。
- 当前仓库改动的核心只是“embedding provider”，不是“检索架构”。

可以拆成三层来看：

1. 不变的是 RAG / 对话式 agent 的整体结构

- 文档仍然先经过 `WebBaseLoader` 加载。
- 仍然会经过 `RecursiveCharacterTextSplitter` 切分。
- 仍然把文本块写入 `Chroma` 向量库。
- 仍然由 `retriever` 提供相似片段，再包装成 `retriever tool` 给 agent 调用。

所以从教程结构上看，它仍然是：

- `embedding -> vector store -> retriever -> tool -> agent`

2. 改变的是“谁来生成向量”

- 官方示例一般使用 `OpenAIEmbeddings`。
- 当前仓库改成自定义 `HuggingFaceBgeM3Embeddings`，底层通过 `InferenceClient(provider="hf-inference")` 调用 `BAAI/bge-m3`。

因此这里的变化不是“LangChain 换了一套 API 逻辑”，而是：

- LangChain 这一层仍然只要求你提供符合 `Embeddings` 接口的对象。
- 真正生成向量的底层服务，从 OpenAI-compatible 渠道换成了 Hugging Face provider。

3. 为什么当前仓库要这样改

- 因为当前学习环境里，聊天模型和 embedding 模型并不一定来自同一个可用渠道。
- 之前已经出现过 `text-embedding-ada-002` 或兼容 embedding 接口不可用的报错。
- 现在已经确认 Hugging Face token 可以访问 `BAAI/bge-m3`，所以当前仓库把这章的向量检索接入切到了 Hugging Face，保证示例可以继续沿着官方教程主线跑通。

所以可以把差异压缩成一句话：

- 官方教程更像是“直接展示标准写法”。
- 当前仓库更像是“保留官方教程结构，但把 embedding provider 替换成当前环境可用的实现”。
