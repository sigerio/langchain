from langchain_openai import ChatOpenAI
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import os

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

# 外部历史存储
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # 按 session_id 获取或创建历史对象
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 包装原始链，让它具备自动管理历史的能力
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

# 会话 abc11：第一次提到自己的名字
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

# 会话 abc11：继续追问，能够利用前一轮历史
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