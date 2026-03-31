from langchain_openai import ChatOpenAI
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import os

# 初始化聊天模型
model = ChatOpenAI(
    model="gpt-5.4",
    base_url="https://api.udcode.cn/v1",
    api_key=os.getenv("OPENAI_API_KEY", "你的API_KEY"),
)

# 第 1 步：不使用提示词模板，直接手写最终消息
# 这里是已经完全展开后的消息列表，会直接发给模型
direct_messages = [
    SystemMessage(
        content="You are a helpful assistant. Answer all questions to the best of your ability in 中文."
    ),
    HumanMessage(content="hi! I'm Todd"),
]

direct_response = model.invoke(direct_messages)
print("=== 第1步：直接手写消息 ===")
print(direct_response)
print(direct_response.content)

# 第 2 步：先定义提示词模板，再传入变量生成真正消息
# 这里的 prompt 还不是最终消息，只是消息结构模板
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
    ),
    # 当前用户输入或历史消息会插入到这个位置
    MessagesPlaceholder(variable_name="messages"),
])

# 传入变量后，模板才会被展开成真正的消息列表
prompt_value = prompt.invoke(
    {
        "messages": [HumanMessage(content="hi! I'm Todd")],
        "language": "中文",
    }
)

print("=== 第2步：查看提示词模板展开结果 ===")
print(prompt_value.to_messages())

# 第 3 步：把提示词模板接入带消息历史的 chatbot 链
chain = prompt | model

# 内存中的聊天历史存储
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # 为不同会话返回不同的消息历史对象
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 因为当前输入是一个字典，不再是单纯的消息列表
# 所以需要显式告诉 LangChain：哪一个字段用来存放消息历史
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

config = {"configurable": {"session_id": "abc11"}}

# 第一次调用：输入消息和额外变量 language
response = with_message_history.invoke(
    {
        "messages": [HumanMessage(content="hi! I'm Todd")],
        "language": "中文",
    },
    config=config,
)
print("=== 第3步：带消息历史的 chatbot ===")
# 输出完整的 AIMessage 对象
print(response)
print(response.content)

# 第二次调用：同一个 session_id 下继续追问
response = with_message_history.invoke(
    {
        "messages": [HumanMessage(content="What's my name?")],
        "language": "中文",
    },
    config=config,
)
print(response.content)

# 第3次调用：不同的session_id
config = {"configurable": {"session_id": "abc01"}}
response = with_message_history.invoke(
    {
        "messages": [HumanMessage(content="What's my name?")],
        "language": "中文",
    },
    config=config,
)

print(response.content)
