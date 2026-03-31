from operator import itemgetter
import os

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.messages import HumanMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI


def build_model() -> ChatOpenAI:
    """创建聊天模型实例。"""
    return ChatOpenAI(
        model="gpt-5.4",
        base_url="https://api.udcode.cn/v1",
        api_key=os.getenv("OPENAI_API_KEY", "你的API_KEY"),
    )


model = build_model()

# 官方思路 1：
# 先定义 prompt，其中 system 指令放在模板里，
# 历史消息和当前输入消息统一走 messages 这个占位符。
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
    ),
    MessagesPlaceholder(variable_name="messages"),
])

# 官方思路 2：
# 在 messages 真正进入 prompt 之前，先通过 trim_messages 做裁剪。
# 这里裁剪的是“history + 当前输入”合并后的 messages 列表。
trimmer = trim_messages(
    max_tokens=65,        # 总上下文超过 65 tokens 时，旧消息会被继续裁掉
    strategy="last",      # 优先保留最后面的新消息，所以更早的历史更容易先被删
    token_counter=model,
    allow_partial=False,  # 不保留半条消息；装不下时直接整条裁掉
    start_on="human",     # 裁剪后的消息通常从 HumanMessage 开始
)

trim_before_prompt = RunnablePassthrough.assign(
    # itemgetter("messages") 取出输入字典中的消息列表
    # 再交给 trimmer 裁剪，最后把裁剪结果重新写回 messages 字段
    messages=itemgetter("messages") | trimmer,
)

# 先修剪 messages，再交给 prompt 生成真正发送给模型的消息
chain = trim_before_prompt | prompt | model

# 外部历史存储：按 session_id 保存不同会话的历史
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # 新会话创建新的历史对象
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 官方思路 3：
# 用 RunnableWithMessageHistory 包装整条链。
# 它会在每次调用前读取历史消息，并把历史消息与当前输入消息
# 一起放进 messages 字段，然后再进入上面的 trim_before_prompt。
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)


def print_messages(title: str, messages) -> None:
    """按顺序打印消息列表，便于观察裁剪前后的结构。"""
    print(title)
    for index, message in enumerate(messages, start=1):
        print(f"{index}. {type(message).__name__}: {message.content}")


def preview_pipeline(session_id: str, language: str, current_messages) -> None:
    """预览 history + 当前输入在进入 prompt 之前的裁剪效果。"""
    history = get_session_history(session_id)

    # 官方包装器真正执行时，会把历史消息和当前输入消息合并后再往下传。
    # 所以这里先手动模拟那一步，方便直接观察“历史是否传进来了”。
    merged_messages = history.messages + current_messages

    # 这里经常让人误解：
    # 如果 max_tokens 太小，虽然 merged_messages 里明明有历史，
    # 但裁剪后 trimmed_messages 里可能已经看不到旧历史了。
    trimmed_messages = trimmer.invoke(merged_messages)

    print_messages("=== 合并后的 messages ===", merged_messages)
    print_messages("=== 裁剪后的 messages ===", trimmed_messages)

    # 预览最终交给模型的完整消息列表
    prompt_value = prompt.invoke(
        {
            "language": language,
            "messages": trimmed_messages,
        }
    )
    print_messages("=== prompt 展开后的最终消息 ===", prompt_value.to_messages())


def invoke_with_preview(session_id: str, language: str, user_text: str) -> None:
    """先预览裁剪结果，再真正调用带消息历史的链。"""
    current_messages = [HumanMessage(content=user_text)]

    # 注意：这一步发生在真正调用模型之前。
    # 因此这里看到的历史，只包含“之前轮次已经写回 store 的历史”，
    # 不包含“当前这一轮调用稍后才会生成的 AIMessage”。
    preview_pipeline(
        session_id=session_id,
        language=language,
        current_messages=current_messages,
    )

    response = with_message_history.invoke(
        {
            "messages": current_messages,
            "language": language,
        },
        config={"configurable": {"session_id": session_id}},
    )

    print("=== 模型返回的 AIMessage ===")
    print(response)
    print("=== 模型返回的文本 ===")
    print(response.content)
    print()


if __name__ == "__main__":
    session_id = "trim-history-demo"

    invoke_with_preview(
        session_id=session_id,
        language="中文",
        user_text="hi! I'm Bob",
    )

    invoke_with_preview(
        session_id=session_id,
        language="中文",
        user_text="我喜欢香草冰淇淋，我最喜欢的数字是7，我也喜欢周末徒步旅行。",
    )

    invoke_with_preview(
        session_id=session_id,
        language="中文",
        user_text="我的名字是什么？我最喜欢的冰淇淋口味是什么？我最喜欢的数字是什么？",
    )
