import os

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


def build_model() -> ChatOpenAI:
    """创建与官方教程一致的聊天模型实例。"""
    return ChatOpenAI(
        model="gpt-5.4",
        base_url="https://api.udcode.cn/v1",
        api_key=os.getenv("OPENAI_API_KEY", "你的API_KEY"),
    )


def build_tools():
    """创建官方教程中使用的搜索工具列表。"""
    search = TavilySearchResults(max_results=2)
    return [search]


def example_bind_tools() -> None:
    """示例 1：只把工具绑定给模型，还不是真正的代理。"""
    model = build_model()
    tools = build_tools()

    # bind_tools 只会把工具描述告诉模型，
    # 模型可能生成 tool_calls，但不会自动帮你执行工具。
    model_with_tools = model.bind_tools(tools)

    response = model_with_tools.invoke(
        [HumanMessage(content="What's the weather in SF?")]
    )

    print("=== 示例 1：model.bind_tools(tools) ===")
    print(response)
    print("=== tool_calls ===")
    print(response.tool_calls)
    print()


def example_agent_invoke() -> None:
    """示例 2：创建真正可执行的代理。"""
    model = build_model()
    tools = build_tools()

    # create_react_agent 会把“模型 + 工具”组装成代理执行器。
    agent_executor = create_react_agent(model, tools)

    result = agent_executor.invoke(
        {
            "messages": [HumanMessage(content="What's the weather in SF?")],
        }
    )

    print("=== 示例 2：create_react_agent(model, tools) ===")
    print(result["messages"])
    print()


def example_agent_memory() -> None:
    """示例 3：给代理增加线程级记忆。"""
    model = build_model()
    tools = build_tools()

    # MemorySaver 会把同一个 thread_id 下的状态保存在当前进程内存中。
    memory = MemorySaver()

    agent_executor = create_react_agent(
        model,
        tools,
        checkpointer=memory,
    )

    # thread_id 用来标识一段会话线程。
    config = {"configurable": {"thread_id": "abc123"}}

    first_result = agent_executor.invoke(
        {"messages": [HumanMessage(content="hi! I'm Bob")]},
        config=config,
    )
    print("=== 示例 3：第一轮，写入线程状态 ===")
    print(first_result["messages"])
    print()

    second_result = agent_executor.invoke(
        {"messages": [HumanMessage(content="What's my name?")]},
        config=config,
    )
    print("=== 示例 3：第二轮，读取同一 thread_id 的历史 ===")
    print(second_result["messages"])
    print()


if __name__ == "__main__":
    example_bind_tools()
    example_agent_invoke()
    example_agent_memory()
