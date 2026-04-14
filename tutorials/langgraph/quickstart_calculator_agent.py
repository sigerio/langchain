"""
LangGraph Quickstart - 计算器代理 (Calculator Agent)
官方教程: https://docs.langchain.com/oss/python/langgraph/quickstart

运行前请确保已设置环境变量，例如:
    export ANTHROPIC_API_KEY="sk-..."
"""

from typing import Literal
import operator

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END

# ========== 步骤 1: 定义工具和模型 ==========

# 初始化支持工具调用的 LLM
# 如果未设置 ANTHROPIC_API_KEY，可替换为其他支持工具的模型，例如 "gpt-4o"
model = init_chat_model("claude-sonnet-4-6", temperature=0)


@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`."""
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`."""
    return a / b


tools = [add, multiply, divide]
# 建立名称到工具对象的映射，便于在 tool_node 中快速查找
tools_by_name = {tool.name: tool for tool in tools}
# 将工具绑定到模型，使模型具备生成 tool_calls 的能力
model_with_tools = model.bind_tools(tools)


# ========== 步骤 2: 定义状态 ==========

class MessagesState(TypedDict):
    # Annotated 配合 operator.add 表示：多个节点更新 messages 时，采用追加合并策略
    messages: Annotated[list[AnyMessage], operator.add]
    # 标量字段，记录 LLM 调用次数，直接覆盖更新
    llm_calls: int


# ========== 步骤 3: 定义模型节点 ==========

def llm_call(state: dict):
    """LLM 决定是调用工具还是直接回答用户。"""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ========== 步骤 4: 定义工具节点 ==========

def tool_node(state: dict):
    """执行 LLM 请求的工具调用，并将结果包装为 ToolMessage 返回。"""
    result = []
    # 从最后一条消息中提取 tool_calls（这是 LLM 输出的工具调用请求）
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": result}


# ========== 步骤 5: 定义结束逻辑（条件边） ==========

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """根据 LLM 是否发起工具调用来决定下一步流向。"""
    last_message = state["messages"][-1]
    # 如果 LLM 请求了工具调用，则进入 tool_node
    if last_message.tool_calls:
        return "tool_node"
    # 否则直接结束
    return END


# ========== 步骤 6: 构建并编译代理 ==========

agent_builder = StateGraph(MessagesState)

# 注册节点
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# 注册边
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END],
)
# 工具执行完后，重新回到 LLM 节点进行下一轮决策
agent_builder.add_edge("tool_node", "llm_call")

# 编译图，生成可执行对象
agent = agent_builder.compile()


# ========== 运行示例 ==========

if __name__ == "__main__":
    messages = [HumanMessage(content="Add 3 and 4.")]
    result = agent.invoke({"messages": messages})

    print("--- 最终结果 ---")
    for m in result["messages"]:
        m.pretty_print()

    print(f"\nLLM 调用次数: {result['llm_calls']}")
