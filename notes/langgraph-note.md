# LangGraph 官方教程学习笔记

本笔记用于记录 LangGraph 官方教程学习过程中的概念理解、代码写法和疑点追踪。

## 记录规则

本笔记用于记录我在学习 LangGraph 官方教程过程中提出的“概念理解”和“代码写法”类问题。
- 记录内容应建立在 LangGraph 官方教程基础上。
- 重点记录“概念理解”和“代码写法”类问题，便于后续回看与改写。
- 后续若沿用 LangGraph 笔记中的问题组织方式，可在开始正式学习时再补充统一的问题 ID 规则。

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

---

## Quickstart

对应源码文件：`tutorials/langgraph/quickstart_calculator_agent.py`

本节基于 LangGraph 官方 Quickstart 教程，构建一个具备加、乘、除能力的计算器代理。

---

### Q-QS-01-calculator-agent-steps
**状态：** `ACT`

**问题：** LangGraph Quickstart 中构建计算器代理的四个步骤（定义工具和模型、定义模型节点、定义工具节点、定义代理）分别是什么意思，应该怎么理解和写？

**定义：**
这四个步骤共同构成了 LangGraph 的 Graph API 最小代理工作流：
1. **定义工具和模型**：为代理准备可执行的工具函数和具备工具调用能力的 LLM。
2. **定义模型节点（`llm_call`）**：代理的"大脑"，负责接收当前状态中的消息，调用 LLM 决定是回复用户还是请求调用工具。
3. **定义工具节点（`tool_node`）**：代理的"手"，负责实际执行 LLM 请求的工具调用，并将执行结果以 `ToolMessage` 的形式返回给状态。
4. **定义代理（`StateGraph`）**：将上述节点和边组装成一张状态图，通过条件边控制 LLM 和工具节点之间的循环，直到任务完成。

**功能：**
- 步骤 1 让 LLM 拥有"能力清单"（加、乘、除）。
- 步骤 2 让 LLM 基于当前对话上下文做出"决策"（继续调用工具或直接回答）。
- 步骤 3 把 LLM 的决策转化为"实际行动"，并将结果回传给 LLM。
- 步骤 4 将决策与行动编排成一个可自动循环执行的图结构，代理可以在 `llm_call` ↔ `tool_node` 之间多轮迭代，直到得出最终答案。

**源码示例：**

```python
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from typing import Literal
import operator

# ========== 步骤 1：定义工具和模型 ==========
model = init_chat_model("claude-sonnet-4-6", temperature=0)

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add a and b."""
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """Divide a by b."""
    return a / b

tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# ========== 步骤 2：定义模型节点 ==========
def llm_call(state: dict):
    """LLM 决定是调用工具还是直接回答"""
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
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# ========== 步骤 3：定义工具节点 ==========
def tool_node(state: dict):
    """执行 LLM 请求的工具调用"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"]
            )
        )
    return {"messages": result}

# ========== 步骤 4：定义代理（组装图） ==========
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """根据 LLM 是否发起工具调用来决定下一步"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END

agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()

# 运行示例
messages = [HumanMessage(content="Add 3 and 4.")]
result = agent.invoke({"messages": messages})
for m in result["messages"]:
    m.pretty_print()
```

**疑点：**
