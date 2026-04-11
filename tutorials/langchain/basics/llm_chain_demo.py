import os

# =========================
# 运行环境开关区
# 官方教程可以开启 LangSmith tracing 观察运行链路
# 为了避免脚本结束时后台线程尚未提交完 trace，这里会在退出前显式等待
# =========================
ENABLE_LANGSMITH_TRACING = True

if not ENABLE_LANGSMITH_TRACING:
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
else:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = "false"

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.langchain import wait_for_all_tracers
from langserve import RemoteRunnable

# =========================
# 示例开关区
# 将需要学习的功能块设置为 True，不需要运行的设置为 False
# 后续新增教程示例时，继续按这个模式追加即可，避免覆盖已有代码
# =========================
RUN_EXAMPLE_MESSAGES_TRANSLATE = False
RUN_EXAMPLE_OUTPUT_PARSER_STEP = False
RUN_EXAMPLE_OUTPUT_PARSER_CHAIN = False
RUN_EXAMPLE_PROMPT_TEMPLATE = False
RUN_EXAMPLE_PROMPT_CHAIN = False
RUN_REMOTE_RUNNABLE_SERVER = True


def build_model() -> ChatOpenAI:
    """创建聊天模型实例。"""
    return ChatOpenAI(
        model="gpt-5.4",
        base_url="https://api.udcode.cn/v1",
        api_key=os.getenv("OPENAI_API_KEY", "你的API_KEY"),
    )


def build_translate_messages():
    """构造教程中的翻译消息。"""
    return [
        SystemMessage(content="Translate the following from English into Italian"),
        HumanMessage(content="hi!"),
    ]


def example_messages_translate():
    """示例 1：直接使用消息列表调用模型。"""
    model = build_model()
    messages = build_translate_messages()

    # 直接调用模型，返回 AIMessage 对象
    response = model.invoke(messages)

    # 输出模型回复中的文本内容
    print("== 示例 1：消息调用 ==")
    print(response.content)


def example_output_parser_step():
    """示例 2：先调用模型，再用输出解析器提取字符串。"""
    model = build_model()
    messages = build_translate_messages()
    parser = StrOutputParser()

    # 第一步：调用模型，得到 AIMessage
    result = model.invoke(messages)

    # 第二步：解析成普通字符串
    text = parser.invoke(result)

    print("== 示例 2：输出解析器分步写法 ==")
    print(result)
    print(text)


def example_output_parser_chain():
    """示例 3：使用 LCEL 将模型与输出解析器串成链。"""
    model = build_model()
    messages = build_translate_messages()
    parser = StrOutputParser()

    # 使用 | 把模型和输出解析器串成一条链
    chain = model | parser

    print("== 示例 3：输出解析器链式写法 ==")
    print(chain.invoke(messages))

def example_prompt_template():
    """示例 4：只使用提示词模板生成消息。"""
    # 定义系统提示词模板，其中 {language} 是动态变量
    system_template = "Translate the following into {language}:"
    
    # 创建聊天提示词模板
    # 最终会根据传入参数生成 system 消息和 user 消息
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", "{text}"),
    ])

    # 向模板传入变量，生成真正的聊天消息
    prompt_value = prompt_template.invoke({
        "language": "italian",
        "text": "hi",
    })

    # 查看模板生成后的消息列表
    print("== 示例 4：提示词模板生成消息 ==")
    print(prompt_value.to_messages())


def example_prompt_chain():
    """示例 5：将提示词模板、模型和输出解析器串成完整链。"""
    model = build_model()
    parser = StrOutputParser()

    # 定义系统提示词模板，其中 {language} 是动态变量
    system_template = "Translate the following into {language}:"

    # 创建聊天提示词模板
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", "{text}"),
    ])

    # 将提示词模板、模型和输出解析器串成完整链
    chain = prompt_template | model | parser

    print("== 示例 5：提示词模板链式写法 ==")
    print(chain.invoke({
        "language": "chinese",
        "text": "hi",
    }))


def example_run_remoter():
    # 连接到已经启动好的 LangServe 服务
    remote_chain = RemoteRunnable("http://localhost:8080/chain/")

    # 像调用本地链一样，远程调用服务端链
    result = remote_chain.invoke({
        "language": "德语",
        "text": "hi",
    })

    # 输出远程服务返回的翻译结果
    print(result)



def main():
    has_enabled_example = False

    try:
        if RUN_EXAMPLE_MESSAGES_TRANSLATE:
            has_enabled_example = True
            example_messages_translate()

        if RUN_EXAMPLE_OUTPUT_PARSER_STEP:
            has_enabled_example = True
            example_output_parser_step()

        if RUN_EXAMPLE_OUTPUT_PARSER_CHAIN:
            has_enabled_example = True
            example_output_parser_chain()

        if RUN_EXAMPLE_PROMPT_TEMPLATE:
            has_enabled_example = True
            example_prompt_template()

        if RUN_EXAMPLE_PROMPT_CHAIN:
            has_enabled_example = True
            example_prompt_chain()

        if RUN_REMOTE_RUNNABLE_SERVER:
            has_enabled_example = True
            example_run_remoter()

        if not has_enabled_example:
            print("当前没有启用任何示例，请将示例开关设置为 True。")
    finally:
        # tracing 开启时，等待所有 trace 提交完成再退出
        if ENABLE_LANGSMITH_TRACING:
            wait_for_all_tracers()


if __name__ == "__main__":
    main()
