import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def build_model() -> ChatOpenAI:
    """创建聊天模型实例。"""
    return ChatOpenAI(
        model="gpt-5.4",
        base_url="https://api.udcode.cn/v1",
        api_key=os.getenv("OPENAI_API_KEY", "你的API_KEY"),
    )


def example_model_stream() -> None:
    """示例 1：直接对聊天模型使用 .stream()."""
    model = build_model()

    messages = [
        SystemMessage(content="Translate the following from English into Italian"),
        HumanMessage(content="hi!"),
    ]

    # stream() 不会一次性返回完整 AIMessage，
    # 而是持续产出多个 AIMessageChunk。
    full = None

    print("=== 示例 1：model.stream(messages) ===")
    for chunk in model.stream(messages):
        # 官方文档推荐直接读取 chunk.text 观察流式文本。
        print(chunk.text, end="", flush=True)

        # 每个 chunk 都可以通过 + 逐步拼回完整消息。
        full = chunk if full is None else full + chunk

    print("\n=== 最终聚合后的完整 AIMessageChunk ===")
    print(full)
    print("=== 最终聚合后的文本 ===")
    print(full.text if full is not None else "")


def example_chain_stream() -> None:
    """示例 2：对提示词模板 + 模型组成的链使用 .stream()."""
    model = build_model()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Translate the following into {language}."),
        ("user", "{text}"),
    ])

    # Runnable 文档说明：通过 | 组合得到的链，天然也支持 stream()。
    chain = prompt | model

    full = None

    print("=== 示例 2：chain.stream({...}) ===")
    for chunk in chain.stream(
        {
            "language": "Italian",
            "text": "I love building with LangChain.",
        }
    ):
        print(chunk.text, end="", flush=True)
        full = chunk if full is None else full + chunk

    print("\n=== 链式调用聚合后的完整结果 ===")
    print(full)
    print("=== 链式调用聚合后的文本 ===")
    print(full.text if full is not None else "")


def example_chain_stream_with_parser() -> None:
    """示例 3：对提示词模板 + 模型 + 输出解析器使用 .stream()."""
    model = build_model()
    parser = StrOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Translate the following into {language}."),
        ("user", "{text}"),
    ])

    # StrOutputParser 官方参考说明它支持 streaming，
    # 因此这条链可以直接流式产出字符串 chunk。
    chain = prompt | model | parser

    full_text = ""

    print("=== 示例 3：chain.stream({...}) + StrOutputParser ===")
    for chunk in chain.stream(
        {
            "language": "Italian",
            "text": "I love building with LangChain.",
        }
    ):
        print(chunk, end="", flush=True)
        full_text += chunk

    print("\n=== 解析后的完整文本 ===")
    print(full_text)


if __name__ == "__main__":
    example_model_stream()
    print()
    example_chain_stream()
    print()
    example_chain_stream_with_parser()
