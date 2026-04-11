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


SOURCE_URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"
HF_EMBEDDING_MODEL = "BAAI/bge-m3"


class HuggingFaceBgeM3Embeddings(Embeddings):
    """使用 Hugging Face Inference Providers 调用 BAAI/bge-m3。"""

    def __init__(self, api_key: str, model_name: str = HF_EMBEDDING_MODEL) -> None:
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
        result = self.client.feature_extraction(
            texts,
            model=self.model_name,
        )
        if hasattr(result, "tolist"):
            result = result.tolist()
        return [list(item) for item in result]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)


def build_model() -> ChatOpenAI:
    """构造聊天模型。

    这里延续当前仓库的 OpenAI-compatible 写法，并显式关闭 responses api，
    以避免兼容渠道和 chat completions / responses API 的差异。
    """
    return ChatOpenAI(
        model="gpt-5.4",
        base_url="https://api.udcode.cn/v1",
        api_key=os.getenv("OPENAI_API_KEY", "你的API_KEY"),
        use_responses_api=False,
        temperature=0,
    )


def build_embeddings() -> HuggingFaceBgeM3Embeddings:
    """构造 Hugging Face 版 embedding 模型。"""
    api_key = os.getenv("HUGGING_EMBEDDING_API_KEY", "")
    return HuggingFaceBgeM3Embeddings(
        api_key=api_key,
        model_name=HF_EMBEDDING_MODEL,
    )


def build_retriever():
    """按官方教程主线构造检索器。

    注意：这一部分依赖 embedding 模型。
    如果当前 provider 不支持 embedding，需要替换为支持 embeddings 的渠道。
    """
    loader = WebBaseLoader(
        web_paths=(SOURCE_URL,),
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
        embedding=build_embeddings(),
    )
    return vectorstore.as_retriever()


def build_agent():
    """构造对话式 RAG agent。"""
    model = build_model()
    retriever = build_retriever()

    # 把检索器包装成工具后，agent 才能在执行过程中决定是否调用它。
    tool = create_retriever_tool(
        retriever,
        "blog_post_retriever",
        "Searches and returns excerpts from the Autonomous Agents blog post.",
    )
    tools = [tool]

    # LangGraph 的 checkpointer 会按 thread_id 保存同一会话的历史状态。
    memory = MemorySaver()

    agent_executor = create_react_agent(
        model,
        tools,
        checkpointer=memory,
    )
    return agent_executor


if __name__ == "__main__":
    agent_executor = build_agent()
    config = {"configurable": {"thread_id": "qa-chat-history-demo"}}

    print("=== 第一轮：普通寒暄，agent 可以选择不检索 ===")
    for step in agent_executor.stream(
        {"messages": [{"role": "user", "content": "Hi! I'm Bob"}]},
        config=config,
    ):
        print(step)
        print("----")

    print("=== 第二轮：知识问题，agent 可以决定调用检索工具 ===")
    for step in agent_executor.stream(
        {"messages": [{"role": "user", "content": "What is Task Decomposition?"}]},
        config=config,
    ):
        print(step)
        print("----")

    print("=== 第三轮：追问，agent 会在同一 thread_id 下继续参考历史 ===")
    for step in agent_executor.stream(
        {"messages": [{"role": "user", "content": "What are common ways of doing it?"}]},
        config=config,
    ):
        print(step)
        print("----")
