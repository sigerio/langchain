import os

import bs4
from huggingface_hub import InferenceClient
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
        # huggingface_hub 可能返回 ndarray，也可能返回普通 list
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
    """创建与当前仓库一致的聊天模型实例。"""
    return ChatOpenAI(
        model="gpt-5.4",
        base_url="https://api.udcode.cn/v1",
        api_key=os.getenv("OPENAI_API_KEY", "你的API_KEY"),
        # 当前渠道是 OpenAI-compatible 接口，显式固定到 chat completions 路径。
        use_responses_api=False,
    )


def build_embeddings() -> HuggingFaceBgeM3Embeddings:
    """构造 Hugging Face 版 embedding 模型。"""
    api_key = os.getenv("HUGGING_EMBEDDING_API_KEY", "")
    return HuggingFaceBgeM3Embeddings(
        api_key=api_key,
        model_name=HF_EMBEDDING_MODEL,
    )


def load_docs():
    """示例 1：加载网页正文，得到 Document 列表。"""
    # 只保留标题、正文和小节标题，尽量贴近官方教程中的写法。
    bs4_strainer = bs4.SoupStrainer(
        class_=("post-title", "post-header", "post-content")
    )
    loader = WebBaseLoader(
        web_paths=(SOURCE_URL,),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()

    print("=== 加载阶段：Document 数量 ===")
    print(len(docs))
    print("=== 第一篇 Document 的 metadata ===")
    print(docs[0].metadata)
    print("=== 第一篇 Document 前 300 个字符 ===")
    print(docs[0].page_content[:300])
    print()

    return docs


def split_docs(docs):
    """示例 2：把长文档切成更适合索引和检索的块。"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)

    print("=== 分割阶段：切分后的块数 ===")
    print(len(all_splits))
    print("=== 第一个块的 metadata ===")
    print(all_splits[0].metadata)
    print("=== 第一个块前 300 个字符 ===")
    print(all_splits[0].page_content[:300])
    print()

    return all_splits


def build_retriever(all_splits):
    """示例 3：使用 Hugging Face embedding 构造向量检索器。"""
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=build_embeddings(),
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6},
    )
    return retriever


def retrieve_docs(retriever, query: str):
    """示例 4：使用向量检索器取回相关文档块。"""
    retrieved_docs = retriever.invoke(query)

    print("=== 检索阶段：命中的文档块数量 ===")
    print(len(retrieved_docs))
    print("=== 第一个检索结果的 metadata ===")
    print(retrieved_docs[0].metadata)
    print("=== 第一个检索结果前 300 个字符 ===")
    print(retrieved_docs[0].page_content[:300])
    print()

    return retriever


def build_rag_chain(retriever):
    """示例 5：把检索结果和问题一起交给模型生成答案。"""
    model = build_model()

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
        # 把多个 Document 的正文拼成一个字符串，再送入提示词。
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return rag_chain


if __name__ == "__main__":
    docs = load_docs()
    all_splits = split_docs(docs)

    query = "What is Task Decomposition?"
    retriever = build_retriever(all_splits)
    retriever = retrieve_docs(retriever, query=query)

    rag_chain = build_rag_chain(retriever)

    print("=== 生成阶段：最终回答 ===")
    answer = rag_chain.invoke(query)
    print(answer)
