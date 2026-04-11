# BM25 RAG Design

**Goal:** 把当前仓库中的 `rag.py` 从依赖 embedding 的向量检索实现，改成不依赖 embedding URL 的 `BM25Retriever` 实现，同时在 `note.md` 中明确记录“向量检索”和“BM25 检索”两种 RAG 方案的区别。

**Background**

当前仓库的 `rag.py` 基于 LangChain 官方 RAG 教程主线，使用了：

- `OpenAIEmbeddings`
- `Chroma`
- `vectorstore.as_retriever(...)`

这条路线要求底层 provider 提供可用的 embedding 模型与 `/embeddings` 接口。当前环境不具备这项能力，因此代码无法按现状跑通。

LangChain 官方同时提供了不依赖 embedding 的检索器实现，例如 `BM25Retriever`。这类方案不走“文本转向量”的语义检索路径，而是基于关键词统计进行检索，因此适合当前环境。

## Proposed Approach

本次改造采用单方案替换，不在仓库里并存两套可执行实现。

- 代码实现统一切换到 `BM25Retriever`
- 保留官方教程里“向量检索主线”的知识解释
- 在笔记中增加“当前仓库的无 embedding 改写版”与“两种方式的区别”

这样做的原因是：

- 当前需求是“先跑通学习示例”，不是做检索方案横向实验
- 同时保留两套实现会增加维护负担，也会让当前学习目标变得分散
- 笔记中保留对比已经足够承载“为什么官方教程这样写、当前仓库为什么这样改”的理解

## Design

### 1. `rag.py` 的结构调整

保留以下阶段不变：

- `load_docs()`
- `split_docs()`
- `build_rag_chain(...)`

替换以下阶段：

- 删除 `build_embeddings()`
- 删除 `build_vectorstore(...)`
- 把 `retrieve_docs(vectorstore, query)` 改成基于 `BM25Retriever` 的实现
- 新增 `build_retriever(all_splits)`，直接从切分后的 `Document` 列表构造 BM25 检索器

新的主流程变为：

1. 加载网页，得到 `Document` 列表
2. 切分文档，得到多个文本块
3. 使用 `BM25Retriever.from_documents(all_splits)` 构造检索器
4. 对用户问题执行 `retriever.invoke(query)` 取回相关块
5. 把检索到的上下文和问题一起交给模型生成答案

### 2. 检索语义的变化

切换到 `BM25Retriever` 后，检索行为从“向量语义相似度”变成“关键词相关性”。

这意味着：

- 不再需要 embedding 模型
- 不再需要向量数据库
- 检索结果更依赖用户问题中的关键词是否与文档块中的词汇匹配

这不是“和官方教程完全等价”的替换，而是“在当前环境约束下的可运行替代方案”。

### 3. `note.md` 的更新方式

`rag` 章节当前的主问题仍保持为：

- `Q-RAG-01-rag-stages`

在该问题下新增 `疑点 / 追加提问`，解释：

- 为什么官方教程使用 embedding + vector store
- 为什么当前仓库改成 BM25
- 两种检索方式在依赖、检索原理、适用场景上的区别

同时把当前 `源码示例` 中的“可运行示例”改为 BM25 版本，并补一个对比示例片段，用于说明向量检索版与 BM25 版的结构差异。

## Files To Change

- Modify: `rag.py`
- Modify: `note.md`
- Modify or replace: `tests/test_rag.py`

## Testing

本次改造优先做“本地可验证、不依赖真实网络或模型服务”的测试：

- 测试 `build_retriever(...)` 返回的对象可用于 `invoke(...)`
- 测试 `rag.py` 不再引用 `OpenAIEmbeddings` 与 `Chroma`
- 继续保留 `python -m py_compile` 的语法检查

不在本次设计里承诺真实运行网页抓取与模型生成，因为这仍依赖外部网络与可用模型。

## Risks

- `BM25Retriever` 的检索效果通常不如语义检索自然，尤其在用户问题与原文措辞差异较大时
- 教程知识线会从“官方标准主线”变成“官方主线 + 当前环境下的替代实现”
- 如果后续你拿到了可用 embedding URL，可能还需要再回切到向量检索实现

## Acceptance Criteria

- `rag.py` 不再依赖 embedding 模型与向量存储
- `rag.py` 能通过语法检查
- `tests/test_rag.py` 覆盖新的 BM25 路径
- `note.md` 明确写出“向量检索 vs BM25 检索”的区别，并解释为什么当前仓库使用 BM25
