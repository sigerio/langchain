# BM25 RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将当前仓库中的 `rag.py` 改成不依赖 embedding 模型的 BM25 检索实现，并在 `note.md` 中补充 BM25 与向量检索的区别。

**Architecture:** 保留官方教程的 `加载 -> 分割 -> 检索 -> 生成` 主链路，但把“向量存储 + embeddings”替换为 `BM25Retriever`。测试只验证本地可确定的行为：检索器构建方式、模块依赖变化与语法正确性。

**Tech Stack:** Python, LangChain, `langchain_community.retrievers.BM25Retriever`, `unittest`

---

### Task 1: Replace Embedding Test With BM25 Test

**Files:**
- Modify: `tests/test_rag.py`
- Test: `tests/test_rag.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest
from unittest.mock import patch

import rag


class FakeBM25Retriever:
    @classmethod
    def from_documents(cls, documents, k):
        return {
            "documents": documents,
            "k": k,
        }


class RagBM25RetrieverTest(unittest.TestCase):
    def test_build_retriever_uses_bm25_from_documents(self):
        documents = ["chunk-1", "chunk-2"]

        with patch.object(rag, "BM25Retriever", FakeBM25Retriever):
            retriever = rag.build_retriever(documents)

        self.assertEqual(retriever["documents"], documents)
        self.assertEqual(retriever["k"], 6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest discover -s tests -p 'test_rag.py' -v`
Expected: FAIL because `rag.build_retriever` does not exist yet

- [ ] **Step 3: Write minimal implementation**

```python
from langchain_community.retrievers import BM25Retriever


def build_retriever(all_splits):
    return BM25Retriever.from_documents(all_splits, k=6)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest discover -s tests -p 'test_rag.py' -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_rag.py rag.py
git commit -m "test: cover BM25 retriever setup"
```

### Task 2: Switch `rag.py` To BM25 Retrieval

**Files:**
- Modify: `rag.py`
- Test: `tests/test_rag.py`

- [ ] **Step 1: Write the failing dependency assertions**

```python
import pathlib
import unittest


class RagSourceStructureTest(unittest.TestCase):
    def test_rag_uses_bm25_not_embeddings(self):
        source = pathlib.Path("rag.py").read_text()
        self.assertIn("BM25Retriever", source)
        self.assertNotIn("OpenAIEmbeddings", source)
        self.assertNotIn("langchain_chroma", source)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest discover -s tests -p 'test_rag.py' -v`
Expected: FAIL because `rag.py` still imports embeddings and chroma

- [ ] **Step 3: Write minimal implementation**

```python
from langchain_community.retrievers import BM25Retriever


def build_retriever(all_splits):
    return BM25Retriever.from_documents(all_splits, k=6)


def retrieve_docs(retriever, query: str):
    retrieved_docs = retriever.invoke(query)
    ...
    return retriever


if __name__ == "__main__":
    docs = load_docs()
    all_splits = split_docs(docs)
    query = "What is Task Decomposition?"
    retriever = build_retriever(all_splits)
    retriever = retrieve_docs(retriever, query=query)
```

- [ ] **Step 4: Run tests and syntax verification**

Run: `python -m unittest discover -s tests -p 'test_rag.py' -v`
Expected: PASS

Run: `python -m py_compile rag.py tests/test_rag.py`
Expected: PASS with no output

- [ ] **Step 5: Commit**

```bash
git add rag.py tests/test_rag.py
git commit -m "refactor: switch rag example to BM25 retrieval"
```

### Task 3: Update Notes For BM25 vs Vector Retrieval

**Files:**
- Modify: `note.md`

- [ ] **Step 1: Update the runnable source example**

```markdown
把当前可运行示例改成：
- `BM25Retriever.from_documents(all_splits, k=6)`
- `retriever.invoke(query)`
- 不再包含 `OpenAIEmbeddings` 与 `Chroma`
```

- [ ] **Step 2: Add a comparison note under `疑点`**

```markdown
##### 追加提问 2：向量检索和 BM25 检索有什么区别？

- 向量检索：依赖 embedding 模型与向量存储，按语义相似度检索
- BM25 检索：不依赖 embedding，按关键词统计相关性检索
- 当前仓库使用 BM25，是因为当前环境没有可用的 embedding URL
```

- [ ] **Step 3: Verify note structure**

Run: `rg -n '^## 大章节：`rag`|^### 问题 1 \\[ACT\\]|^##### 追加提问 2：向量检索和 BM25 检索有什么区别？' note.md`
Expected: shows the `rag` section, active question, and new follow-up entry

- [ ] **Step 4: Commit**

```bash
git add note.md
git commit -m "docs: explain BM25 rag variant and retrieval differences"
```
