# LangGraph Prep Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将当前以 LangChain 为主的学习仓库整理为清晰的归档结构，为后续 LangGraph 学习预留独立目录与独立笔记文件。

**Architecture:** 采用“归档迁移而非行为改造”的方式重组仓库。根目录只保留仓库级入口与规则，现有 LangChain 示例、测试和笔记迁移到按主题划分的子目录，并通过导航文档维护旧路径到新路径的映射。

**Tech Stack:** Python, pytest/unittest, Markdown, git worktrees

---

### Task 1: 建立新目录骨架与仓库总入口

**Files:**
- Create: `README.md`
- Create: `notes/langgraph-note.md`
- Create: `docs/archive/langchain-index.md`

- [ ] **Step 1: 写出需要存在的新目录与入口文件草稿**

```text
README.md
notes/langgraph-note.md
docs/archive/langchain-index.md
```

- [ ] **Step 2: 创建根目录 README**

```markdown
# LangChain / LangGraph Learning Repo

本仓库用于按官方教程学习 LangChain 与 LangGraph。
当前 `tutorials/langchain/` 保存已归档的 LangChain 示例；
后续 `tutorials/langgraph/` 用于新的 LangGraph 学习内容。
```

- [ ] **Step 3: 创建 LangGraph 独立笔记模板**

```markdown
# LangGraph 官方教程学习笔记

本笔记用于记录 LangGraph 官方教程学习过程中的概念理解、代码写法和疑点追踪。
```

- [ ] **Step 4: 创建 LangChain 归档导航文档**

```markdown
# LangChain Archive Index

记录旧文件路径、新文件路径和对应官方教程主题，便于回看与检索。
```

- [ ] **Step 5: 运行文件存在性检查**

Run: `find README.md notes docs/archive -maxdepth 2 -type f | sort`
Expected: 输出新增入口文件列表

- [ ] **Step 6: Commit**

```bash
git add README.md notes/langgraph-note.md docs/archive/langchain-index.md
git commit -m "docs: add repository reorganization entrypoints"
```

### Task 2: 迁移 LangChain 笔记与示例目录

**Files:**
- Modify: `notes/langchain-note.md`
- Create: `tutorials/langchain/basics/`
- Create: `tutorials/langchain/rag/`
- Create: `tutorials/langchain/history/`
- Create: `tutorials/langchain/agents/`
- Create: `tutorials/langchain/serve/`

- [ ] **Step 1: 将旧笔记重命名到新位置**

```bash
mv note.md notes/langchain-note.md
```

- [ ] **Step 2: 创建 LangChain 主题目录**

```bash
mkdir -p tutorials/langchain/{basics,rag,history,agents,serve}
```

- [ ] **Step 3: 迁移基础示例与服务化示例**

```bash
mv demo.py tutorials/langchain/basics/llm_chain_demo.py
mv server.py tutorials/langchain/serve/langserve_server.py
```

- [ ] **Step 4: 迁移 RAG、历史、代理示例**

```bash
mv rag.py tutorials/langchain/rag/rag_demo.py
mv chatbot.py tutorials/langchain/history/chatbot_basic.py
mv chatbot-stream.py tutorials/langchain/history/chatbot_stream.py
mv chatbot-history.py tutorials/langchain/history/chatbot_history.py
mv chatbot-trim-history.py tutorials/langchain/history/chatbot_trim_history.py
mv agents.py tutorials/langchain/agents/agents_demo.py
mv qa_chat_history.py tutorials/langchain/agents/qa_chat_history.py
```

- [ ] **Step 5: 运行目录结构检查**

Run: `find tutorials/langchain -maxdepth 3 -type f | sort`
Expected: 输出所有迁移后的 LangChain 示例文件

- [ ] **Step 6: Commit**

```bash
git add notes/langchain-note.md tutorials/langchain
git commit -m "refactor: archive langchain examples by topic"
```

### Task 3: 迁移测试并修正导入与路径引用

**Files:**
- Create: `tests/langchain/test_rag.py`
- Create: `tests/langchain/test_qa_chat_history.py`
- Modify: `tests/langchain/test_rag.py`
- Modify: `tests/langchain/test_qa_chat_history.py`

- [ ] **Step 1: 迁移现有测试文件到 LangChain 子目录**

```bash
mkdir -p tests/langchain
mv tests/test_rag.py tests/langchain/test_rag.py
mv tests/test_qa_chat_history.py tests/langchain/test_qa_chat_history.py
```

- [ ] **Step 2: 先写失败测试，要求它从新路径加载模块**

```python
import importlib.util
import pathlib

RAG_PATH = pathlib.Path("tutorials/langchain/rag/rag_demo.py")
assert RAG_PATH.exists()
```

- [ ] **Step 3: 运行测试确认旧路径假设失效或新路径断言尚未补齐**

Run: `pytest tests/langchain/test_rag.py tests/langchain/test_qa_chat_history.py -q`
Expected: FAIL，提示导入路径或源码路径仍引用旧位置

- [ ] **Step 4: 用最小修改让测试从新文件位置加载源码**

```python
spec = importlib.util.spec_from_file_location("rag_demo", RAG_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
```

- [ ] **Step 5: 再次运行迁移后的测试**

Run: `pytest tests/langchain/test_rag.py tests/langchain/test_qa_chat_history.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/langchain
git commit -m "test: relocate langchain tests with source path loading"
```

### Task 4: 更新规则文档与归档导航

**Files:**
- Modify: `AGENTS.md`
- Modify: `notes/langchain-note.md`
- Modify: `docs/archive/langchain-index.md`
- Modify: `README.md`

- [ ] **Step 1: 更新 AGENTS 记录规则目标文件**

```markdown
- 当讨论 LangChain 时，读取并记录到 `notes/langchain-note.md`
- 当讨论 LangGraph 时，读取并记录到 `notes/langgraph-note.md`
```

- [ ] **Step 2: 调整 LangChain 笔记开头说明和内部源码路径引用**

```markdown
# LangChain 官方教程学习笔记

说明：本笔记对应 `tutorials/langchain/` 下的归档示例。
```

- [ ] **Step 3: 补全归档导航的旧路径到新路径映射**

```markdown
| 旧路径 | 新路径 | 主题 |
| --- | --- | --- |
| `demo.py` | `tutorials/langchain/basics/llm_chain_demo.py` | 基础链路 |
```

- [ ] **Step 4: 在 README 中说明根目录约定与后续 LangGraph 扩展位**

```markdown
- `tutorials/langchain/`: 已归档的 LangChain 示例
- `tutorials/langgraph/`: 后续 LangGraph 学习目录
- `notes/`: 分主题学习笔记
```

- [ ] **Step 5: 运行文本搜索确认旧路径基本已替换**

Run: `rg -n "note\\.md|demo\\.py|server\\.py|rag\\.py|qa_chat_history\\.py" README.md AGENTS.md notes docs tests tutorials`
Expected: 仅保留导航文档中的旧路径映射，不再把旧路径当作当前主路径使用

- [ ] **Step 6: Commit**

```bash
git add AGENTS.md README.md notes docs/archive
git commit -m "docs: align archive docs and note targets"
```

### Task 5: 验证整理结果

**Files:**
- Verify only

- [ ] **Step 1: 运行迁移后的测试**

Run: `pytest tests/langchain/test_rag.py tests/langchain/test_qa_chat_history.py -q`
Expected: PASS

- [ ] **Step 2: 运行 Python 语法检查**

Run: `python -m py_compile $(find tutorials/langchain -name '*.py' | sort)`
Expected: PASS，无输出

- [ ] **Step 3: 检查最终目录结构**

Run: `find . -maxdepth 3 -type f | sort`
Expected: 根目录仅保留仓库级入口文件，LangChain 示例集中在 `tutorials/langchain/`

- [ ] **Step 4: 检查 git 状态**

Run: `git status --short`
Expected: 仅显示本次整理相关改动

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: reorganize repo for langgraph prep"
```
