# Modular RAG MCP Server

一个基于多阶段检索增强生成（RAG）与模型上下文协议（MCP）的可扩展智能知识检索框架。

> 本项目是一个**实战驱动的学习平台**，将 RAG 核心工程实践融入代码设计，适合用于技术学习、简历项目与面试备战。

---

## 项目特点

- **混合检索（Hybrid Search）**：BM25 稀疏检索 + Dense Embedding 语义检索，通过 RRF 算法融合，兼顾查准率与查全率
- **两段式精排架构**：粗排（低成本泛召回）→ 精排（Cross-Encoder / LLM Rerank），大幅提升 Top-K 精准度
- **全链路可插拔**：LLM、Embedding、向量数据库、分块策略、评估框架均通过统一抽象接口封装，配置文件一键切换，零代码修改
- **MCP 标准接口**：作为 MCP Server 运行，可直接对接 GitHub Copilot、Claude Desktop 等主流 AI 客户端
- **多模态支持**：Image-to-Text 策略，利用 Vision LLM 生成图像描述并融入检索链路
- **全链路可观测**：结构化追踪覆盖 Ingestion 与 Query 两条完整流水线，配套 Streamlit 可视化管理面板

---

## 技术栈

| 层次 | 技术选型 |
|------|---------|
| 文档解析 | MarkItDown（PDF → Markdown） |
| 文本分块 | LangChain RecursiveCharacterTextSplitter |
| 向量数据库 | ChromaDB（本地嵌入式） |
| 稀疏检索 | BM25（自研实现） |
| Embedding | OpenAI / Azure OpenAI / Ollama（可插拔） |
| LLM | OpenAI / Azure / DeepSeek / Ollama（可插拔） |
| MCP 服务 | Python 官方 MCP SDK |
| 可视化面板 | Streamlit |
| 评估框架 | Ragas / 自定义指标（可插拔） |

---

## 快速开始

### 环境要求

- Python >= 3.10
- （可选）Ollama，用于本地 Embedding 模型

### 安装

```bash
pip install -e .
```

### 配置

复制并编辑配置文件：

```bash
cp config/test_credentials.yaml.example config/settings.yaml
```

`config/settings.yaml` 关键配置项：

```yaml
llm:
  provider: openai       # openai | azure | ollama | deepseek
  model: gpt-4o
  api_key: YOUR_API_KEY

embedding:
  provider: ollama       # openai | azure | ollama
  model: nomic-embed-text
  base_url: http://localhost:11434

rerank:
  enabled: false
  provider: none         # none | cross_encoder | llm
```

### 文档摄取

```bash
# 摄取单个 PDF
python scripts/ingest.py --path documents/report.pdf --collection my_docs

# 摄取整个目录
python scripts/ingest.py --path documents/ --collection my_docs

# 强制重新处理
python scripts/ingest.py --path documents/report.pdf --force
```

### 知识库查询

```bash
# 基础查询
python scripts/query.py --query "你的问题" --collection my_docs

# 详细模式（展示 Dense/Sparse/Fusion/Rerank 各阶段结果）
python scripts/query.py --query "你的问题" --verbose

# 禁用重排
python scripts/query.py --query "你的问题" --no-rerank
```

### 启动可视化面板

```bash
python scripts/start_dashboard.py
# 或
streamlit run scripts/start_dashboard.py
```

面板提供：系统总览、数据浏览器、Ingestion 管理、Query 追踪、Ingestion 追踪、评估面板。

---

## 项目结构

```
.
├── config/
│   ├── settings.yaml          # 主配置文件
│   └── prompts/               # LLM Prompt 模板
├── data/
│   └── db/                    # 本地数据库（Chroma、BM25、SQLite）
├── scripts/
│   ├── ingest.py              # 文档摄取 CLI
│   ├── query.py               # 知识库查询 CLI
│   ├── evaluate.py            # 评估脚本
│   └── start_dashboard.py     # 启动可视化面板
├── src/
│   ├── core/                  # 核心类型、配置、查询引擎、追踪
│   │   └── query_engine/      # HybridSearch、Reranker、QueryProcessor
│   ├── ingestion/             # 摄取流水线（Loader→Split→Transform→Embed→Upsert）
│   ├── libs/                  # 可插拔组件（LLM、Embedding、VectorStore、Evaluator）
│   └── observability/         # 日志、追踪、Streamlit Dashboard
├── logs/
│   └── traces.jsonl           # 结构化追踪日志
└── main.py                    # MCP Server 入口
```

---

## MCP 接入

将本项目作为 MCP Server 接入 GitHub Copilot 或 Claude Desktop：

```json
{
  "mcpServers": {
    "knowledge-hub": {
      "command": "python",
      "args": ["main.py"],
      "cwd": "/path/to/this/project"
    }
  }
}
```

暴露的核心工具：

| 工具名 | 功能 |
|--------|------|
| `query_knowledge_hub` | 混合检索 + Rerank，返回带引用的结构化结果 |
| `list_collections` | 列举可用的文档集合 |
| `get_document_summary` | 获取指定文档的摘要与元信息 |

---

## RAG 流水线概览

```
文档摄取（Ingestion）
  PDF → MarkItDown → RecursiveCharacterTextSplitter
      → LLM Transform（Refine / Metadata Enrichment / Image Captioning）
      → Dense + Sparse 双路编码
      → Chroma（向量） + BM25（倒排） + SQLite（历史记录）

知识检索（Query）
  Query → 关键词提取 → Dense 召回 + Sparse 召回
        → RRF 融合 → Rerank（可选）→ 返回 Top-K + 引用
```

---

## 可插拔组件切换

修改 `config/settings.yaml` 即可切换任意组件，无需改代码：

```yaml
# 切换 LLM
llm:
  provider: ollama   # 改为本地模型

# 切换 Embedding
embedding:
  provider: openai

# 开启重排
rerank:
  enabled: true
  provider: cross_encoder
```

---

## 许可证

MIT License
