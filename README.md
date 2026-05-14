# VeriMind

VeriMind 是一个面向学术知识库的可信 Agentic RAG 原型系统，重点探索多粒度检索、意图感知路由、回答忠实度审计和证据不足时的保守拒答机制。

项目关注的问题不是“如何让模型总能回答”，而是“如何让系统在证据不足、检索不可靠或回答超出上下文时及时暴露风险”。

## 1. Research Motivation

标准 RAG 系统常能召回看似相关的文本片段，但仍可能生成证据不支持的结论。对于学术研究、专业知识库和低容错问答场景，仅仅把文档切块后交给大模型并不够。

VeriMind 尝试通过多粒度索引、意图感知路由和回答后置审计，将普通检索问答推进到 evidence-aware generation：

- 用户问题先被路由到合适的检索粒度。
- 系统从不同粒度的知识索引中召回证据。
- 回答生成后再通过忠实度审计进行风险检查。
- 当证据不足时，系统优先输出保守说明，而不是强行生成。

## 2. Core Ideas

### Multi-granularity Retrieval

系统围绕不同问题类型建立多粒度索引：

- `DETAIL`：面向参数、年份、定义等细节问题。
- `CONCEPT`：面向概念、方法、机制解释。
- `CONTEXT`：面向论文总结、多文档对比和长上下文问题。

这种设计用于缓解固定切块带来的语义截断和上下文不足问题。

### Intent-aware Routing

系统根据用户问题的意图选择更合适的检索粒度，而不是对所有问题使用同一种 chunk 大小和同一套召回策略。

### Answer Auditing

回答生成后，系统通过 LLM-as-a-Judge 的方式检查回答是否忠实于检索证据，并结合检索相关性形成 Trust-Score。

### Conservative Refusal

当检索证据不足或回答可能超出上下文时，系统倾向于说明证据不足，而不是生成没有来源支撑的结论。

## 3. Tech Stack

- Python
- LlamaIndex
- Alibaba DashScope / Qwen
- ChromaDB
- Streamlit
- Conda

## 4. Quick Start

### 1. Create environment

```bash
conda create -n verimind python=3.10
conda activate verimind
```

### 2. Install dependencies

```bash
pip install streamlit llama-index llama-index-llms-dashscope llama-index-embeddings-dashscope chromadb python-dotenv tqdm
```

### 3. Configure API key

Create a `.env` file in the project root:

```env
DASHSCOPE_API_KEY=sk-xxxx
```

### 4. Build index

Put Markdown documents into `./knowledge_mds`, then run:

```bash
python build_index.py
```

### 5. Start app

```bash
streamlit run rag_app_final.py
```

## 5. Repository Scope

This repository focuses on text-based academic knowledge bases. It is not intended to solve all document intelligence problems.

In particular:

- Multimodal chart/table understanding is not included in this repository.
- Production-grade access control, monitoring, and replay systems are not included yet.
- Evaluation results should be interpreted together with raw logs, benchmark files, and manuscript materials when available.

## 6. Roadmap

- Hybrid retrieval with sparse + dense search.
- Reranking for noisy top-k results.
- Claim-evidence alignment for key answer statements.
- Structured audit logs and replayable traces.
- Failure taxonomy for retrieval miss, unsupported claim, over-refusal and under-refusal.
- MM-RAG / VeriSight extension for chart-text evidence alignment.

## 7. License

This project is released under the MIT License.
