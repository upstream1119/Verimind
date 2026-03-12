# 🛡️ VeriMind Pro: Trustworthy Academic Agentic RAG

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![LlamaIndex](https://img.shields.io/badge/Framework-LlamaIndex-orange.svg)](https://www.llamaindex.ai/)
[![DashScope](https://img.shields.io/badge/LLM-Qwen--Plus-green.svg)](https://help.aliyun.com/zh/dashscope/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**VeriMind Pro** 是一款专为学术研究与专业知识库设计的 **Agentic RAG (代理式检索增强生成)** 系统。通过多粒度索引技术、意图感知路由以及实时信任评分机制，VeriMind Pro 为国内专业用户提供了一个高效、透明且极低幻觉的智能问答解决方案。

---

## ✨ 核心竞争力：为什么选择 VeriMind Pro？

1.  **🧠 意图感知的动态路由 (Intent-Aware Routing)**
    不同于传统的固定切片检索，VeriMind 会实时分析用户的提问意图：
    -   **DETAIL (128 Tokens)**: 针对参数、年份、具体定义等精确细节。
    -   **CONCEPT (512 Tokens)**: 针对概念原理、方法论解释。
    -   **CONTEXT (1024 Tokens)**: 针对论文总结、多文档对比、优缺点分析。
    系统自动匹配最佳检索粒度，确保“找得准且看得全”。

2.  **⚖️ 实时双重审计机制 (Dual-Verify Logic)**
    每一条生成的答案都会经过“质检员”的审查：
    -   **检索相关性评分**: 评估检索到的片段与问题的契合度。
    -   **忠实度审计 (Faithfulness)**: 采用 **LLM-as-a-Judge** 模式，严格对比生成的答案是否包含上下文之外的“私货”。
    最终生成 **Trust-Score (信任分)**，可视化展示结果的可信度。

3.  **🛡️ 幻觉“短路”拦截策略**
    在处理国内论文或特定技术文档时，如果检索内容不足，系统会触发“短路”机制：优先承认“未找到对应证据”并解释原因，而非强行回答。

4.  **💎 极致交互深度美学**
    界面基于 **Streamlit** 深度定制，通过 Frosted Glass (毛玻璃) 效果、霓虹状态灯和动态推理链条展示，让 RAG 的黑盒过程完全透明化。

---

## 🛠️ 技术栈与架构 (Tech Stack)

-   **核心框架**: [LlamaIndex](https://github.com/run-llama/llama_index) (数据编排与 Agent 流程控制)
-   **后端大模型**: [Alibaba DashScope](https://help.aliyun.com/zh/dashscope/) (主力模型: `qwen-plus`)
-   **向量数据库**: [ChromaDB](https://www.trychroma.com/) (本地持久化向量存储)
-   **UI 框架**: [Streamlit](https://streamlit.io/) (带高级 CSS 注入的 UI)
-   **环境配置**: 支持 Conda 虚拟环境隔离，确保依赖一致性。

---

## 🚀 快速上手 (Quick Start)

### 1. 环境准备
```bash
# 创建并激活 Conda 环境
conda create -n verimind python=3.10
conda activate verimind

# 安装依赖
pip install streamlit llama-index llama-index-llms-dashscope llama-index-embeddings-dashscope chromadb python-dotenv tqdm
```

### 2. 配置 API Key
在项目根目录创建 `.env` 文件，并填入您的通义千问 Key：
```env
DASHSCOPE_API_KEY=sk-xxxx...
```

### 3. 构建索引
将您的 Markdown 论文放入 `./knowledge_mds` 文件夹，然后运行：
```bash
python build_index.py
```

### 4. 启动应用
```bash
streamlit run rag_app_final.py
```

---

## 📈 未来规划 (Roadmap)

- [ ] **多路召回 (Hybrid Search)**: 引入 BM25 算法平衡关键词与向量的匹配权重。
- [ ] **重排序 (Reranker)**: 集成 BGE-Reranker 以进一步优化由于 Top-K 噪声引起的精度损失。
- [ ] **多模态解析**: 支持直接对 PDF 图片内容进行结构化理解。

---

## 📄 许可证
本项目采用 MIT 许可证。
