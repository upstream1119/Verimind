import streamlit as st
import os
import time
import re
import chromadb
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.dashscope import DashScopeEmbedding
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# --- 1. Page Config & CSS ---
st.set_page_config(
    page_title="VeriMind | Trustworthy Academic RAG",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS: Frosted glass, neon glow, dynamic cards
st.markdown("""
<style>
    /* Global background and font */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }

    /* Sidebar style */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }

    /* Core metric card container */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }

    /* Score color definitions */
    .score-high { color: #00FF41; text-shadow: 0 0 10px rgba(0, 255, 65, 0.4); }
    .score-mid { color: #FFD700; text-shadow: 0 0 10px rgba(255, 215, 0, 0.4); }
    .score-low { color: #FF3333; text-shadow: 0 0 10px rgba(255, 51, 51, 0.4); }

    /* Reference source tag */
    .source-tag {
        font-size: 0.8em;
        padding: 2px 8px;
        border-radius: 4px;
        background: #21262D;
        border: 1px solid #30363D;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Configuration & Initialization ---
# ⚠️ Ensure API Key is set in PyCharm env vars or here
if "DASHSCOPE_API_KEY" not in os.environ:
    # os.environ["DASHSCOPE_API_KEY"] = "sk-..." # Uncomment here temporarily if local run fails
    pass


# Initialize models
def init_models(temperature=0.1):
    Settings.llm = DashScope(
        model_name="qwen-plus",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        max_tokens=2048,
        temperature=temperature
    )
    Settings.embed_model = DashScopeEmbedding(
        model_name="text-embedding-v1",
        api_key=os.environ.get("DASHSCOPE_API_KEY")
    )


DATA_DIR = "./knowledge_mds"
INDEX_CONFIGS = {
    "DETAIL": {"path": "./chroma_db_md_128", "collection": "rag_md_128"},
    "CONCEPT": {"path": "./chroma_db_md_512", "collection": "rag_md_512"},
    "CONTEXT": {"path": "./chroma_db_md_1024", "collection": "rag_md_1024"},
}


@st.cache_resource(show_spinner=False)
def initialize_index():
    """Load Indices"""
    if not os.path.exists(DATA_DIR): return None
    all_indices = {}
    try:
        for name, config in INDEX_CONFIGS.items():
            db = chromadb.PersistentClient(path=config['path'])
            chroma_collection = db.get_or_create_collection(name=config['collection'])
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            if chroma_collection.count() > 0:
                all_indices[name] = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        return all_indices
    except Exception as e:
        return None


# --- 3. Core Logic: Routing & Scoring ---

def get_query_intent(query: str) -> str:
    q = query.lower()
    # 加入中文细节类关键词
    if any(k in q for k in ["how many", "year", "parameter", "who", "when", "哪一年", "多少", "谁", "年份", "参数",
                            "时间"]): return "DETAIL"
    # 加入中文宏观类关键词
    if any(k in q for k in
           ["compare", "difference", "summarize", "evaluate", "pros and cons", "比较", "区别", "总结", "对比",
            "优缺点"]): return "CONTEXT"
    return "CONCEPT"


def evaluate_faithfulness(query, answer, context):
    """Faithfulness Judge (LLM-as-a-Judge) - 工业加固版"""

    # 1. 确定性短路拦截：只要生成器老实承认了没找到，直接给 10 分，不劳烦裁判大模型纠结！
    refusal_flags = ["information not found", "未找到", "未提及", "无证据", "不知道", "没有提及"]
    if any(flag in answer.lower() for flag in refusal_flags):
        return 10

    # 2. 如果不是拒答，才走 LLM 裁判的严格审查
    prompt = f"""You are a rigorous academic peer reviewer. Evaluate if the [Generated Answer] is strictly and exclusively derived from the [Retrieved Context].
Instructions:
1. If the answer introduces external knowledge, specific examples, or entities NOT present in the context, score it 0.
2. Output ONLY a numerical integer score from 0 to 10. Do not output any other text, explanations, or rules.

[Retrieved Context]: {context}
[Generated Answer]: {answer}
Score:"""
    try:
        res = Settings.llm.complete(prompt).text.strip()
        score_match = re.search(r'(?:^|\s)(10|[0-9])(?:\s|$)', res)
        if score_match:
            score = int(score_match.group(1))
        else:
            score = int(re.search(r'\d+', res).group())
        return min(10, max(0, score))
    except Exception as e:
        print(f"Audit Error: {e}")
        return 6


# --- 4. UI Layout ---

with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/security-checked.png", width=60)
    st.title("VeriMind Pro")
    st.caption("🚀 Agentic RAG Framework")
    st.divider()

    st.markdown("### 🎛️ Console")
    temp = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.1)
    top_k = st.slider("Retrieval Depth (Top-K)", 1, 8, 3)

    init_models(temperature=temp)
    indices = initialize_index()

    if indices:
        st.success(f"✅ Knowledge Base Mounted ({len(indices)} Granularity Levels)")
    else:
        st.error("❌ Knowledge Base Disconnected")

    st.divider()
    st.markdown("### 📊 Real-time Monitor")
    st.info("Routing Mode: **Adaptive**")
    st.info("Scoring Model: **Dual-Verify**")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metrics" in msg:
            # Render dashboard in history
            m = msg["metrics"]
            # Simplified display
            st.caption(
                f"🛡️ Trust-Score: **{m['trust']}** ({m['delta']}) | Retrieval: {m['retrieval']} | Faithfulness: {m['faithfulness']}")

# Input processing
if prompt := st.chat_input("Enter a scientific query (e.g.: 比较 ReAct 和 Reflexion...)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if indices:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()

            # --- 🌟 Core Highlight: Reasoning Chain Visualization (The Hook) ---
            with st.status("🧠 VeriMind Agent Reasoning...", expanded=True) as status:

                # Step 1: Intent Analysis
                st.write("🔍 **Step 1: Semantic Intent Analysis**")
                intent = get_query_intent(prompt)
                time.sleep(0.5)  # Delay for demonstration
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;`Identified`: **{intent}** (Granularity Match)")

                # Step 2: Dynamic Retrieval
                st.write(f"📂 **Step 2: Dynamic Route Retrieval ({intent} Layer)**")
                index = indices.get(intent, indices["CONCEPT"])
                retriever = index.as_retriever(similarity_top_k=top_k)
                nodes = retriever.retrieve(prompt)

                # Simulate retrieval score calculation
                raw_scores = [n.score for n in nodes]
                avg_score = sum(raw_scores) / len(raw_scores) if raw_scores else 0
                calibrated_ret = min(10.0, max(0.0, (avg_score - 0.3) * 20))  # Simple mapping

                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;`Nodes Hit`: {len(nodes)} | `Avg Similarity`: {avg_score:.4f}")

                # Step 3: Generation
                st.write("✍️ **Step 3: Knowledge-Augmented Generation**")

                # 给生成器戴上“更聪明的紧箍咒”，彻底拦截幻觉并保持学者风度
                QA_PROMPT_TMPL = PromptTemplate(
                    "You are a strict academic assistant. Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information and NOT prior knowledge, answer the query in Chinese. "
                    "If the exact answer cannot be found in the context, you MUST explicitly start your response with '根据检索到的上下文，未找到足够的信息。' and then briefly explain what the context actually talks about and why it lacks the specific answer. "
                    "Do NOT use parametric memory or guess.\n"
                    "Query: {query_str}\n"
                    "Answer: "
                )

                query_engine = index.as_query_engine(
                    similarity_top_k=top_k,
                    text_qa_template=QA_PROMPT_TMPL
                )
                response = query_engine.query(prompt)

                # Step 4: Audit
                st.write("⚖️ **Step 4: Dual-Credibility Audit**")
                context_str = "\n".join([n.text for n in nodes])
                faith_score = evaluate_faithfulness(prompt, str(response), context_str)

                # Final calculation
                trust_score = 0.4 * calibrated_ret + 0.6 * faith_score

                status.update(label="✅ Reasoning Complete", state="complete", expanded=False)

            # --- Result Rendering ---
            st.markdown(str(response))

            # --- 🌟 Dashboard Rendering (CSS Enhanced) ---
            st.divider()

            # Color Logic
            if trust_score >= 7.5:
                color_class = "score-high"
                label = "TRUSTED"
                bg_color = "rgba(0, 255, 65, 0.1)"
            elif trust_score >= 5.0:
                color_class = "score-mid"
                label = "WARNING"
                bg_color = "rgba(255, 215, 0, 0.1)"
            else:
                color_class = "score-low"
                label = "REJECTED"
                bg_color = "rgba(255, 51, 51, 0.1)"

            c1, c2, c3 = st.columns([1, 1, 1.5])

            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:12px; color:#888;">📡 Retrieval Relevance</div>
                    <div style="font-size:24px; font-weight:bold; color:#00d4ff;">{calibrated_ret:.1f}</div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:12px; color:#888;">⚖️ Faithfulness Audit</div>
                    <div style="font-size:24px; font-weight:bold; color:#d600ff;">{faith_score}/10</div>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                st.markdown(f"""
                <div class="metric-card" style="background: {bg_color}; border-color: {bg_color};">
                    <div style="font-size:12px; color:#aaa;">🛡️ TRUST-SCORE</div>
                    <div class="{color_class}" style="font-size:28px; font-weight:bold;">{trust_score:.1f}</div>
                    <div class="{color_class}" style="font-size:12px; font-weight:bold;">{label}</div>
                </div>
                """, unsafe_allow_html=True)

            # --- Source Display ---
            with st.expander("📚 Click to View Reference Sources"):
                for idx, node in enumerate(nodes):
                    st.markdown(f"""
                    **[{idx + 1}] {node.metadata.get('file_name', 'Unknown Source')}** <span class="source-tag">{node.metadata.get('category', 'General')}</span> 
                    <span style="color:#888; font-size:0.8em;">Similarity: {node.score:.4f}</span>
                    """, unsafe_allow_html=True)
                    st.caption(node.text[:200] + "...")
                    st.divider()

            # Record History
            st.session_state.messages.append({
                "role": "assistant",
                "content": str(response),
                "metrics": {
                    "trust": f"{trust_score:.1f}",
                    "delta": label,
                    "retrieval": f"{calibrated_ret:.1f}",
                    "faithfulness": f"{faith_score}"
                }
            })