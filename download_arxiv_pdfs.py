import os
import urllib.request
import time
import re
import xml.etree.ElementTree as ET

# 根目录设定
BASE_DIR = "./knowledge_pdfs"

# 🌟 核心：使用字典将论文严格分类映射
CATEGORIZED_PAPERS = {
    "01_Agents": [
        "2309.17288",  # AutoAgents
        "2309.07870",  # Agents
        "2308.03688",  # AgentBench
        "2308.10848",  # AgentVerse
        "2307.16789",  # ToolLLM
        "2310.10634",  # OpenAgents
        "2405.15793",  # SWE-agent
        "2305.04091",  # Plan-and-Solve
        "2112.09332",  # WebGPT
        "2305.15334",  # Gorilla
        "2311.14658",  # TaskWeaver
        "2310.12823",  # AgentTuning
        "2305.14325",  # MultiAgent Debate
        "2307.07924",  # ChatDev
    ],
    "02_RAG_Optimization": [
        "2112.04426",  # RETRO
        "2007.01282",  # FiD
        "2208.03299",  # Atlas
        "2004.04906",  # DPR
        "2004.12832",  # ColBERT
        "2112.11440",  # ColBERTv2
        "2007.00808",  # ANCE
        "2301.12652",  # RePlug
        "2205.12006",  # RetroMAE
        "2006.11673",  # GAR
        "2408.05141",  # HybridRAG
        "2406.15319",  # LongRAG
        "2312.10997",  # Modular RAG (Survey)
        "2402.03367",  # RAG-Fusion
        "2305.15294",  # Iterative RAG
        "2309.01431",  # RAG Benchmark (RGB)
    ],
    "03_Reasoning_Prompting": [
        "2205.10625",  # Least-to-Most
        "2211.12588",  # Program-of-Thoughts
        "2211.10435",  # PAL
        "2303.17651",  # Self-Refine
        "2301.13379",  # Faithful-CoT
        "2210.02406",  # Decomposition Prompting
        "2312.07910",  # PromptBench
        "2305.14325",  # Debate Prompting
        "2402.03620",  # Self-Discover
        "2305.14992",  # Reasoning via Planning
    ],
    "04_Evaluation_Safety": [
        "2109.07958",  # TruthfulQA
        "2305.11747",  # HaluEval
        "1803.05355",  # FEVER
        "1809.09600",  # HotpotQA
        "1903.04559",  # NaturalQuestions
        "2308.14508",  # LongBench
        "2211.09110",  # HELM
        "2206.04615",  # BIG-Bench
        "2306.05685",  # MT-Bench
        "2406.11939",  # Arena-Hard
        "2308.07201",  # ChatEval
        "1910.12840",  # FactCC
        "2004.04228",  # QAGS
        "2303.16634",  # G-Eval
    ],
    "05_Foundations": [
        "2005.14165",  # GPT-3
        "1910.10683",  # T5
        "1907.11692",  # RoBERTa
        "1909.11942",  # ALBERT
        "2010.11929",  # Vision Transformer
        "1412.6980",  # Adam
        "1711.05101",  # AdamW
        "2106.09685",  # LoRA
        "2305.14314",  # QLoRA
        "2205.14135",  # FlashAttention
        "1701.06538",  # Mixture-of-Experts
        "2101.03961",  # Switch Transformer
    ]
}


def get_paper_title(arxiv_id):
    """调用 arXiv API 获取官方论文标题，并清洗为合法文件名"""
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            # arXiv API 的命名空间
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entry = root.find('atom:entry', ns)

            if entry is not None:
                title = entry.find('atom:title', ns).text
                # 1. 移除标题中可能附带的换行符，并将多余空格替换为单空格
                title = re.sub(r'\s+', ' ', title.replace('\n', ' ')).strip()
                # 2. 核心：将 Windows/Mac 文件系统不支持的特殊符号全部替换为下划线 '_'
                # 包含：冒号、问号、星号、双引号、尖括号、斜杠、反斜杠、竖线
                safe_title = re.sub(r'[\\/*?:"<>|]', '_', title)
                return safe_title
    except Exception as e:
        print(f"     ⚠️ 无法获取元数据 ({e})，降级使用 ID 命名")

    return arxiv_id  # 如果网络原因获取失败，则保底使用数字 ID


def download_arxiv_papers():
    print("🚀 开始按物理分类批量下载学术论文 (启用自动命名机制)...")

    total_papers = sum(len(ids) for ids in CATEGORIZED_PAPERS.values())
    downloaded_count = 0

    for category, arxiv_ids in CATEGORIZED_PAPERS.items():
        if not arxiv_ids:
            continue

        category_dir = os.path.join(BASE_DIR, category)
        os.makedirs(category_dir, exist_ok=True)

        print(f"\n📂 正在处理分类目录: [{category}] (共 {len(arxiv_ids)} 篇)")

        for paper_id in arxiv_ids:
            downloaded_count += 1

            # 先去查询真实论文名字
            print(f"  [{downloaded_count}/{total_papers}] 🔍 正在获取 {paper_id} 的元数据...")
            paper_title = get_paper_title(paper_id)

            pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
            # 拼接带真实名字的文件路径
            file_path = os.path.join(category_dir, f"{paper_title}.pdf")

            if os.path.exists(file_path):
                print(f"     ⏩ 已存在，跳过: {paper_title}.pdf")
                continue

            print(f"     ⬇️ 下载中: {paper_title}.pdf ...")
            try:
                # 伪装请求头防拦截
                req = urllib.request.Request(pdf_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
                    out_file.write(response.read())
                print(f"     ✅ 成功: 已保存")

                # 稍微增加一点延时，防止连续调用 API 和下载 PDF 被 arXiv 封禁 IP
                time.sleep(3)
            except Exception as e:
                print(f"     ❌ 失败 {paper_id}: {e}")


if __name__ == "__main__":
    download_arxiv_papers()