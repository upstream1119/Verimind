import os
import fitz  # PyMuPDF
from pathlib import Path

# --- 配置 ---
PDF_DIR = "./knowledge_pdfs"
MD_DIR = "./knowledge_mds"

def convert_all():
    print("🚀 启动全量清洗程序 (保持物理分类树结构)...")

    # 1. 扫描所有 PDF，包括子文件夹
    pdf_files = list(Path(PDF_DIR).rglob("*.pdf"))
    print(f"📄 共发现 {len(pdf_files)} 篇 PDF 文档，准备按目录流转...")

    success_count = 0

    for pdf_path in pdf_files:
        try:
            # 构造对应的 markdown 输出路径 (完美映射子文件夹，例如 01_Agents)
            relative_path = pdf_path.relative_to(PDF_DIR)
            md_path = Path(MD_DIR) / relative_path.with_suffix(".md")

            # 确保 MD 对应的子文件夹存在
            md_path.parent.mkdir(parents=True, exist_ok=True)

            # 获取当前分类名称用于打印日志
            category_name = md_path.parent.name

            # --- 核心：强力提取 ---
            doc = fitz.open(pdf_path)
            full_text = []
            for page in doc:
                text = page.get_text("text")
                full_text.append(text)

            final_content = "\n\n".join(full_text)
            header = f"# {pdf_path.stem}\n\n> Auto-converted plain text. Category: {category_name}\n\n"

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(header + final_content)

            print(f"✅ [OK] {category_name} -> {pdf_path.name}")
            success_count += 1

        except Exception as e:
            print(f"❌ [Fail] {pdf_path.name}: {e}")

    print(f"\n🎉 完美！成功将 {success_count} 篇文档转换并同步分类至 {MD_DIR} 目录。")

if __name__ == "__main__":
    convert_all()