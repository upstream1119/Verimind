"""
step1_prefill_rejected.py
用法：python step1_prefill_rejected.py
输入：blind_audit_sheet.csv
输出：blind_audit_sheet_to_annotate.csv（REJECTED 行自动预填 0，其余待人工标注）
"""
import csv

INPUT  = "../csv/blind_audit_sheet.csv"
OUTPUT = "../csv/blind_audit_sheet_to_annotate.csv"

SYSTEM_MAP = {
    "System_A": "VeriMind_Full",
    "System_B": "Vanilla_RAG",
    "System_C": "Self_RAG",
    "System_D": "HyDE",
}

with open(INPUT, encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

auto_filled = 0
for row in rows:
    # 解盲 system_id
    row["system_id"] = SYSTEM_MAP.get(row["system_id"], row["system_id"])
    # REJECTED 行预填 0
    if row["stratum"] == "REJECTED":
        row["human_label"] = "0"
        row["human_notes"] = "auto-prefilled: honest rejection = no hallucination"
        auto_filled += 1

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

need_manual = sum(1 for r in rows if r["human_label"] == "")
print(f"✅ System 映射已解盲：A=VeriMind_Full, B=Vanilla_RAG, C=Self_RAG, D=HyDE")
print(f"✅ 自动预填 REJECTED：{auto_filled} 条 → human_label = 0")
print(f"📝 剩余需人工标注：{need_manual} 条（TRUSTED + WARNING）")
print(f"📄 输出文件：{OUTPUT}")
print()
print("标注说明：")
print("  0 = 无幻觉（答案与 context 一致，或诚实拒绝）")
print("  1 = 有幻觉（引入了 context 之外的事实/数字/实体）")