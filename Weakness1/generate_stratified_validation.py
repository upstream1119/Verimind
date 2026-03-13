"""
generate_stratified_validation.py
==================================
生成 W1 "150题分层交叉验证集"

分层定义（基于 VeriMind_Full 的输出状态）：
  Trusted  (T): VeriMind 给出了实质性回答 AND trust_score >= 7.0  → 64条（全取）
  Warning  (W): VeriMind 输出 "Information not found" AND trust_score < 6.0  → 12条（全取）
  Rejected (R): VeriMind 输出 "Information not found" AND trust_score >= 7.0  → 74条（从224条中抽取）

总计：64T + 12W + 74R = 150题

输出文件（写入 /home/claude/）：
  1. stratified_val_set.json        — 结构化验证集（含bench题目 + 各系统回答）
  2. blind_audit_sheet.csv          — 人工盲审表（隐去系统名称，用于W4 Kappa计算）
  3. stratified_val_metadata.json   — 抽样统计摘要（供论文陈述使用）
"""

import csv, json, random
from collections import defaultdict, Counter
import os

# ─── 配置 ────────────────────────────────────────────────────────────────────
SEED = 42
CSV_PATH = "/mnt/user-data/uploads/experiment_results_sci_v3.csv"
JSON_PATH = "/mnt/user-data/uploads/verimind_bench_v3.json"
OUT_DIR   = "/home/claude"

# 验证集中保留的系统（对应论文 W1 要求：覆盖多系统横评）
SYSTEMS   = ["VeriMind_Full", "Vanilla_RAG", "Self-RAG", "HyDE"]

# 分层阈值
TRUSTED_TS_MIN  = 7.0   # 有实质回答 + trust_score >= 此值 → Trusted
WARNING_TS_MAX  = 6.0   # not-found + trust_score < 此值  → Warning
REJECTED_TS_MIN = 7.0   # not-found + trust_score >= 此值 → Rejected

# 目标抽取数量
N_TRUSTED  = None   # None = 全取
N_WARNING  = None   # None = 全取
N_REJECTED = 74
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)

# 1. 读取数据
print("▶ 读取 CSV 和 JSON 文件...")
with open(CSV_PATH, encoding='utf-8-sig') as f:
    all_rows = list(csv.DictReader(f))

with open(JSON_PATH, encoding='utf-8') as f:
    bench = json.load(f)

bench_map = {str(i): bench[i] for i in range(len(bench))}

# 按系统 + question_id 建立索引
by_strat = defaultdict(dict)
for r in all_rows:
    by_strat[r['strategy']][r['question_id']] = r

vm = by_strat['VeriMind_Full']

# 2. 分层
print("▶ 执行分层...")
trusted_ids, warning_ids, rejected_ids = [], [], []

for qid, r in vm.items():
    ans = r['generated_answer'].lower()
    ts  = float(r['trust_score'])
    is_not_found = 'information not found' in ans

    if not is_not_found and ts >= TRUSTED_TS_MIN:
        trusted_ids.append(qid)
    elif is_not_found and ts < WARNING_TS_MAX:
        warning_ids.append(qid)
    elif is_not_found and ts >= REJECTED_TS_MIN:
        rejected_ids.append(qid)

print(f"  可用池: Trusted={len(trusted_ids)}, Warning={len(warning_ids)}, Rejected={len(rejected_ids)}")

# 3. 抽样
sampled_t = sorted(trusted_ids)   if N_TRUSTED  is None else random.sample(trusted_ids,  N_TRUSTED)
sampled_w = sorted(warning_ids)   if N_WARNING  is None else random.sample(warning_ids,  N_WARNING)
sampled_r = random.sample(rejected_ids, N_REJECTED)

all_selected = sampled_t + sampled_w + sampled_r
assert len(all_selected) == 150, f"抽样总数应为150，实为{len(all_selected)}"
print(f"  抽样结果: Trusted={len(sampled_t)}, Warning={len(sampled_w)}, Rejected={len(sampled_r)}, 总计={len(all_selected)}")

# 4. 为每题打上分层标签
label_map = {}
for qid in sampled_t: label_map[qid] = 'TRUSTED'
for qid in sampled_w: label_map[qid] = 'WARNING'
for qid in sampled_r: label_map[qid] = 'REJECTED'

# 5. 构建结构化验证集
print("▶ 构建结构化验证集...")
val_set = []
for qid in sorted(all_selected, key=lambda x: int(x)):
    b = bench_map.get(qid, {})
    entry = {
        "question_id"  : qid,
        "stratum"      : label_map[qid],
        "category"     : b.get('category', 'N/A'),
        "difficulty"   : b.get('difficulty', 'N/A'),
        "question"     : b.get('question', ''),
        "context"      : b.get('context', ''),
        "gold_answer"  : b.get('gold_answer', ''),
        "source_file"  : b.get('source_file', ''),
        "systems"      : {}
    }
    for sys_name in SYSTEMS:
        r = by_strat[sys_name].get(qid, {})
        entry["systems"][sys_name] = {
            "generated_answer"  : r.get('generated_answer', ''),
            "trust_score"       : r.get('trust_score', ''),
            "faith_score"       : r.get('faith_score', ''),
            "hallucination_rate": r.get('hallucination_rate', ''),
            "retrieval_hit"     : r.get('retrieval_hit', ''),
        }
    val_set.append(entry)

out_json = os.path.join(OUT_DIR, "stratified_val_set.json")
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(val_set, f, ensure_ascii=False, indent=2)
print(f"  ✅ 已写入: {out_json}")

# 6. 生成盲审 CSV（W4 人工标注用）
# 每条记录 = 1道题 × 1个系统，共 150×4=600条（或只保留VeriMind=150条）
# 按照W1需求，这里生成 VeriMind 的 150条盲审表（隐去系统名）
print("▶ 生成人工盲审表 (blind_audit_sheet.csv)...")
blind_rows = []
sample_id  = 1
for entry in val_set:
    for sys_name in SYSTEMS:
        s = entry["systems"][sys_name]
        if not s["generated_answer"]:
            continue
        blind_rows.append({
            "sample_id"         : sample_id,
            "question_id"       : entry["question_id"],
            "stratum"           : entry["stratum"],          # 保留用于分析
            "category"          : entry["category"],
            "difficulty"        : entry["difficulty"],
            "question"          : entry["question"],
            "context"           : entry["context"][:500] + ("..." if len(entry["context"]) > 500 else ""),
            "gold_answer"       : entry["gold_answer"],
            "generated_answer"  : s["generated_answer"],
            "system_id"         : f"System_{['A','B','C','D'][SYSTEMS.index(sys_name)]}",  # 盲化
            # ── 以下留给人工填写 ──
            "human_label"       : "",   # 0=幻觉/不忠实, 1=无幻觉/忠实
            "human_notes"       : "",
        })
        sample_id += 1

out_csv = os.path.join(OUT_DIR, "blind_audit_sheet.csv")
fieldnames = list(blind_rows[0].keys())
with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(blind_rows)
print(f"  ✅ 已写入: {out_csv}  (共 {len(blind_rows)} 条待标注记录)")

# 7. 生成统计摘要 JSON（供论文陈述使用）
print("▶ 生成统计摘要...")

def dist(ids, key):
    return dict(Counter(bench_map[qid].get(key, 'N/A') for qid in ids if qid in bench_map))

metadata = {
    "seed": SEED,
    "total_questions": len(all_selected),
    "stratum_counts": {
        "TRUSTED" : len(sampled_t),
        "WARNING" : len(sampled_w),
        "REJECTED": len(sampled_r),
    },
    "stratum_definitions": {
        "TRUSTED" : f"VeriMind gave substantive answer AND trust_score >= {TRUSTED_TS_MIN}",
        "WARNING" : f"VeriMind output 'Information not found' AND trust_score < {WARNING_TS_MAX}",
        "REJECTED": f"VeriMind output 'Information not found' AND trust_score >= {REJECTED_TS_MIN}",
    },
    "note_on_warning_scarcity": (
        "Only 12 WARNING samples exist in the full 300-question test set, "
        "reflecting VeriMind's deterministic gating: the system rarely operates "
        "in a low-confidence regime. This is consistent with the bimodal trust_score "
        "distribution documented in the paper."
    ),
    "category_distribution": {
        "TRUSTED" : dist(sampled_t, 'category'),
        "WARNING" : dist(sampled_w, 'category'),
        "REJECTED": dist(sampled_r, 'category'),
        "TOTAL"   : dist(all_selected, 'category'),
    },
    "difficulty_distribution": {
        "TRUSTED" : dist(sampled_t, 'difficulty'),
        "WARNING" : dist(sampled_w, 'difficulty'),
        "REJECTED": dist(sampled_r, 'difficulty'),
        "TOTAL"   : dist(all_selected, 'difficulty'),
    },
    "systems_covered": SYSTEMS,
    "output_files": {
        "structured_val_set" : "stratified_val_set.json",
        "blind_audit_sheet"  : "blind_audit_sheet.csv",
        "metadata"           : "stratified_val_metadata.json",
    }
}

out_meta = os.path.join(OUT_DIR, "stratified_val_metadata.json")
with open(out_meta, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
print(f"  ✅ 已写入: {out_meta}")

# 8. 打印摘要报告
print("\n" + "="*60)
print("📊 抽样摘要（可直接用于论文 Section 5.2 陈述）")
print("="*60)
print(f"  总题数   : {len(all_selected)}")
print(f"  TRUSTED  : {len(sampled_t):3d}  ({len(sampled_t)/len(all_selected)*100:.1f}%)")
print(f"  WARNING  : {len(sampled_w):3d}  ({len(sampled_w)/len(all_selected)*100:.1f}%)")
print(f"  REJECTED : {len(sampled_r):3d}  ({len(sampled_r)/len(all_selected)*100:.1f}%)")
print(f"\n  Category 总分布: {dist(all_selected, 'category')}")
print(f"  Difficulty 总分布: {dict(Counter(bench_map[q].get('difficulty') for q in all_selected if q in bench_map))}")
print(f"\n  盲审表行数（多系统）: {len(blind_rows)}")
print(f"  VeriMind独立行数 : {len([r for r in blind_rows if r['system_id']=='System_A'])}")
print("="*60)
print("✅ 全部完成！")
