"""Build evidence-linked final reports and machine-readable summary from offline outputs."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pct(x):
    return f"{100 * x:.2f}%"


def main():
    router = json.loads((OUT / "router_regret_summary.json").read_text(encoding="utf-8"))
    fusion = json.loads((OUT / "fusion_summary.json").read_text(encoding="utf-8"))
    scifact = json.loads((OUT / "scifact_label_consistency_summary.json").read_text(encoding="utf-8"))
    route_rows = csv_rows(OUT / "router_regret_per_question.csv")
    fusion_rows = csv_rows(OUT / "fusion_per_question.csv")
    label_rows = csv_rows(OUT / "scifact_label_trace.csv")
    tie_rows = csv_rows(OUT / "fusion_tie_sensitivity.csv")
    cheap = csv_rows(ROOT / "revision/artifacts/1_17_qasper_cheap_gate/cheap_gate_ranking_comparison.csv")
    cascade = csv_rows(ROOT / "revision/artifacts/1_18_qasper_cost_aware_cascade/qasper_cost_aware_summary.csv")
    cheap_by_name = {r["score"]: r for r in cheap}
    frozen = {r["policy"]: r for r in cascade if r["risk_budget"] == "0.55"}
    assert len(route_rows) == 237 and len(fusion_rows) == 300 and len(label_rows) == 450
    assert int(cheap_by_name["cheap_score"]["candidate_answers_n"]) == 96
    assert int(frozen["Cost-Aware Cascade"]["accepted_n"]) == 23
    assert int(frozen["Cost-Aware Cascade"]["verifier_calls"]) == 13
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "execution": "Offline only; no LLM/API calls; no answer or retrieval generation",
        "commands": ["python -B run_router.py", "python -B run_fusion.py", "python -B run_tie_sensitivity.py", "python -B run_scifact.py", "python -B run_report.py"],
        "task1_router": router, "task2_fusion": fusion, "task3_scifact": scifact,
        "heldout_resume_evidence": {
            "dataset": "QASPER", "split": "disjoint 150-question held-out", "candidate_n": 96,
            "cheap_score_spearman_answer_f1": float(cheap_by_name["cheap_score"]["spearman_with_answer_f1"]),
            "cheap_score_observed_aurc": float(cheap_by_name["cheap_score"]["aurc_observed"]),
            "legacy_trust_observed_aurc": float(cheap_by_name["legacy_trust_score"]["aurc_observed"]),
            "cascade_accepted_n": int(frozen["Cost-Aware Cascade"]["accepted_n"]),
            "cascade_coverage": float(frozen["Cost-Aware Cascade"]["coverage"]),
            "cascade_mean_answer_f1": float(frozen["Cost-Aware Cascade"]["mean_answer_f1"]),
            "cascade_verifier_calls": int(frozen["Cost-Aware Cascade"]["verifier_calls"]),
            "cascade_estimated_verifier_input_tokens": int(float(frozen["Cost-Aware Cascade"]["estimated_verifier_input_tokens"])),
            "always_verify_estimated_verifier_input_tokens": int(float(frozen["Always Verify"]["estimated_verifier_input_tokens"])),
            "estimated_token_savings": float(frozen["Cost-Aware Cascade"]["token_savings_vs_always_verify"]),
            "source_note": "Token counts are character/4 offline estimates, not measured billing or latency."
        },
        "validation": {
            "router_input_unique": len({r["question_id"] for r in route_rows}) == 237,
            "fusion_input_unique": len({r["question_id"] for r in fusion_rows}) == 300,
            "scifact_candidate_unique": len({r["sample_id"] for r in label_rows}) == 450,
            "fusion_at5_exactly_fixed512_all_evaluable": all(r["fusion_recall_at5"] == r["recall_512_at5"] for r in fusion_rows if r["recall_512_at5"]),
            "all_six_rrf_tie_orders_same_aggregate": all(
                max(float(r["mean_fusion_recall"]) for r in tie_rows if r["subset"] == subset and int(r["k"]) == k) -
                min(float(r["mean_fusion_recall"]) for r in tie_rows if r["subset"] == subset and int(r["k"]) == k) < 1e-12
                for subset in ("all", "text_retrievable") for k in (3, 5)),
            "raw_evaluator_parser_mismatch_n": scifact["raw_evaluator_parser_mismatches"],
            "context_alignment_error_n": scifact["id_or_context_alignment_errors"],
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in (
                ROOT / "revision/artifacts/oracle_granularity_20260927/retrieval.jsonl",
                ROOT / "revision/artifacts/supplement_20260927_full/items.json",
                ROOT / "revision/artifacts/supplement_20260927_full/retrieval.jsonl",
                ROOT / "revision/artifacts/supplement_20260927_full/answers.jsonl",
                ROOT / "revision/artifacts/1_17_qasper_cheap_gate/cheap_gate_ranking_comparison.csv",
                ROOT / "revision/artifacts/1_18_qasper_cost_aware_cascade/qasper_cost_aware_summary.csv")
        }
    }
    assert all(summary["validation"][key] for key in (
        "router_input_unique", "fusion_input_unique", "scifact_candidate_unique",
        "fusion_at5_exactly_fixed512_all_evaluable", "all_six_rrf_tie_orders_same_aggregate"))
    assert summary["validation"]["raw_evaluator_parser_mismatch_n"] == 0
    assert summary["validation"]["context_alignment_error_n"] == 0
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    a = router["all"]
    t = router["text_retrievable"]
    f = {(r["subset"], r["k"]): r for r in fusion["rows"]}
    audit = f"""# SciFact supported 标签来源与一致性审计

## 必须修复

- 当前 `supported` 不宜作为真实证据支持的 ground truth：450 条候选中，**{scifact['flag_counts_overlapping']['BOOLEAN_REASON_CONFLICT']} 条**原始 evaluator 输出的 `supported=true` 与其文字理由中的明确“证据不足”信号相冲突；**{scifact['supported_true_evidence_sufficient_false_n']} 条**同时出现 `supported=true`、`evidence_sufficient=false`。前者包含在后者之中。关键词仅用于定位待复核记录，不生成修正标签。
- 以 Judge 分数 ≥5 作为*比较代理阈值*，与 evaluator `supported` 布尔值不一致 **{scifact['flag_counts_overlapping']['JUDGE_EVALUATOR_CONFLICT']} 条**；这些只是两个自动判断器的分歧，不是经人工确认的错判数。
- Judge=0 共 **{scifact['judge_score_zero_n']} 条**，其中 **{scifact['judge_score_zero_supported_true_n']} 条**被 evaluator 标为 supported；TrustScore [0.30,0.40) 子集中 Judge=0 为 **{scifact['trust_030_040_judge_zero_n']} 条**、其中 supported=true 为 **{scifact['trust_030_040_judge_zero_supported_true_n']} 条**。这两组分母不同，不要混写。

## 应修复

- 对冲突记录安排独立盲审并建立明确的 supported 定义；此前 30 条内容审计是模型执行的初步判读，不能外推为 450 条的人工真值。
- 将 `supported`、`evidence_sufficient`、`correct` 三个字段分别审核，避免用 gold verdict 的正确性替代检索证据支持性。

## 可选

- 对 Judge 0–10 分数做带有独立标注集的校准；不要用这 450 条自动标签直接训练后再评价。

## 证据缺口与代码溯源

1. 当前运行使用 `paired-v1`：GLM 生成、DeepSeek 给出 0–10 的 `faith_score`、Qwen 独立返回包含 `supported` 的 JSON。生成和评分写入同一条 `answers.jsonl` 记录。源头在 `revision/experiments/supplement_20260927.py` 的 `generate_and_evaluate`；`parse_evaluation` 只解析 JSON 并检查布尔类型，不从文字理由映射 `supported`。
2. `offline_gate_diagnostics.read_rows` 直接读取 `answer["evaluation"]["supported"]`；`aggregate`/`case_metrics.csv` 也直接用该字段。未发现当前路径存在 label merge、旧标签覆盖或理由关键词映射。
3. 450/450 原始 evaluator JSON 与保存解析对象一致；450/450 原始 judge 整数与保存分数一致；450/450 UID、dataset 和检索上下文 SHA-256 对齐。全部记录来自同一 `answers.jsonl`，时间戳日期均为 2026-09-27。**未发现确定性的 parser、ID 对齐或版本混用 bug**；这不能排除模型自身的输出矛盾或上游未记录的差异。
4. 样本来自 SciFact train（{scifact['source_splits'].get('train',0)} 条）和 dev（{scifact['source_splits'].get('dev',0)} 条）的候选回答，不能称为独立 held-out test。当前产物中 `ID_ALIGNMENT_ERROR`、`PARSER_ERROR` 检出数为 0；未发现版本混用证据，但不能证明上游不存在未记录的版本差异。
5. `BOOLEAN_REASON_CONFLICT` 的文字模式是保守检索信号；`PARTIAL_SUPPORT_AMBIGUITY_CANDIDATE` 也只是待审标记。CSV 的 `derived_supported_boolean` 是从**同一原始 JSON 重新解析**，不是 corrected label；`corrected_supported_label` 留空。

## 建议下一步与指标边界

- 可继续作为开发集描述统计使用：检索 Recall@K、与 gold verdict 确定性比较的 label accuracy、覆盖率/拒答率、原始分数分布；写清数据切分和指标定义。
- 暂停把 SciFact 的 `supported`/`unsupported`、Faithfulness、Unsupported AURC 和以它们构成的 safe AUROC 写成论文或简历的真实事实性/幻觉率结论，直至独立复核或重新标注。
- 逐例证据见 `scifact_label_trace.csv`，冲突子集见 `scifact_conflicts.csv`。冲突类型允许重叠，不能相加当作独立问题数。
"""
    (OUT / "scifact_label_audit_report.md").write_text(audit, encoding="utf-8")
    def row(subset, k):
        r = f[(subset, k)]
        return f"| {subset} | @{k} | {r['128']:.3f} | {r['512']:.3f} | {r['1024']:.3f} | {r['router']:.3f} | {r['fusion']:.3f} | {r['fusion_exact_node']:.3f} | {r['oracle']:.3f} |"
    report = f"""# VeriMind 最终离线诊断与简历可引用边界

本轮只读取既有 QASPER/SciFact 实验产物及本地索引；没有调用 LLM/API、生成答案或重新检索。新结果均保存在本目录。QASPER 的 300 题属于开发实验；另列的 150 题来自独立 held-out 产物，两组数据不可混称。

## 1. Router

**核心数字。** QASPER 全量 300 题中，237 题有可评估 gold evidence；文本可检索子集为 251 题，其中 188 题可评估。全量 mean Router Regret 为 **{a['mean_regret_at3']:.4f} (@3)** / **{a['mean_regret_at5']:.4f} (@5)**，median 均为 0。@5 有 **{a['regret_positive_n_at5']}/{a['evaluable_n']} ({pct(a['regret_positive_rate_at5'])})** 题出现正 regret；Router 达到最优为 **{a['router_optimal_n_at5']}/{a['evaluable_n']} ({pct(a['router_optimal_rate_at5'])})**，但三粒度完全相同就有 **{a['all_three_equal_n_at5']}** 题，Router *唯一最佳且选对*为 **{a['router_unique_best_n_at5']}**。@3 相应为最优 {a['router_optimal_n_at3']}/{a['evaluable_n']}、正 regret {a['regret_positive_n_at3']} 题、唯一最佳且选对 {a['router_unique_best_n_at3']} 题。

文本子集 @5 mean regret **{t['mean_regret_at5']:.4f}**，正 regret **{t['regret_positive_n_at5']}/{t['evaluable_n']}**，Router 最优 **{t['router_optimal_n_at5']}/{t['evaluable_n']}**，三粒度全相同 **{t['all_three_equal_n_at5']}**。全量 @5 的唯一最佳粒度是 128（{a['unique_best_counts_at5']['128']} 题）；512、1024 独占最佳均为 0。最优粒度出现并列时每个粒度都计入对应的“包含并列最优”次数，不强行分配。

**问题分布。** Oracle 相对固定 512 的 @5 收益只发生在 **22/237** 题；其中收益最大的 10 题贡献 **61.73%** 总增益。Router 正 regret 的 26 题中，前 10 题贡献 **{pct(a['top_10_regret_share_at5'])}** 总 regret。当前 `route_query` 是关键词规则 Router，**不是 LLM Router**。没有可靠的独立 factual/reasoning 问题类型标签；现有 DETAIL/CONCEPT/CONTEXT 是 Router 规则类别，不能用来证明题目本体类型。按规则类别，@5 平均 regret：DETAIL 17 题为 **0.1373**，CONCEPT 210 题为 **0.0771**，CONTEXT 10 题为 **0**；CONCEPT 因样本多贡献主要绝对 regret。

**可安全下的结论：** 在这批开发问题里当前 Router 的检索 Recall 略低于固定 512，Oracle 上限的改善集中于少数可获益题。**不可下的结论：** “Router 准确率 89%”“动态多粒度提高召回”“DETAIL 问题天然最难”。当前 Router 不宜作为简历或论文的核心性能贡献；可以作为实现与消融对象。

## 2. Multi-granularity Retrieval

RRF 使用 `k_rrf=60`。主结果只将同一论文中**重建后的原始 parent 段落文本完全相同**的 chunk 合并，以论文 ID 和完整段落哈希作为键；同一路由重复 chunk 只取其最高排名。没有语义相似去重，也没有使用 gold evidence 排序。对照列“仅精确 chunk ID”不做跨粒度合并。并列分数按预定粒度顺序 128→512→1024 处理；另试遍六种固定粒度顺序，两个子集的汇总 Recall@3/5 均未变化，详见 `fusion_tie_sensitivity.csv`。

| 范围 | Recall | 128 | Fixed 512 | 1024 | Router | RRF 段落去重 | RRF 仅 chunk ID | Oracle |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{row('all',3)}
{row('all',5)}
{row('text_retrievable',3)}
{row('text_retrievable',5)}

全量可评估题上，段落去重 RRF @3 为 **{f[('all',3)]['fusion']:.3f}**，低于固定 512 的 **{f[('all',3)]['512']:.3f}** 和 Router 的 **{f[('all',3)]['router']:.3f}**；@5 为 **{f[('all',5)]['fusion']:.3f}**，与固定 512 **逐题相同**，比 Router **{f[('all',5)]['router']:.3f}** 略高，距 Oracle **{f[('all',5)]['oracle']:.3f}** 仍差 **{f[('all',5)]['oracle']-f[('all',5)]['fusion']:.3f}**。文本子集 @5 同样与固定 512 相同（**{f[('text_retrievable',5)]['fusion']:.3f}**），距 Oracle **{f[('text_retrievable',5)]['oracle']-f[('text_retrievable',5)]['fusion']:.3f}**。仅 chunk ID 去重时全量 @3/@5 降为 **{f[('all',3)]['fusion_exact_node']:.3f}/{f[('all',5)]['fusion_exact_node']:.3f}**，说明跨粒度重复证据处理对融合结果影响很大。

**最终判断：** 三粒度 Oracle 显示问题特异性的检索空间，但当前**关键词规则 Router**与这个无模型 RRF 都未超过固定 512。Fusion 未接近 Oracle，不能据此推断未来学习式/LLM Router 一定无用；只能说**当前规则 Router 没有证明必要性或性能收益**。Oracle 的收益主要来自少数在 128 下更容易召回 gold 的题，后续若研究路由应针对这些题预先学习或识别，而非事后选优。

固定 512 逐题重算与既有 Oracle/冻结结果完全一致；输入 300 题一致；Fusion 排序不读 gold。`fusion_retrieval.jsonl` 包含逐题两种 RRF 排序与命中信息，`fusion_per_question.csv`、`fusion_summary.csv/json` 给出复算值。

## 3. SciFact Label Reliability

450 条 SciFact+NEI 候选来自 train 249 / dev 201。`supported` 来源是 Qwen 原始 JSON；解析器没有从 reason 生成布尔值。原始 JSON、保存解析对象、judge 原始整数、样本 ID 与检索上下文哈希 **均 450/450 对齐**；**未发现确定性的解析、merge、ID 或版本混用 bug**。但有 **20 条**布尔值与明确“证据不足”的文字理由冲突，**28 条** `supported=true` 且 `evidence_sufficient=false`，以及 **160 条** Judge≥5 的代理判断与 evaluator 布尔值冲突（类型可重叠）。Judge=0 共 **{scifact['judge_score_zero_n']}** 条，其中 **{scifact['judge_score_zero_supported_true_n']}** 条被 evaluator 标为 supported；TrustScore [0.30,0.40) 的 Judge=0 子集为 **{scifact['trust_030_040_judge_zero_n']}** 条，其中 **{scifact['trust_030_040_judge_zero_supported_true_n']}** 条标 supported。主要问题是**自动 evaluator 输出内部不一致与不同判断器的定义/判断分歧**；现有证据不能把冲突全部归因于 Judge，也不能生成 corrected label。

**暂停引用**以 SciFact 自动 `supported` 为真值的 faithfulness、unsupported rate、Unsupported AURC、safe AUROC，以及“真实幻觉率下降”。可继续引用明确标注为开发集描述统计的检索 Recall、gold verdict 的确定性准确率、coverage/拒答比例和原始分数分布。详见 `scifact_label_audit_report.md` 和两份逐例 CSV。

## 4. Resume-safe Metrics

下列数字来自与上述 300 题开发分析**分开的 150 题 QASPER held-out**，已从 `1_17_qasper_cheap_gate/cheap_gate_ranking_comparison.csv` 和 `1_18_qasper_cost_aware_cascade/qasper_cost_aware_summary.csv` 复核：

| 可引用数字 | Dataset / N | Metric 与评价范围 | 必要限制 |
|---|---|---|---|
| Cheap score observed AURC **0.2915**；legacy TrustScore **0.3840** | QASPER held-out 150 题中的 **96 条自然回答** | 按归一化 Answer F1 排序，observed candidate-answer coverage 范围 | 不等于事实支持性或生产环境性能 |
| Cheap score 与 Answer F1 的 Spearman **0.4137** | 同上，96 条候选 | 候选答案质量排序 | 不表示 Answer F1 本身达到 0.4137 |
| Frozen cost-aware cascade 接受 **23/150（15.33%）**，已答平均 Answer F1 **0.5046** | QASPER held-out 150 题 | 预先在 calibration 选定的 ε=0.55 工作点 | 不宣称优于所有 Always Verify 工作点 |
| Verifier 调用 **13/150**；估计输入 tokens **6,811 vs 56,777（约少 88%）** | QASPER held-out 150 题；Always Verify 对 96 条自然回答调用 | 离线字符数/4 的 verifier 输入量估计 | 非真实 API 账单、测得延迟或端到端成本 |

不把本轮开发集 Oracle 写成实际系统性能；不把 17/30 初步审计外推；不把 SciFact 自动 unsupported 率称为真实幻觉率；不把开发题称为 held-out。

## 5. Recommended Resume Wording

- 构建 VeriMind 科研文献问答原型，打通 Chroma 检索、证据追踪、答案生成与可拒答评估链路，并为 QASPER 开发集建立逐题可复算的检索与消融产物。
- 设计不依赖 LLM Judge 输出的轻量答案质量评分；在独立 QASPER held-out 150 题中的 96 条候选回答上，按 Answer F1 排序的 observed AURC 为 **0.2915**，优于旧版 TrustScore 的 **0.3840**。
- 实现校准后冻结阈值的按需验证级联；在 QASPER held-out 150 题上回答 **23 题**、调用 verifier **13 次**，估计 verifier 输入量较对全部 96 条自然回答验证减少约 **88%**（离线 token 估计）。
- 建立逐样本审计与回归校验：对 300 道 QASPER 开发题完成三粒度 Recall/Oracle/RRF 对照，并追踪 450 条 SciFact 候选的原始评分、解析值和上下文哈希，识别自动标签内部不一致。

## 审计优先级与复现

**必须修复：** SciFact 自动支持性标签的独立复核；简历和论文停止把它当真实幻觉率。**应修复：** 若继续研究多粒度，设计独立训练/校准的路由器并与固定 512、无模型融合公平比较。**可选：** 在更大 held-out 上复核检索与工作点。**证据缺口：** 独立 SciFact 人工真值、Router 的 held-out 收益、生产调用成本。**下一检查：** 先复核 28 条内部布尔冲突，再确定标签修订协议。

复现顺序：`python -B run_router.py`、`python -B run_fusion.py`、`python -B run_tie_sensitivity.py`、`python -B run_scifact.py`、`python -B run_report.py`。`run_fusion.py` 仅读本地 Chroma 索引；无模型或网络调用。输入哈希、运行时间和所有汇总见 `summary.json`。
"""
    (OUT / "final_resume_diagnostics_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"report_written": True, "summary_written": True,
                      "validation": summary["validation"]}))


if __name__ == "__main__":
    main()
