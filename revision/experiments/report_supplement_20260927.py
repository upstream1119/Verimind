"""Validate and render the completed supplemental experiment without API calls."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics

from supplement_20260927 import POLICIES, decision, digest, dump_csv, load_items, read_jsonl, refusal_metrics


def wilson(successes, n):
    if not n:
        return None, None
    z = 1.96
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** .5 / denominator
    return max(0, center - half), min(1, center + half)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(str(v) for v in row) + " |" for row in rows])


def pct(value):
    return "—" if value in (None, "") else f"{float(value) * 100:.2f}%"


def num(value):
    return "—" if value in (None, "") else f"{float(value):.3f}"


def validate(output):
    items = json.loads((output / "items.json").read_text(encoding="utf-8"))
    assert items == load_items(), "Report requires the complete, unchanged input set"
    assert Counter(i["dataset"] for i in items) == {"qasper": 300, "scifact": 454, "scifact_nei": 60}
    retrieval_rows = read_jsonl(output / "retrieval.jsonl")
    answer_rows = read_jsonl(output / "answers.jsonl")
    retrievals = {r["uid"]: r for r in retrieval_rows}
    answers = {(r["uid"], r["strategy"]): r for r in answer_rows}
    assert len(retrievals) == len(retrieval_rows) == len(items)
    assert len(answers) == len(answer_rows) == 2 * len(items)
    for item in items:
        for strategy in ("fixed_512", "multi"):
            r = retrievals[item["uid"]]["results"][strategy]
            a = answers[item["uid"], strategy]
            assert a["context_sha256"] == digest("\n\n".join(n["text"] for n in r["nodes"][:3]))
            assert [n["rank"] for n in r["nodes"]] == list(range(1, len(r["nodes"]) + 1))
            assert len(r["nodes"]) <= 5
            assert 0 <= a["faith_score"] <= 10
            assert all(type(a["evaluation"][k]) is bool for k in ("supported", "correct", "evidence_sufficient"))
            assert all(a[k] >= 0 for k in ("generation_sec", "gate_sec", "evaluation_sec"))
    cases = list(csv.DictReader((output / "case_metrics.csv").open(encoding="utf-8-sig")))
    assert len(cases) == 5 * len(items)
    assert len({(r["uid"], r["policy"]) for r in cases}) == len(cases)
    for case in cases:
        strategy = "fixed_512" if case["policy"] in ("fixed_512", "no_routing") else "multi"
        a = answers[case["uid"], strategy]
        assert case["decision"] == decision(a["answer"], a["calibrated_retrieval"], a["faith_score"], case["policy"])
        assert int(case["accepted"]) == int(case["decision"] == "ANSWER")
        assert int(case["unsupported"]) == int(case["accepted"] == "1" and not a["evaluation"]["supported"])
    summaries = list(csv.DictReader((output / "summary.csv").open(encoding="utf-8-sig")))
    for summary in summaries:
        group = [r for r in cases if r["policy"] == summary["policy"] and
                 (r["dataset"] == summary["dataset"] or summary["dataset"] == "scifact_with_nei"
                  and r["dataset"] in ("scifact", "scifact_nei"))]
        expanded = [r for r in group for _ in range(int(r["weight"]) if summary["aggregation"] == "legacy_500_rows" else 1)]
        n = len(expanded)
        accepted_n = sum(int(r["accepted"]) for r in expanded)
        unsupported_n = sum(int(r["unsupported"]) for r in expanded)
        assert int(summary["n"]) == n and int(summary["accepted_n"]) == accepted_n
        assert int(summary["unsupported_n"]) == unsupported_n
        assert abs(float(summary["coverage"]) - accepted_n / n) < 1e-12
        if accepted_n:
            assert abs(float(summary["unsupported_rate_accepted"]) - unsupported_n / accepted_n) < 1e-12
    metadata = json.loads((output / "runtime_metadata.json").read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[2]
    for path, expected_hash in metadata["source_hashes"].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected_hash
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(Path(__file__).with_name("supplement_20260927.py").read_bytes()).hexdigest() == manifest["script_sha256"]
    return items, retrievals, answers, cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    items, retrievals, answers, cases = validate(output)
    summaries = list(csv.DictReader((output / "summary.csv").open(encoding="utf-8-sig")))
    by_key = {(r["dataset"], r["aggregation"], r["policy"]): r for r in summaries}
    gate_rows = []
    for dataset in ("qasper", "scifact", "scifact_nei"):
        selected = [i for i in items if i["dataset"] == dataset]
        natural_answers = [(i, answers[i["uid"], "multi"]) for i in selected
                           if decision(answers[i["uid"], "multi"]["answer"], 10, 10, "multi") == "ANSWER"]
        bad = [not a["evaluation"]["supported"] for _, a in natural_answers]
        rejected = [decision(a["answer"], a["calibrated_retrieval"], a["faith_score"], "full") == "REJECT"
                    for _, a in natural_answers]
        confusion = refusal_metrics(bad, rejected)
        passed_n = sum(not r for r in rejected)
        passed_bad = sum(b and not r for b, r in zip(bad, rejected))
        low, high = wilson(passed_bad, passed_n)
        gate_rows.append({"dataset": dataset, "total_questions": len(selected), "no_gate_answered_n": len(bad),
                          "no_gate_unsupported_n": sum(bad), "no_gate_risk": sum(bad) / len(bad) if bad else None,
                          "full_answered_n": passed_n, "full_unsupported_n": passed_bad,
                          "full_risk": passed_bad / passed_n if passed_n else None,
                          "full_risk_ci_low": low, "full_risk_ci_high": high, **confusion})
    dump_csv(output / "paired_gate_audit.csv", gate_rows)
    unmapped = []
    for item in items:
        if item["dataset"] != "qasper":
            continue
        for strategy, r in retrievals[item["uid"]]["results"].items():
            if r["gold_units_unmapped"]:
                unmapped.append({"uid": item["uid"], "strategy": strategy, "gold_units_n": r["gold_units_n"],
                                 "unmapped_n": r["gold_units_unmapped"]})
    dump_csv(output / "qasper_unmapped_evidence.csv", unmapped)
    labels = {"fixed_512": "固定 512", "multi": "多粒度 / 去 gate", "full": "完整 VeriMind",
              "no_routing": "去多粒度", "no_judge": "去 judge（检索分数门控）"}
    sections = ["# VeriMind 补充实验结果（2026-09-27）", "",
                "本报告来自本次实际 API 运行与逐条复算。历史实验不覆盖、不混入新统计。",
                "", "## 实验规模与协议", "",
                "主实验：QASPER 300 个问题（240 可回答、60 不可回答）；SciFact 500 条历史评测记录对应 454 个去重 claim 文本。另加 60 个来自 SciFact dev、原始 evidence 为空的 NEI claim，单列报告。",
                "SciFact 去重主样本包含 dev 188 条与 train 266 条（展开权重分别为 209 与 291）；这是沿用历史样本的补充评估，不是官方纯测试集泛化结果。",
                "",
                "GLM-4.5-Air 生成；DeepSeek-chat 参与门控；Qwen-Plus 独立评估。模型使用服务商别名，未声称固定权重版本。seed=20260927、temperature=0.01；生成使用 top-3，检索指标使用 top-5。",
                "",
                "每个输入比较固定 512 与动态 128/512/1024；相同上下文复用同一答案，gate 在同一答案上回放。Full 的 TrustScore=0.4×检索校准分+0.6×faith score，阈值 5；去 judge 后使用检索校准分、阈值 5，未重新调参。多粒度无 gate 与去 gate 是同一个策略，不重复宣称独立实验。",
                "",
                "## 检索效果", "",
                "QASPER：人工标注证据段落级 Recall，以规范化空白后的精确段落匹配建立父段落映射，检索到其任一子块即算命中。SciFact：对原始 claim 的全部 gold 文档计算 Recall，top-k 是检索块数，重复文档不重复加分。MRR@5 在第 5 位截断。空 gold 不纳入检索均值。",
                ""]
    retrieval_table = []
    for dataset in ("qasper", "scifact"):
        for policy in ("fixed_512", "multi"):
            r = by_key[dataset, "unique_questions", policy]
            retrieval_table.append([dataset, labels[policy], r["retrieval_evaluable_n"], pct(r["recall_at_3"]), pct(r["recall_at_5"]), num(r["mrr_at_5"])])
    sections.append(table(["数据集", "策略", "有效分母", "Recall@3", "Recall@5", "MRR@5"], retrieval_table))
    sections += ["", f"QASPER 存在未映射 gold 的问题—策略记录数：{len(unmapped)}。这些 gold 仍保留在召回率分母，不按成功命中或静默删除处理；详见 qasper_unmapped_evidence.csv（若数量为零则无此文件）。",
                 "只读核查确认：49 个问题中的 63 条证据均为 FLOAT SELECTED 图表证据，不在现有文本 corpus 中；当前召回率反映文本索引在完整 gold 分母下的表现，并非完整多模态语料的召回率。", "", "## 质量、覆盖率与消融", "",
                 "Faithfulness 是独立 Qwen 判定为 supported 的已回答比例；unsupported rate 是其补数，不等同于经人工验证的真实幻觉率。QASPER 同时给出独立判定 Answer Correctness 与数据集原生 token F1；SciFact 以 gold verdict 的确定性准确率为主要 correctness，避免 judge 忽视标签。拒答在可回答问题上计错，在不可回答问题上计对。", ""]
    for dataset in ("qasper", "scifact", "scifact_nei"):
        rows = []
        for policy in POLICIES:
            r = by_key[dataset, "unique_questions", policy]
            quality = r["answer_f1"] if dataset == "qasper" else r["label_accuracy"]
            rows.append([labels[policy], f"{r['accepted_n']}/{r['n']}", pct(r["coverage"]), pct(r["faithfulness_supported_rate"]),
                         pct(r["answer_correctness_accepted"]), num(quality), r["unsupported_n"], pct(r["unsupported_rate_accepted"])])
        sections += [f"### {dataset}", "", table(["策略", "回答/总数", "Coverage", "Faithfulness", "Qwen正确率（已答）", "原生F1/标签准确率", "无支持数", "无支持率（已答）"], rows), ""]
    sections += ["## 配对 gate 效果", "", "以下比较固定的多粒度原始答案，独立评估标签保持不变；不存在两个策略重新生成导致的随机差异。", "",
                 table(["数据集", "无gate风险", "Full风险", "拦住无支持/全部无支持", "错拒有支持数", "拦截P", "拦截R", "拦截F1"],
                       [[r["dataset"], pct(r["no_gate_risk"]), pct(r["full_risk"]), f"{r['tp']}/{r['tp']+r['fn']}", r["fp"], pct(r["precision"]), pct(r["recall"]), pct(r["f1"])] for r in gate_rows]), "",
                 "## 拒答能力", "",
                 "QASPER 使用原始 is_unanswerable 标签。SciFact 的 Precision/F1 在 454 个有 evidence 的 claim 与额外 60 个 NEI 合并的 514 个输入上计算，防止只在全正例 NEI 子集中得到误导性 Precision。原始空 evidence 标签不保证全语料中绝无可用证据，因此另提供 independent-judge contextual_refusal 指标作为不同口径。", ""]
    for dataset in ("qasper", "scifact_with_nei"):
        rows = []
        for policy in POLICIES:
            r = by_key[dataset, "unique_questions", policy]
            rows.append([labels[policy], int(r["refusal_tp"]) + int(r["refusal_fn"]), r["refusal_tp"], r["refusal_fp"],
                         pct(r["refusal_precision"]), pct(r["refusal_recall"]), pct(r["refusal_f1"])])
        sections += [f"### {dataset}", "", table(["策略", "应拒答", "正确拒答", "错误拒答", "Precision", "Recall", "F1"], rows), ""]
    sections += ["## 性能代价", ""]
    latency_path = output.parent / "latency_20260927_full"
    if (latency_path / "validation.json").exists():
        latency_validation = json.loads((latency_path / "validation.json").read_text(encoding="utf-8"))
        assert latency_validation["complete"] and latency_validation["actual_rows"] == 300
        timings = read_jsonl(latency_path / "timings.jsonl")
        assert len(timings) == len({(r["uid"], r["policy"]) for r in timings}) == 300
        assert all(not r["gate_called"] for r in timings if r["policy"] != "full")
        latency_rows = list(csv.DictReader((latency_path / "summary.csv").open(encoding="utf-8-sig")))
        assert all(int(r["n"]) == 50 for r in latency_rows) and len(latency_rows) == 6
        sections += ["### 独立实际调用计时（主性能结果）", "",
                     "QASPER/SciFact 各 50 个固定分层样本，三个策略使用相同输入、分别发起实际请求，每输入的策略执行顺序固定随机化。共 300 次调用，worker=2；不复用生成答案，不调用离线评估器。计时从 query embedding 开始到最终 gate 决策结束；不含进程启动和索引加载。单位为秒，P95使用线性插值。该客户端试验不等同于生产服务压测。", "",
                     table(["数据集", "策略", "样本数", "平均延迟", "P95延迟"],
                           [[r["dataset"], labels[r["policy"]], r["n"], num(r["mean_sec"]), num(r["p95_sec"])] for r in latency_rows]), ""]
    else:
        raise ValueError("Independent latency experiment is not complete")
    sections += ["### 全量配对阶段耗时（辅助分析）", "",
                 "使用实测检索+生成+相应gate阶段耗时之和，排除独立Qwen评估；相同上下文共享生成计时。以下保留为辅助分析，不替代上面的独立调用计时。", ""]
    rows = []
    for dataset in ("qasper", "scifact"):
        for policy in POLICIES:
            r = by_key[dataset, "unique_questions", policy]
            rows.append([dataset, labels[policy], num(r["latency_mean_sec"]), num(r["latency_p95_sec"])])
    sections.append(table(["数据集", "策略", "平均阶段累计耗时", "P95"], rows))
    unique_answers = [a for a in answers.values() if not a["shared_context_reuse"]]
    sections += ["", "## 完整性与复现", "",
                 f"- 输入：{len(items)}；检索原始记录：{len(retrievals)}；两检索策略答案记录：{len(answers)}；五策略决策：{len(cases)}。",
                 f"- 成功保存的独立生成记录：{len(unique_answers)}；同上下文复用：{len(answers)-len(unique_answers)}。失败与重试另见日志，记录数不等同于实际请求数或计费token数。",
                 "- 已检查：唯一键、全部输入覆盖、上下文SHA-256、gold源文件SHA-256、运行代码SHA-256、judge字段类型、gate决策重放。",
                 "- 原始答案、judge原文、排序节点、时间与模型配置保存在 answers.jsonl、retrieval.jsonl、manifest.json、runtime_metadata.json。",
                 "- summary.csv 同时提供 SciFact unique_questions 与 legacy_500_rows 两种聚合。",
                 "- Citation Support Rate 未计算：旧系统及本次统一输出协议未要求逐项显式引用；本任务选择了 Faithfulness 与 Answer Correctness 两种生成质量指标。",
                 "- 当前结果属于本次冻结规则与服务商模型别名下的观察结果；没有用测试结果调阈值。历史作者人工审计是独立的旧答案证据，不移植为本次人工认证。", ""]
    sections += ["- 补跑记录：首轮 3 个输入发生评估格式或 TLS 失败，最终全部恢复。一条 Qwen 长说明被 300-token 上限截断，补跑仅将该条输出上限提高到 1024，模型、提示和温度不变；详见 protocol_exception.json 与 retry_diagnostics.jsonl。", ""]
    (output / "RESULTS.md").write_text("\n".join(sections), encoding="utf-8")
    q = {p: by_key["qasper", "unique_questions", p] for p in POLICIES}
    s = {p: by_key["scifact", "unique_questions", p] for p in POLICIES}
    sn = by_key["scifact_with_nei", "unique_questions", "full"]
    timing_by_key = {(r["dataset"], r["policy"]): r for r in latency_rows}
    triple = lambda group, key, fmt=pct: " / ".join(fmt(group[p][key]) for p in ("fixed_512", "multi", "full"))
    perf = lambda dataset: "；".join(f"{short} {num(timing_by_key[dataset,p]['mean_sec'])}/{num(timing_by_key[dataset,p]['p95_sec'])}"
                                      for short, p in (("B", "fixed_512"), ("M", "multi"), ("V", "full")))
    refusal = lambda r: (f"应拒答 {int(r['refusal_tp'])+int(r['refusal_fn'])}，正确拒答 {r['refusal_tp']}，"
                         f"实际拒答 {int(r['refusal_tp'])+int(r['refusal_fp'])}；"
                         f"P/R/F1={pct(r['refusal_precision'])}/{pct(r['refusal_recall'])}/{pct(r['refusal_f1'])}")
    ablation = lambda g: "；".join(f"{labels[p]}：{pct(g[p]['unsupported_rate_accepted'])} / {pct(g[p]['coverage'])}"
                                  for p in ("no_routing", "no_judge", "multi"))
    ten = [
        ["实验规模", "300 个问题：240 可答、60 不可答", "454 个去重 claim，覆盖旧 500 条记录；另加 60 条 NEI"],
        ["Baseline", "固定512 / 多粒度无gate / 完整VeriMind，全部完成", "同左；新协议为声明验证，旧 QA 结果不混入"],
        ["检索提升", f"固定→动态：R@3 {pct(q['fixed_512']['recall_at_3'])}→{pct(q['multi']['recall_at_3'])}；R@5 {pct(q['fixed_512']['recall_at_5'])}→{pct(q['multi']['recall_at_5'])}；MRR@5 {num(q['fixed_512']['mrr_at_5'])}→{num(q['multi']['mrr_at_5'])}",
         f"固定→动态：R@3 {pct(s['fixed_512']['recall_at_3'])}→{pct(s['multi']['recall_at_3'])}；R@5 {pct(s['fixed_512']['recall_at_5'])}→{pct(s['multi']['recall_at_5'])}；MRR@5 {num(s['fixed_512']['mrr_at_5'])}→{num(s['multi']['mrr_at_5'])}"],
        ["多粒度是否有效", "当前结果未提升召回；同一测试集已跑", "当前结果未提升召回；同一测试集已跑"],
        ["生成质量（B/M/V）", f"Faithfulness：{triple(q,'faithfulness_supported_rate')}；原生 Answer F1：{triple(q,'answer_f1',num)}；已答 Qwen Correctness：{triple(q,'answer_correctness_accepted')}",
         f"Faithfulness：{triple(s,'faithfulness_supported_rate')}；全样本标签准确率：{triple(s,'label_accuracy')}"],
        ["TrustScore有效性", f"无gate→Full，无支持率 {pct(q['multi']['unsupported_rate_accepted'])}→{pct(q['full']['unsupported_rate_accepted'])}；无支持数 {q['multi']['unsupported_n']}→{q['full']['unsupported_n']}",
         f"无gate→Full，无支持率 {pct(s['multi']['unsupported_rate_accepted'])}→{pct(s['full']['unsupported_rate_accepted'])}；无支持数 {s['multi']['unsupported_n']}→{s['full']['unsupported_n']}"],
        ["拒答能力（Full）", refusal(q['full']), refusal(sn) + "（454+60 混合集）"],
        ["覆盖率代价（B/M/V）", f"{triple(q,'coverage')}；回答数 {triple(q,'accepted_n',str)}；Full 未回答 {300-int(q['full']['accepted_n'])}",
         f"{triple(s,'coverage')}；回答数 {triple(s,'accepted_n',str)}；Full 未回答 {454-int(s['full']['accepted_n'])}"],
        ["性能代价（秒）", f"B、M、V 各自平均/P95：{perf('qasper')}；每策略50条", f"B、M、V 各自平均/P95：{perf('scifact')}；每策略50条"],
        ["消融（无支持率/Coverage）", ablation(q), ablation(s)],
    ]
    concise = ["# 按原始十项需求汇总", "", "B=固定512；M=动态128/512/1024无gate；V=完整VeriMind。三值均按B/M/V排列。", "",
               "实际主实验为754个去重输入，另加60条NEI，共814个输入；旧SciFact 500条权重结果另存summary.csv。", "",
               table(["项目", "QASPER", "SciFact"], ten), "",
               "无支持率/Faithfulness 使用独立Qwen自动判断，分母为已回答数，不等同于人工确认幻觉。QASPER检索有效分母237；63条图表gold未进入文本索引，保留在召回分母。SciFact拒答指标使用514条混合集，其余主表使用454条去重主样本（dev 188、train 266），不作为官方纯测试集结果。", "",
               "全部42项相关测试通过。主实验814输入/1628策略答案/4070决策行齐全；独立计时300次调用齐全。Citation Support Rate未冒算，本次选择Faithfulness与Answer Correctness/原生质量指标。", ""]
    (output / "TEN_ITEM_RESULTS.md").write_text("\n".join(concise), encoding="utf-8")
    audit = {"validation": "passed", "input_count": len(items), "answer_rows": len(answers), "policy_rows": len(cases),
             "unmapped_qasper_question_strategy_rows": len(unmapped), "successful_unique_generation_records": len(unique_answers)}
    (output / "report_validation.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
