# 进度日志

## 2026-04-12：初始化与规划
- 完成项目结构阅读与任务拆解。
- 建立三文档：`revision/task_plan.md`、`revision/progress.md`、`revision/findings.md`。
- 明确阶段主线：`1.1 -> 1.2 -> 1.3 -> 2.x -> 3.x -> 4`。

## 2026-04-13：1.1 外部验证主线完成

### 1.1.1 SciFact 数据准备（完成）
- 下载并清洗 SciFact 数据，构建 500 条评测集。
- 产物：`revision/artifacts/1_1_scifact/dataset/scifact_bench.json`。

### 1.1.2 SciFact 索引构建（完成）
- 构建 128/512/1024 三粒度向量库。
- 产物：
  - `revision/scifact/chroma_db_sf_128/`
  - `revision/scifact/chroma_db_sf_512/`
  - `revision/scifact/chroma_db_sf_1024/`

### 1.1.3 SciFact 对比实验（完成）
- 运行 500 × 3 系统实验并落盘。
- 产物：`revision/artifacts/1_1_scifact/results/scifact_experiment_results.csv`（1500 行）。
- 统计：McNemar 结论已可用于论文外部验证部分。

## 2026-04-13 ~ 2026-04-14：1.2 基线对等重构

### 1.2.1 十策略统一主实验（完成）
- 将 `CRAG_Local`、`CoT_RAG` 纳入主管线。
- 统一输出字段，满足 cross-eval 依赖。
- 将 `Weakness3` 相关脚本改为 `.env` 读取 API key。
- smoke 产物：
  - `revision/artifacts/1_2_baseline/1_2_1_main/experiment_results_sci_v3_smoke.csv`（100 行）
- full 产物：
  - `revision/artifacts/1_2_baseline/1_2_1_main/experiment_results_sci_v3.csv`（3000 行）

### 1.2.2 三评判器 cross-judge（完成）
- 新建并运行 `revision/cross_eval/run_cross_judge.py`。
- smoke 产物：
  - `revision/artifacts/1_2_baseline/1_2_2_cross_judge/cross_judge_results_smoke.csv`（300 行）
- full 产物：
  - `revision/artifacts/1_2_baseline/1_2_2_cross_judge/cross_judge_results.csv`（9000 行）
- 校验通过：
  - judge 覆盖 `qwen/glm/deepseek`
  - `(question_id, strategy)` 覆盖完整
  - 关键字段无空值

### 1.2.3 一致性统计（full 完成）
- 执行命令：
  - `D:\anaconda\envs\rag_domestic_env\python.exe revision/cross_eval/compute_agreement.py --input_csv revision/artifacts/1_2_baseline/1_2_2_cross_judge/cross_judge_results.csv --output_csv revision/artifacts/1_2_baseline/1_2_3_agreement/cross_judge_agreement.csv --output_txt revision/artifacts/1_2_baseline/1_2_3_agreement/cross_judge_agreement.txt`
- 正式产物：
  - `revision/artifacts/1_2_baseline/1_2_3_agreement/cross_judge_agreement.csv`
  - `revision/artifacts/1_2_baseline/1_2_3_agreement/cross_judge_agreement.txt`
- 结果摘要：
  - `Triplet coverage = 3000/3000 = 1.0000`
  - `Cohen's Kappa: qwen-glm=0.559403, qwen-deepseek=0.546824, glm-deepseek=0.700133`
  - `Fleiss' Kappa: qwen-glm-deepseek=0.601333`
- 结论：`1.2.3` 已从 smoke 验证进入 full 可引用结果。

## 2026-04-14：1.3.1 多模型抽象与跨架构入口（完成）
- 新增 `revision/cross_arch/llm_clients.py`：
  - `LLMClient` 统一接口；
  - `QwenClient/GLMClient/DeepSeekClient` 三实现；
  - 统一重试退避、拒答捷径、分数解析、异常格式；
  - 环境变量固定为 `DASHSCOPE_API_KEY/ZHIPU_API_KEY/DEEPSEEK_API_KEY`。
- 新增 `revision/cross_arch/run_cross_arch_experiment.py`：
  - 固定 CLI：`--input_json --output_csv --generator --judge --limit --strategies --resume --dry_run`；
  - 复用既有检索与策略逻辑，仅替换生成/评判客户端为可注入；
  - `resume` 按 `(question_id, strategy)`，并强制单输出文件对应单一 generator/judge 组合。
- 验证结果：
  - 客户端工厂创建通过（qwen/glm/deepseek）；
  - 缺失 key 报错均包含具体环境变量名；
  - dry-run `--limit 5` 通过并落盘 `cross_arch_dry_run.csv`；
  - 第二次 dry-run 成功识别 resume，跳过 15 条已完成项。
- 边界说明：
  - 本步仅交付抽象层与入口脚本，不产出 E2-E5 实验结果。

## 2026-04-14：目录与资产整理
- 将 1.1/1.2 关键产物统一归档到 `revision/artifacts/`。
- 建立路径索引：`revision/index/current_artifacts.md`。
- 历史不一致产物归档到 `revision/artifacts/legacy/`。

## 2026-04-14 ~ 2026-04-15：运行可观测工具与论文证据流程

### 工具链（已落地）
- `revision/scripts/start_job.ps1`（默认前台可见输出）
- `revision/scripts/shortcuts.ps1`（短命令入口）
- `revision/scripts/tail_job.ps1`
- `revision/scripts/start_background_job.ps1`
- `revision/scripts/job_status.ps1`
- `revision/scripts/stop_background_job.ps1`
- `revision/scripts/notifier_workflow.md`（中文使用说明）

### 论文证据沉淀（已落地）
- 新增 `revision/paper_revision_evidence.md`。
- 新增 `.agent/rules/paper_evidence_workflow.md`。
- 已回填：1.1.3、1.2.1、1.2.2、1.2.3 对论文直接可用证据。

## 2026-04-15：阶段核对结论
- `1.1.1`：完成
- `1.1.2`：完成
- `1.1.3`：完成
- `1.2.1`：完成
- `1.2.2`：完成
- `1.2.3`：完成（full 统计已产出）

## 2026-04-15：1.3.2 E2（GLM 生成 + Qwen 评判）完成
- 执行入口：
  - `revision/cross_arch/run_cross_arch_experiment.py`
  - 参数：`--generator glm --judge qwen --limit 300 --strategies Vanilla_RAG,HyDE,VeriMind_Full --resume true`
- 产物：
  - `revision/artifacts/1_3_cross_arch/cross_arch_E2.csv`
- 规模校验：
  - 最终 `900/900`，`300` 题 × `3` 策略，`(question_id,strategy)` 无缺失、无重复。
- 过程异常与修复：
  - 异常：`need to escape, but no escapechar set` 导致中间缺失 7 行（893/900）。
  - 修复：在 `run_cross_arch_experiment.py` 的 `to_csv` 增加 `escapechar="\\\\"`。
  - 补跑：使用 `--resume true` 成功补齐至 900 行。
- 指标摘要（E2）：
  - `Vanilla_RAG`：faith=5.4533，hal=0.4600，f1=0.2584
  - `HyDE`：faith=5.9433，hal=0.4133，f1=0.2680
  - `VeriMind_Full`：faith=9.9700，hal=0.0033，f1=0.1054
- 结论：
  - VeriMind 在跨架构 E2 下仍保持极低幻觉率优势，但回答更保守（拒答比例高）。

## 2026-04-15：1.3.3 E3（Qwen 生成 + DeepSeek 评判）完成
- 执行入口：
  - `revision/cross_arch/run_cross_arch_experiment.py`
  - 参数：`--generator qwen --judge deepseek --limit 300 --strategies Vanilla_RAG,HyDE,VeriMind_Full --resume true`
- 产物：
  - `revision/artifacts/1_3_cross_arch/cross_arch_E3.csv`
- 规模校验：
  - 最终 `900/900`，`300` 题 × `3` 策略，`(question_id,strategy)` 无缺失、无重复。
- 指标摘要（E3）：
  - `Vanilla_RAG`：faith=2.6200，hal=0.7267，f1=0.2251
  - `HyDE`：faith=2.8133，hal=0.7067，f1=0.2330
  - `VeriMind_Full`：faith=9.9667，hal=0.0000，f1=0.1064，gated_reject=33
- 与 E2 对比：
  - `Vanilla_RAG`：`hal +0.2667`，`faith -2.8333`，`f1 -0.0333`
  - `HyDE`：`hal +0.2933`，`faith -3.1300`，`f1 -0.0350`
  - `VeriMind_Full`：`hal -0.0033`，`faith -0.0033`，`f1 +0.0010`
- 结论：
  - DeepSeek 评判下基线策略更严格且幻觉率更高，但 VeriMind 仍保持近 0 幻觉率。

## 2026-04-15：1.3.4 E4（DeepSeek 生成 + GLM 评判）完成
- 执行入口：
  - `revision/cross_arch/run_cross_arch_experiment.py`
  - 参数：`--generator deepseek --judge glm --limit 300 --strategies Vanilla_RAG,HyDE,VeriMind_Full --resume true`
- 产物：
  - `revision/artifacts/1_3_cross_arch/cross_arch_E4.csv`
- 规模校验：
  - 最终 `900/900`，`300` 题 × `3` 策略，`(question_id,strategy)` 无缺失、无重复。
- 指标摘要（E4）：
  - `Vanilla_RAG`：faith=3.3000，hal=0.6700，f1=0.2545
  - `HyDE`：faith=3.3333，hal=0.6667，f1=0.2757
  - `VeriMind_Full`：faith=10.0000，hal=0.0000，f1=0.1132，gated_reject=29
- 结论：
  - 在 `DeepSeek -> GLM` 组合下，VeriMind 继续保持近 0 幻觉率，跨架构稳定性进一步增强。

## 2026-04-15：1.3.5 E5（GLM-4.7 生成 + DeepSeek 评判）完成
- 执行入口：
  - `revision/cross_arch/run_cross_arch_experiment.py`
  - 参数：`--generator glm --generator_model glm-4.7 --judge deepseek --judge_model deepseek-chat --limit 100 --strategies Vanilla_RAG,HyDE,VeriMind_Full --resume true`
- 产物：
  - `revision/artifacts/1_3_cross_arch/cross_arch_E5.csv`
- 规模校验：
  - 最终 `300/300`，`100` 题 × `3` 策略，`(question_id,strategy)` 无缺失、无重复。
- 指标摘要（E5）：
  - `Vanilla_RAG`：faith=2.2400，hal=0.7600，f1=0.2877
  - `HyDE`：faith=1.9400，hal=0.7700，f1=0.3118
  - `VeriMind_Full`：faith=9.9700，hal=0.0000，f1=0.1222，gated_reject=7
- 结论：
  - E5 再次支持 VeriMind 的低幻觉稳健性；但充分性指标仍需以 trade-off 方式保守表述。

## 2026-04-15：1.3.6 跨架构汇总分析（E1-E5）完成
- 汇总输入：
  - `E1`：`revision/artifacts/1_2_baseline/1_2_1_main/experiment_results_sci_v3.csv`（3 策略 900 行）
  - `E2`：`revision/artifacts/1_3_cross_arch/cross_arch_E2.csv`（900 行）
  - `E3`：`revision/artifacts/1_3_cross_arch/cross_arch_E3.csv`（900 行）
  - `E4`：`revision/artifacts/1_3_cross_arch/cross_arch_E4.csv`（900 行）
  - `E5`：`revision/artifacts/1_3_cross_arch/cross_arch_E5.csv`（300 行）
- 新产物：
  - `revision/artifacts/1_3_cross_arch/cross_arch_E1_E5_summary.csv`
  - `revision/artifacts/1_3_cross_arch/cross_arch_fleiss_E1_E5.csv`
  - `revision/artifacts/1_3_cross_arch/cross_arch_fleiss_E1_E5.txt`
- 完整性校验：
  - E1/E2/E3/E4/E5 均 `(question_id,strategy)` 无缺失、无重复。
  - E5 公共子集（100 题）用于 5 评判位 Fleiss 统计，样本 `300 items × 5 raters`。
- 关键统计（common100）：
  - overall Fleiss' Kappa = `0.521101`
  - `Vanilla_RAG` Fleiss' Kappa = `0.270416`
  - `HyDE` Fleiss' Kappa = `0.329063`
  - `VeriMind_Full` Fleiss' Kappa = `-0.002004`（近零方差导致的统计特性）
- 稳定性摘要（common100，hallucination_rate 区间）：
  - `Vanilla_RAG`: `0.38 -> 0.76`
  - `HyDE`: `0.40 -> 0.77`
  - `VeriMind_Full`: `0.00 -> 0.01`
- 结论：
  - `1.3` 阶段证据链闭环：VeriMind 在 E1-E5 组合下持续近零幻觉，跨架构稳定性得到进一步支撑。

## 2026-04-15：2.1 主结果 BERTScore 补齐（完成）
- 新增脚本：
  - `revision/analysis/add_bertscore_main.py`
- 执行流程：
  - smoke：`--limit 30` 验证脚本可运行并产出结构化结果
  - full：`3000` 行主表全量计算 BERTScore
- 产物：
  - `revision/artifacts/2_1_bertscore/experiment_results_sci_v3_with_bertscore.csv`（3000 行）
  - `revision/artifacts/2_1_bertscore/table7_bertscore_summary.csv`（10 策略汇总）
  - 运行日志：`revision/logs/2_1_bertscore_full.log`
- 完整性校验：
  - 输出行数 `3000`，策略覆盖 `10` 个
  - `bert_score_f1` 无空值，`-1.0` 数量为 `0`
- Table 7 关键数值（BERTScore F1 均值）：
  - `Vanilla_RAG`：`0.795088`
  - `HyDE`：`0.796373`
  - `VeriMind_Full`：`0.789337`
  - 其余策略位于 `0.782794 ~ 0.792067`
- 结论：
  - 在主表口径下补齐了语义相似度指标，VeriMind 的低幻觉优势与可接受的语义相似度可并行报告。

## 2026-04-15：2.2 Cohen's h 效应量（完成）
- 新增脚本：
  - `revision/analysis/cohen_h.py`
- 输入：
  - `revision/artifacts/2_1_bertscore/experiment_results_sci_v3_with_bertscore.csv`
- 输出：
  - `revision/artifacts/2_2_cohen_h/cohen_h_vs_verimind.csv`
  - `revision/artifacts/2_2_cohen_h/cohen_h_vs_verimind.txt`
- 规模与分布：
  - 比较对象共 `9` 个（锚定 `VeriMind_Full`）
  - `large=4`，`small=3`，`negligible=2`
- 关键效应量（按 |h| 排序）：
  - `Long-Context_LLM`: `h=1.772154`（large）
  - `VeriMind_No_Gating`: `h=1.342142`（large）
  - `Vanilla_RAG`: `h=1.328430`（large）
  - `HyDE`: `h=1.273085`（large）
  - `Self-RAG`: `h=0.402716`（small）
  - `CRAG_Local`: `h=0.385351`（small）
  - `CoT_RAG`: `h=0.367208`（small）
- 结论：
  - 相比主要基线（Vanilla/HyDE/Long-Context），VeriMind 的低幻觉优势对应“大效应量”，可用于强化论文统计效力论证。

## 2026-04-15：2.3 阈值敏感性分析（完成）
- 新增脚本：
  - `revision/analysis/threshold_sensitivity.py`
- 输入：
  - `revision/artifacts/2_1_bertscore/experiment_results_sci_v3_with_bertscore.csv`
  - 使用 `VeriMind_Full` + `VeriMind_No_Gating` 的同题配对进行阈值仿真
- 输出：
  - `revision/artifacts/2_3_threshold/threshold_sensitivity_curve.csv`
  - `revision/artifacts/2_3_threshold/threshold_sensitivity_summary.txt`
  - `revision/artifacts/2_3_threshold/threshold_sensitivity_curve.svg`
- 关键结果：
  - 当前阈值 `t=5.0`：`hal=0.000000`，`f1=0.133966`，`accept_rate=0.966667`
  - 最优 F1 阈值：`t=0.0`，`f1=0.144875`，但 `hal=0.030000`
  - 最低幻觉阈值区间：`t>=4.0` 时 `hal=0.000000`
  - 当阈值升至 `t=8.0`：`accept_rate=0.710000`，`f1=0.117097`
- 结论：
  - 阈值提高会持续压低风险但明显牺牲回答充分性；
  - 结合当前结果，`t=4.0~5.0` 是“零幻觉 + 可接受回答率”的合理工作区间。

## 2026-04-15：3.1 数据标注流程补充（完成）
- 新增脚本：
  - `revision/analysis/annotation_audit_3_1.py`
- 输入：
  - `csv/human_validation_50_samples_scored.csv`
  - `csv/relabeled_audit_sheet.csv`
  - `csv/relabeled_audit_sheet2.csv`
- 输出：
  - `revision/artifacts/3_1_annotation/annotation_audit_summary.csv`
  - `revision/artifacts/3_1_annotation/annotation_audit_summary.txt`
  - `revision/annotation_process_3_1.md`
- 流程规模：
  - 复标审计样本 `600` 条（4 系统均衡覆盖，每系统 `150`）
  - 分层分布：`TRUSTED=256`，`WARNING=48`，`REJECTED=296`
- 关键一致性结果：
  - `LLM vs Human (n=50)`：`agreement=0.9600`，`kappa=0.9151`
  - `Annotator A vs B overall (n=600)`：`agreement=0.9033`，`kappa=0.7767`
  - `TRUSTED`：`agreement=0.9844`，`kappa=0.9686`
  - `WARNING`：`agreement=1.0000`，`kappa=1.0000`
  - `REJECTED`：`agreement=0.8176`，`kappa=0.0000`
- 结论：
  - 可支撑“标注一致性整体较高”的论文表述；
  - `REJECTED` 分层需在正文解释边际分布极不均衡导致的 kappa 敏感性。

## 2026-04-15：3.2 AI 使用声明修订（完成）
- 修改文件：
  - `revision/🛠️ 核心战役：全面修改方案与执行计划.md`
- 修订内容：
  - 删除残留 `GLM-4.6V` 预算项，避免与“已弃用”表述冲突；
  - 将 `3.2` 从占位语句升级为正式声明：明确实验模型口径为 `Qwen-Plus / GLM-4.5-Air / GLM-4.7(E5) / DeepSeek-V3.2`；
  - 明确 AI 工具用途边界与“以 artifacts 为准”的证据原则；
  - checklist 中“AI 使用声明修正”改为完成。
- 结论：
  - `3.2` 已完成，模型版本叙述与实际实验执行状态一致。

## 2026-04-15：3.3 过度宣称降调（完成）
- 修改文件：
  - `revision/🛠️ 核心战役：全面修改方案与执行计划.md`
- 修订内容：
  - 将“建立铁证”改为“形成经验性证据链”；
  - 将“证明最强模型也纳入验证”改为“验证强模型条件下结论趋势是否一致”；
  - 将 “Table 7 条件完全一致” 改为 “在统一协议下尽量对齐”；
  - 将 “目标达成” 语句增加范围限定（`在当前评测范围内`）；
  - `3.3` checklist 勾选完成。
- 结论：
  - `3.3` 已完成，强断言表述已降调为“经验性 + 范围限定”口径。

## 2026-04-15：3.4 Related Work 更新（完成）
- 修改文件：
  - `revision/🛠️ 核心战役：全面修改方案与执行计划.md`
- 修订内容：
  - 将 `3.4` 从占位式三行清单升级为“主线化 Related Work”结构；
  - 补齐 2024-2025 方向映射：`Adaptive-RAG / Self-RAG / CRAG / RAGChecker / 2025 延展`；
  - 明确写作边界为“互补定位 + 非首创化 + 可复现实证证据链”。
- 结论：
  - 阶段三（`3.1~3.4`）已全部完成，下一步进入阶段四总验收收口。

## 2026-04-15：4.1 参考文献核对（BibTeX，完成）
- 新增文件：
  - `revision/references/related_work_2024_2025.bib`
- 已补齐条目（2024，已核对）：
  - Adaptive-RAG（NAACL 2024）
  - Self-RAG（ICLR 2024）
  - Corrective RAG / CRAG（arXiv 2024）
  - RAGChecker（NeurIPS 2024 Datasets & Benchmarks）
- 执行边界：
  - 2025 文献不在本步强行写入，已标注为 `4.2` 二次核对任务，避免错误引用。

## 2026-04-15：4.2 2025 参考文献二次核对（完成）
- 更新文件：
  - `revision/references/related_work_2024_2025.bib`
- 新增并核对的 2025 条目：
  - FaithfulRAG（ACL 2025）
  - Retrieval-Augmented Generation with Estimation of Source Reliability（EMNLP 2025）
  - Fact, Fetch, and Reason: A Unified Evaluation of Retrieval-Augmented Generation（NAACL 2025）
- 核对口径：
  - 以 ACL Anthology 主页面为准，补齐 `doi/pages/booktitle/year` 等核心字段。
- 结论：
  - Related Work 的 2024-2025 BibTeX 基础条目已可直接并入论文引用库。

## 2026-04-15：4.3 论文写回包与证据索引收口（完成）
- 新增文件：
  - `revision/final_writeback_packet.md`
- 更新文件：
  - `revision/index/current_artifacts.md`
  - `revision/paper_revision_evidence.md`
  - `revision/task_plan.md`
  - `revision/progress.md`
  - `revision/findings.md`
- 本步目标：
  - 将 1.1~3.4 与 4.1~4.2 的关键证据收敛为“可直接写回论文”的统一入口；
  - 消除 `current_artifacts.md` 中“历史 partial / 待重跑”等过期描述；
  - 明确阶段边界：4.3 完成的是“写回材料收口”，不是“正文已改写”。
- 关键产出：
  - 写回总入口：`revision/final_writeback_packet.md`
  - 最新资产索引：`revision/index/current_artifacts.md`
  - 论文证据链补充：`revision/paper_revision_evidence.md`（新增 4.3 小节）
- 结论：
  - 阶段四已完成资料收口，可直接进入正文写回执行；
  - 下一阻塞项为论文源文件接入（`.tex`/`.docx`）。

## 当前待办（严格顺序）
1. 当前状态：确认论文如何修改，等待执行（你在 Overleaf 开始落地后触发 4.4）。
2. 待触发：按 `revision/final_writeback_packet.md` 逐段写回正文。
3. 待触发：写回完成后执行终稿润色与一致性校对（图表编号、引用、术语统一）。

---

## 关键错误与修复记录（保留）
- GLM 思考模式导致 `content` 为空：通过禁用 thinking 解决。
- BERTScore 初始化联网失败：通过离线模式与本地缓存规避。
- BERTScore 依赖链 DLL 导入异常：通过沙箱外执行 + 脚本级最小兼容处理完成。
- 全量主表尾段写盘异常（escapechar）：通过补写阶段修复并完成 3000 行闭环。
- 文档编码异常（中文乱码）：已执行整份中文重建，恢复可读与可维护状态。

---

*最后更新：2026-04-15*

## 2026-06-22：JIIS 投稿版收口

- 已完成：
  - 新增 QASPER 数据准备、索引、实验与统计分析流程。
  - 完成 1800 行 QASPER 全量实验并生成 McNemar 配对检验。
  - 重算 VeriMind-Bench 与 SciFact 的出版级指标。
  - 将原 IJMLC 稿独立重构为 JIIS 版本。
  - 将正文从 33 页压缩至 25 页，切换为 `smallcondensed` 与数字引用。
  - 删除旧等覆盖阈值论证、冗余图表、价格敏感的成本小节和重复审计表。
  - 创建 cover letter、投稿清单和复现材料目录。
- 当前稿件：
  - `revision/paper_jiis/main.tex`
  - `revision/paper_jiis/main.pdf`
- 下一步：
  1. 生成并校验 Overleaf 源码包。
  2. 生成并校验复现材料包。
  3. 投稿前由作者确认署名、声明和非一稿多投状态。

*最后更新：2026-06-22*

## 2026-06-22：QASPER 多裁判复评与正文写回

- 完成 802 个 QASPER 已放行答案的三裁判复评，共 2406 条唯一记录。
- 评判器：DeepSeek `deepseek-chat`、Qwen-Plus、GLM-4.5-Air。
- 完整性审计：缺失分数 0、越界分数 0、阈值标签错配 0、API 失败 0。
- 一致性：三裁判一致率 `89.65%`，Fleiss' Kappa `0.5041`。
- 多数票结果：
  - VeriMind Full：`1/133`（`0.75%`）
  - Vanilla RAG：`20/147`（`13.61%`）
  - HyDE：`9/143`（`6.29%`）
  - No Gating：`18/149`（`12.08%`）
- 已写回 `revision/paper_jiis/main.tex`，以公开 QASPER 多裁判结果替换旧
  stress-test 多裁判数字。
- 已删除论文中的 `domain experts`、600 条 independent dual-annotator
  audit，以及既有 50 条 human-label 一致性结论。
- 50 条 QASPER 作者盲审表已生成但尚未填写，因此论文未报告人工审计结果。

*最后更新：2026-06-22*


## 2026-09-27：补充实验执行中

- 新增 `revision/qasper/export_retrieval_audit.py`：5 问题 smoke 通过；300 问题 × 2 检索策略完整运行结束。
- 新增 `revision/experiments/supplement_20260927.py` 与 `test_supplement_20260927.py`：12 项指标测试由失败转为通过。
- `_ctypes` 在沙箱内拒绝访问；相同环境在获准的沙箱外运行成功，未安装或升级依赖。
- GLM、DeepSeek、Qwen 最小接口检查通过；端到端 smoke 正在运行，尚无全量新生成结果。


### 2026-09-27：全分支 smoke 已验证，准备全量

- `supplement_20260927_smoke_routes/validation.json`：12 输入、24 策略答案、60 policy 行，complete=true；三数据组均覆盖 DETAIL/CONCEPT/CONTEXT。
- `python -B -m unittest revision.experiments.test_supplement_20260927`：14 tests passed。
- 全量范围：QASPER 300 + SciFact 去重 454 + 单列 NEI 60 = 814 输入；每输入两种检索策略，五种决策策略。
- SciFact correctness 主指标使用原始 gold label 的确定性匹配，Qwen correctness 作为辅助，避免评判器忽略标签差异。
- 新输出：`revision/artifacts/supplement_20260927_full/`。正文与简历的全量量化结论仍待完成后验证。


### 2026-09-27：QASPER 全量阶段完成与语料限制

- 本次 full 产物已包含 300 个 QASPER 输入的 600 条策略答案；全任务仍在运行，SciFact/NEI 尚未完成。
- 237 个问题具有非空 evidence；49 个问题共 63 条 gold 证据标为 `FLOAT SELECTED`，全部不在现有 qasper_corpus 文本段落中。
- SQLite 只读源段落重建与 corpus 交叉核对表明，当前未映射证据数恰好等于 63 条图表证据，不是文本段落映射程序遗漏。
- 这些 gold 保留在标准召回分母；报告必须注明当前是文本索引，不能把图表缺失解释为路由模块单独导致的失败。
- 正文候选说明：QASPER 检索评估使用完整标注证据分母，而现有索引只覆盖文本段落，因此图表证据的缺失构成语料覆盖上限。


### 2026-09-27：独立计时入口与回归验证

- 新增 `revision/experiments/latency_20260927.py`、`test_latency_20260927.py`；计划 QASPER/SciFact 各 50 个固定分层样本 × 3 策略，策略顺序固定随机化。
- baseline/multi 在线路径不调用 judge；Full 仅计入在线 gate，不计入 Qwen 离线评估。每次独立调用，不复用答案。
- 计时 smoke：2 输入、6 次实际调用，全部完成。
- 新增行为测试 16 项通过；现有 selective-risk、cheap-gate、cost-aware-cascade、gate-v2 回归测试 26 项通过，共 42 项。
- 证据：`revision/artifacts/supplement_20260927_full/test_verification.json`、`revision/artifacts/latency_20260927_smoke/validation.json`。
- 全量主任务仍在运行；已发现两条评判器非 JSON 回复，保留失败记录，结束后只补齐缺失条目，不用默认分数填充。


### 2026-09-27：主实验全量完成，独立性能试验进行中

- `supplement_20260927_full/validation.json`：814 输入、814 检索行、1628 策略答案、4070 policy 行，complete=true。
- 首轮 3 个输入未完成（2 条评估 JSON 截断/格式异常、1 条 TLS 中断）；保留 errors.jsonl，补跑后无缺失输入。
- 最后一条诊断确认 Qwen 的长说明被 300-token 上限截断；仅该条将输出上限增至 1024，原模型/提示/温度不变。详见 protocol_exception.json、retry_diagnostics.jsonl。
- 主结论草稿：动态路由未改善当前基准检索召回；QASPER gate 在独立评判口径下未降低已回答无支持比例；SciFact 风险降低伴随覆盖率下降。完整定量报告等待独立计时完成后一次交付。
- 独立计时作业：`latency_20260927_full`，100 个固定输入 × 3 策略，原始 timings.jsonl 单独保存。


## 2026-09-27：十项补充实验全部完成并验证

- [x] 全量：QASPER 300 + SciFact 454 去重 claim + NEI 60 = 814 输入；1628 策略答案、4070 决策行完整，无未解决缺失。900 份成功独立生成记录复用为两检索策略结果；失败重试另存日志。
- [x] 独立计时：两数据集各 50 输入 × 3 策略，共 300 次实际调用，无失败；baseline/multi 不调用在线 judge。
- [x] 42 项测试通过（16 新测试 + 26 回归）；报告审计通过唯一键、上下文哈希、源文件哈希、代码哈希、gate重放、summary分母。
- 交付：`revision/artifacts/supplement_20260927_full/TEN_ITEM_RESULTS.md`、`RESULTS.md`、`summary.csv`、`paired_gate_audit.csv`；性能：`revision/artifacts/latency_20260927_full/summary.csv`。
- 检索固定→动态：QASPER R@3 42.92%→42.11%、R@5 50.33%→49.35%；SciFact R@3 85.18%→84.52%、R@5 89.74%→89.44%。当前无多粒度召回提升证据。
- 无gate→Full，Qwen无支持率：QASPER 9.72%→9.92%，Coverage 48.00%→43.67%；SciFact 6.92%→4.33%，Coverage 92.29%→61.01%。不写成普遍降低真实幻觉。
- Full 拒答：QASPER TP=54/60、FP=115，P/R/F1=31.95%/90.00%/47.16%；SciFact 454+60混合集 TP=53/60、FP=177，P/R/F1=23.04%/88.33%/36.55%。
- 独立平均/P95秒：QASPER baseline 1.264/2.026，Full 1.598/2.955；SciFact baseline 1.330/1.638，Full 1.960/2.782。均为每策略50样本客户端测量，不是生产SLA。
- 边界：QASPER 63条图表gold不在文本索引；SciFact主样本dev188/train266，并非纯官方test；Faithfulness与无支持率是独立Qwen自动判定，未宣称新答案经人工认证。
- 一个评估JSON截断案例只增加输出上限300→1024，其模型/提示/温度不变，protocol_exception.json记录；所有原始失败与补跑记录保留。
- 论文候选表述：在固定输入与配对答案的补充评估中，动态粒度路由未提高检索召回；信任门控在SciFact降低了自动判定的已答无支持率，但减少了回答覆盖，在QASPER未观察到该条件风险改善。结果说明当前门控仍存在较高误拒，需同时报告风险、覆盖率与延迟。未据此作统计显著性结论。
- 未修改论文正文、应用代码或依赖；未覆盖历史实验原始结果。


## 2026-09-27：Gate 离线诊断完成

- 证据：`revision/artifacts/gate_diagnostics_20260927/REPORT.md`、`scores.csv`、`threshold_sweep.csv`、`gate_activation.csv`、`debug50.csv`、`provenance.json`。
- 814条冻结 multi 原始答案，QASPER 300；SciFact 454+NEI 60。50条分层debug；无API调用、无重新生成。
- retrieval gate阈值0.5：QASPER实际新增拦截0（4条低分已主动拒答），SciFact+NEI新增拦截0；与no-gate决策差异均0。极端/边界断言与P10-P90 coverage单调性通过，支持阈值未触发而不是已证实实现bug。
- QASPER：retrieval预测answerable AUROC 0.592；judge预测supported 0.500；Trust预测safe 0.532。SciFact+NEI分别为0.768/0.634/0.757。对应AUPRC与正例率详见scores.csv。
- 0.1–0.9 Trust阈值表已保存。QASPER无稳定风险收益；SciFact+NEI在0.3→0.4时coverage 84.4%→56.2%，表明校准不平滑。当前不依据测试集选择新阈值。
- 必须保留口径：judge/Trust AUROC只在候选答案上计算，主动拒答默认score=10排除；支持标签来自独立Qwen，safe还要求答案正确；AUPRC受高正例率影响。
- 论文候选写法：当前retrieval-only阈值未产生额外拒答，门控消融与无门控重合；离线分数诊断表明QASPER的现有judge/Trust信号区分力弱，而SciFact的融合分数具有一定排序信息但存在显著的coverage跳变。后续应在独立校准集调参并在留出集验证。
