# 任务计划：VeriMind 论文修改执行方案

> 目标期刊：EAAI（首选）/ Applied Soft Computing（备选）
> 核心策略：消除三大硬伤（外部验证、基线对等、跨架构泛化）
> 资源原则：GLM-4.7 严格节省，主力由 GLM-4.5-Air 承担

---

## 0. 当前状态总览（2026-04-15）
- 阶段 1.1：已完成
- 阶段 1.2.1：已完成（10 策略主表 3000 行）
- 阶段 1.2.2：已完成（三评判器 cross-judge 9000 行）
- 阶段 1.2.3：已完成（full 一致性统计已产出）
- 阶段 1.3.1：已完成（统一客户端抽象与跨架构入口）
- 阶段 1.3.2：已完成（E2，900 行）
- 阶段 1.3.3：已完成（E3，900 行）
- 阶段 1.3.4：已完成（E4，900 行）
- 阶段 1.3.5：已完成（E5，300 行，100×3）
- 阶段 1.3.6：已完成（E1-E5 汇总与 Fleiss 报告）
- 阶段 2.1：已完成（主表 BERTScore 补齐 + Table 7 素材）
- 阶段 2.2：已完成（Cohen's h 效应量）
- 阶段 2.3：已完成（阈值敏感性曲线）
- 下一步：确认论文如何修改，等待执行（4.4 写回步骤已整理到 `revision/final_writeback_packet.md`）

---

## 1. 阶段一：修复致命硬伤

### 1.1 SciFact 外部基准验证（已完成）

#### 1.1.1 下载与预处理 SciFact 数据集
- [x] 通过可用镜像下载 SciFact 数据（规避 HuggingFace 网络问题）
- [x] 筛选有 evidence 的 claim，构建 500 条评测样本
- [x] 输出为与主流程对齐格式
- [x] 产物：`revision/artifacts/1_1_scifact/dataset/scifact_bench.json`

#### 1.1.2 构建 SciFact 向量索引
- [x] 完成 128/512/1024 三粒度索引构建
- [x] 支持断点续传与批处理
- [x] 产物：
  - `revision/scifact/chroma_db_sf_128/`
  - `revision/scifact/chroma_db_sf_512/`
  - `revision/scifact/chroma_db_sf_1024/`

#### 1.1.3 运行 SciFact 对比实验
- [x] 完成 500 题 × 3 系统实验
- [x] 产物：`revision/artifacts/1_1_scifact/results/scifact_experiment_results.csv`（1500 行）
- [x] McNemar 检验脚本与统计结论已产出

### 1.2 重构基线对比实验（10 系统全纳入）

#### 1.2.1 统一主实验管线（已完成）
- [x] `run_sci_experiment_main.py` 纳入 `CRAG_Local` 与 `CoT_RAG`
- [x] 统一 10 策略输出 schema（含 cross-eval 所需字段）
- [x] `Weakness3` 相关脚本改为 `.env` 读取 key
- [x] smoke 通过：
  - `revision/artifacts/1_2_baseline/1_2_1_main/experiment_results_sci_v3_smoke.csv`（100 行）
  - `revision/artifacts/1_2_baseline/1_2_2_cross_judge/cross_judge_results_smoke.csv`（300 行）
- [x] full 主表完成：
  - `revision/artifacts/1_2_baseline/1_2_1_main/experiment_results_sci_v3.csv`（3000 行）

#### 1.2.2 三评判器 cross-judge（已完成）
- [x] 新建 `revision/cross_eval/run_cross_judge.py`
- [x] smoke 通过（100 × 3 = 300）
- [x] full 完成：
  - `revision/artifacts/1_2_baseline/1_2_2_cross_judge/cross_judge_results.csv`（9000 行）

#### 1.2.3 一致性统计（已完成）
- [x] 新建 `revision/cross_eval/compute_agreement.py`
- [x] 实现两两 Cohen's Kappa + 三方 Fleiss' Kappa
- [x] 基于 full 9000 行输出正式一致性文件
  - `revision/artifacts/1_2_baseline/1_2_3_agreement/cross_judge_agreement.csv`
  - `revision/artifacts/1_2_baseline/1_2_3_agreement/cross_judge_agreement.txt`
  - `Triplet coverage = 3000/3000 = 1.0000`
  - `Cohen's Kappa = 0.559403 / 0.546824 / 0.700133`
  - `Fleiss' Kappa = 0.601333`

### 1.3 三厂商交叉实验 E2-E5（已完成）

#### 1.3.1 多模型客户端抽象（已完成）
- [x] 新建 `revision/cross_arch/llm_clients.py`
- [x] 新建 `revision/cross_arch/run_cross_arch_experiment.py`
- [x] 支持 `--generator/--judge --resume --dry_run` 固定入口
- [x] dry-run 结构验收通过：`revision/artifacts/1_3_cross_arch/cross_arch_dry_run.csv`（15 行）

#### 1.3.2 E2：GLM-4.5-Air 生成 + Qwen-Plus 评判（300 条）
- [x] 执行并产出 `revision/artifacts/1_3_cross_arch/cross_arch_E2.csv`（900 行，300×3）
- [x] 运行中出现 CSV 转义异常，已修复并补齐缺失 7 行（最终完整 900/900）

#### 1.3.3 E3：Qwen-Plus 生成 + DeepSeek 评判（300 条）
- [x] 执行并产出 `revision/artifacts/1_3_cross_arch/cross_arch_E3.csv`（900 行，300×3）

#### 1.3.4 E4：DeepSeek 生成 + GLM-4.5-Air 评判（300 条）
- [x] 执行并产出 `revision/artifacts/1_3_cross_arch/cross_arch_E4.csv`（900 行，300×3）

#### 1.3.5 E5：GLM-4.7 生成 + DeepSeek 评判（100 条分层抽样）
- [x] 执行并产出 `revision/artifacts/1_3_cross_arch/cross_arch_E5.csv`（300 行，100×3）

#### 1.3.6 汇总分析
- [x] 合并 E1-E5 并计算核心指标
- [x] 产出正式 Fleiss' Kappa 报告
  - `revision/artifacts/1_3_cross_arch/cross_arch_E1_E5_summary.csv`
  - `revision/artifacts/1_3_cross_arch/cross_arch_fleiss_E1_E5.csv`
  - `revision/artifacts/1_3_cross_arch/cross_arch_fleiss_E1_E5.txt`

---

## 2. 阶段二：评估维度与统计效力（已完成）

### 2.1 引入 BERTScore
- [x] 在主结果中补齐 BERTScore F1
- [x] 更新论文 Table 7（素材表已产出）
  - `revision/artifacts/2_1_bertscore/experiment_results_sci_v3_with_bertscore.csv`
  - `revision/artifacts/2_1_bertscore/table7_bertscore_summary.csv`

### 2.2 补充效应量（Cohen's h）
- [x] 新建 `revision/analysis/cohen_h.py`
- [x] 输出各基线对 VeriMind 的效应量
  - `revision/artifacts/2_2_cohen_h/cohen_h_vs_verimind.csv`
  - `revision/artifacts/2_2_cohen_h/cohen_h_vs_verimind.txt`

### 2.3 阈值敏感性分析
- [x] 新建 `revision/analysis/threshold_sensitivity.py`
- [x] 产出阈值-指标曲线图
  - `revision/artifacts/2_3_threshold/threshold_sensitivity_curve.csv`
  - `revision/artifacts/2_3_threshold/threshold_sensitivity_summary.txt`
  - `revision/artifacts/2_3_threshold/threshold_sensitivity_curve.svg`

---

## 3. 阶段三：学术表述与严谨性重构（进行中）

### 3.1 数据标注流程补充
- [x] 补充标注人数、流程、Iaa/一致性信息
- [x] 产物：
  - `revision/analysis/annotation_audit_3_1.py`
  - `revision/artifacts/3_1_annotation/annotation_audit_summary.csv`
  - `revision/artifacts/3_1_annotation/annotation_audit_summary.txt`
  - `revision/annotation_process_3_1.md`

### 3.2 AI 使用声明修订
- [x] 校正模型版本描述（避免错误版本名）
- [x] 已修订：`revision/🛠️ 核心战役：全面修改方案与执行计划.md`（3.2 段）

### 3.3 过度宣称降调
- [x] 将“数学保证”类表述统一改为“经验验证”类表述
- [x] 已修订：`revision/🛠️ 核心战役：全面修改方案与执行计划.md`（3.3 段与关键措辞）

### 3.4 Related Work 更新
- [x] 补充 2024-2025 关键工作（含 Adaptive-RAG、RAGCHECKER 等）
- [x] 已修订：`revision/🛠️ 核心战役：全面修改方案与执行计划.md`（3.4 段）

---

## 4. 阶段四：投稿前总验收（未完成）
### 4.1 参考文献核对（BibTeX，已完成）
- [x] 新增 Related Work BibTeX 总表：`revision/references/related_work_2024_2025.bib`
- [x] 补齐并核对 2024 核心条目（Adaptive-RAG / Self-RAG / CRAG / RAGChecker）
- [x] 标注 2025 条目二次核对任务（4.2）

### 4.2 2025 参考文献二次核对（已完成）
- [x] 补齐 2025 BibTeX 条目（faithfulness / reliability / unified evaluation）
- [x] 2025 新增条目已写入：`revision/references/related_work_2024_2025.bib`
- [x] 条目来源核对：ACL Anthology 主页面（含 DOI/页码/会议信息）

### 4.3 论文写回包与证据索引收口（已完成）
- [x] 新增论文写回包：`revision/final_writeback_packet.md`
- [x] 更新资产索引：`revision/index/current_artifacts.md`
- [x] 同步论文证据总表：`revision/paper_revision_evidence.md`（补充 4.3）
- [x] 同步阶段文档：`revision/task_plan.md`、`revision/progress.md`、`revision/findings.md`
- [x] 边界确认：本步仅完成“写回材料收口”，未直接写入论文正文（仓库暂无 `.tex/.docx` 源文件）

- [ ] 外部验证章节与表格写回论文
- [ ] 10 基线对等评测全部写回
- [ ] 三厂商交叉实验写回
- [ ] 一致性统计写回
- [ ] BERTScore/效应量/阈值敏感性写回
- [x] 表述降调与 Related Work 更新完成
- [ ] 全文润色完成

---

## 5. 关键文件映射

### 已有与复用
- `run_sci_experiment_main.py`
- `revision/cross_eval/run_cross_judge.py`
- `revision/cross_eval/compute_agreement.py`
- `revision/scifact/run_scifact_experiment.py`
- `revision/analysis/add_bertscore_main.py`
- `revision/analysis/cohen_h.py`
- `revision/analysis/annotation_audit_3_1.py`
- `Weakness3/run_crag_local.py`
- `Weakness3/run_cot_rag.py`

### 当前关键产物
- `revision/artifacts/1_1_scifact/results/scifact_experiment_results.csv`
- `revision/artifacts/1_2_baseline/1_2_1_main/experiment_results_sci_v3.csv`
- `revision/artifacts/1_2_baseline/1_2_2_cross_judge/cross_judge_results.csv`
- `revision/artifacts/1_2_baseline/1_2_3_agreement/cross_judge_agreement.csv`
- `revision/artifacts/1_2_baseline/1_2_3_agreement/cross_judge_agreement.txt`
- `revision/artifacts/1_2_baseline/1_2_3_agreement/cross_judge_agreement_smoke.csv`
- `revision/artifacts/1_2_baseline/1_2_3_agreement/cross_judge_agreement_smoke.txt`
- `revision/artifacts/1_3_cross_arch/cross_arch_E2.csv`
- `revision/artifacts/1_3_cross_arch/cross_arch_E3.csv`
- `revision/artifacts/1_3_cross_arch/cross_arch_E4.csv`
- `revision/artifacts/1_3_cross_arch/cross_arch_E5.csv`
- `revision/artifacts/1_3_cross_arch/cross_arch_E1_E5_summary.csv`
- `revision/artifacts/1_3_cross_arch/cross_arch_fleiss_E1_E5.csv`
- `revision/artifacts/1_3_cross_arch/cross_arch_fleiss_E1_E5.txt`
- `revision/artifacts/2_1_bertscore/experiment_results_sci_v3_with_bertscore.csv`
- `revision/artifacts/2_1_bertscore/table7_bertscore_summary.csv`
- `revision/artifacts/2_2_cohen_h/cohen_h_vs_verimind.csv`
- `revision/artifacts/2_2_cohen_h/cohen_h_vs_verimind.txt`
- `revision/artifacts/2_3_threshold/threshold_sensitivity_curve.csv`
- `revision/artifacts/2_3_threshold/threshold_sensitivity_summary.txt`
- `revision/artifacts/2_3_threshold/threshold_sensitivity_curve.svg`
- `revision/artifacts/3_1_annotation/annotation_audit_summary.csv`
- `revision/artifacts/3_1_annotation/annotation_audit_summary.txt`

---

## 6. 资源与成本（当前估算）
- GLM-4.5-Air：主力承担评判与部分生成
- GLM-4.7：仅 E5 小规模高质量验证
- Qwen-Plus / DeepSeek：按量付费，优先控制在论文关键实验范围

---

## 7. 固定执行规则
- 每完成一个小任务，先同步三文档再进入下一步
- 所有代码修改先给预览，再落盘
- 长任务默认前台可见输出；仅在明确要求时启用后台
- 论文证据按任务编号持续追加到 `revision/paper_revision_evidence.md`

---

## 8. 附加工具与流程（已落地）

### 8.1 实验运行工具（最终版）
- [x] 默认前台实时输出（可见进度）
- [x] 显式后台模式（按需启用，支持状态/日志/停止）
- [x] 统一入口：`revision/scripts/start_job.ps1` + `revision/scripts/shortcuts.ps1`

### 8.2 论文证据沉淀流程
- [x] 新增 `revision/paper_revision_evidence.md`
- [x] 已回填 1.1.3 / 1.2.1 / 1.2.2 / 1.2.3 的论文可用证据
- [x] 新增 `.agent/rules/paper_evidence_workflow.md` 固化流程

---

*最后更新：2026-04-15*


## 2026-09-27：实习项目证据补齐（进行中）

- [x] 审计历史 QASPER 300 问题与 SciFact 500 条目（454 条不同 claim 文本）。
- [x] 指标行为测试 12 项通过；QASPER 固定/动态 top-5 导出完成。
- [x] 三数据组端到端 smoke 验证。
- [x] 完成 QASPER 300、SciFact 454 去重 claim、独立 SciFact NEI 60 的配对生成与独立评估。
- [x] 汇总五策略、检索/质量/拒答/覆盖率/阶段计时，核对原始记录。
- 入口：`revision/experiments/supplement_20260927.py`；seed=20260927。
- 用户已要求补齐全部测试后按十项表格统一交付；新产物独立保存，历史结果保留。


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
