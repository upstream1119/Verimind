# 2026-09-27 离线 Gate 诊断

本报告只回放冻结的 multi/no-gate 原始答案，不调用模型、不重新生成答案。QASPER n=300；SciFact n=514，其中454个有证据 claim 加60个NEI。

## 分数区分能力

| 数据集 | 分数 | 正例定义 | n / 正例 | AUROC | AUPRC(AP) | 正例率 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| QASPER | retrieval_score | answerable | 300 / 240 | 0.592 | 0.859 | 80.0% |
| QASPER | judge_score | supported | 144 / 130 | 0.500 | 0.903 | 90.3% |
| QASPER | trust_score | safe | 144 / 91 | 0.532 | 0.638 | 63.2% |
| SciFact+NEI | retrieval_score | answerable | 514 / 454 | 0.768 | 0.959 | 88.3% |
| SciFact+NEI | judge_score | supported | 450 / 414 | 0.634 | 0.942 | 92.0% |
| SciFact+NEI | trust_score | safe | 450 / 374 | 0.757 | 0.929 | 83.1% |

AUROC 对并列分数按半胜处理；AUPRC 为按同分数组计算的 Average Precision。judge/TrustScore 仅在生成器实际给出候选答案的样本上评估，主动拒答的默认 judge=10 不进入这些 AUROC。safe = 可回答、独立Qwen判为supported、且答案正确；QASPER正确性来自Qwen，SciFact来自gold verdict。AUPRC必须与正例率一起读，不能单看数值大小。

### 相同 safe 正例下的三分数比较

| 数据集 | Retrieval AUROC/AP | Judge AUROC/AP | Trust AUROC/AP | n |
| --- | ---: | ---: | ---: | ---: |
| QASPER | 0.432/0.592 | 0.608/0.692 | 0.532/0.638 | 144 |
| SciFact+NEI | 0.616/0.877 | 0.737/0.909 | 0.757/0.929 | 450 |

## 当前 retrieval gate 是否触发

| 数据集 | 全部/候选 | 分数低于0.5（全部） | 实际新增拦截 | 与no-gate最终决策不同 |
| --- | ---: | ---: | ---: | ---: |
| QASPER | 300/144 | 4 | 0 | 0 |
| SciFact+NEI | 514/450 | 0 | 0 | 0 |

QASPER四条分数低于0.5的输入已被生成器主动拒答；SciFact+NEI没有分数低于0.5的输入。极端输入和阈值两侧的分支断言通过；retrieval P10–P90扫描的coverage单调下降，详见retrieval_quantiles.csv。因此相同结果由当前阈值下没有额外拦截解释，并非已证实的门控代码故障。

## TrustScore 0.1–0.9 阈值扫描

Coverage=最终回答数/全部输入；Selective Accuracy=已回答且正确/已回答；Unsupported=已回答中Qwen判无支持的比例。拒答P/R/F1以QASPER不可回答或SciFact NEI为正类，包含生成器主动拒答。相同原始答案回放，不代表新模型策略的前瞻性验证。

| 数据集 | 阈值 | Coverage | Selective Accuracy | Unsupported | 拒答P | 拒答R | 拒答F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QASPER | 0.1 | 48.0% | 63.2% | 9.7% | 32.7% | 85.0% | 47.2% |
| QASPER | 0.2 | 48.0% | 63.2% | 9.7% | 32.7% | 85.0% | 47.2% |
| QASPER | 0.3 | 44.7% | 65.7% | 9.7% | 32.5% | 90.0% | 47.8% |
| QASPER | 0.4 | 43.7% | 64.9% | 9.9% | 32.0% | 90.0% | 47.2% |
| QASPER | 0.5 | 43.7% | 64.9% | 9.9% | 32.0% | 90.0% | 47.2% |
| QASPER | 0.6 | 43.7% | 64.9% | 9.9% | 32.0% | 90.0% | 47.2% |
| QASPER | 0.7 | 42.0% | 66.7% | 9.5% | 31.0% | 90.0% | 46.2% |
| QASPER | 0.8 | 38.7% | 68.1% | 9.5% | 29.9% | 91.7% | 45.1% |
| QASPER | 0.9 | 17.7% | 66.0% | 15.1% | 23.1% | 95.0% | 37.1% |
| SciFact+NEI | 0.1 | 87.5% | 83.6% | 8.0% | 45.3% | 48.3% | 46.8% |
| SciFact+NEI | 0.2 | 87.5% | 83.6% | 8.0% | 45.3% | 48.3% | 46.8% |
| SciFact+NEI | 0.3 | 84.4% | 84.3% | 7.8% | 41.2% | 55.0% | 47.1% |
| SciFact+NEI | 0.4 | 56.2% | 93.1% | 5.5% | 23.6% | 88.3% | 37.2% |
| SciFact+NEI | 0.5 | 55.3% | 93.3% | 5.3% | 23.0% | 88.3% | 36.6% |
| SciFact+NEI | 0.6 | 55.3% | 93.3% | 5.3% | 23.0% | 88.3% | 36.6% |
| SciFact+NEI | 0.7 | 54.5% | 93.2% | 5.0% | 22.6% | 88.3% | 36.1% |
| SciFact+NEI | 0.8 | 52.5% | 93.3% | 5.2% | 22.1% | 90.0% | 35.5% |
| SciFact+NEI | 0.9 | 45.5% | 94.4% | 4.3% | 20.4% | 95.0% | 33.5% |

## 判断与边界

- QASPER：Judge 对supported的AUROC约0.500；Trust对safe约0.532；把阈值从0.1扫到0.9也未形成稳定的无支持风险下降。因此现有证据更指向分数区分力不足，单独调阈值难以解决。
- SciFact+NEI：Trust对safe的AUROC约0.757，说明有一定排序信息；但0.3到0.4时coverage从84.4%跳到56.2%，属于分数离散和校准不平滑。阈值位置影响很大，需在独立校准集选定后再到留出集验证。
- AUPRC正例高度不均衡：QASPER supported正例率90.3%，SciFact+NEI supported正例率92.0%；Judge AP高值不能解释为很强的辨别能力。
- 这些标签仍是自动评估/现有gold，不是新的人工作者盲审。本文档不将当前测试集上挑出的最佳阈值当作已验证改进。

原始输出：scores.csv、threshold_sweep.csv、gate_activation.csv、debug50.csv、distributions.csv、retrieval_quantiles.csv；验证记录：validation.json。
