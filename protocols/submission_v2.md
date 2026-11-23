**投稿改进研究协议草案 v2 — 2026-10-29**

状态：P0 草案，未冻结。机器可读配置在 [protocol_draft.json](../configs/submission_v2/protocol_draft.json)。本文件规定后续研究，不表示这些实验已经运行。P0 的重新拟合仍按旧协议进行，仅为复现旧表。

**目标与事件**

核心问题为内部表示相对可观察历史的预测增量，以及事实 QA 到 agent step 的迁移边界。主结果不以获得显著正效应为验收标准。

当前步骤标签 y_t 沿用旧定义作为 primary operational label：首次检索 supporting document；Finish 的 token F1≥0.5；invalid 为 0。宽松重复检索、Finish F1≥0.8 和 exact match 是事先计划的敏感性条件。标签审计通过前，不将其等同于推理正确性或因果贡献。

最终成功 Y 为冻结 agent policy 下最终答案正确；未回答为失败。所有步骤正确的联合事件与 Y 不同，product/min 仅作启发式。当前步骤分数不得改名为最终成功概率。

**数据角色与标签访问**

已有两模型共用相同的 600 train / 300 legacy test 问题。原 test 已用于分析，在新研究中标为 legacy development。保留原数据字段用于旧实验复现；新配置显式映射角色，不修改旧文件。

实际 ID 清单位于 [split_manifest.json](../manifests/submission_v2/p0/split_manifest.json)。两模型 TriviaQA 使用同一确定性规则划分 1600 source-train / 400 source-validation：对 `p0-source-split-v1:` 与 ID 拼接字符串取 SHA-256 后排序。该规则不看任何标签或方法效果。

新 Hotpot calibration 暂定 200，final test 暂定 600，至多规划 1000；具体数量按旧开发集上的精度模拟，在新测试结果解封前固定。当前没有选择新问题、读取新测试结果或生成新轨迹。新问题必须排除全部旧 900 IDs；source/target 需检查问题文本重叠。P0 只做规范化精确文本筛查，近重复筛查仍待实施。

strict zero-shot 的 pooling、layer、训练、概率映射都只使用源域标签。测试标签只交给独立 scorer。不得沿用 RQ1 的 agent-label-informed 层选择。旧设定保留为 target-informed reproduction，不列为严格 zero-shot。

few-shot 预算为 25/50/100/200 个 target step labels，默认每个 episode 最多一条；每预算 20 次配对抽样。选择、拟合、校准都计入总预算；不能额外借用全局 200-question calibration。抽到单类的样本按预设规则回退并报告，不能通过无限重抽隐式查看更多标签。

**信息可用性**

| 时点 | 可以使用 | 不得使用 |
| --- | --- | --- |
| pre-step | question、既往动作/observation、位置、既往行为统计、该前缀状态 | 当前 thought/action/type、当前检索结果、gold、未来 |
| pre-action | pre-step 输入 + 已生成 thought | 还未生成的 action/type、当前执行结果 |
| action-generated | 已生成 action 与其 hidden/log-prob、此前历史 | 当前执行后的结果、gold、未来 |
| post-observation（单列） | 实际已返回的工具结果 | gold correctness/supporting labels、未来 |

BM25/resolve 若需要实际检索，必须计入工具成本或放在相应时点，不能作为所有时点的免费输入。sampling consistency 是本步生成后信息，不能放入 pre-step 的公平 baseline。

P1 对两模型各至少 30 个分层旧 episode 做 prefix-only 检查；比较独立 tokenize 的 token IDs、span、隐藏状态与重复运行数值噪声。结构完整和高 cosine 不等于因果读取有效。

**主对比与控制**

主对比是 legal-history + legal-output + hidden-state score 相对 legal-history + legal-output 的增量，按 pre-step/action-generated 分开。历史基线含可用时点的位置、重复检索、文档计数、动作类别、长度、检索匹配信息。完整 feature list 在 P1 冻结，确保不含 gold。

保留与 consistency/verbalized/log-prob 的旧对比，但不是唯一证据。增加一个合理的浅层非线性历史基线、完整重训的标签置乱控制以及最近邻 probe 方法的适配对照。最后一项在方法细节核查后确定，不能只凭论文标题声称完成复现。

Source-only selection、零标签迁移、有限预算校准、同预算 in-domain probe 均要分别记录监督假设；报告按动作类别和位置分层结果，检验迁移是否仅由类别/位置解释。

**验证与 RQ3**

按 question/环境实例做 outer 5-fold 验证。所有 probe scaling、PCA、层选择及 calibration 在相应外层训练内完成；元模型使用真正 cross-fitted scores。按 step 随机划分不允许。部署预测不读取真实 previous error，oracle 条件分析如保留须单列。

M0/M1 加入相同的位置/可观察历史控制，再研究 previous-error 的额外信息。forward recursion 必须交代其条件模型假设：分数本身含历史，并不自动构成完整的 Bayesian filtering。终局比较包括 M0 last、M1 forward 和直接 prefix-success predictor。

pointwise PSIS-LOO/SR 不再是可部署收益的主证据。held-out Brier/log loss 和 episode 配对差异是主要概率评价；序列 log score 如提供，说明条件/边际/联合定义。报告 R-hat、ESS、divergence 及每次拟合的状态。关联系数不得解释为因果传播。

**统计与决策规则**

主要 CI 为 95% paired episode bootstrap，默认 2000 replicates。同一题的多次 rollout 一起采样；两个模型的同题比较使用同一 index。明确区分测试抽样 CI、训练 seed 方差和 few-shot 抽样 SD。

主表同时给 AUROC、proper score；error AUPRC 使用正确方向并报告 error prevalence。ECE 默认 10 等宽 bins；5/15 bins、等频分箱仅作敏感性分析。点估计来自原样本，bootstrap mean 和 CI 分字段存储。多项主假设的数量、调整方法及实用效应阈值在冻结时写入，不按最终显著性选择。

early-abort 主约束为 calibration 上选择约 95% successful-episode retention，90% 为次要分析；这是经验目标，不是有限样本保证。阈值固定后报告 test 实际 retention、coverage、kept accuracy、token/tool/time 成本与 CI。所有 timeout 纳入分母，超时 deferral 与早期预警分开。

离线回放只支持停止型策略，未来 success 仅作事后标签。重规划、替代动作、人类接管效果需要另外运行，不能从原轨迹反事实推得。

**第二环境与标签审计**

第二环境候选是 ALFWorld 纯文本，pilot 尚未执行。默认用于不同交互结构下的 prefix-success / early-abort 分析。action validity 只作辅助标签，不与 Hotpot 检索 usefulness 混同。

人工审计从旧数据约 200 steps + 40 complete episodes 开始，覆盖位置、重复、答案阈值、恢复与超时；作者与独立标注者分别标注并保留分歧。第二位标注者未落实前，不写成已完成的双人审计。

**版本、资源与冻结条件**

旧 run 元数据未固定模型/tokenizer/dataset revision；旧权重可能已删除。用户补充的 `reproducibility/environment.json` 已提供数据 revision，下载后的文件 SHA-256、旧样本选择、步骤 observation/操作性标签与 TriviaQA 标签均核验一致，见 `manifests/submission_v2/p0/dataset_verification.json`。模型/tokenizer revision 仍待确认。恢复权重时记录明确 revision，重新生成的结果作为新实验，不声称逐位复现旧运行。新的 GPU pilot 在权重恢复前保留 pending，不用历史日志冒充新测速。

新结果只写 `results/submission_v2/`，旧结果保留。source hash、library versions、BLAS backend/线程数写入每次分析。当前严格复现容差保留，微小重新拟合差异通过线程敏感性诊断解释，不事后改宽容差来宣称逐项精确一致。

P0 实测：原生 16 线程设置使 342/342 项标量比较通过，最大绝对差约 1.11e-16；1 和 4 线程分别有 18 和 11 项超出原定容差，但全部保留三位小数一致。新分析暂定固定 16 线程并记录 backend；这不构成任意硬件上逐位复现的保证。

冻结前必须完成：F1–F6 处理方案；prefix-only 检查；标签 rubric；信息时点与基线清单；完整标签预算；样本量精度模拟；第二环境进入/退出决定；主要对比及效果阈值；模型/数据 revision；可执行成本记录。此后新增分析标为 exploratory，bug fixes 留变更记录。

**P0 之外的工作尚未启动**

本草案不表示已修复旧模型的统计设计。P1/P2 将实施实际改动。P0 优先证明已有材料可用、旧数字可追溯，并将阻塞及未确认事项具体化。
