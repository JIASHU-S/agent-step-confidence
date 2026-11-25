**主结果确认协议 v1（MC1）— 2026-11-25**

状态：**草案已提交，未冻结。** 分析代码（`scripts/23_confirm_main.py`）与本文件一起提交；git 提交时间早于任何评分。冻结 = 在 Windows 机器上运行 `python scripts/23_confirm_main.py --freeze`，写入 `manifests/main_confirmation/v1_frozen_inputs.json`。冻结后的修改另起版本并说明原因。

**1. 目的**

论文 RQ1–RQ3 的数字全部来自 legacy test（300 题 × 2 模型）。这个集合被反复分析过，RQ1 的读出也是用 agent 训练标签选的。本协议在 held-out final 600 题上，用**冻结的训练集 probe 和基线**做一次确认评分，把论文主表改为 final 上的结果，legacy 结果降为开发集。

**2. 数据角色与已有暴露（如实记录）**

- 训练：legacy 运行 `runs/<m>` 的 train 划分（600 题，与论文相同），不使用 legacy test。
- 目标：`runs/<m>-final`（v1.2 采集的 600 题 final，清单 sha256 `0a782a75…`）。
- **已有暴露**：v1.2 源替换确认（`results/source_swap_confirmation.json`）已经在 final 标签上计算过三个源探针（S0/C1/S2-QA）按错误类型的同步骤 AUROC，并输出了各类步骤数。**在本协议提交前，final 上没有计算过任何同域 probe、一致性 / 口述置信度 / log-prob、可见文本基线、位置基线或轨迹层面的指标。** 本协议的分析全部是论文已有分析的原样复用，读出和基线都事先固定，几乎没有研究者自由度。论文中照此说明。

**3. 固定内容**

| | Qwen2.5-7B | Llama-3.1-8B |
| --- | --- | --- |
| action-token probe 读出（`probe_choice.json`） | mean L20 | last L23 |
| pre-step / pre-action 读出（`preaction.csv`） | L20 / L20 | L18 / L18 |
| TriviaQA probe，源内选读出（`review_checks.json`） | last L20 | last L14 |
| TriviaQA probe，agent 标签选读出（= action probe 读出） | mean L20 | last L23 |

- 脚本在 legacy train 上用原规则（GroupKFold OOF AUROC，{mean,last}×8 层）**重新选择**一次读出，与上表不一致则拒绝评分（只检查一致性，不改表）。
- probe：标准化 → PCA 64 → logistic，在 legacy train 全部步骤上拟合一次，直接给 final 步骤打分。
- 基线（全部只在 legacy train 上拟合）：
  - 一致性、口述置信度、log-prob mean/min：原始分数；概率指标先做 Platt 校准。
  - 可见文本 + 历史：`scripts/16_review_checks.py` 的 14 个特征，标准化 + logistic。
  - 位置：步号 one-hot（1–6），以及步号 + 动作类型 one-hot，logistic（C=1e4）。
- RQ3：M0/M1 在 legacy train 的 OOF probe 分数上拟合（与论文相同：NUTS 4 链 × 1000，seed 0/1），用于 final 上的轨迹置信度；另在 final 步骤（冻结 probe 分数）上重新拟合 M0/M1，用于估计耦合系数 g。
- 统计：final 600 题按题 bootstrap 2000 次（seed 0），同一模型内所有信号共用一组重采样；百分位区间。点估计来自原样本。

**4. 预先登记的检验**

**主要（3 项，每项都要求两个模型同时成立；3 项之间做 Bonferroni，即看 98.33% 区间下界 > 0）：**

- **P1**：AUROC(action probe) − AUROC(一致性) > 0。
- **P2**：AUROC(action probe) − AUROC(可见文本 + 历史) > 0。
- **P3**：AUROC(pre-step probe) − AUROC(步号) > 0。（需要先在 final 上运行 07 回放，见第 6 节。）

**次要（95% 区间，两个模型都满足才算成立，无论结果都报告）：**

- **S1**：Search 步骤上 action probe − 一致性 > 0。
- **S2**：同题配对 AUROC（只比较同一轨迹内一对正确 / 错误步骤）上 action probe − 一致性 > 0。
- **S3**：action probe − 步号 + 动作类型 > 0。
- **S4**：轨迹 ECE(M0 product) − ECE(M1 forward) > 0（连乘低估的修复）。
- **S5**：在 final 步骤上拟合的 M1 中，g 的 90% 后验区间上界 < 0。

**只估计（不检验，报告点估计和 95% 区间）：**

- Finish 步骤 action probe − 一致性；pre-step probe − 一致性；pre-action probe 的各项差值。
- 口述置信度、log-prob 与 probe 的差值；各信号 ECE / Brier。
- 每个步号内的 AUROC（probe、一致性）。
- 宽松标签（重复检索不算错）下 probe − 一致性。
- RQ2：TriviaQA probe 在两种读出下的零样本 AUROC 和原始 ECE；用 legacy train 中 50 步做 Platt（20 次抽样，seed 0）后的 ECE；同样 50 步训练的同域 probe 的 AUROC。
- RQ3：M1 forward 与 M0 last 的 AUROC / ECE / Brier 差值；hand-off 风险–覆盖曲线下面积差值（M1 running vs M0 running product）；同时报告"未完成的轨迹记为失败、置信度取最后一步"的版本。
- final 上 M0 vs M1 的 PSIS-LOO（ΔELPD、SE、Pareto k）。

**不做**：不在 final 上重新选层、不换基线、不加新特征。任何新增分析标为探索性。

**5. 结果与论文表述（事先写定）**

| 结果 | 论文表述 |
| --- | --- |
| P1、P2、P3 全部成立 | 主表改用 final 数字；摘要的三条 RQ1 结论保留 |
| P1 成立，P2 不成立 | 删除"probe 读到可见文本以外的信息"这一主张，只保留优于输出类信号 |
| P3 不成立 | 删除"步骤生成前即可检测"的主张，pre-step 结果移到附录作描述 |
| P1 不成立 | RQ1 主结论不复现；作为负结果报告，论文重心转到 RQ2 / RQ3 |
| S4 成立 | 保留"forward recursion 修复连乘的低估" |
| S5 成立 | 保留"错误耦合"的描述；不成立则改为"耦合只在 legacy 上观察到" |

forward 与 last 的差值无论落在哪里都只作估计；不声称两者等价，也不声称 forward 更好，除非区间整体在 0 的一侧，此时按方向如实描述。

**6. 执行顺序（Windows，GPU 只用于第 2 步）**

1. 本文件与 `scripts/23_confirm_main.py` 已提交到 git（时间戳即预登记时间）。
2. 对两个 final 运行做 pre-action 回放（不读标签）：
   `python scripts/07_preaction_features.py --run runs/qwen7b-final`；Llama 同理。07 现在会从 meta 的 `ids_file` 读取固定版本的 HotpotQA，并使用记录的模型 revision。
3. `python scripts/23_confirm_main.py --freeze`：记录代码、legacy 训练输入、final 目标输入的 sha256。已有确认输出时拒绝冻结。
4. `python scripts/23_confirm_main.py`：逐项核对哈希和无标签完整性检查（与 v1.2 相同：600 题齐全、步号连续、特征有限且形状正确、feats_pre 齐全），全部通过后评分一次，写 `results/main_confirmation.json` 和 `results/<m>-final/main_*.csv`。再次运行只返回已有结果的位置。

**7. 功效（按 legacy 的效应和 1/√n 换算到 600 题）**

legacy 上 P1/P2/P3 的差值为 0.09–0.20，95% 区间半宽约 0.03；600 题时半宽约 0.024，98.33% 区间半宽约 0.029。若真实效应为 legacy 的一半（约 0.05），单项单模型功效仍约 0.9。S5 在 legacy 上 g ≈ −0.6，后验 90% 区间半宽约 0.25，600 题时约 0.18。
