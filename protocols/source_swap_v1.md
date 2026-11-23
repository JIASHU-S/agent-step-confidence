**源任务替换实验协议 v1 — 2026-11-23**

状态：**v1.1 已于 2026-11-23 冻结**（第 8 节；第 1–6 节为原始设计，第 7 节为 pilot 和修订依据，第 10 节为第一阶段结果）。**v1.2 已于 2026-11-24 冻结**（第 11 节；冻结清单 sha256 `4cabcfd8a567…`）。冻结后的修改另起版本并说明原因。

**1. 动机（来自探索性结果）**

[scripts/18_transfer_decomposition.py](../scripts/18_transfer_decomposition.py) 在 legacy test 上的探索结果（只作假说来源）：

| 目标对比（同步骤 AUROC） | Qwen 迁移 / 同域 | Llama 迁移 / 同域 |
| --- | ---: | ---: |
| Finish | 0.705 / 0.709 | 0.691 / 0.679 |
| Search 正确 vs 重复 | 0.741 / 0.813 | 0.828 / 0.936 |
| Search 正确 vs 新的非支撑段落（distractor） | 0.638 / 0.785 | 0.587 / 0.853 |

迁移 = TriviaQA 探针在同域读出层；同域探针不依赖 TriviaQA 方向（INLP 移除后变化 ≤0.008），各动作的正确性方向近似正交。

**研究问题**：QA→agent 迁移覆盖哪类步骤错误，是由源标签度量的内容（content）决定，还是由提示格式（format）决定？

**2. 设计：内容 × 格式 2×2 源任务**

| 源 | 内容：标签度量什么 | 格式 | 数据 | 标签 |
| --- | --- | --- | --- | --- |
| S0（已有） | 答案知识：闭卷能否答对 | QA 短答 | TriviaQA 2000 | 别名匹配 |
| C1 | 答案知识 | ReAct，生成 `Finish[...]`，无文档 | TriviaQA 同 2000 题 | 别名匹配 |
| S2 | 查找选择：下一步检索是否取回新的支撑段落 | ReAct，生成 `Search[...]` | 2WikiMultihopQA 约 4000 题 | 与 env.py 同一规则 |
| S2-QA | 查找选择 | QA 短答："应先查哪篇文档？"回答一个标题 | 2Wiki，与 S2 不重叠的同等数量题 | 解析到的标题是否为新的支撑段落 |

- 全部为模型**自己生成**的单步决策及其正误，与 S0、目标域一致；不用 teacher-forced 的金标准/干扰标题作正负例。
- S2 / S2-QA 各半为第一跳与第二跳决策。第二跳的历史用模板构造：`Thought 1: I should look up <gold title 1>.`、`Action 1: Search[<gold title 1>]`、`Observation 1: <该段落>`，然后由模型生成第 2 步。重复检索在第二跳可能自然出现，单独记录，不作主分析。
- 读出与目标域一致：ReAct 源取动作片段 mean / last；QA 源取答案片段 mean / last；层与 agent 运行相同。
- 选 2Wiki 而不用 HotpotQA，以保持源域与目标数据集分离；MuSiQue 保留作确认目标。MuSiQue 部分由 SQuAD 等单跳数据组成，所以源任务不用 SQuAD。2Wiki 使用前固定 revision、核对许可，并对 HotpotQA 900 题和未来 MuSiQue 测试题做规范化文本与近重复筛查。

**3. 源域内部的处理（零目标标签）**

1. 每个源按 `sha256("source-swap-v1:" + id)` 排序，划分 80% source-train / 20% source-validation。
2. **样本量与基率对齐**：S2 第一跳的正确率预计约 84%（参照 Hotpot 第一步检索 0.845 / 0.836）。主分析中每个源用同一确定性规则抽取平衡子样本：正负例各 m 个，m = 所有源中负例最少者，至少 300。全量数据作敏感性分析。
3. 每个源独立地用 source-validation 的 OOF AUROC 选 pooling 和层，再在该源全部训练样本上拟合；不读取任何 agent 标签。完整层曲线作描述性结果。
4. 探针结构与主分析相同：标准化 → PCA 64 → logistic。

**4. 预先登记的假说与判定**

主要指标：目标域**同步骤 AUROC**（只比较 t 相同的正确/错误步骤对），按题 bootstrap 2000 次（seed 0），源间差值使用同一组重采样。

- **H1（主要，交互效应）**：在格式相同的一对源（S2 vs C1）上，
  [AUROC_distractor(S2) − AUROC_distractor(C1)] − [AUROC_finish(S2) − AUROC_finish(C1)] > 0。
  判定：两个模型的 95% 区间都在 0 以上才算成立（两个模型同时成立，不另作多重校正）。
- **H2（次要）**：AUROC_distractor(S2) > AUROC_distractor(S0)，且 AUROC_finish(S0) ≥ AUROC_finish(S2) − 0.05。
- **H3（格式，估计而非检验）**：报告 C1 − S0、S2 − S2-QA 在各对比上的差值及区间。Finish 负例只有 104 / 59 个，样本量不足以建立等效，只报告估计值，不宣称"格式无关"。
- **探索性**：重复检索对比；层曲线；多源合并探针（S0+C1+S2）与同域探针的差距；S2 第一跳与第二跳分开报告。

**结果解读规则（事先写定）**：
- H1 成立：迁移的范围由源标签的内容决定。
- S2 和 C1 在两类对比上都相近，而 S2-QA 明显更差：格式或近任务效应占主导，不能作内容解释。
- S2 在 distractor 上也不提升：说明 distractor 的信号依赖多步历史，单步查找标签学不到。这一点作为有界的负结果报告，后续实验改用更长构造历史的源。

**5. 数据角色与确认**

- **第一阶段**：在 legacy test 上评估。目标数据已被分析过，但源任务、预测和判定都在采集前写定，所以结果作为对假说的第一次检验，仍标为开发阶段。
- **第二阶段（确认）**：在新 Hotpot final test 和 MuSiQue 封存测试上，用冻结的源探针和本节判定规则一次性评估，不再改源、改层或改判定。

**6. 计算与实现**

- 速率参照：TriviaQA 在 4070 Ti 上约 0.3–0.4 s/题。ReAct 源的提示更长，第二跳还带段落，实际速率在 pilot 中测量。每个源每个模型预计在若干 GPU 小时以内，4070 Ti 或 Colab T4 都可以。两个模型共约 4 个新源 × 2 个模型。
- 同一个模型 × 源的数据在同一种 GPU 上采完，记录 GPU 型号、库版本、模型和数据 revision。
- 代码：新增 `scripts/19_source_features.py`（在 02 的基础上加 `--source {trivia_react, wiki_search, wiki_search_qa}`，沿用逐题检查点和断点续跑）；新增 `scripts/20_source_swap.py`（复用 `agentconf/errtypes.py` 和 18 的 bootstrap 表）。S2 的标签直接调用 `HotpotEnv` 并把 2Wiki 转成同一 schema，用单元测试覆盖。
- Pilot：每个源 50 题，检查解析失败率、基率、负例数是否够用以及速率。pilot 数据不进入主分析。

**7. Pilot 结果（Qwen2.5-7B，2026-11-23）与修订提案**

先做权重核验：固定 revision `a09a3545…` 重算 30 个旧 agent 步骤，最小余弦 0.9999999，log-prob 差 0，通过。

| 源 | n | 正确率 | 负例 | 负例构成 | s/题 |
| --- | ---: | ---: | ---: | --- | ---: |
| C1 trivia_react | 50 | 0.46 | 27 | 闭卷答错 | 1.21 |
| S2 wiki_search | 100 | 0.95 | 5 | Search 只有 2 个（1 个 distractor、1 个重复）；3 个过早 Finish | 2.95 |
| S2-QA wiki_search_qa | 100 | 0.87 | 13 | 全部是 distractor；hop 1 正确率 0.80，hop 2 为 0.95 | 0.52 |

- **标签修正**：2Wiki 有 8.7% 的支撑标题带非 ASCII 字符，模型常常省略重音，HotpotEnv 会因此判为 no-match。HotpotQA 目标域中 no-match 每个模型只有 1 次。所以源环境改为不区分变音符号的 `FoldedEnv`，其余规则不变。修正前的 50 题 pilot 保存在 `*-pilot-v0-unfolded`。
- **S2 不可行**：ReAct 格式下，2Wiki 单步查找的 Search 错误率约为 2%（2/96）。而 HotpotQA 目标第一步的 distractor 率是 15–16%。即使用完 dev 中全部可用的 6,188 题，Search 负例预计只有约 130 个，低于 m≥300。2Wiki 的问题是模板生成的，实体名直接写在问题里，带 Thought 的 ReAct 几乎不会选错。要达到 300 个负例需约 15,000 题（需要 train 划分，约每模型 12 GPU 小时），而且负例仍会以过早 Finish 和重复为主，不是要操纵的 distractor 内容。
- **S2-QA 可行**：负例 13%，全部是 distractor；hop 1 的错误率（20%）接近目标第一步。按 6,188 题估计约 800 个负例，耗时约 54 分钟/模型。

**修订提案（待确认，确认后才算冻结）**：主假说改用 QA 格式下格式一致的一对源，S0（答案知识）vs S2-QA（查找选择）：
H1′：[AUROC_distractor(S2-QA) − AUROC_distractor(S0)] − [AUROC_finish(S2-QA) − AUROC_finish(S0)] > 0，判定规则同 H1。C1 − S0 继续估计答案内容上的格式效应。S2 从主分析移除；"ReAct 下单步查找几乎不出错"作为描述性结果报告，用来说明目标域的 distractor 错误更多来自 HotpotQA 问题的间接指代和多步过程，而不是单步查找本身。

**8. 冻结：v1.1（2026-11-23）**

第 7 节的修订已经确认。以本节为准；第 4 节中与本节冲突的部分（H1、H2 中的 S2）作废。冻结时还没有采集任何主数据，也没有用新源计算任何目标域指标。代码尚未提交，提交后补记提交号。

- **源**：S0（`runs/<m>/trivia.jsonl` 与 `trivia_feats.npz`，2000 题）；C1 `trivia_react-main`（与 S0 同 2000 题）；S2-QA `wiki_search_qa-main`（4000 题，从 pilot 保留区之后的确定性顺序中选取）。S2 不采主数据。
- **平衡**：m = 三个源各自 min(正例数, 负例数) 中的最小值。每个源按 `sha256("source-swap-v1-balance:" + id)` 排序，取前 m 个正例和前 m 个负例。若 m < 300，在查看任何目标指标之前，先按同一顺序增加 S2-QA 的题数。
- **源内划分与选层**：平衡样本按 `sha256("source-swap-v1:" + id)` 排序，前 80% 作 source-train，后 20% 作 source-val。在 {mean, last} × 8 层上，用 source-train 拟合（标准化 → PCA 64 → logistic），以 source-val 的 AUROC 选读出；再用该读出在全部 2m 个样本上重新拟合。全过程不读取 agent 标签。
- **目标评分**：legacy test 的 agent 步骤；同步骤 AUROC（只比较 t 相同的正确/错误步骤对）；按题 bootstrap 2000 次（seed 0），所有源共用同一组重采样。
- **H1′（主要）**：I = [A_distractor(S2-QA) − A_distractor(S0)] − [A_finish(S2-QA) − A_finish(S0)]。其中 distractor 指"正确 Search vs 新的非支撑段落"。两个模型的 I 的 95% 区间都在 0 以上，才算成立。
- **H2′（次要）**：A_distractor(S2-QA) − A_distractor(S0) 的区间在 0 以上，且 A_finish(S0) − A_finish(S2-QA) 的区间下界大于 −0.05。
- **格式估计**：C1 − S0 在每个对比上的差值及区间，只作估计。
- **探索性**：重复检索对比；pooled AUROC；源选读出以外各层在目标上的曲线（描述性）。
- **报告顺序**：先报告 Qwen。Llama 需要用户完成 gated 授权后按同一流程采集；H1′ 的判定要等两个模型都完成后才做出。

**9. 待决定**

- m 的最低值 300 是否足够：以 pilot 的基率为准；若不够，增加 S2 的题数，而不是放宽平衡规则。
- 是否加入开卷阅读理解源 S1（给定 2Wiki 10 段落作答）：它与 Finish 的内容更接近，但不在 2×2 主设计中，默认作为可选的探索性分析。

**10. 第一阶段结果（legacy test，2026-11-24）**

按第 8 节执行，未改任何规则。两个模型的权重核验都通过（Llama 固定 revision `0e9e39f2…`，最小余弦 0.9999999，log-prob 差 0）。输出在 `results/<m>/source_swap_*`。

| | Qwen2.5-7B | Llama-3.1-8B |
| --- | --- | --- |
| m | 446 | 693 |
| 源选读出 S0 / C1 / S2-QA | mean L20 / last L20 / mean L16 | last L14 / mean L14 / last L14 |
| A_finish S0 / S2-QA | 0.718 / 0.518 | 0.658 / 0.546 |
| A_distractor S0 / S2-QA | 0.604 / 0.755 | 0.717 / 0.684 |
| **I**（95% 区间） | **+0.351 [0.224, 0.480]** | **+0.080 [−0.044, 0.208]** |
| H2′ distractor：S2-QA − S0 | +0.151 [0.053, 0.245] | −0.032 [−0.090, 0.021] |
| H2′ finish：S0 − S2-QA | +0.200 [0.103, 0.298] | +0.112 [0.015, 0.221] |
| 格式 C1 − S0（finish / distractor） | −0.101 / +0.031 | −0.096 / −0.113 |

**判定：H1′ 不成立**（Llama 的区间包含 0），H2′ 也只在 Qwen 上成立。按预先写定的规则，不能宣称"迁移范围由源标签内容决定"。

**探索性（事后，不作判定）**：源选读出以外的层曲线显示，两个模型的上半层（Qwen 16–28、Llama 18–32，mean/last 共 8 个读出）上，I 在 16 个读出中全部为正（Qwen 0.01–0.31，Llama 0.14–0.27；上半层平均 0.16 / 0.20），S2-QA 在 distractor 上平均高于 S0 约 0.15 / 0.13。Llama 的主分析结果主要来自三个源都在 source-val 上选中了 L14，而 S0 在该层的 distractor AUROC 是局部高点。这说明按源内验证集选层与目标迁移不一致；该模式只能在新数据上用事先写定的、对层稳健的指标确认。

**11. v1.2：对层稳健的确认分析（2026-11-24 冻结）**

**为什么要改**：第 10 节显示 v1.1 的结论取决于读出选择。按 source-val 选层得到的层与目标迁移最好的层不一致：Llama 三个源都选中 L14，I 的区间包含 0；而在上半层的 16 个读出上，I 全部为正。v1.2 改用事先固定的一组读出，不再按单个读出作判定。**这一规则是看过第一阶段层曲线之后定的**，所以第一阶段的数字（11.6）只作参考和功效估计，不算证据；检验只在尚未生成的确认目标上做一次。

另外更正一处方向：v1 / v1.1 的 H2 finish 条件检查的是 `A_fin(S0) − A_fin(S2-QA)` 的下界大于 −0.05，即 S2-QA 在 Finish 上没有大幅领先 S0；它不能支持"换成检索源后 Finish 损失小于 0.05"。v1.1 的结果仍按其原规则报告（第 10 节），v1.2 改为下面 11.4 的方向。

**11.1 固定内容（与 v1.1 相同，不重新采集）**

- 源数据：两个模型的 S0、C1、S2-QA 主数据（第 8 节）。平衡（Qwen m = 446，Llama m = 693）、标准化 → PCA 64 → logistic、在全部 2m 个样本上拟合，都与 v1.1 相同。
- 冻结清单：`python scripts/22_confirm_source_swap.py --freeze` 写入 [v1.2_frozen_inputs.json](../manifests/source_swap/v1.2_frozen_inputs.json)：源数据文件、分析与采集代码（scripts 01/20/21/22 及其 agentconf 依赖）、题目清单的 sha256，以及**模型集合（Qwen、Llama）及各自的 final 运行目录**、revision 和存储层。已有任何确认目标轨迹时，脚本拒绝冻结。确认时逐项重算，任一不一致即不评分。

**11.2 读出规则**

- 每个源使用同一组 8 个读出：存储层的上半部分 × {mean, last}，即 Qwen L16/20/24/28、Llama L18/23/27/32。不在源内选层。
- 每个读出单独拟合源探针，计算目标同步骤 AUROC；**A 为 8 个读出 AUROC 的等权平均**。bootstrap 在每次重采样内取同样的平均。

**11.3 确认目标与数据角色**

- **题目清单（已生成，[heldout_manifest.json](../manifests/heldout_v1/heldout_manifest.json)，sha256 `0a782a75…`）**：由 [scripts/21_select_heldout.py](../scripts/21_select_heldout.py) 按 [agentconf/heldout.py](../agentconf/heldout.py) 的规则一次性生成，只读取题号和题目文本。
  - 题池：固定版本的 HotpotQA distractor validation（revision `1908d6af…`，7,405 题，sha256 已核对；与旧 900 题的题目和答案逐一一致）。
  - 排除：旧 900 个 ID（两个模型相同）；题池内规范化文本重复；与参照题规范化文本完全相同；与任一参照题 TF-IDF（规范化文本，词 1–2 gram）余弦 ≥ 0.8。参照题为旧 900 题、TriviaQA validation 全部 9,960 题、2Wiki dev 全部 12,576 题。
  - 结果：排除旧 ID 900 题，文本重复与近重复均为 0；入选 800 题与参照题的最大余弦为 0.48。`python scripts/21_select_heldout.py --verify` 可从原始文件重新推导并与清单比对（已核对一致）。按 `sha256("heldout-v1:" + id)` 排序，前 200 题为 calibration（submission_v2 使用），后 600 题为 **final**。
- **采集**：两个模型都用 `scripts/01_run_agent.py --ids-file manifests/heldout_v1/heldout_manifest.json --splits final --revision <冻结 revision>`，其余设置与旧运行相同（max_steps 6、k_samples 5、temperature 0.7、自动 8 层）。脚本核对题池文件 sha256，并把清单 sha256、split 和 revision 写入 meta，断点续跑时逐项核对。预计每个模型约 9–11 小时（旧运行 52–64 秒/题），在同一张 4070 Ti 上完成。
- **盲态与评分**：采集期间只检查轨迹是否完整，不计算任何探针分数或 AUROC。两个模型都完成后，运行一次 `python scripts/22_confirm_source_swap.py`。它只接受冻结清单中的完整模型集合，不能用单个模型给出正式结论。评分前先对所有模型完成全部无标签检查：冻结的哈希；模型、revision、存储层、清单 sha256 和 split；episode 集合与 600 个 final 题号完全一致（无缺失、无多余）；每个 episode 的第 1…n 步都在且 split 正确；特征文件可读、数值有限、形状为 [n, 层数, d] 且 d 与源特征一致。全部通过后才评分；评分结果先写入暂存目录，所有模型都评分成功后才移入各运行目录并写出汇总。两个模型使用同一份题目清单和同一个 seed，所以各源之间、两个模型之间的 bootstrap 重采样都按题配对。
- **停止规则**：两个模型各完成同一份 600 题 final 清单后，在已保存的轨迹上联合评分一次。达到样本量即报告估计和区间，包括阴性和不确定结果。中断时在原目录恢复，沿用原清单、模型和判定规则。重复启动已完成的评分会返回原汇总的位置。
- **MuSiQue**：本轮结论限于 HotpotQA。若以后研究跨数据集泛化，MuSiQue 单独登记为 v1.3，先冻结环境、标签和题目清单，再生成轨迹。

**11.4 预先登记的检验**

所有 A 都是 11.2 的平均同步骤 AUROC；按题 bootstrap 2000 次（seed 0）；95% 百分位区间。

- **H1″（主要）**：I = [A_dis(S2-QA) − A_dis(S0)] − [A_fin(S2-QA) − A_fin(S0)]。final test 上两个模型的 I 的 95% 区间都在 0 以上，才算成立。只有这一个主要检验，不另作多重校正。
- **检索增益（次要）**：G = A_dis(S2-QA) − A_dis(S0)，区间下界 > 0。
- **Finish 代价（次要）**：C = A_fin(S2-QA) − A_fin(S0)。
  - "Finish 损失小于 0.05"：C 的区间下界 > −0.05；
  - "存在 Finish 损失"：C 的区间上界 < 0；
  - 两者都不满足时为不确定。
- **估计（不检验）**：格式效应 C1 − S0；重复检索对比；pooled AUROC；v1.1 规则（source-val 选单个读出）在 final test 上的 I、G、C，与 v1.2 的结果在同一次评分中计算并一起报告，用来说明结论对选层规则的依赖程度。
- 汇总文件对每个模型分别列出 I、G、C 的估计和区间，无论落在 11.5 的哪一行。

**11.5 结果与论文表述的对应（事先写定）**

由 `scripts/22_confirm_source_swap.py` 按下表自动判定，每个条件都要求两个模型同时满足：

| 结果 | 表述 | MuSiQue |
| --- | --- | --- |
| H1″ 成立，G > 0，Finish 损失 < 0.05 | 泛化：检索源在小 Finish 代价下扩展了对 distractor 错误的覆盖 | 要推广到 HotpotQA 以外时**必须**做（v1.3） |
| H1″ 成立，G > 0，存在 Finish 损失 | 这两个训练源表现出权衡，检索错误识别改善，Finish 错误识别下降 | 可选 |
| H1″ 成立，G > 0，Finish 不确定或两模型不一致 | 只声称检索增益，对 Finish 不下结论 | 可选 |
| H1″ 成立，G 未在两个模型都通过 | 交互效应已确认、跨模型检索增益未确认：分别报告 I、G 和 Finish 差值 C，不把交互归因于 distractor 一侧（例如正交互可能主要来自 Finish 下降） | 否 |
| H1″ 不成立，G > 0 | 报告检索增益估计；交互效应未确认 | 否 |
| H1″ 不成立，G 也未在两个模型都通过 | 第一阶段的层模式不能复现，作为负结果报告 | 否 |

**11.6 第一阶段参考值与功效（legacy test，已经看过，不是证据）**

| `--readout upper` | Qwen | Llama |
| --- | --- | --- |
| A_finish S0 / S2-QA | 0.625 / 0.619 | 0.679 / 0.612 |
| A_distractor S0 / S2-QA | 0.570 / 0.724 | 0.626 / 0.757 |
| I | +0.160 [0.087, 0.229] | +0.199 [0.127, 0.274] |
| G | +0.154 [0.097, 0.208] | +0.132 [0.085, 0.181] |
| C = A_fin(S2-QA) − A_fin(S0) | −0.006 [−0.056, 0.045]：不确定 | −0.068 [−0.129, −0.010]：存在损失 |

按 legacy 的数字，Qwen 的 Finish 代价不确定，Llama 的 Finish 代价区间低于零。若 final 重现这组判定，联合表述为“只声称检索增益”。

功效（按 1/√n 从 300 题换算到 600 题，假设错误率相近；读出规则是看过数据后定的，真实效应可能更小）：
- I 的标准误约 0.026。真实 I = 0.08 时，单个模型功效约 0.87，两个模型都成立约 0.75；I = 0.10 时约 0.97 和 0.94。
- C 的标准误约 0.018（Qwen）/ 0.021（Llama）。真实 C = 0 时，"损失 < 0.05"的功效约 0.78 / 0.64；真实 C 接近 Llama 的 −0.068 时几乎不会通过。

**11.7 冻结记录**

2026-11-24 冻结：运行 `python scripts/22_confirm_source_swap.py --freeze`，生成 [v1.2_frozen_inputs.json](../manifests/source_swap/v1.2_frozen_inputs.json)（sha256 `4cabcfd8a567…`）。冻结时还没有任何 final 轨迹；随后在同一张 RTX 4070 Ti 上依次开始 Qwen、Llama 的 final 采集。
