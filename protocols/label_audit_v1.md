**步骤标签人工审计协议 v1 — 2026-11-25**

状态：草案。抽样在 Windows 上运行一次（`scripts/24_label_audit.py sample`），之后不重抽。标注说明见 [label_audit_guidelines.md](label_audit_guidelines.md)（英文，便于第二位标注者使用）。

**1. 目的**

论文的步骤标签是自动规则：Search 取回此前未取回的支撑段落才算对，Finish 的 token F1 ≥ 0.5 才算对，无法解析的动作算错。审稿人会问：这些标签与人对"这一步是否有用 / 是否答对"的判断一致到什么程度？重复检索算错、F1 阈值、非支撑但相关的段落，各自造成多少分歧？本审计给出按层（stratum）的一致率、总体加权一致率，以及两位标注者之间的 κ。

**2. 抽样（只用 legacy 开发运行，两个模型，不碰 final）**

- **完整轨迹**：每个模型 20 条，按"是否成功 × 步数 ≤2 / >2"分 4 组，每组 5 条；按 `sha256("label-audit-v1:ep:<model>:<qid>")` 排序选取。这些轨迹的全部步骤也进入步骤表。
- **分层步骤**：其余题目中，每个模型按下表抽取，每层每题最多一步，按 `sha256("label-audit-v1:pick:<model>:<qid>:<t>")` 排序：

| 层 | 定义 | 每模型 |
| --- | --- | ---: |
| search_correct | Search，取回新的支撑段落 | 25 |
| search_distractor | Search，取回新的非支撑段落 | 25 |
| search_repeat | Search，取回已取回过的段落 | 15 |
| finish_correct_exact | Finish，F1 = 1 | 7 |
| finish_correct_partial | Finish，0.5 ≤ F1 < 1 | 8 |
| finish_wrong_partial | Finish，0 < F1 < 0.5 | 8 |
| finish_wrong_zero | Finish，F1 = 0 | 7 |
| invalid | 无法解析 | 5 |

  合计每模型 100 个分层步骤，加上约 60–70 个轨迹步骤，两个模型约 330 个步骤、40 条轨迹。F1 边界附近和重复检索被有意过采样，因为分歧最可能出现在这里。

- **盲态**：标注表不显示模型、自动标签、F1 和支撑段落标记；条目按哈希打乱，两个模型交错。答案键单独写在 `key.json`，标注完成前不给标注者看。

**3. 人工判断与比较**

- 严格读法：`S-USEFUL`、`F-CORRECT` 为 1；`S-UNHELPFUL`、`S-REPEAT`、`F-WRONG`、`I-INVALID` 为 0；`S-PARTIAL`、`F-PARTIAL`、`F-GOLD-ISSUE`、`I-VALID`、`UNCLEAR` 不计入。
- 宽松读法：另把 `S-PARTIAL`、`F-PARTIAL` 记为 1。
- 报告：每层的一致率和 Wilson 95% 区间；用各层在开发数据中的步骤数加权的总体标签准确率（只用分层部分，按模型分别报告）；自动标签 × 人工判断的混淆表；轨迹成功与 `E-*` 编码的一致率；两位标注者在编码上的 Cohen's κ 和分歧条目清单。
- **事先写定的解读**：
  - search_repeat 层中若多数被标为 `S-REPEAT`，说明"重复算错"与人工判断一致，宽松标签只是敏感性分析。
  - search_distractor 层中 `S-PARTIAL` 的比例，就是"自动标签把有用的桥接检索记为错误"的比例；论文的 Limitations 引用这个数字。
  - Finish 两个部分匹配层的一致率决定 F1 阈值 0.5 是否合理；若 finish_correct_partial 的人工 `F-WRONG` 比例超过 30%，敏感性分析改用 F1 ≥ 0.8 作为主阈值之外的报告项。

**4. 标注者**

作者（A）和一位独立标注者（B）各自独立标注全部条目，完成前不讨论。没有落实第二位标注者之前，论文只写"作者单人审计"，不写成双人审计。分歧在双方都完成后讨论，讨论结果单独记录，不覆盖原始标注。

**5. 执行**

```powershell
.\scripts\python_local.ps1 scripts/24_label_audit.py sample --runs runs/qwen7b runs/llama8b --out audit/label_audit_v1
# 复制 annotate_steps.csv、annotate_episodes.csv 给每位标注者；填完放入 audit/label_audit_v1/A 与 .../B
.\scripts\python_local.ps1 scripts/24_label_audit.py score --out audit/label_audit_v1 --annotator A=audit/label_audit_v1/A --annotator B=audit/label_audit_v1/B
```

**6. 工作量**

约 330 个步骤 × 约 1 分钟，加 40 条轨迹 × 约 2 分钟，每位标注者约 7 小时，可以分几次完成。
