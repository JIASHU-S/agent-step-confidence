**最近邻文献与对照矩阵 — P0 草案，2026-10-29**

本矩阵用于约束贡献及选择对照，不表示已复现任何外部方法。以下事实据链接版本的作者摘要核对；具体训练样本、输入边界、选层、校准预算和实现需要在 P1 查全文与代码。未确认项明确留空，不从摘要猜测。

| 工作与版本 | 环境 / 预测目标 | 表示与时点 | 监督 / 成本口径 | 对本项目的影响和拟议对照 |
| --- | --- | --- | --- | --- |
| [ReProbe, v5](https://arxiv.org/abs/2511.06209v5) | 数学、规划、知识 QA；推理步骤可信度 | 冻结 LLM 内部状态；生成过程中验证步骤 | 小于 10M 参数的 transformer probe；大模型或原模型提供标注；具体标注与推理开销待核 | 不能把“内部状态预测步骤可信度”本身作为新贡献。考察结构化 probe 适配；与本项目 QA→agent 迁移的监督条件逐项比较。 |
| [Do Agents Know When They Succeed?, v2](https://arxiv.org/abs/2609.09448v2) | Bash、SQL、Python；eventual task success | ARP 用动作决策处表示；LTD 总结轨迹 residual-stream 变化 | 训练预算待核；作者称不改 prompt、不需多次 rollout | 本项目必须加入同事件的 prefix-success predictor；线性适配是本项目基线，核实实现前不能称 ARP 完整复现。LTD 输入是否含整条轨迹须先核清。 |
| [Doomed from the Start, v2](https://arxiv.org/abs/2607.06503v2) | TextCraft、WebShop；早期 eventual failure 与停止策略 | 早期轮次的线性 hidden-state probes，组成 cascade | 独立选择/认证数据；论文提出全局成功保留率保证，具体假设与样本量待核 | 最直接的 early-abort 最近邻。先加入单 gate、同可观察历史基线，再评估 cascade 是否可公平适配；本项目经验 95% retention 目标不能冒用其理论保证。 |

上述“拟议对照”是本项目研究设计判断，不是原论文结论。共同比较字段应包括：预测事件、合法输入时点、source/target 监督、总标签预算、问题级 split、是否包含 timeout、训练及检索成本、阈值校准与统计保证。任何无法对齐的字段须在主文或附录披露。

P1 待完成：读取对应版本全文、核查官方代码可用性与许可、确定可复现或仅可适配的对照、记录每项差异。完整相关工作还需覆盖报告已有参考文献；本表只列优先处理的三个最近邻。
