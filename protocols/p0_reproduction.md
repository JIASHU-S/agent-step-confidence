**P0 复现入口 — 2026-10-29**

在 `F:/Jiashu/llm-uq-project/agent-conf` 的 PowerShell 中运行。使用已有 `.venv`；以下均不调用模型推理、不下载权重，不改旧结果。

```powershell
.venv/Scripts/python.exe -m pytest tests/test_asset_audit.py -q
.venv/Scripts/python.exe scripts/10_p0_audit.py
.venv/Scripts/python.exe scripts/11_p0_reproduce.py --threads 16 --out results/submission_v2/p0_threads16
.venv/Scripts/python.exe scripts/12_p0_costs.py
.venv/Scripts/python.exe scripts/13_p0_numerics.py
.venv/Scripts/python.exe scripts/14_p0_datasets.py
```

`13_p0_numerics.py` 读取保留的 1、4、16 线程结果。需要重跑线程敏感性时分别用 `--threads 1 --out results/submission_v2/p0_threads1` 和 `--threads 4 --out results/submission_v2/p0`。这两项有严格容差不匹配，预期退出码为 1；记录保留用于解释数值敏感性，不改宽容差。

`14_p0_datasets.py` 验证本地固定版本数据、重建旧抽样和 split、回放已存动作核对 observation/标签，并准备 pilot 输入；本地文件不存在时可加 `--download` 恢复两份公开 validation parquet（合计 34,787,896 bytes），只接受来源记录中的固定 URL、大小与 SHA-256。此选项会联网，不下载模型权重。首次恢复后重跑 `10_p0_audit.py`，将新文件和配置纳入 hash 清单。

`10_p0_audit.py` 检查原始行键、全部 NPZ 内容、标签字段与结果关联、输出副本、文件 hash，并保存 split manifest。它不证明 Search 标签的语义有效性或表示的因果读取边界。

`11_p0_reproduce.py` 重算旧协议的主要标量、选定 probe 拟合和 bootstrap 区间。默认验证目录为 `p0_threads16`。直接指标绝对容差 1e-8，重新拟合比较 1e-6；不使用相对容差。它不重跑模型生成、全层选择网格、Bayesian 后验、PSIS-LOO、全部 hand-off bands 或宽松标签拟合。

`12_p0_costs.py` 从历史日志估算资源。GPU pilot 配置只在缺失时创建，以免覆盖后来录入的 revision。新增/修改协议后应重跑资产审计，刷新 source hash。

唯一旧结果入口为 `runs/qwen7b/results`、`runs/llama8b/results`；其他旧导出已做一致性比较。P0 输出仅证明旧数字来源可追溯，不能作为修正六个设计问题后的新结果。

权重恢复时同时固定 model 与 tokenizer 的完整 commit SHA，保存配置及文件 hash。Hugging Face 默认取 main 的最新版本；指定 `revision` 才能取得特定版本，见 [官方说明](https://huggingface.co/docs/huggingface_hub/guides/download)。找不到旧 SHA 时，为新实验固定一个明确版本，并单列历史来源缺口。

数据版本核对除 repo revision 外，还包括 HotpotQA `distractor` / `validation`、TriviaQA `rc.nocontext` / `validation`、既有 question IDs、问题/答案和 Hotpot 的十段 context。package version、dataset builder version 和 repository commit 不互相替代。
