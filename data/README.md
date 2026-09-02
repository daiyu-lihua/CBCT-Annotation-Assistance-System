# data/ —— 数据层

- `inputs/`：原始 CBCT 等输入数据（**脱敏后**使用；大型/原始数据不入库）。
- `outputs/`：推理/标注/导出的运行产物（mock 预测 mask、导出训练数据包等）。

二者均已被 `.gitignore` 排除，不上传 GitHub；如需共享数据请走脱敏后的单独通道。

> 2026-09-02：为推送 GitHub，已有推理产物（jobs、export、agent_log.jsonl）整体移出仓库至项目同级 `CBCT_data/outputs/` 保存；服务端下次运行会在本目录重建 `outputs/`，历史产物不受影响。模型权重的位置说明见仓库根 `模型权重放置说明.md`。