# agent/ —— 标注流程 Agent（预留）

本目录预留给 B 组 Agent 调度模块：病例状态管理、推理模式推荐、标签质量检查、经验沉淀。

当前 mock 服务端已在 `server/inference` 中提供 `/agent/recommend_mode`、
`/agent/log` 等接口占位；正式 Agent 逻辑（含标签质检 label_checker 等）规划在此实现。