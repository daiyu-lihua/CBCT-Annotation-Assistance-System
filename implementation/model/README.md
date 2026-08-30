# model/ —— 模型接入模块（预留）

本目录预留给 A 组模型侧：真实牙齿分割模型的推理接入代码与配置。

- `interfaces/`：推理封装接口（占位，待 A 组提供模型推理脚本后接入）
- `weights/`：模型权重目录（**已被 .gitignore 排除，不上传 GitHub**；大文件/权重放这里）

当前阶段前端与 `server/inference`（mock）已打通，真实模型到位后，
把 `server/inference` 中 `/predict` 的实现替换为对本模块推理接口的调用即可。