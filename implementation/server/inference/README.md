# implementation/server/inference —— 本地推理服务端（含 mock 假后端）

本目录承载 B 组的**本地推理服务端**。当前提供开发联调用**假后端（mock server）**，
在真实模型 / A 组推理脚本接入前，先用它把 3D Slicer 前端"发送请求 → 接收 mask →
显示"整条链路调通。接口与仓库「统一接口协议」完全一致。

## 依赖环境

运行环境是 Conda 的 `nninteractive` 环境（已有 fastapi / uvicorn / nibabel / numpy）：
```
<你的 conda 目录>\envs\nninteractive\python.exe
```
（`<你的 conda 目录>` 替换为你本机的 Anaconda/Miniconda 安装位置；
若你在终端里先用 `conda activate nninteractive`，则可直接写 `python`。）

若在别的机器上没有该环境，需保证 Python 3.9+ 且安装依赖：
```bash
pip install fastapi uvicorn nibabel numpy
```

## 启动

在项目根目录执行：

```powershell
python implementation\server\inference\mock_server.py
```
（或用上面的完整 python 路径代替 `python`；也可双击 `user\launcher\一键启动服务.bat`）

启动成功会看到：
```
CBCT mock server -> http://127.0.0.1:8000/api/v1
INFO: Uvicorn running on http://127.0.0.1:8000
```

服务地址（前端默认填这个）：`http://127.0.0.1:8000/api/v1`

> mock 的输入 CBCT 需放在本机绝对路径（如 `<你的数据路径>\1.nii`），
> 因为 /predict 需服务端能读到该文件；在 Slicer 前端选影像时选你本机的文件即可。

## 停止

- 前台运行：`Ctrl + C`
- 后台运行：找到 python 进程结束，或关闭启动它的终端

## 接口一览（9 个，前缀 /api/v1）

| 方法 | 路径 | 作用 |
|------|------|------|
| GET | /status | 服务/模型/硬件状态 |
| GET | /config | 可用模型、推理模式、标签模板 |
| POST | /cases | 创建病例，返回 case_id |
| POST | /images/inspect | 读 CBCT 的形状/间距 |
| POST | /agent/recommend_mode | Agent 按 ROI 大小推荐推理模式 |
| POST | /predict | **核心**：ROI 内生成假分割 mask |
| POST | /check_label | 标签质检（返回问题列表） |
| POST | /export | 导出训练数据包 |
| POST | /agent/log | 记录标注事件 |

## 输出目录

- mask / confidence：`data/outputs/mock_masks/`
- 标签：`data/outputs/labels/`
- 导出包：`data/outputs/mock_masks/export/`

这些目录已在 `.gitignore` 中忽略（`data/outputs/`、`*.nii`、`*.nii.gz`、`*.nrrd`），
不会误推送到 GitHub。

## 3D Slicer 前端对接

1. 先启动 mock server
2. Slicer 插件「服务连接」卡，地址保持默认 `http://127.0.0.1:8000/api/v1`
3. 点「连接测试」→ 绿字即通
4. 再依次：加载配置 → 导入 CBCT → 框 ROI → 智能推荐/开始分割 → 修正 → 质检 → 导出

## 注意事项

- mock 的 `/predict` 只是在 ROI 内生成一个**实心方块假 mask**，不代表真实分割效果，
  仅用于链路联调。
- 不要在本目录存放真实患者 CBCT 原始数据，Demo 数据必须脱敏。
- 后续接入 A 组真实模型时，只需把 `/predict` 的实现替换为真实推理脚本，
  接口契约保持不变即可，前端无需改动。