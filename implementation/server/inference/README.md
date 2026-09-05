# implementation/server/inference —— 本地推理服务端

本目录承载 B 组的**本地推理服务端**。当前项目按 ToothSeg 的 `Dataset121` 语义分割分支推进：模型直接输出牙位语义标签图，用于后续显示、人工修正、规则检查和导出。完整 ToothSeg 双分支实例流程暂不运行，只保留未来高级模式接口。

## 依赖环境

运行环境建议使用 Conda 的 `nnInteractive` 环境。该环境需要包含：

```text
fastapi / uvicorn / nibabel / numpy / SimpleITK / torch / nnunetv2
```

本机目前推荐使用：

```
<你的 conda 目录>\envs\nnInteractive\python.exe
```

若只跑假模型联调，安装基础依赖即可：

```bash
pip install fastapi uvicorn nibabel numpy
```

若要跑 ToothSeg 真实语义分割，还必须保证 nnU-Net v2、PyTorch CUDA、SimpleITK 可用，
并且本机存在 ToothSeg 语义分支权重目录：

```text
<nnUNet_results>/Dataset121_ToothFairy2_Teeth
```

`nnUNet_results` 指“包含 Dataset121 子目录的那一层”。服务端会优先读取
`nnUNet_results`、`TOOTHSEG_NNUNET_RESULTS`、`CBCT_NNUNET_RESULTS` 环境变量；
如果没有设置，会自动检查 `implementation/model/weights/`、`ToothSeg/nnUNet_results/`、
仓库同级外部权重目录和 `D:\ToothSegWork\nnUNet_results/`。

## 启动

在项目根目录执行：

```powershell
python implementation\server\inference\toothseg_server.py
```
（或用上面的完整 python 路径代替 `python`；也可双击 `user\launcher\一键启动服务.bat`）

启动成功会看到：
```
CBCT ToothSeg server -> http://127.0.0.1:8000/api/v1
INFO: Uvicorn running on http://127.0.0.1:8000
```

服务地址（前端默认填这个）：`http://127.0.0.1:8000/api/v1`

> 推荐把输入 CBCT 放在英文路径下（例如 `D:\CBCTData\inputs\case_0001.nii.gz`）。
> 如果从中文路径导入，服务端会先复制到英文 runtime 缓存后再读取，避免 SimpleITK / nnU-Net 在 Windows 下读取中文路径失败。

若只想秒级验证前端流程，也可以手动启动 mock 服务：

```powershell
python implementation\server\inference\mock_server.py
```

## 停止

- 前台运行：`Ctrl + C`
- 后台运行：找到 python 进程结束，或关闭启动它的终端

## 接口一览（9 个，前缀 /api/v1）

| 方法 | 路径 | 作用 |
|------|------|------|
| GET | /status | 服务/模型/硬件状态，包含 ToothSeg 是否可用 |
| GET | /config | 可用模型、推理模式、标签模板 |
| POST | /cases | 创建病例，返回 case_id |
| POST | /images/inspect | 读 CBCT 的形状/间距 |
| POST | /agent/recommend_mode | Agent 按 ROI 大小推荐推理模式 |
| POST | /predict | **核心**：当前调用 ToothSeg 语义分割；完整 ToothSeg 高级模式仅保留接口 |
| POST | /check_label | 标签质检（返回问题列表） |
| POST | /export | 导出训练数据包 |
| POST | /agent/log | 记录标注事件 |

## 模型选择

前端“模型与配置”下拉框当前会看到两个模型：

```text
toothseg-semantic-05mm  ToothSeg 语义分割模型，当前主流程
toothseg-full           ToothSeg 完整双分支，未来高级模式占位，当前未启用
```

选择 `toothseg-semantic-05mm` 后，点击“开始分割”，服务端会自动执行：

```text
输入 CBCT 图像
  -> 复制到英文工作目录
  -> 重采样到 0.5mm
  -> 调用 ToothSeg Dataset121
  -> 输出语义标签图 .nii.gz
  -> 返回 mask_path 给 Slicer 显示
```

当前 ToothSeg 语义分割不会使用 ROI 裁剪整图，ROI 暂时只作为流程参数记录。

服务端返回给前端的 `mask_path` 是项目标签规范版结果，不是 ToothSeg 原始 1-32 标签：

```text
ToothSeg 1-16   -> 本项目 dense 1-16   -> 101-116 上颌天然牙
ToothSeg 17-32  -> 本项目 dense 49-64  -> 401-416 下颌天然牙
背景 0          -> 背景 0
```

同一目录会额外保存 `raw_mask_path` 指向的 ToothSeg 原始语义输出，以及 `label_mapping.json`、`tooth_locations.json`。

推理模式含义：

```text
fast      0.75mm 降采样，速度优先
balanced  0.5mm 降采样，默认推荐
fine      当前与 0.5mm 语义推理一致，预留未来更高质量配置
```

## 输出目录

- 输入缓存：默认 `D:\ToothSegWork\_runtime\input_cache\`
- ToothSeg 复用包：默认 `D:\ToothSegWork\_runtime\reuse_packages\`
- ToothSeg 语义分割结果：默认位于对应复用包的 `semantic\<task_key>\final\`
- 关闭复用时的临时推理结果：默认 `D:\ToothSegWork\_runtime\semantic_predictions\`
- 服务端 jobs / export / agent 日志：默认 `D:\ToothSegWork\_runtime\server_outputs\`
- mock 联调输出：默认 `D:\ToothSegWork\_runtime\mock_outputs\`

这些运行目录应保持英文路径。可通过 `CBCT_TOOTHSEG_RUNTIME` 覆盖统一 runtime 根目录，
也可用 `CBCT_SERVER_OUTPUT`、`CBCT_MOCK_OUTPUT` 单独覆盖服务端和 mock 输出目录。

## 3D Slicer 前端对接

1. 先启动 ToothSeg 语义服务端
2. Slicer 插件「服务连接」卡，地址保持默认 `http://127.0.0.1:8000/api/v1`
3. 点「连接测试」→ 绿字即通
4. 再依次：加载配置 → 导入 CBCT → 框 ROI → 智能推荐/开始分割 → 修正 → 质检 → 导出

## 注意事项

- `mock-simple-cube` 只是在 ROI 内生成一个**实心方块假 mask**，仅用于链路联调。
- `toothseg-semantic-05mm` 是当前真实模型主流程，只运行语义分割分支，不运行完整双分支。
- `toothseg-full` 是未来高级模式接口占位，当前选择后会返回“暂未启用”。
- 不要在本目录存放真实患者 CBCT 原始数据，Demo 数据必须脱敏。
- 本机 RTX 4060 Laptop 8GB 显存建议默认使用 0.5mm 降采样，不建议直接整图原分辨率推理。
