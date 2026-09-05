# model/ —— 模型接入模块

本目录用于放置 A 组模型侧的推理接入代码与配置。当前项目按照 ToothSeg 的
`Dataset121` 语义分割分支推进，用于给 3D Slicer 前端提供真实 AI 自动标注结果。
本阶段不运行 ToothSeg 实例分支，也不运行完整双分支后处理。

- `toothseg_semantic.py`：ToothSeg 语义分割适配模块，由后端 `/predict` 自动调用。
- `toothseg/run_toothseg.py`：完整 ToothSeg 入口，当前仅作为未来高级模式代码储备。
- `interfaces/`：推理封装接口（规划预留）
- `weights/`：模型权重目录（**已被 .gitignore 排除，不上传 GitHub**；当前只要求语义分支权重）

## 当前模型调用方式

前端不直接运行模型。用户在 3D Slicer 插件中点击“开始分割”后，插件调用：

```text
POST /api/v1/predict
```

如果请求中的 `model_id` 是：

```text
toothseg-semantic-05mm
```

后端会调用 `toothseg_semantic.py`，执行如下流程：

```text
1. 检查输入 CBCT 文件是否存在
2. 复制图像到英文工作目录，避免中文路径导致 nnU-Net / SimpleITK 读取异常
3. 将图像重采样到 0.5mm 或保守模式指定 spacing
4. 按 nnU-Net 规范生成 imagesTs 输入文件
5. 调用 ToothSeg Dataset121 语义分割模型
6. 检查 ToothSeg 原始输出 .nii.gz 是否存在且非空
7. 将 ToothSeg 语义标签映射为本项目天然牙 dense 标签
8. 把项目标签版 mask_path 返回给 3D Slicer 前端显示
```

完成后的主要产物是：

```text
<case>.nii.gz                   ToothSeg 原始语义分割标签图
<case>_project_labels.nii.gz    映射到本项目规范后的天然牙标签图
label_mapping.json              ToothSeg 标签到本项目标签的映射记录
tooth_locations.json            每颗牙的中心点、包围盒和体素数
result.json              本次推理参数、输出路径、spacing、标签统计
reuse_card.json          复用包索引，记录同一图像同一模式下是否已有可复用结果
readable_summary.txt     便于人工查看的运行摘要
```

当前标签映射规则：

```text
ToothSeg 1-16   -> 本项目 dense 1-16   -> 101-116 上颌天然牙
ToothSeg 17-32  -> 本项目 dense 49-64  -> 401-416 下颌天然牙
背景 0          -> 背景 0
```

## 当前限制

- ROI 只作为流程记录，不裁剪输入；语义模型按整张降采样 CBCT 推理。
- `fast` 使用 0.75mm 降采样，`balanced` 使用 0.5mm 降采样，`fine` 当前与 0.5mm 语义推理一致。
- 当前只需要 `Dataset121_ToothFairy2_Teeth` 权重。
- `Dataset123_ToothFairy2fixed_teeth_spacing02_brd3px` 实例分支权重是未来高级模式预留，当前运行不依赖。

## 未来高级模式接口

未来如果要恢复完整 ToothSeg，可在服务端重新启用 `toothseg-full`，调用：

```text
implementation/model/toothseg/run_toothseg.py --mode full
```

该模式会运行语义分支、实例分支和牙位编号后处理，耗时和显存压力都明显高于当前语义模式。
