# 牙科 CBCT 交互式实例级辅助标注系统

本仓库用于开发“基于自进化 Agent 端侧推理的牙科 CBCT 交互式实例级辅助标注系统”。

项目目标是构建一套面向牙科 CBCT 数据集建设的本地化人机协同辅助标注系统。系统以 3D Slicer 作为医学影像可视化与人工修正前端，以本地 AI 推理服务作为后端，以半监督牙齿实例分割模型作为核心算法，以标注流程 Agent 负责流程调度、标签质检、经验记录和推理模式推荐。

成熟流程如下：

```text
CBCT 数据导入
  -> 标签模板加载
  -> ROI 选择
  -> Agent 状态检查与推理模式推荐
  -> 本地 AI 初分割
  -> 3D Slicer 显示结果
  -> 人工修正
  -> 标签质量检查
  -> 标准结果导出
  -> 修正数据反哺模型训练
```

## 目录说明

```text
.
├── docs/
├── configs/
├── data/
├── data_tools/
├── training/
├── inference_server/
├── slicer_extension/
├── agent/
├── quality_control/
├── deployment/
├── tests/
├── scripts/
├── assets/
├── outputs/
└── README.md
```

## docs/

用于存放项目文档。

建议内容：

- 项目架构说明；
- API 接口协议；
- 标签规范；
- 开发流程说明；
- 会议记录；
- 阶段总结；
- 软著、专利、结题材料草稿。

子目录说明：

```text
docs/architecture/     系统架构、模块关系、运行流程图
docs/api/              前后端接口协议、请求响应格式、错误码说明
docs/labeling/         牙位编号规则、标签模板、标注规范
docs/meeting_notes/    组会记录、任务分配、阶段复盘
```

## configs/

用于存放项目配置文件。

建议内容：

- 标签编号配置；
- 模型配置；
- 推理模式配置；
- 服务端配置；
- 数据路径配置。

子目录说明：

```text
configs/labels/        牙位标签、牙髓标签、种植体标签等配置
configs/models/        模型结构、权重路径、输入尺寸等配置
configs/inference/     快速、均衡、精细三种推理模式配置
```

## data/

用于组织项目数据。

注意：真实患者原始数据和敏感医学影像不应上传到公开 GitHub 仓库。该目录可以保留结构，但真实数据应根据实验室或医院要求保存在本地、内网或受控存储中。

子目录说明：

```text
data/raw/              原始 CBCT 数据，例如 DICOM、nii.gz、nrrd
data/processed/        预处理后的图像数据，例如重采样、裁剪后的数据
data/labels/           人工标注或人工修正后的标签
data/predictions/      AI 模型生成的初分割结果
data/reports/          标签质量检查报告和数据统计报告
data/splits/           训练集、验证集、测试集划分文件
data/demo_cases/       脱敏演示病例或示例数据说明
```

## data_tools/

用于存放数据处理工具脚本。

建议内容：

- DICOM 转 nii.gz；
- nrrd、nii.gz、LabelMap 格式转换；
- 图像重采样；
- 灰度归一化；
- ROI 裁剪；
- 数据集划分；
- 标签文件检查；
- 数据统计。

## training/

用于存放模型训练相关代码。

建议内容：

- PyTorch、MONAI 或 nnU-Net 训练代码；
- 数据集读取代码；
- 模型结构；
- 损失函数；
- 评价指标；
- 半监督训练流程；
- 伪标签生成与筛选；
- 模型导出代码。

子目录说明：

```text
training/datasets/         数据集读取和数据增强逻辑
training/models/           3D U-Net、nnU-Net、2.5D 模型等模型结构
training/losses/           Dice Loss、Cross Entropy 等损失函数
training/metrics/          Dice、IoU、实例识别准确率等评价指标
training/experiments/      实验配置、训练记录、对比实验说明
training/semi_supervised/  半监督学习、伪标签生成、伪标签筛选
training/export/           PyTorch 权重导出、ONNX 导出
```

## inference_server/

用于存放本地 AI 推理服务端代码。

该模块建议使用 FastAPI 实现。FastAPI 服务负责接收 3D Slicer 或 Agent 的请求，完成图像预处理、模型推理、后处理和结果返回。

子目录说明：

```text
inference_server/api/          接口路由，例如 /predict、/status、/check_label
inference_server/schemas/      请求和响应的数据结构定义
inference_server/services/     推理服务、图像读取、配置管理等业务逻辑
inference_server/runtime/      模型加载、ONNX Runtime、TensorRT 等运行逻辑
inference_server/postprocess/  连通域分析、孔洞填补、小碎片去除、实例编号整理
```

## slicer_extension/

用于存放 3D Slicer 插件代码。

3D Slicer 插件负责提供桌面 GUI 操作入口，包括加载 CBCT、选择 ROI、调用后端服务、显示 AI 分割结果、人工修正和导出标签。

子目录说明：

```text
slicer_extension/CBCTAnnotator/  项目自定义 Slicer 插件主体
slicer_extension/resources/      插件图标、界面资源、示例配置
```

## agent/

用于存放标注流程 Agent 相关代码与规则。

本项目中的 Agent 是流程助手，不直接进行医学诊断。它主要负责状态记录、流程提示、推理模式推荐、标签质检调用、经验沉淀和任务日志管理。

子目录说明：

```text
agent/memory/    当前病例状态、长期规则记忆、历史任务摘要
agent/skills/    可复用操作经验，例如 ROI 选择建议、质检建议
agent/rules/     流程规则、推理模式推荐规则、标签检查规则
agent/logs/      Agent 任务日志、用户操作记录、经验候选记录
```

## quality_control/

用于存放标签质量检查相关代码。

建议内容：

- 空标签检查；
- 重复编号检查；
- 标签编号范围检查；
- 连通域检查；
- 异常体积检查；
- 小碎片检查；
- 左右侧或上下颌混淆检查；
- 导出文件可读性检查；
- 质量报告生成。

## deployment/

用于存放模型部署与端侧适配相关内容。

子目录说明：

```text
deployment/onnx/      ONNX 导出、ONNX Runtime 推理验证
deployment/tensorrt/  TensorRT 加速、NVIDIA GPU 或 Jetson 适配
deployment/ascend/    昇腾等国产化算力平台适配资料和脚本
```

## tests/

用于存放测试代码。

建议内容：

- 数据预处理测试；
- API 接口测试；
- 推理服务测试；
- 标签质检测试；
- 3D Slicer 与后端联调测试；
- 完整流程集成测试。

子目录说明：

```text
tests/unit/         单个函数或单个模块测试
tests/integration/  多模块联调测试，例如 Slicer -> Agent -> 推理服务
```

## scripts/

用于存放项目辅助脚本。

建议内容：

- 启动本地推理服务；
- 初始化目录；
- 批量转换数据；
- 批量生成报告；
- 清理临时文件；
- 运行 Demo。

## assets/

用于存放非代码资源。

建议内容：

- 项目 Logo；
- 系统截图；
- 流程图；
- 架构图；
- 演示素材；
- 图标资源。

## outputs/

用于存放运行输出和阶段性结果。

子目录说明：

```text
outputs/models/       训练得到的模型权重或导出模型
outputs/exports/      系统导出的标准训练数据包
outputs/screenshots/  系统演示截图
outputs/reports/      实验报告、质检报告、模型评估报告
```

## 统一接口协议

本项目采用 3D Slicer 桌面前端 + 本地 FastAPI 推理服务端的架构。这里的接口协议用于“系统与 Agent 组”和“模型与数据组”之间对接。

接口中的 `/` 表示路径层级，不表示多选一。例如 `/api/v1/predict` 是一个完整接口地址。

接口中的 `GET` 和 `POST` 是 HTTP 请求方法：

```text
GET  = 获取信息，例如查看服务状态、读取配置
POST = 提交任务或数据，例如创建病例、执行推理、标签检查
```

### 接口统一约定

```text
Base URL: http://127.0.0.1:8000/api/v1
Content-Type: application/json
坐标顺序: [x, y, z]
图像输出优先格式: nii.gz
标签输出优先格式: nii.gz，同时兼容 seg.nrrd
质检报告输出: quality_report.json + quality_report.md
```

路径约定：

```text
GET  /status
GET  /config
POST /cases
POST /images/inspect
POST /agent/recommend_mode
POST /predict
POST /check_label
POST /export
POST /agent/log
```

完整调用地址示例：

```text
GET  http://127.0.0.1:8000/api/v1/status
POST http://127.0.0.1:8000/api/v1/predict
POST http://127.0.0.1:8000/api/v1/check_label
```

### 1. 服务状态接口

```text
GET /status
```

用途：检查后端服务是否启动、模型是否加载、硬件资源是否可用。

响应示例：

```json
{
  "service": "running",
  "version": "1.0.0",
  "device": {
    "type": "cuda",
    "name": "RTX 4090",
    "memory_free_mb": 18000
  },
  "model": {
    "loaded": true,
    "model_id": "tooth_seg_v1",
    "runtime": "pytorch"
  },
  "agent": {
    "enabled": true
  },
  "supported_formats": ["dicom", "nii.gz", "nrrd"]
}
```

### 2. 配置接口

```text
GET /config
```

用途：读取可用模型、推理模式和标签模板。

响应示例：

```json
{
  "models": [
    {
      "model_id": "tooth_seg_v1",
      "task": "tooth_instance_segmentation"
    }
  ],
  "inference_modes": {
    "fast": {
      "patch_size": [96, 96, 64],
      "overlap": 0.25
    },
    "balanced": {
      "patch_size": [128, 128, 96],
      "overlap": 0.5
    },
    "fine": {
      "patch_size": [160, 160, 128],
      "overlap": 0.625
    }
  },
  "label_templates": [
    {
      "template_id": "adult_fdi_v1",
      "path": "configs/labels/adult_fdi.yaml"
    }
  ]
}
```

### 3. 病例初始化接口

```text
POST /cases
```

用途：创建一次标注任务，生成 `case_id`，记录原始图像路径和标签模板。

请求示例：

```json
{
  "image_path": "D:/cases/case_0001/raw/image.nii.gz",
  "image_format": "nii.gz",
  "label_template_id": "adult_fdi_v1",
  "operator": "member_a"
}
```

响应示例：

```json
{
  "case_id": "case_0001",
  "status": "created",
  "case_state_path": "data/demo_cases/case_0001/state.json"
}
```

### 4. 图像信息检查接口

```text
POST /images/inspect
```

用途：读取 CBCT 图像基础信息，供 3D Slicer 前端和 Agent 判断数据大小、体素间距和推理策略。

请求示例：

```json
{
  "case_id": "case_0001",
  "image_path": "D:/cases/case_0001/raw/image.nii.gz"
}
```

响应示例：

```json
{
  "shape": [512, 512, 320],
  "spacing": [0.3, 0.3, 0.3],
  "direction": "RAS",
  "intensity_range": [-1000, 3000],
  "estimated_size_mb": 310
}
```

### 5. 推理模式推荐接口

```text
POST /agent/recommend_mode
```

用途：Agent 根据 ROI 范围、图像大小和硬件状态推荐推理模式。

请求示例：

```json
{
  "case_id": "case_0001",
  "roi": {
    "start": [40, 60, 20],
    "size": [160, 160, 128]
  },
  "target": ["tooth", "pulp", "implant"]
}
```

响应示例：

```json
{
  "recommended_mode": "balanced",
  "reason": "ROI 尺寸适中，当前 GPU 显存充足，推荐均衡模式。",
  "fallback_mode": "fast"
}
```

### 6. AI 初分割接口

```text
POST /predict
```

用途：执行牙科 CBCT 的 AI 初分割。

请求示例：

```json
{
  "case_id": "case_0001",
  "image_path": "D:/cases/case_0001/raw/image.nii.gz",
  "roi": {
    "start": [40, 60, 20],
    "size": [160, 160, 128]
  },
  "model_id": "tooth_seg_v1",
  "mode": "balanced",
  "targets": ["tooth", "pulp", "implant"],
  "output_format": "nii.gz",
  "output_dir": "D:/cases/case_0001/predictions"
}
```

响应示例：

```json
{
  "status": "success",
  "case_id": "case_0001",
  "prediction_id": "pred_0001",
  "mask_path": "D:/cases/case_0001/predictions/ai_pred.nii.gz",
  "confidence_path": "D:/cases/case_0001/predictions/confidence.nii.gz",
  "runtime": {
    "mode": "balanced",
    "elapsed_ms": 18500,
    "device": "cuda"
  },
  "message": "AI 初分割完成。"
}
```

### 7. 标签质量检查接口

```text
POST /check_label
```

用途：检查人工修正后的标签是否符合项目标注规范。

请求示例：

```json
{
  "case_id": "case_0001",
  "label_path": "D:/cases/case_0001/labels/corrected_label.nii.gz",
  "label_template_id": "adult_fdi_v1",
  "checks": [
    "empty",
    "duplicate_id",
    "component",
    "volume",
    "tiny_fragment",
    "format"
  ]
}
```

响应示例：

```json
{
  "status": "warning",
  "report_json": "D:/cases/case_0001/reports/quality_report.json",
  "report_md": "D:/cases/case_0001/reports/quality_report.md",
  "summary": {
    "error": 0,
    "warning": 2,
    "passed": 18
  },
  "issues": [
    {
      "level": "warning",
      "label": 112,
      "type": "multi_component",
      "message": "标签 112 存在多个不连续区域。"
    }
  ]
}
```

### 8. 结果导出接口

```text
POST /export
```

用途：导出训练可用的标准数据包。

请求示例：

```json
{
  "case_id": "case_0001",
  "image_path": "D:/cases/case_0001/raw/image.nii.gz",
  "label_path": "D:/cases/case_0001/labels/corrected_label.nii.gz",
  "export_format": ["nii.gz", "seg.nrrd"],
  "include_report": true,
  "output_dir": "D:/cases/case_0001/export"
}
```

响应示例：

```json
{
  "status": "success",
  "export_dir": "D:/cases/case_0001/export",
  "files": [
    "image.nii.gz",
    "label.nii.gz",
    "label.seg.nrrd",
    "quality_report.md",
    "metadata.json"
  ]
}
```

### 9. 任务日志接口

```text
POST /agent/log
```

用途：记录一次病例标注流程，用于 Agent 经验沉淀和半监督数据回流。

请求示例：

```json
{
  "case_id": "case_0001",
  "event": "label_corrected",
  "operator": "member_a",
  "payload": {
    "prediction_id": "pred_0001",
    "corrected_label_path": "D:/cases/case_0001/labels/corrected_label.nii.gz",
    "notes": "修正了 112 和 113 牙根边界。"
  }
}
```

响应示例：

```json
{
  "status": "success",
  "log_id": "log_0001"
}
```

### 统一错误格式

所有接口失败时统一返回如下格式：

```json
{
  "status": "error",
  "error_code": "MODEL_NOT_LOADED",
  "message": "模型尚未加载，请检查 /status。",
  "details": {}
}
```

建议统一错误码：

```text
IMAGE_NOT_FOUND       图像文件不存在
UNSUPPORTED_FORMAT    图像或标签格式不支持
INVALID_ROI           ROI 坐标或尺寸不合法
MODEL_NOT_LOADED      模型尚未加载
CUDA_OUT_OF_MEMORY    GPU 显存不足
PREDICTION_FAILED     模型推理失败
LABEL_NOT_FOUND       标签文件不存在
LABEL_FORMAT_ERROR    标签格式错误
EXPORT_FAILED         结果导出失败
```

## 开发协作原则

建议团队按两条主线并行推进：

```text
模型与数据组：
数据整理 -> 预处理 -> 基础模型训练 -> 半监督优化 -> 模型导出

系统与 Agent 组：
3D Slicer 插件 -> FastAPI 服务 -> Agent 流程 -> 标签质检 -> 结果导出
```

两组之间通过统一接口协议对接。系统组前期可以使用假 mask 或已有模型调通流程，模型组不需要等待系统完全完成即可独立训练模型。

## 注意事项

- 不要将真实患者原始 CBCT 数据上传到公开仓库。
- 不要把大体积模型权重直接提交到 GitHub，必要时使用本地共享盘、网盘或 Git LFS。
- 修改接口格式前必须通知两组成员。
- 每个模块的输入、输出、文件路径和错误信息都应尽量清晰。
- 优先跑通完整闭环，再逐步提高模型精度和 Agent 智能程度。
