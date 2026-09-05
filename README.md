# 牙科 CBCT 交互式实例级辅助标注系统

本仓库用于开发“基于自进化 Agent 端侧推理的牙科 CBCT 交互式实例级辅助标注系统”。

项目目标是构建一套面向牙科 CBCT 数据集建设的本地化人机协同辅助标注系统。当前阶段以 3D Slicer 作为医学影像可视化与人工修正前端，以本地 AI 推理服务作为后端，以 ToothSeg 语义分割模型作为核心算法，先完成“牙齿位置语义标注 + 标签导出 + 人工修正”的闭环。标注流程 Agent 负责流程调度、标签质检、经验记录和推理模式推荐；完整实例级 ToothSeg 流程保留为未来高级模式。

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
├── README.md
├── user/                    用户层
│   ├── plugin/              3D Slicer 插件本体 (CBCTAnnotator.py, lib/ApiClient.py)
│   └── launcher/            启动入口：一键启动服务、环境说明
├── implementation/          实现层：后台服务、模型接入、Agent
│   ├── server/inference/    本地推理服务端（mock 假后端 + ToothSeg 语义推理, FastAPI）
│   ├── model/               模型接入（当前主流程为 ToothSeg 语义分割）
│   └── agent/               Agent 调度（预留）
├── docs/                    各类文档（项目介绍 / 使用指南 / 接口方案）
├── data/                    数据层：inputs 原始数据、outputs 运行产物（均不入库）
├── configs/                 配置文件（规划预留）
├── data_tools/              数据处理脚本（规划预留）
├── training/                模型训练代码（规划预留，A 组）
├── quality_control/         标签质检（规划预留）
├── deployment/              端侧部署（规划预留）
├── tests/                   测试（规划预留）
├── scripts/                 辅助脚本（规划预留）
└── assets/                  非代码资源（规划预留）
```

## docs/

用于存放项目文档。

建议内容：

- 项目架构说明；

- 使用指南与用户手册；

- API 接口协议；

- 标签规范；

- 阶段总结、软著、专利、结题材料草稿。

当前已有子目录：

```text
docs/项目介绍/       项目背景、开发分工与实现说明
docs/使用指南/       插件使用手册（用户如何操作）
docs/接口与开发/     前后端接口协议、请求响应格式、错误码说明
```

规划预留子目录：

```text
docs/architecture/     系统架构、模块关系、运行流程图
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

当前落盘目录（已被 .gitignore 排除，不入库）：

```text
data/inputs/     本次使用的原始 CBCT 输入（脱敏）
data/outputs/    推理/标注/导出运行产物（mock 预测 mask、导出训练包等）
```

规划子目录：

```text
data/processed/        预处理后的图像数据，例如重采样、裁剪后的数据
data/labels/           人工标注或人工修正后的标签
data/predictions/      AI 模型生成的初分割结果
data/reports/          标签质量检查报告和数据统计报告
data/splits/           训练集、验证集、测试集划分文件
data/demo_cases/       脱敏演示病例或示例数据说明
```

## data\_tools/

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

## implementation/server/inference/

用于存放本地 AI 推理服务端代码（FastAPI 实现，统一接口协议 v1）。

服务端负责接收 3D Slicer 或 Agent 的请求，完成图像读取、推理调用、结果返回。当前有两个实现（9 个接口均就绪）：

```text
mock_server.py      本地假后端（B 组前端联调用，返回 ROI 内实心立方体假 mask）
toothseg_server.py  ToothSeg 语义推理服务端（调用 implementation/model/toothseg_semantic.py）
```

**ToothSeg 语义推理服务端**启动方式（在 nninteractive 环境中）：

```powershell
E:\miniconda3\envs\nninteractive\python.exe implementation\server\inference\toothseg_server.py
```

或双击 `user\launcher\一键启动服务.bat`。地址同为 `http://127.0.0.1:8000/api/v1`（mock 与 ToothSeg 语义服务二选一运行）。

要点：

- `/predict` 当前只运行 ToothSeg Dataset121 语义分割分支，插件侧 `ApiClient.predict()` 在子线程调用，前端不会卡死；

- 同一时刻仅允许一个推理任务（GPU 独占），重复请求返回 `PREDICT_IN_PROGRESS`；

- 推理由语义适配器调用 nnU-Net，服务进程不初始化 CUDA；

- 输入优先使用 NIfTI（.nii/.nii.gz）；服务端会复制到英文工作目录并生成 nnU-Net 需要的 `_0000.nii.gz` 输入；ROI 记录不裁剪；

- 产物：项目标签规范版语义标签图 `.nii.gz`，默认位于输入图像旁的复用包目录或 `D:\ToothSegWork\_runtime\semantic_predictions\`；返回字段包含 `mask_path`、`raw_mask_path`、`spacing_mm`、`mask_info`、`mapping_info` 和复用信息。

语义标签映射规则：

```text
ToothSeg 1-16   -> 本项目 dense 1-16   -> 101-116 上颌天然牙
ToothSeg 17-32  -> 本项目 dense 49-64  -> 401-416 下颌天然牙
背景 0          -> 背景 0
```

规划子目录：

```text
implementation/server/inference/api/          接口路由，例如 /predict、/status、/check_label
implementation/server/inference/schemas/      请求和响应的数据结构定义
implementation/server/inference/services/     推理服务、图像读取、配置管理等业务逻辑
implementation/server/inference/runtime/      模型加载、ONNX Runtime、TensorRT 等运行逻辑
implementation/server/inference/postprocess/  连通域分析、孔洞填补、小碎片去除、实例编号整理
```

## implementation/model/

用于存放模型接入代码。当前主流程只接入 ToothSeg 语义分割模型（2026-09 语义模式版）；完整双分支代码保留为未来高级模式接口储备。

```text
implementation/model/toothseg/                  ToothSeg 推理代码（自包含）
  ├── run_toothseg.py                           参数化端到端推理入口（full/sem/inst，当前不由主服务调用 full）
  ├── memsafe_inference.py                      MemSafe 低显存滑窗推理器
  ├── postprocess_predictions/                  后处理：实例化/重采样/牙位编号（未来高级模式预留）
  └── toothseg/                                 迷你工具包（copy_geometry 等）
implementation/model/weights/                   nnU-Net 权重（.gitignore 排除，不入库）
  ├── Dataset121_ToothFairy2_Teeth/...          语义分支（当前必需）
  └── Dataset123_ToothFairy2fixed_teeth_spacing02_brd3px/...  实例分支（未来高级模式预留，当前非必需）
```

环境要求：Conda `nninteractive` 环境（包含 nnU-Net v2、PyTorch CUDA、SimpleITK 等）。模型实际读取的原始缓存、预处理文件、分割结果和复用包默认放在英文 runtime 目录 `D:\ToothSegWork\_runtime\`，可用 `CBCT_TOOTHSEG_RUNTIME` 覆盖。推荐原始 CBCT 数据也放在英文路径下，例如 `D:\CBCTData\inputs\`，避免 SimpleITK / nnU-Net 在 Windows 下读取中文路径失败。

## 模型权重路径配置

模型权重不上传 GitHub，需要每个成员在本机单独放置。当前代码会自动寻找 ToothSeg 语义分割使用的 nnU-Net 权重。

需要放置的是 `nnUNet_results` 这一层目录。它下面应该能找到如下文件：

```text
nnUNet_results/
└── Dataset121_ToothFairy2_Teeth/
    └── nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_256_bs8_ctnorm/
        └── fold_5/
            └── checkpoint_final.pth
```

推荐放置路径：

```text
D:\ToothSegWork\nnUNet_results
```

也就是说，最终权重文件应位于：

```text
D:\ToothSegWork\nnUNet_results\Dataset121_ToothFairy2_Teeth\nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_256_bs8_ctnorm\fold_5\checkpoint_final.pth
```

当前代码的读取逻辑如下：

```text
1. 优先读取环境变量：
   nnUNet_results
   TOOTHSEG_NNUNET_RESULTS
   CBCT_NNUNET_RESULTS

2. 如果没有环境变量，就自动检查常见候选路径：
   implementation/model/weights
   implementation/model/weights/nnUNet_results
   D:/ToothSegWork/nnUNet_results
   项目根目录/ToothSeg/nnUNet_results
   项目上级目录/model_weights
   项目上级目录/model_weights/nnUNet_results
   项目上级目录/nnUNet_results

3. 哪个候选目录里能找到 checkpoint_final.pth，就使用哪个目录。

4. 如果全部找不到，/status 会返回 checkpoint_exists=false，前端会显示模型未完整加载。
```

建议团队统一使用 `D:\ToothSegWork\nnUNet_results`。这样既不会把大权重放进 GitHub，也能减少中文路径造成的兼容问题。

## user/plugin/

用于存放 3D Slicer 插件（用户层前端）代码。

插件负责提供桌面 GUI 操作入口：加载 CBCT、选择 ROI、调用后端服务、显示 AI 分割结果、人工修正和导出标签。

```text
user/plugin/CBCTAnnotator.py   插件主模块（数据导入 / ROI / 分割 / 修正 / 质检 / 导出 等卡片界面）
user/plugin/lib/ApiClient.py   后端接口封装层
```

## implementation/agent/

用于存放标注流程 Agent 相关代码与规则。

本项目中的 Agent 是流程助手，不直接进行医学诊断。它主要负责状态记录、流程提示、推理模式推荐、标签质检调用、经验沉淀和任务日志管理。

子目录说明：

```text
agent/memory/    当前病例状态、长期规则记忆、历史任务摘要
agent/skills/    可复用操作经验，例如 ROI 选择建议、质检建议
agent/rules/     流程规则、推理模式推荐规则、标签检查规则
agent/logs/      Agent 任务日志、用户操作记录、经验候选记录
```

## quality\_control/

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
  "status": "ok",
  "service": {
    "name": "cbct-toothseg-semantic-server",
    "version": "1.0.0"
  },
  "device": {
    "type": "cuda",
    "name": "RTX 4090",
    "memory_total_mb": 24576,
    "memory_free_mb": 18000
  },
  "model": {
    "loaded": true,
    "name": "toothseg-semantic-05mm",
    "default": "toothseg-semantic-05mm",
    "advanced_available": false,
    "advanced_model_id": "toothseg-full"
  }
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
      "model_id": "toothseg-semantic-05mm",
      "task": "tooth_semantic_segmentation"
    },
    {
      "model_id": "toothseg-full",
      "task": "tooth_instance_segmentation",
      "enabled": false,
      "note": "未来高级模式接口占位，当前不运行完整双分支。"
    }
  ],
  "inference_modes": {
    "fast": {
      "spacing_mm": 0.75,
      "description": "速度优先"
    },
    "balanced": {
      "spacing_mm": 0.5,
      "description": "默认推荐"
    },
    "fine": {
      "spacing_mm": 0.5,
      "description": "当前与均衡模式一致，预留未来增强"
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
  "model_id": "toothseg-semantic-05mm",
  "mode": "balanced",
  "spacing_mm": 0.75,
  "targets": ["tooth", "pulp", "implant"],
  "output_format": "nii.gz",
  "output_dir": "D:/cases/case_0001/predictions"
}
```

字段说明：

```text
spacing_mm  降采样间距，单位 mm。值越大，体素数量越少，显存压力越低，但细节会减少。
            当前前端允许 0.50-2.00，推荐先用 0.75 或 1.00 跑通流程。
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
    "spacing_mm": 0.75,
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
  "export_format": "nii.gz",
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

## 当前分支文件说明（slicer-ui-reuse-label-update）

本节用于团队成员理解当前 GitHub 分支里每个主要文件的用途。这里说明的是“应该进入仓库的工程文件”，不包括本地病例数据、模型权重、运行缓存和第三方完整下载包。

### 根目录文件

| 文件 | 功能 | 当前更新内容 |
| --- | --- | --- |
| `.gitattributes` | Git 文本格式规则。 | 保持基础配置，用于减少不同系统换行差异。 |
| `.gitignore` | Git 忽略规则。 | 明确排除真实影像、模型权重、运行产物、`ToothSeg.zip`、`ToothSeg/`、`02/`、`demo_case/`、`__pycache__/` 等不可上传内容。 |
| `README.md` | 项目总说明文档。 | 整理当前项目定位、目录结构、接口协议、文件说明、分支上传说明和不可上传清单。 |
| `改动说明.md` | 当前阶段代码改动记录。 | 记录 ToothSeg 接入、Slicer 前端、复用包、标签集合、降采样参数和低显存策略。 |
| `项目开发分工与实现说明.md` | 面向四人团队的分工与实现路线说明。 | 作为团队协作和任务分派参考文档。 |
| `CBCT_ITKSNAP_Label_Description_dense_idx_3digit_code.txt` | 本项目标签命名规则参考文件。 | 用于理解 `101_UpperTooth_Pos01` 这类标签形式。 |

### data

| 文件 | 功能 | 当前更新内容 |
| --- | --- | --- |
| `data/README.md` | 数据目录说明。 | 说明数据目录只保留结构和说明，真实医学影像、标签、输出结果不入库。 |

### docs

| 文件 | 功能 | 当前更新内容 |
| --- | --- | --- |
| `docs/项目介绍/项目开发分工与实现说明.md` | 项目定位、整体架构、实现路径、团队分工说明。 | 给初学者团队看，帮助理解模型组和系统组如何并行推进。 |
| `docs/使用指南/插件使用手册.md` | Slicer 插件使用说明。 | 说明如何启动服务、连接插件、导入影像、执行分割、人工修正和导出。 |
| `docs/接口与开发/前端插件任务清单与接口方案.md` | 前后端接口对接方案。 | 作为系统组和模型组协作时的接口参考。 |

### user/launcher

| 文件 | 功能 | 当前更新内容 |
| --- | --- | --- |
| `user/launcher/一键启动服务.bat` | Windows 下一键启动本地 ToothSeg 语义推理服务。 | 固定启动 `toothseg_server.py`，启动后服务地址为 `http://127.0.0.1:8000/api/v1`。 |
| `user/launcher/环境说明.md` | 本地运行环境说明。 | 说明需要 Python、FastAPI、nnU-Net、PyTorch、SimpleITK 等依赖，以及常见端口占用问题。 |

### user/plugin

| 文件 | 功能 | 当前更新内容 |
| --- | --- | --- |
| `user/plugin/CBCTAnnotator.py` | 3D Slicer 插件主界面。 | 使用更接近 Slicer/nnInteractive 的原生 Qt 风格；支持服务自动检测、标签集合下拉管理、重命名/删除标签集合、空标签列表逻辑、分割进度条、中止分割、复用包管理、可调降采样间距。 |
| `user/plugin/lib/ApiClient.py` | 插件访问后端的 HTTP 客户端。 | 封装 `/status`、`/config`、`/cases`、`/images/inspect`、`/predict`、`/predict/progress`、`/predict/cancel`、`/reuse/status`、`/reuse/delete`、`/check_label`、`/export`、`/agent/log` 等接口；新增 `spacing_mm` 参数传递。 |

### implementation/server/inference

| 文件 | 功能 | 当前更新内容 |
| --- | --- | --- |
| `implementation/server/inference/README.md` | 后端推理服务说明。 | 说明真实 ToothSeg 服务和 mock 服务的启动、接口、依赖和常见问题。 |
| `implementation/server/inference/toothseg_server.py` | 真实 ToothSeg 语义推理 FastAPI 服务。 | 接收 Slicer 插件请求，调用 ToothSeg 语义分割；支持进度查询、中止请求、复用包检测/删除、标签质检、导出；新增 `spacing_mm` 降采样参数。 |
| `implementation/server/inference/mock_server.py` | 假后端联调服务。 | 不依赖真实模型，用 ROI 生成测试 mask，便于前端开发；保持与真实服务一致的接口字段。 |
| `implementation/server/inference/assets/label_spec_96.txt` | 后端使用的 96 类标签规范。 | 用于返回标签模板和标签颜色、名称信息。 |

### implementation/model

| 文件 | 功能 | 当前更新内容 |
| --- | --- | --- |
| `implementation/model/README.md` | 模型接入说明。 | 说明当前主流程是 ToothSeg Dataset121 语义分割，完整双分支为未来高级模式。 |
| `implementation/model/toothseg_semantic.py` | ToothSeg 语义分割适配层。 | 完成输入复制、整图哈希、复用包管理、降采样、nnU-Net 调用、结果检查、标签映射和结果登记；新增可配置 `spacing_mm`。 |
| `implementation/model/toothseg/run_toothseg.py` | ToothSeg 完整流程入口。 | 保留 full/sem/inst 三种模式入口，当前主服务暂不直接运行 full。 |
| `implementation/model/toothseg/memsafe_inference.py` | 低显存推理器。 | 作为未来完整双分支高级模式的显存保护方案。 |
| `implementation/model/toothseg/postprocess_predictions/__init__.py` | 后处理包初始化。 | 保留包结构。 |
| `implementation/model/toothseg/postprocess_predictions/assign_majority_tooth_labels.py` | 多数投票牙位分配脚本。 | 未来实例级后处理预留。 |
| `implementation/model/toothseg/postprocess_predictions/assign_mincost_tooth_labels.py` | 最小代价牙位分配脚本。 | 未来将实例分割结果映射到牙位编号时使用。 |
| `implementation/model/toothseg/postprocess_predictions/border_core_to_instances.py` | 将边界/核心预测转为实例的脚本。 | 未来完整 ToothSeg 实例分支预留。 |
| `implementation/model/toothseg/postprocess_predictions/resize_predictions.py` | 将预测结果重采样回参考图像空间。 | 未来完整流程输出回原始 CBCT 空间时使用。 |
| `implementation/model/toothseg/postprocess_predictions/sitk_compat.py` | SimpleITK 兼容工具。 | 解决不同环境下 SimpleITK 接口差异。 |
| `implementation/model/toothseg/toothseg/__init__.py` | ToothSeg 迷你包初始化。 | 保留必要包结构。 |
| `implementation/model/toothseg/toothseg/datasets/__init__.py` | 数据集工具包初始化。 | 保留必要包结构。 |
| `implementation/model/toothseg/toothseg/datasets/inhouse_dataset/__init__.py` | 内部数据集工具包初始化。 | 保留必要包结构。 |
| `implementation/model/toothseg/toothseg/datasets/inhouse_dataset/utils.py` | 图像几何信息复制等工具函数。 | 后处理脚本依赖的轻量工具。 |
| `implementation/model/toothseg/toothseg/datasets/toothfairy2/__init__.py` | ToothFairy2 数据集工具包初始化。 | 保留必要包结构。 |
| `implementation/model/toothseg/toothseg/datasets/toothfairy2/fdi_pair_distrs.json` | 牙位关系先验数据。 | 完整双分支牙位分配阶段预留。 |
| `implementation/model/toothseg/toothseg/datasets/toothfairy2/gt_instances.py` | ToothFairy2 实例标签处理脚本。 | 未来训练/后处理研究预留。 |
| `implementation/model/toothseg/toothseg/datasets/toothfairy2/splits_final.json` | 数据集划分文件。 | ToothSeg 原项目训练流程参考。 |
| `implementation/model/toothseg/toothseg/datasets/toothfairy2/toothfairy2.py` | ToothFairy2 数据集转换脚本。 | 未来理解 ToothSeg 数据准备流程时参考。 |

### implementation/agent

| 文件 | 功能 | 当前更新内容 |
| --- | --- | --- |
| `implementation/agent/README.md` | Agent 层说明。 | 当前为预留说明，后续用于记录推理模式推荐、日志总结、标签质检和半监督回流策略。 |

## 不上传到 GitHub 的内容

以下内容不要提交到普通 GitHub 仓库：

```text
ToothSeg.zip                         下载的模型/项目压缩包，体积过大
ToothSeg/                            完整第三方 ToothSeg 下载目录，易混入权重和缓存
02/                                  本地病例数据目录，包含医学影像/标签
demo_case/                           演示病例数据目录，包含医学影像
data/inputs/                         原始输入数据
data/outputs/                        推理、导出、日志等运行产物
outputs/                             运行产物
implementation/model/weights/        nnU-Net / ToothSeg 权重
implementation/model/work/           nnU-Net 工作缓存
D:\ToothSegWork\_runtime\            本机运行缓存、复用包、预测结果
__pycache__/                         Python 编译缓存
*.nii / *.nii.gz / *.nrrd / *.mha    医学影像或标签文件
*.zip                                大型压缩包，默认不入库
```

如果确实需要共享模型权重或演示病例，应使用受控网盘、学校内网盘、阿里云 OSS、Git LFS 或私有存储，并明确脱敏要求。
