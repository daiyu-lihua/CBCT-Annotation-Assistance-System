"""本地假后端（mock server）。

B 组前端联调用：实现与「统一接口协议」一致的全部 9 个后端接口，
返回模拟结果（/predict 会在 ROI 内生成一个实心立方体假 mask）。

在真实模型 / A 组推理脚本接入前，用它先把 Slicer 前端
"发送请求 -> 接收 mask -> 显示"整条链路调通。

启动：
    python implementation/server/inference/mock_server.py
服务地址：http://127.0.0.1:8000/api/v1
"""

import os
import time
import uuid
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

BASE_PATH = "/api/v1"
# 输出目录：项目仓库 data/outputs/mock_masks（不入库，按 .gitignore 排除）
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "outputs" / "mock_masks"

app = FastAPI(title="CBCT Mock Server", version="0.1.0")


def _log(event: str, payload: dict):
    print(f"[mock-server] {event}: {payload}", flush=True)


# ---------- 请求体 ----------

class CreateCaseRequest(BaseModel):
    image_path: str
    image_format: str
    label_template_id: str
    operator: str


class InspectRequest(BaseModel):
    case_id: str
    image_path: str


class RecommendModeRequest(BaseModel):
    case_id: str
    roi: dict
    target: str


class PredictRequest(BaseModel):
    case_id: str
    image_path: str
    roi: dict                 # {"start": [x,y,z], "size": [x,y,z]} (IJK)
    model_id: str
    mode: str
    targets: list
    output_format: str = "nii.gz"
    output_dir: Optional[str] = None


class CheckLabelRequest(BaseModel):
    case_id: str
    label_path: str
    label_template_id: str
    checks: Optional[list] = None


class ExportRequest(BaseModel):
    case_id: str
    image_path: str
    label_path: str
    export_format: str
    include_report: bool = True
    output_dir: Optional[str] = None


class LogRequest(BaseModel):
    case_id: str
    event: str
    operator: str
    payload: dict = {}


# ---------- 通用工具 ----------

def _err(error_code: str, message: str, details=None):
    return {"status": "error", "error_code": error_code,
            "message": message, "details": details or {}}


def _ok(**kw):
    return {"status": "ok", **kw}


def _load_image(path: str):
    """用 nibabel 读取 nii，返回 (data, affine)。"""
    img = nib.load(str(path))
    return img.get_fdata().astype(np.float32), img.affine


# ---------- 1. 服务状态 ----------

@app.get(BASE_PATH + "/status")
def status():
    return _ok(
        service={"name": "cbct-mock-server", "version": "0.1.0"},
        model={"loaded": True, "name": "mock-simple-cube"},
        device={"type": "cpu", "data": "mock"},
    )


# ---------- 2. 配置 ----------

@app.get(BASE_PATH + "/config")
def config():
    return _ok(
        models=[
            {"model_id": "teeth-seg-unet", "name": "牙齿分割 3D U-Net"},
        ],
        modes=[
            {"id": "fast", "name": "快速模式"},
            {"id": "balanced", "name": "均衡模式"},
            {"id": "fine", "name": "精细模式"},
        ],
        label_templates=[
            {"template_id": "teeth-16", "name": "16 颗牙模板"},
            {"template_id": "teeth-32", "name": "32 颗牙模板"},
        ],
    )


# ---------- 3. 病例初始化 ----------

@app.post(BASE_PATH + "/cases")
def create_case(req: CreateCaseRequest):
    case_id = "case-" + uuid.uuid4().hex[:8]
    state_path = str(OUTPUT_DIR / f"{case_id}_state.json")
    _log("create_case", {"case_id": case_id, "image": req.image_path})
    return _ok(case_id=case_id, case_state_path=state_path)


# ---------- 4. 图像信息检查 ----------

@app.post(BASE_PATH + "/images/inspect")
def inspect_image(req: InspectRequest):
    if not os.path.exists(req.image_path):
        return _err("FILE_NOT_FOUND", f"图像不存在: {req.image_path}")
    try:
        img = nib.load(req.image_path)
        shape = [int(s) for s in img.shape[:3]]
        spacing = [float(z) for z in img.header.get_zooms()[:3]]
    except Exception as e:
        return _err("READ_IMAGE_FAILED", f"读取图像失败: {e}")
    return _ok(shape=shape, spacing=spacing)


# ---------- 5. 推理模式推荐 ----------

@app.post(BASE_PATH + "/agent/recommend_mode")
def recommend_mode(req: RecommendModeRequest):
    # 规则式推荐：按 ROI 体积大小选模式
    sx, sy, sz = req.roi.get("size", [1, 1, 1])
    volume = sx * sy * sz
    if volume <= 32 ** 3:
        mode = "fine"
        reason = "目标区域较小，可选用精细模式提升细节"
    elif volume <= 96 ** 3:
        mode = "balanced"
        reason = "目标区域适中，均衡模式性价比最高"
    else:
        mode = "fast"
        reason = "目标区域较大，为省算力建议快速模式"
    return _ok(mode=mode, reason=reason)


# ---------- 6. AI 初分割 ----------

@app.post(BASE_PATH + "/predict")
def predict(req: PredictRequest):
    if not os.path.exists(req.image_path):
        return _err("FILE_NOT_FOUND", f"图像不存在: {req.image_path}")
    try:
        data, affine = _load_image(req.image_path)
    except Exception as e:
        return _err("READ_IMAGE_FAILED", f"读取图像失败: {e}")

    dh, dw, dz = data.shape
    start = [int(v) for v in req.roi.get("start", [0, 0, 0])]
    size = [int(v) for v in req.roi.get("size", [dh, dw, dz])]
    # 钳制到图像边界
    start = [min(s, d - 1) for s, d in zip(start, (dh, dw, dz))]
    end = [min(s + si, d) for s, si, d in zip(start, size, (dh, dw, dz))]

    # 假 mask：全 0，仅 ROI 范围填 label（模拟"AI 分割出牙齿区域"）
    label = np.zeros((dh, dw, dz), dtype=np.uint8)
    label[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = 1

    out_dir = Path(req.output_dir) if req.output_dir else OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    pred_id = uuid.uuid4().hex[:8]
    # mock 统一输出 .nii.gz（nibabel 按扩展名识别格式）
    mask_path = out_dir / f"{pred_id}_mask.nii.gz"
    nib.save(nib.Nifti1Image(label, affine), str(mask_path))
    conf_path = out_dir / (pred_id + "_confidence.nii.gz")
    nib.save(nib.Nifti1Image(
        np.full((dh, dw, dz), 0.95, dtype=np.float32), affine), str(conf_path))

    _log("predict", {"prediction_id": pred_id, "roi_start": start,
                     "roi_size": size})
    return _ok(
        prediction_id=pred_id,
        mask_path=str(mask_path),
        confidence_path=str(conf_path),
        model_id=req.model_id,
        mode=req.mode,
    )


# ---------- 7. 标签质量检查 ----------

@app.post(BASE_PATH + "/check_label")
def check_label(req: CheckLabelRequest):
    issues = []  # mock 返回空问题
    return _ok(checked=[c for c in (req.checks or [])], issues=issues,
               passed=len(issues) == 0)


# ---------- 8. 结果导出 ----------

@app.post(BASE_PATH + "/export")
def export(req: ExportRequest):
    out_dir = Path(req.output_dir) if req.output_dir else OUTPUT_DIR / "export"
    os.makedirs(out_dir, exist_ok=True)
    t = time.strftime("%Y%m%d_%H%M%S")
    export_dir = out_dir / f"{req.case_id}_{t}"
    os.makedirs(export_dir, exist_ok=True)
    files = []
    if req.label_path and os.path.exists(req.label_path):
        dst = export_dir / os.path.basename(req.label_path)
        import shutil
        shutil.copy(req.label_path, dst)
        files.append(str(dst))
    if req.include_report:
        report = export_dir / "report.json"
        report.write_text(
            '{"summary": "mock 导出报告", "issues": []}',
            encoding="utf-8")
        files.append(str(report))
    return _ok(export_dir=str(export_dir), files=files, format=req.export_format)


# ---------- 9. 任务日志 ----------

@app.post(BASE_PATH + "/agent/log")
def task_log(req: LogRequest):
    _log("agent/log", {"case_id": req.case_id, "event": req.event,
                       "operator": req.operator, "payload": req.payload})
    return _ok(recorded=1, event=req.event)


if __name__ == "__main__":
    import uvicorn
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"CBCT mock server -> http://127.0.0.1:8000{BASE_PATH}")
    # 宿主机访问，绑定 127.0.0.1
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")