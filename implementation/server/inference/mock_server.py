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
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

BASE_PATH = "/api/v1"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from implementation.model.toothseg_semantic import (  # noqa: E402
    PredictionCancelled,
    RUNTIME_ROOT,
    delete_reuse_package,
    inspect_reuse_package,
    run_toothseg_semantic,
    stage_image_for_reading,
    toothseg_status,
)

# 输出目录：默认放入英文 runtime，避免联调时读取中文路径失败。
OUTPUT_DIR = Path(os.environ.get("CBCT_MOCK_OUTPUT", str(RUNTIME_ROOT / "mock_outputs")))

# 新的 96 类 ITKSNAP 稠密标签规范（队友提供，已收录进项目 assets）
LABEL_SPEC_FILE = Path(__file__).resolve().parent / "assets" / "label_spec_96.txt"
# 统一模板 ID：1xx-6xx 共 96 类（上/下颌 × 牙体/牙髓/种植体 × 16 位置）
LABEL_TEMPLATE_ID = "teeth-dense-96"

app = FastAPI(title="CBCT Mock Server", version="0.1.0")
_PROGRESS_LOCK = threading.Lock()
_PROGRESS_BY_CASE = {}
_CANCEL_LOCK = threading.Lock()
_CANCEL_EVENTS = {}
_CANCEL_KEEP_REUSE = {}


def _log(event: str, payload: dict):
    print(f"[mock-server] {event}: {payload}", flush=True)


# ---------- 96 类标签规范 ----------

def _parse_label_spec(path: Path):
    """解析 ITKSNAP 标签描述文件，返回 [{id, code, category, position, name, color}]。

    文件行形如："    1    84  130  242     1.00  1  1    "101_UpperTooth_Pos01""
    LABEL 语义码：1xx=上颌牙体 2xx=上颌牙髓 3xx=上颌种植体
                4xx=下颌牙体 5xx=下颌牙髓 6xx=下颌种植体；xx=01-16 位置号
    IDX 稠密 1-96，供模型训练使用；标注语义以 LABEL 语义码为准。
    """
    labels = []
    if not path.exists():
        return labels
    line_re = re.compile(r'^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+[\d.]+\s+\d+\s+\d+\s+"(.*)"\s*$')
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = line_re.match(raw)
        if not m or m.group(1) == "0":
            continue  # 跳过表头 / 注释 / Clear Label(0)
        idx = int(m.group(1))
        color = [int(m.group(2)), int(m.group(3)), int(m.group(4))]
        name = m.group(5)
        parts = name.split("_")
        code = parts[0]
        category = parts[1] if len(parts) > 1 else ""
        position = parts[2] if len(parts) > 2 else ""
        labels.append({
            "id": idx,
            "code": code,
            "category": category,
            "position": position,
            "name": name,
            "color": color,
        })
    return labels


LABEL_SPEC = _parse_label_spec(LABEL_SPEC_FILE)


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
    keep_reuse: bool = True
    spacing_mm: Optional[float] = None


class ReuseRequest(BaseModel):
    image_path: str
    model_id: str = "toothseg-semantic-05mm"
    mode: str = "balanced"
    spacing_mm: Optional[float] = None


class CancelPredictRequest(BaseModel):
    case_id: str
    image_path: Optional[str] = None
    model_id: str = "toothseg-semantic-05mm"
    mode: str = "balanced"
    keep_reuse: bool = True
    spacing_mm: Optional[float] = None


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


def _set_progress(case_id: str, percent: int, stage: str, message: str,
                  run_status: str = "running", details=None):
    record = {
        "case_id": case_id,
        "percent": max(0, min(100, int(percent))),
        "stage": stage,
        "message": message,
        "run_status": run_status,
        "updated_at": time.strftime("%F %T"),
        "details": details or {},
    }
    with _PROGRESS_LOCK:
        prev = _PROGRESS_BY_CASE.get(case_id, {})
        if prev.get("started_at") and run_status in {"running", "success", "error", "cancelling", "cancelled"}:
            record["started_at"] = prev["started_at"]
        elif run_status == "running":
            record["started_at"] = time.strftime("%F %T")
        _PROGRESS_BY_CASE[case_id] = record
    _log("progress", record)
    return record


def _register_cancel_event(case_id: str):
    with _CANCEL_LOCK:
        event = threading.Event()
        _CANCEL_EVENTS[case_id] = event
        _CANCEL_KEEP_REUSE.pop(case_id, None)
        return event


def _request_cancel(case_id: str, keep_reuse: bool):
    with _CANCEL_LOCK:
        _CANCEL_KEEP_REUSE[case_id] = bool(keep_reuse)
        event = _CANCEL_EVENTS.get(case_id)
        if event is None:
            return False
        event.set()
        return True


def _cancel_keep_reuse(case_id: str, default: bool):
    with _CANCEL_LOCK:
        return bool(_CANCEL_KEEP_REUSE.get(case_id, default))


def _clear_cancel_event(case_id: str):
    with _CANCEL_LOCK:
        _CANCEL_EVENTS.pop(case_id, None)
        _CANCEL_KEEP_REUSE.pop(case_id, None)


def _load_image(path: str):
    """用 nibabel 读取 nii，返回 (data, affine)。"""
    img = nib.load(str(path))
    return img.get_fdata().astype(np.float32), img.affine


# ---------- 1. 服务状态 ----------

@app.get(BASE_PATH + "/status")
def status():
    toothseg = toothseg_status()
    return _ok(
        service={"name": "cbct-local-inference-server", "version": "0.2.0"},
        model={
            "loaded": True,
            "default": "mock-simple-cube",
            "toothseg_semantic_available": toothseg["available"],
            "toothseg": toothseg,
            "reuse_package_supported": True,
        },
        device={"type": "cuda", "data": "cuda", "fallback": "cpu"},
    )


@app.get(BASE_PATH + "/predict/progress/{case_id}")
def predict_progress(case_id: str):
    with _PROGRESS_LOCK:
        progress = dict(_PROGRESS_BY_CASE.get(case_id, {}))
    if not progress:
        return _ok(
            case_id=case_id,
            percent=0,
            stage="idle",
            message="当前病例没有正在记录的推理进度。",
            run_status="idle",
            details={},
        )
    return _ok(**progress)


@app.post(BASE_PATH + "/predict/cancel")
def cancel_predict(req: CancelPredictRequest):
    requested = _request_cancel(req.case_id, req.keep_reuse)
    if requested:
        _set_progress(
            req.case_id,
            0,
            "cancel_requested",
            "已收到中止请求，正在通知后端推理进程停止。",
            "cancelling",
            {
                "keep_reuse": bool(req.keep_reuse),
                "model_id": req.model_id,
                "mode": req.mode,
            },
        )
        return _ok(
            case_id=req.case_id,
            cancel_requested=True,
            keep_reuse=bool(req.keep_reuse),
            message="已请求中止当前推理任务。",
        )
    return _ok(
        case_id=req.case_id,
        cancel_requested=False,
        keep_reuse=bool(req.keep_reuse),
        message="当前病例没有正在运行的后端推理任务。",
    )


# ---------- 2.1 复用包检测 / 删除 ----------

@app.post(BASE_PATH + "/reuse/status")
def reuse_status(req: ReuseRequest):
    try:
        info = inspect_reuse_package(req.image_path, req.model_id, req.mode, req.spacing_mm)
    except FileNotFoundError as e:
        return _err("FILE_NOT_FOUND", str(e))
    except Exception as e:
        return _err("REUSE_CHECK_FAILED", f"复用包检测失败: {e}")
    return _ok(**info)


@app.post(BASE_PATH + "/reuse/delete")
def reuse_delete(req: ReuseRequest):
    try:
        result = delete_reuse_package(req.image_path)
    except Exception as e:
        return _err("REUSE_DELETE_FAILED", f"复用包删除失败: {e}")
    return _ok(**result)


# ---------- 2. 配置 ----------

@app.get(BASE_PATH + "/config")
def config():
    return _ok(
        models=[
            {"model_id": "mock-simple-cube", "name": "假模型：ROI 实心方块联调"},
            {"model_id": "toothseg-semantic-05mm", "name": "ToothSeg 语义分割 0.5mm"},
        ],
        modes=[
            {"id": "fast", "name": "快速模式"},
            {"id": "balanced", "name": "均衡模式"},
            {"id": "fine", "name": "精细模式"},
        ],
        downsample={
            "field": "spacing_mm",
            "unit": "mm",
            "min": 0.5,
            "max": 2.0,
            "step": 0.05,
            "default": 0.75,
            "note": "mock 服务接收该参数但不真正降采样；真实 ToothSeg 服务会使用它。",
        },
        label_templates=[
            {
                "template_id": LABEL_TEMPLATE_ID,
                "name": "96 类牙体稠密标签 (ITKSNAP)",
                "label_count": len(LABEL_SPEC),
                "labels": LABEL_SPEC,
            },
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
        staged_path = stage_image_for_reading(req.image_path, "input_cache")
        img = nib.load(staged_path)
        shape = [int(s) for s in img.shape[:3]]
        spacing = [float(z) for z in img.header.get_zooms()[:3]]
    except Exception as e:
        return _err("READ_IMAGE_FAILED", f"读取图像失败: {e}")
    return _ok(shape=shape, spacing=spacing, staged_path=staged_path)


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
    _set_progress(req.case_id, 1, "accepted", "后端已接收分割请求，正在进入校验流程。")
    if not os.path.exists(req.image_path):
        _set_progress(req.case_id, 0, "error", f"图像不存在: {req.image_path}", "error")
        return _err("FILE_NOT_FOUND", f"图像不存在: {req.image_path}")

    _log("predict_request", {
        "case_id": req.case_id,
        "model_id": req.model_id,
        "mode": req.mode,
        "roi": req.roi,
        "image": req.image_path,
    })

    if req.model_id == "toothseg-semantic-05mm":
        cancel_event = _register_cancel_event(req.case_id)
        try:
            def progress_callback(event: dict):
                _set_progress(
                    req.case_id,
                    event.get("percent", 0),
                    event.get("stage", "running"),
                    event.get("message", "后端正在处理。"),
                    "running",
                    event.get("details", {}),
                )

            result = run_toothseg_semantic(
                image_path=req.image_path,
                case_id=req.case_id,
                mode=req.mode,
                spacing_mm=req.spacing_mm,
                output_dir=req.output_dir,
                device="cuda",
                keep_reuse=req.keep_reuse,
                progress_callback=progress_callback,
                cancel_checker=cancel_event.is_set,
            )
        except FileNotFoundError as e:
            _set_progress(req.case_id, 0, "error", str(e), "error")
            return _err("FILE_NOT_FOUND", str(e))
        except PredictionCancelled as e:
            keep_reuse_after_cancel = _cancel_keep_reuse(req.case_id, req.keep_reuse)
            reuse_action = {"keep_reuse": keep_reuse_after_cancel}
            if keep_reuse_after_cancel:
                try:
                    reuse_action["reuse_status"] = inspect_reuse_package(
                        req.image_path, req.model_id, req.mode, req.spacing_mm)
                except Exception as reuse_error:
                    reuse_action["reuse_status_error"] = str(reuse_error)
            else:
                try:
                    reuse_action["delete_result"] = delete_reuse_package(req.image_path)
                except Exception as delete_error:
                    reuse_action["delete_error"] = str(delete_error)
            _set_progress(req.case_id, 0, "cancelled", "用户已中止本次分割。", "cancelled", reuse_action)
            return _err("PREDICTION_CANCELLED", str(e), reuse_action)
        except Exception as e:
            _set_progress(req.case_id, 0, "error", str(e), "error", {"model_id": req.model_id})
            return _err("PREDICTION_FAILED", str(e), {"model_id": req.model_id})
        finally:
            _clear_cancel_event(req.case_id)
        _set_progress(req.case_id, 100, "done", "后端分割完成，结果已返回前端。", "success",
                      {"prediction_id": result["prediction_id"]})
        return _ok(
            prediction_id=result["prediction_id"],
            mask_path=result["mask_path"],
            confidence_path=None,
            model_id=req.model_id,
            mode=req.mode,
            spacing_mm=result.get("spacing_mm"),
            work_dir=result.get("work_dir"),
            log_path=result.get("log_path"),
            mask_info=result.get("mask_info"),
            roi_used=False,
            message="ToothSeg 语义分割完成。当前版本按整张降采样 CBCT 推理，ROI 暂用于流程记录。",
            reuse_dir=result.get("reuse_dir"),
            reuse_card=result.get("reuse_card"),
            task_key=result.get("task_key"),
            reused=result.get("reused", False),
        )

    _log("mock_predict_start", {"message": "当前选择的是假模型，将在 ROI 内生成实心方块 mask"})
    _set_progress(req.case_id, 20, "read_image", "mock 正在读取输入影像。")
    try:
        data, affine = _load_image(req.image_path)
    except Exception as e:
        _set_progress(req.case_id, 0, "error", f"读取图像失败: {e}", "error")
        return _err("READ_IMAGE_FAILED", f"读取图像失败: {e}")

    _set_progress(req.case_id, 55, "make_mock_mask", "mock 正在生成测试 mask。")
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
                     "roi_size": size, "mask_path": str(mask_path)})
    _set_progress(req.case_id, 100, "done", "mock 分割完成，结果已返回前端。", "success",
                  {"prediction_id": pred_id})
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
