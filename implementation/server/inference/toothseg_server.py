# -*- coding: utf-8 -*-
"""CBCT ToothSeg 语义分割推理服务端。

与 mock_server.py 实现相同的 9 个统一接口（统一接口协议 v1），
区别在于 /predict 真正调用 ToothSeg 语义分割模型：

    implementation/model/toothseg_semantic.py        (Dataset121 语义分支适配)
    自动发现 nnUNet_results 下的 Dataset121...       (语义分支权重)

设计要点
--------
1. 当前主流程只运行 ToothSeg Dataset121 语义分割分支，用于自动生成牙位语义
   标签图；不运行实例分支，也不运行完整 ToothSeg 后处理。
2. GPU 独占：同一时刻只允许一个 /predict（线程锁），第二个请求立即
   返回 PREDICT_IN_PROGRESS，避免并发推理挤爆显存。
3. 本服务进程自身绝不 import torch / 不初始化 CUDA（/status 用
   nvidia-smi 探测显卡），避免长期占用 8GB 显存。
4. 语义适配器会把输入复制到英文工作目录、按模式降采样，并生成 nnU-Net
   需要的 *_0000.nii.gz 输入；ROI 仅记录不裁剪。
5. 未来高级模式 toothseg-full 只保留接口占位，当前不会执行完整双分支。

启动：
    E:\\miniconda3\\envs\\nninteractive\\python.exe implementation/server/inference/toothseg_server.py
服务地址：http://127.0.0.1:8000/api/v1
"""

import json
import os
import re
import subprocess
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
    set_custom_model_root,
    stage_image_for_reading,
    toothseg_status,
)

# 运行产物目录：默认放入英文 runtime，避免医学影像库读取中文路径失败。
OUTPUT_DIR = Path(os.environ.get("CBCT_SERVER_OUTPUT", str(RUNTIME_ROOT / "server_outputs")))
JOBS_DIR = OUTPUT_DIR / "jobs"
AGENT_LOG = OUTPUT_DIR / "agent_log.jsonl"

# 96 类标签规范（与 mock_server 共用 implementation/server/inference/assets/）
LABEL_SPEC_FILE = Path(__file__).resolve().parent / "assets" / "label_spec_96.txt"
LABEL_TEMPLATE_ID = "teeth-dense-96"

# ToothSeg 模型接入位置。weights 是旧版本地兜底目录，实际权重根目录由
# implementation.model.toothseg_semantic.toothseg_status() 自动解析。
TOOTHSEG_DIR = PROJECT_ROOT / "implementation" / "model" / "toothseg"
LEGACY_WEIGHTS_DIR = PROJECT_ROOT / "implementation" / "model" / "weights"
SEMSEG_CP = (LEGACY_WEIGHTS_DIR / "Dataset121_ToothFairy2_Teeth" /
             "nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_256_bs8_ctnorm" /
             "fold_5" / "checkpoint_final.pth")
INSTSEG_CP = (LEGACY_WEIGHTS_DIR / "Dataset123_ToothFairy2fixed_teeth_spacing02_brd3px" /
              "nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8_ctnorm" /
              "fold_5" / "checkpoint_final.pth")

SEMANTIC_MODEL_ID = "toothseg-semantic-05mm"
ADVANCED_MODEL_ID = "toothseg-full"
MODEL_ID = SEMANTIC_MODEL_ID
# FDI 牙位合法值（11-18, 21-28, 31-38, 41-48）
FDI_LABELS = {q * 10 + i for q in (1, 2, 3, 4) for i in range(1, 9)}

app = FastAPI(title="CBCT ToothSeg Server", version="1.0.0")
_PREDICT_LOCK = threading.Lock()
_PROGRESS_LOCK = threading.Lock()
_PROGRESS_BY_CASE = {}
_CANCEL_LOCK = threading.Lock()
_CANCEL_EVENTS = {}
_CANCEL_KEEP_REUSE = {}


def _log(event: str, payload: dict):
    print(f"[toothseg-server] {event}: {payload}", flush=True)


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


def _get_progress(case_id: str):
    with _PROGRESS_LOCK:
        return dict(_PROGRESS_BY_CASE.get(case_id, {}))


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


def _gpu_info():
    """用 nvidia-smi 探测 GPU（绝不初始化 CUDA context）。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8)
        if out.returncode == 0 and out.stdout.strip():
            name, total, free = [x.strip() for x in out.stdout.splitlines()[0].split(",")]
            return {"type": "cuda", "name": name,
                    "memory_total_mb": int(float(total)),
                    "memory_free_mb": int(float(free))}
    except Exception:
        pass
    return {"type": "cpu", "name": "nvidia-smi 不可用"}


# ---------- 96 类标签规范（解析逻辑与 mock_server 一致） ----------

def _parse_label_spec(path: Path):
    labels = []
    if not path.exists():
        return labels
    line_re = re.compile(
        r'^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+[\d.]+\s+\d+\s+\d+\s+"(.*)"\s*$')
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = line_re.match(raw)
        if not m or m.group(1) == "0":
            continue
        idx = int(m.group(1))
        color = [int(m.group(2)), int(m.group(3)), int(m.group(4))]
        name = m.group(5)
        parts = name.split("_")
        labels.append({
            "id": idx, "code": parts[0],
            "category": parts[1] if len(parts) > 1 else "",
            "position": parts[2] if len(parts) > 2 else "",
            "name": name, "color": color,
        })
    return labels


LABEL_SPEC = _parse_label_spec(LABEL_SPEC_FILE)


# ---------- 请求体（与统一接口协议一致） ----------

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
    model_id: str = SEMANTIC_MODEL_ID
    mode: str = "balanced"
    spacing_mm: Optional[float] = None


class CancelPredictRequest(BaseModel):
    case_id: str
    image_path: Optional[str] = None
    model_id: str = SEMANTIC_MODEL_ID
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


class SetModelPathRequest(BaseModel):
    model_path: str


# ---------- 1. 服务状态 ----------

@app.get(BASE_PATH + "/status")
def status():
    toothseg = toothseg_status()
    return _ok(
        service={"name": "cbct-toothseg-semantic-server", "version": "1.0.0"},
        model={
            "loaded": bool(toothseg["available"]),
            "name": MODEL_ID,
            "default": SEMANTIC_MODEL_ID,
            "advanced_available": False,
            "advanced_model_id": ADVANCED_MODEL_ID,
            "toothseg_semantic": toothseg,
            "missing_checkpoints": (
                [] if toothseg.get("checkpoint_exists")
                else [toothseg.get("checkpoint_path", str(SEMSEG_CP))]
            ),
            "help_message": toothseg.get("help_message", ""),
        },
        device=_gpu_info(),
    )


@app.post(BASE_PATH + "/model/set_path")
def set_model_path(req: SetModelPathRequest):
    try:
        set_custom_model_root(req.model_path)
        toothseg = toothseg_status()
        return _ok(
            message="模型目录设置成功" if toothseg["checkpoint_exists"] else "已更新模型目录，但未能检测到有效权重",
            model_path=req.model_path,
            toothseg=toothseg,
        )
    except Exception as exc:
        return _err("SET_MODEL_PATH_FAILED", f"设置模型目录失败: {exc}")



@app.get(BASE_PATH + "/predict/progress/{case_id}")
def predict_progress(case_id: str):
    progress = _get_progress(case_id)
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


# ---------- 2. 配置 ----------

@app.get(BASE_PATH + "/config")
def config():
    return _ok(
        models=[
            {
                "model_id": SEMANTIC_MODEL_ID,
                "name": "ToothSeg 语义分割（当前主流程，只输出牙位语义标签）",
                "task": "tooth_semantic_segmentation",
                "enabled": True,
            },
            {
                "model_id": ADVANCED_MODEL_ID,
                "name": "ToothSeg 完整双分支（未来高级模式，当前未启用）",
                "task": "tooth_instance_segmentation",
                "enabled": False,
                "note": "保留接口占位；当前项目不运行实例分支。",
            },
        ],
        modes=[
            {"id": "fast", "name": "快速模式（语义分割 0.75mm 降采样）"},
            {"id": "balanced", "name": "均衡模式（语义分割 0.5mm 降采样，默认）"},
            {"id": "fine", "name": "精细模式（当前与 0.5mm 语义推理一致，预留增强）"},
        ],
        downsample={
            "field": "spacing_mm",
            "unit": "mm",
            "min": 0.5,
            "max": 2.0,
            "step": 0.05,
            "default": 0.75,
            "note": "前端传入 spacing_mm 时优先使用该值；值越大，显存占用越低，细节越少。",
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


# ---------- 2.1 复用包检测 / 删除 ----------

@app.post(BASE_PATH + "/reuse/status")
def reuse_status(req: ReuseRequest):
    try:
        info = inspect_reuse_package(req.image_path, req.model_id, req.mode, req.spacing_mm)
    except FileNotFoundError as e:
        return _err("IMAGE_NOT_FOUND", str(e))
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


# ---------- 3. 病例初始化 ----------

@app.post(BASE_PATH + "/cases")
def create_case(req: CreateCaseRequest):
    case_id = "case-" + uuid.uuid4().hex[:8]
    case_dir = JOBS_DIR / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    state_path = str(case_dir / "case_state.json")
    (case_dir / "case_state.json").write_text(json.dumps({
        "case_id": case_id, "image_path": req.image_path,
        "image_format": req.image_format,
        "label_template_id": req.label_template_id,
        "operator": req.operator, "created_at": time.strftime("%F %T"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    _log("create_case", {"case_id": case_id, "image": req.image_path})
    return _ok(case_id=case_id, case_state_path=state_path)


# ---------- 4. 图像信息检查 ----------

@app.post(BASE_PATH + "/images/inspect")
def inspect_image(req: InspectRequest):
    if not os.path.exists(req.image_path):
        return _err("IMAGE_NOT_FOUND", f"图像不存在: {req.image_path}")
    try:
        staged_path = stage_image_for_reading(req.image_path, "input_cache")
        img = nib.load(staged_path)
        shape = [int(s) for s in img.shape[:3]]
        spacing = [float(z) for z in img.header.get_zooms()[:3]]
    except Exception as e:
        return _err("UNSUPPORTED_FORMAT", f"读取图像失败: {e}")
    return _ok(shape=shape, spacing=spacing, staged_path=staged_path)


# ---------- 5. 推理模式推荐 ----------

@app.post(BASE_PATH + "/agent/recommend_mode")
def recommend_mode(req: RecommendModeRequest):
    # 当前 ToothSeg 语义分割按全图推理；ROI 只影响推荐，不裁剪输入。
    sx, sy, sz = req.roi.get("size", [1, 1, 1])
    volume = abs(sx * sy * sz)
    if volume <= 128 ** 3:
        mode, reason = "balanced", "当前语义模型按全图推理；ROI 较小，默认使用 0.5mm 均衡模式"
    elif volume <= 256 ** 3:
        mode, reason = "balanced", "当前语义模型按全图推理；ROI 适中，建议 0.5mm 均衡模式"
    else:
        mode, reason = "fast", "当前语义模型按全图推理；区域较大时建议 0.75mm 快速模式缩短等待"
    return _ok(mode=mode, reason=reason)


# ---------- 6. AI 初分割（当前只运行 ToothSeg 语义分割） ----------

@app.post(BASE_PATH + "/predict")
def predict(req: PredictRequest):
    with _PROGRESS_LOCK:
        _PROGRESS_BY_CASE.pop(req.case_id, None)
    _set_progress(req.case_id, 1, "accepted", "后端已接收分割请求，正在进入校验流程。")
    model_id = (req.model_id or SEMANTIC_MODEL_ID).strip()
    if model_id == ADVANCED_MODEL_ID:
        _set_progress(req.case_id, 0, "error", "当前完整双分支模式暂未启用。", "error",
                      {"model_id": model_id})
        return _err(
            "MODEL_NOT_ENABLED",
            "当前项目阶段只运行 ToothSeg 语义分割；完整双分支高级模式已保留接口但暂未启用。",
            {"model_id": model_id, "enabled_model": SEMANTIC_MODEL_ID},
        )
    if model_id not in {SEMANTIC_MODEL_ID, "tooth_seg_v1"}:
        _set_progress(req.case_id, 0, "error", f"未知模型: {model_id}", "error",
                      {"supported": [SEMANTIC_MODEL_ID, ADVANCED_MODEL_ID]})
        return _err(
            "MODEL_NOT_FOUND",
            f"未知模型: {model_id}",
            {"supported": [SEMANTIC_MODEL_ID, ADVANCED_MODEL_ID]},
        )
    _set_progress(req.case_id, 3, "check_runtime", "正在检查 ToothSeg 运行环境。")
    toothseg = toothseg_status()
    if not toothseg.get("checkpoint_exists"):
        _set_progress(req.case_id, 0, "error", "ToothSeg 语义分支权重缺失。", "error",
                      {"missing": [toothseg.get("checkpoint_path", str(SEMSEG_CP))]})
        return _err(
            "MODEL_NOT_LOADED",
            "ToothSeg 语义分支权重缺失，请检查 nnUNet_results / TOOTHSEG_NNUNET_RESULTS / CBCT_NNUNET_RESULTS，或项目内 ToothSeg/nnUNet_results。",
            {
                "missing": [toothseg.get("checkpoint_path", str(SEMSEG_CP))],
                "searched": toothseg.get("nnunet_results_candidates", []),
            },
        )
    if not toothseg.get("predict_exe"):
        _set_progress(req.case_id, 0, "error", "未找到 nnUNetv2_predict。", "error",
                      {"error": toothseg.get("error")})
        return _err("MODEL_RUNTIME_NOT_READY", "未找到 nnUNetv2_predict，请使用包含 nnU-Net v2 的环境启动服务",
                    {"error": toothseg.get("error")})
    if not Path(req.image_path).exists():
        _set_progress(req.case_id, 0, "error", f"图像不存在: {req.image_path}", "error")
        return _err("IMAGE_NOT_FOUND", f"图像不存在: {req.image_path}")

    if not _PREDICT_LOCK.acquire(blocking=False):
        _set_progress(req.case_id, 0, "busy", "已有 ToothSeg 推理任务在运行。", "error",
                      {"hint": "GPU 独占，请等待当前任务完成后再试"})
        return _err("PREDICT_IN_PROGRESS",
                    "已有 ToothSeg 语义推理任务在运行（GPU 独占），请等待其完成后再试",
                    {"hint": "可检测当前图像的复用包，或等待任务完成"})
    cancel_event = _register_cancel_event(req.case_id)
    try:
        t0 = time.time()

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
        elapsed = round(time.time() - t0, 1)

        _log("predict_done", {"prediction_id": result["prediction_id"], "mask": result["mask_path"],
                              "elapsed_sec": elapsed})
        _set_progress(req.case_id, 100, "done", "后端分割完成，结果已返回前端。", "success",
                      {"prediction_id": result["prediction_id"], "elapsed_sec": elapsed})
        return _ok(
            prediction_id=result["prediction_id"],
            mask_path=result["mask_path"],
            raw_mask_path=result.get("raw_mask_path"),
            confidence_path=None,
            model_id=SEMANTIC_MODEL_ID,
            mode=req.mode,
            spacing_mm=result.get("spacing_mm"),
            work_dir=result.get("work_dir"),
            log_path=result.get("log_path"),
            result_path=result.get("result_path"),
            reuse_dir=result.get("reuse_dir"),
            reuse_card=result.get("reuse_card"),
            task_key=result.get("task_key"),
            reused=result.get("reused", False),
            mask_info=result.get("mask_info"),
            mapping_info=result.get("mapping_info"),
            roi_used=False,
            roi_note="当前 ToothSeg 语义分割按整张降采样 CBCT 推理，ROI 仅作为流程记录。",
            elapsed_sec=elapsed,
        )
    except FileNotFoundError as e:
        _set_progress(req.case_id, 0, "error", str(e), "error")
        return _err("IMAGE_NOT_FOUND", str(e))
    except PredictionCancelled as e:
        keep_reuse_after_cancel = _cancel_keep_reuse(req.case_id, req.keep_reuse)
        reuse_action = {"keep_reuse": keep_reuse_after_cancel}
        if keep_reuse_after_cancel:
            try:
                reuse_action["reuse_status"] = inspect_reuse_package(
                    req.image_path, model_id, req.mode, req.spacing_mm)
            except Exception as reuse_error:
                reuse_action["reuse_status_error"] = str(reuse_error)
        else:
            try:
                reuse_action["delete_result"] = delete_reuse_package(req.image_path)
            except Exception as delete_error:
                reuse_action["delete_error"] = str(delete_error)
        _set_progress(
            req.case_id,
            0,
            "cancelled",
            "用户已中止本次分割。",
            "cancelled",
            reuse_action,
        )
        return _err("PREDICTION_CANCELLED", str(e), reuse_action)
    except Exception as e:
        _set_progress(req.case_id, 0, "error", f"ToothSeg 语义分割失败: {e}", "error",
                      {"model_id": SEMANTIC_MODEL_ID})
        return _err("PREDICTION_FAILED", f"ToothSeg 语义分割失败: {e}",
                    {"model_id": SEMANTIC_MODEL_ID})
    finally:
        _clear_cancel_event(req.case_id)
        _PREDICT_LOCK.release()


# ---------- 7. 标签质量检查（真实检查） ----------

@app.post(BASE_PATH + "/check_label")
def check_label(req: CheckLabelRequest):
    checks = req.checks or []
    issues = []
    if not os.path.exists(req.label_path):
        return _err("LABEL_NOT_FOUND", f"标签文件不存在: {req.label_path}")
    try:
        img = nib.load(req.label_path)
        data = np.asanyarray(img.dataobj)
    except Exception as e:
        return _err("LABEL_FORMAT_ERROR", f"读取标签失败: {e}")
    vals = np.unique(data.astype(np.int64))
    vals = vals[vals > 0]
    if vals.size == 0:
        issues.append({"severity": "error", "check": "non_empty",
                       "message": "标签为空（无任何前景体素）"})
    if "label_range" in checks or not checks:
        allowed = set(range(1, 97)) | FDI_LABELS
        bad = sorted(int(v) for v in vals if int(v) not in allowed)
        if bad:
            issues.append({"severity": "warning", "check": "label_range",
                           "message": "存在不在标签规范内的取值",
                           "values": bad[:20]})
    issues.append({"severity": "info", "check": "summary",
                   "message": "标签统计",
                   "n_labels": int(vals.size),
                   "values": [int(v) for v in vals[:96]]})
    passed = not any(i.get("severity") == "error" for i in issues)
    return _ok(checked=checks, issues=issues, passed=passed)


# ---------- 8. 结果导出 ----------

@app.post(BASE_PATH + "/export")
def export(req: ExportRequest):
    out_dir = Path(req.output_dir) if req.output_dir else OUTPUT_DIR / "export"
    t = time.strftime("%Y%m%d_%H%M%S")
    export_dir = out_dir / f"{req.case_id}_{t}"
    export_dir.mkdir(parents=True, exist_ok=True)
    files = []
    if req.label_path and os.path.exists(req.label_path):
        dst = export_dir / os.path.basename(req.label_path)
        shutil.copy2(req.label_path, dst)
        files.append(str(dst))
    if req.include_report:
        report = {"summary": "ToothSeg 分割结果导出",
                  "case_id": req.case_id,
                  "label_path": req.label_path,
                  "image_path": req.image_path,
                  "export_format": req.export_format,
                  "model_stage": "toothseg_semantic_only",
                  "exported_at": time.strftime("%F %T")}
        try:
            lab = nib.load(req.label_path)
            d = np.asanyarray(lab.dataobj).astype(np.int64)
            vals = np.unique(d[d > 0])
            report["label_stats"] = {
                "shape": [int(s) for s in d.shape],
                "spacing": [float(z) for z in lab.header.get_zooms()[:3]],
                "n_foreground_voxels": int((d > 0).sum()),
                "unique_labels": [int(v) for v in vals[:96]],
            }
        except Exception as e:
            report["label_stats_error"] = str(e)
        rp = export_dir / "report.json"
        rp.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")
        files.append(str(rp))
    _log("export", {"case_id": req.case_id, "files": files})
    return _ok(export_dir=str(export_dir), files=files, format=req.export_format)


# ---------- 9. 任务日志 ----------

@app.post(BASE_PATH + "/agent/log")
def task_log(req: LogRequest):
    rec = {"ts": time.strftime("%F %T"), "case_id": req.case_id,
           "event": req.event, "operator": req.operator,
           "payload": req.payload}
    try:
        AGENT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(AGENT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        _log("agent/log_write_failed", {"error": str(e)})
    _log("agent/log", rec)
    return _ok(recorded=1, event=req.event)


if __name__ == "__main__":
    import uvicorn
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    toothseg = toothseg_status()
    gpu = _gpu_info()
    ckpt_ok = bool(toothseg.get("checkpoint_exists"))
    predict_ok = bool(toothseg.get("predict_exe"))

    print("\n" + "=" * 68, flush=True)
    if ckpt_ok and predict_ok:
        print("  >>> [SUCCESS] 模型权重检测成功！(ToothSeg 语义分割已就绪)", flush=True)
        print(f"  >>> [WEIGHTS] 权重路径: {toothseg.get('checkpoint_path')}", flush=True)
        if gpu.get("type") == "cuda":
            print(f"  >>> [DEVICE ] GPU 加速就绪: {gpu.get('name')} (空闲显存: {gpu.get('memory_free_mb')} MB)", flush=True)
        else:
            print("  >>> [DEVICE ] 当前运行于 CPU 模式", flush=True)
        print("  >>> [READY  ] 状态就绪！可在 3D Slicer 中正常点击【开始分割】", flush=True)
    elif not ckpt_ok:
        print("  >>> [FAILED ] 未检测到模型权重文件！", flush=True)
        print("  >>> [GUIDE  ] 1. 请将 ToothSeg 解压放入项目根目录下的 models/ 文件夹中", flush=True)
        print("  >>> [GUIDE  ] 2. 或在 3D Slicer 插件界面点击【📁 模型目录】手动指定", flush=True)
    else:
        print("  >>> [FAILED ] 未找到 nnUNetv2_predict 推理环境！", flush=True)
        print("  >>> [GUIDE  ] 请确保使用包含 nnU-Net v2 的 Conda (nninteractive) 环境启动", flush=True)

    print("=" * 68, flush=True)
    print(f"  服务监听地址 : http://127.0.0.1:8000{BASE_PATH}", flush=True)
    print(f"  本地运行目录 : {RUNTIME_ROOT}", flush=True)
    print("=" * 68 + "\n", flush=True)


    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

