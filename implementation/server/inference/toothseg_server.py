# -*- coding: utf-8 -*-
"""CBCT ToothSeg 真实推理服务端。

与 mock_server.py 实现相同的 9 个统一接口（统一接口协议 v1），
区别在于 /predict 真正调用 ToothSeg 模型：

    implementation/model/toothseg/run_toothseg.py   (MemSafe 双分支推理 + 后处理)
    implementation/model/weights/Dataset121...      (语义分支权重)
    implementation/model/weights/Dataset123...      (实例分支权重)

设计要点
--------
1. 同步长请求：完整推理约 30-90 分钟，/predict 阻塞直到完成；
   插件侧 ApiClient.predict() 读超时为 None(无限等待) 且在子线程调用，
   前端界面不会卡死。FastAPI 同步端点跑在线程池，/status 等轻接口
   在推理期间仍然可用。
2. GPU 独占：同一时刻只允许一个 /predict（线程锁），第二个请求立即
   返回 PREDICT_IN_PROGRESS。推理在本进程的独立子进程中运行
   （run_toothseg.py full 模式再拆语义/实例两个孙进程，父进程不初始化
   CUDA，规避 Windows 父子进程 GPU 竞争导致的 0xC0000005）。
3. 本服务进程自身绝不 import torch / 不初始化 CUDA（/status 用
   nvidia-smi 探测显卡），避免长期占用 8GB 显存。
4. 输入约定：ToothSeg 要求文件名带 "_0000" 后缀，服务端自动把用户
   图像复制到任务目录并重命名；ROI 仅记录不裁剪（全图推理保证牙位
   编号空间先验正确）。
5. 产物：data/outputs/toothseg/jobs/<pred_id>/output/final_prediction/*.nii.gz
   (FDI 11-48 牙位标签)，mask_path 直接返回给插件加载。

启动：
    E:\\miniconda3\\envs\\nninteractive\\python.exe implementation/server/inference/toothseg_server.py
服务地址：http://127.0.0.1:8000/api/v1
"""

import json
import os
import re
import shutil
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
# 运行产物目录：项目仓库 data/outputs/toothseg（不入库）
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "toothseg"
JOBS_DIR = OUTPUT_DIR / "jobs"
AGENT_LOG = OUTPUT_DIR / "agent_log.jsonl"

# 96 类标签规范（与 mock_server 共用 implementation/server/inference/assets/）
LABEL_SPEC_FILE = Path(__file__).resolve().parent / "assets" / "label_spec_96.txt"
LABEL_TEMPLATE_ID = "teeth-dense-96"

# ToothSeg 模型接入位置（README 规范：implementation/model/toothseg + weights）
TOOTHSEG_DIR = PROJECT_ROOT / "implementation" / "model" / "toothseg"
RUN_SCRIPT = TOOTHSEG_DIR / "run_toothseg.py"
WEIGHTS_DIR = PROJECT_ROOT / "implementation" / "model" / "weights"
SEMSEG_CP = (WEIGHTS_DIR / "Dataset121_ToothFairy2_Teeth" /
             "nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_256_bs8_ctnorm" /
             "fold_5" / "checkpoint_final.pth")
INSTSEG_CP = (WEIGHTS_DIR / "Dataset123_ToothFairy2fixed_teeth_spacing02_brd3px" /
              "nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8_ctnorm" /
              "fold_5" / "checkpoint_final.pth")

MODEL_ID = "toothseg-full"
# 推理模式 -> 滑窗步长（小 = 更精细但更慢）。fast 模式整流程约 30-50 分钟，
# balanced 为交接文档验证的默认配置。
MODE_STEP = {"fast": 0.8, "balanced": 0.5, "fine": 0.4}
# FDI 牙位合法值（11-18, 21-28, 31-38, 41-48）
FDI_LABELS = {q * 10 + i for q in (1, 2, 3, 4) for i in range(1, 9)}

app = FastAPI(title="CBCT ToothSeg Server", version="1.0.0")
_PREDICT_LOCK = threading.Lock()


def _log(event: str, payload: dict):
    print(f"[toothseg-server] {event}: {payload}", flush=True)


def _err(error_code: str, message: str, details=None):
    return {"status": "error", "error_code": error_code,
            "message": message, "details": details or {}}


def _ok(**kw):
    return {"status": "ok", **kw}


def _log_tail(path: Path, n: int = 30):
    """读取日志文件最后 n 行（推理失败时附带在 details 里便于排查）。"""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-n:]
    except Exception:
        return []


def _nifti_stem(name: str):
    """返回 (stem, suffix)：支持 .nii.gz / .nii；非 NIfTI 返回 None。"""
    low = name.lower()
    if low.endswith(".nii.gz"):
        return name[:-7], name[-7:]
    if low.endswith(".nii"):
        return name[:-4], name[-4:]
    return None


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


# ---------- 1. 服务状态 ----------

@app.get(BASE_PATH + "/status")
def status():
    loaded = SEMSEG_CP.exists() and INSTSEG_CP.exists()
    missing = [str(p) for p in (SEMSEG_CP, INSTSEG_CP) if not p.exists()]
    return _ok(
        service={"name": "cbct-toothseg-server", "version": "1.0.0"},
        model={"loaded": loaded, "name": MODEL_ID,
               "missing_checkpoints": missing},
        device=_gpu_info(),
    )


# ---------- 2. 配置 ----------

@app.get(BASE_PATH + "/config")
def config():
    return _ok(
        models=[
            {"model_id": MODEL_ID,
             "name": "ToothSeg 牙齿分割（双分支 + 牙位编号 FDI 11-48）"},
        ],
        modes=[
            {"id": "fast", "name": "快速模式（滑窗步长 0.8，约 30-50 分钟）"},
            {"id": "balanced", "name": "均衡模式（滑窗步长 0.5，交接默认，约 1 小时）"},
            {"id": "fine", "name": "精细模式（滑窗步长 0.4，更精细、更慢）"},
        ],
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
        img = nib.load(req.image_path)
        shape = [int(s) for s in img.shape[:3]]
        spacing = [float(z) for z in img.header.get_zooms()[:3]]
    except Exception as e:
        return _err("UNSUPPORTED_FORMAT", f"读取图像失败: {e}")
    return _ok(shape=shape, spacing=spacing)


# ---------- 5. 推理模式推荐 ----------

@app.post(BASE_PATH + "/agent/recommend_mode")
def recommend_mode(req: RecommendModeRequest):
    # ToothSeg 按全图推理，ROI 只影响推荐：体积越大越建议快速模式
    sx, sy, sz = req.roi.get("size", [1, 1, 1])
    volume = abs(sx * sy * sz)
    if volume <= 128 ** 3:
        mode, reason = "fine", "目标区域较小，可用精细模式提升细节"
    elif volume <= 256 ** 3:
        mode, reason = "balanced", "目标区域适中，建议均衡模式（交接默认配置）"
    else:
        mode, reason = "fast", "区域较大/全图推理，建议快速模式缩短等待"
    return _ok(mode=mode, reason=reason)


# ---------- 6. AI 初分割（真实 ToothSeg 推理） ----------

@app.post(BASE_PATH + "/predict")
def predict(req: PredictRequest):
    if not RUN_SCRIPT.exists():
        return _err("MODEL_NOT_LOADED", f"推理入口不存在: {RUN_SCRIPT}")
    if not (SEMSEG_CP.exists() and INSTSEG_CP.exists()):
        return _err("MODEL_NOT_LOADED", "模型权重缺失，请检查 implementation/model/weights",
                    {"missing": [str(p) for p in (SEMSEG_CP, INSTSEG_CP)
                                 if not p.exists()]})
    src = Path(req.image_path)
    if not src.exists():
        return _err("IMAGE_NOT_FOUND", f"图像不存在: {req.image_path}")
    parsed = _nifti_stem(src.name)
    if parsed is None:
        return _err("UNSUPPORTED_FORMAT",
                    "ToothSeg 仅支持 NIfTI (.nii / .nii.gz)。请在 Slicer 中把图像"
                    "另存为 .nii.gz 后重试（或把文件放入 data/inputs）",
                    {"image_path": req.image_path})

    if not _PREDICT_LOCK.acquire(blocking=False):
        return _err("PREDICT_IN_PROGRESS",
                    "已有 ToothSeg 推理任务在运行（GPU 独占），请等待其完成后再试",
                    {"hint": "可在 data/outputs/toothseg/jobs 下查看运行中的任务"})
    try:
        t0 = time.time()
        pred_id = "pred-" + uuid.uuid4().hex[:8]
        job_dir = JOBS_DIR / pred_id
        input_dir = job_dir / "input"
        output_root = job_dir / "output"
        input_dir.mkdir(parents=True, exist_ok=True)

        # ToothSeg 要求 "_0000" 后缀：复制用户图像到任务输入目录并改名
        stem, suffix = parsed
        staged_name = stem if stem.endswith("_0000") else stem + "_0000"
        staged = input_dir / (staged_name + suffix)
        _log("predict_stage", {"pred_id": pred_id, "src": str(src),
                               "staged": str(staged)})
        shutil.copy2(str(src), str(staged))

        step = MODE_STEP.get(str(req.mode).lower(), 0.5)
        log_path = job_dir / "server_predict.log"
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["TOOTHSEG_HOME"] = str(TOOTHSEG_DIR)
        env.setdefault("TOOTHSEG_NNUNET_RESULTS", str(WEIGHTS_DIR))
        cmd = [sys.executable, str(RUN_SCRIPT),
               "-i", str(input_dir), "-o", str(output_root),
               "--mode", "full", "--step-size", str(step), "--np", "2"]
        _log("predict_start", {"pred_id": pred_id, "mode": req.mode,
                               "step_size": step, "cmd": " ".join(cmd)})
        with open(log_path, "w", encoding="utf-8") as log_f:
            proc = subprocess.Popen(cmd, stdout=log_f,
                                    stderr=subprocess.STDOUT, env=env,
                                    cwd=str(TOOTHSEG_DIR))
            rc = proc.wait()

        if rc != 0:
            _log("predict_failed", {"pred_id": pred_id, "exit": rc})
            return _err("PREDICTION_FAILED",
                        f"ToothSeg 推理失败 (exit={rc})，日志: {log_path}",
                        {"log_tail": _log_tail(log_path),
                         "log_path": str(log_path), "exit_code": rc})

        final_dir = output_root / "final_prediction"
        masks = sorted(final_dir.glob("*.nii.gz"))
        if not masks:
            return _err("PREDICTION_FAILED",
                        "推理完成但未找到 final_prediction/*.nii.gz 输出",
                        {"log_tail": _log_tail(log_path),
                         "log_path": str(log_path)})
        mask_path = masks[0]
        elapsed = round(time.time() - t0, 1)

        # 任务元信息落盘（便于回溯）
        (job_dir / "job_meta.json").write_text(json.dumps({
            "pred_id": pred_id, "case_id": req.case_id,
            "image_path": req.image_path, "staged_input": str(staged),
            "mask_path": str(mask_path), "mode": req.mode,
            "step_size": step, "roi": req.roi, "roi_used": False,
            "exit_code": rc, "elapsed_sec": elapsed,
            "finished_at": time.strftime("%F %T"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        _log("predict_done", {"pred_id": pred_id, "mask": str(mask_path),
                              "elapsed_sec": elapsed})
        return _ok(
            prediction_id=pred_id,
            mask_path=str(mask_path),
            confidence_path=None,   # ToothSeg 最终输出不含置信度图
            model_id=req.model_id,
            mode=req.mode,
            roi_used=False,
            roi_note="ToothSeg 按全图推理（保证牙位编号空间先验正确），"
                     "ROI 已记录在 job_meta.json",
            elapsed_sec=elapsed,
        )
    finally:
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
    print(f"CBCT ToothSeg server -> http://127.0.0.1:8000{BASE_PATH}", flush=True)
    print(f"  模型代码 : {TOOTHSEG_DIR}", flush=True)
    print(f"  权重目录 : {WEIGHTS_DIR}", flush=True)
    print(f"  产物目录 : {JOBS_DIR}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
