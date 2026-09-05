"""ToothSeg semantic branch adapter used by the local inference server.

This module is not a user-facing script. The FastAPI /predict endpoint imports
and calls it when the frontend selects the ToothSeg semantic model.
"""

from __future__ import annotations

import json
import hashlib
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


class PredictionCancelled(RuntimeError):
    """Raised when the frontend requests cancellation of an active prediction."""


MODEL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_ROOT.parents[1]
ASCII_TOOTHSEG_ROOT = Path("D:/ToothSegWork")
TOOTHSEG_ROOT = Path(
    os.environ.get("TOOTHSEG_ROOT")
    or os.environ.get("TOOTHSEG_HOME")
    or ASCII_TOOTHSEG_ROOT
)
if not TOOTHSEG_ROOT.exists():
    TOOTHSEG_ROOT = PROJECT_ROOT / "ToothSeg"


def _path_is_ascii(path: Path | str) -> bool:
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _default_runtime_root() -> Path:
    configured = os.environ.get("CBCT_TOOTHSEG_RUNTIME")
    if configured:
        return Path(configured)
    if _path_is_ascii(TOOTHSEG_ROOT):
        return TOOTHSEG_ROOT / "_runtime"
    return ASCII_TOOTHSEG_ROOT / "_runtime"


RUNTIME_ROOT = _default_runtime_root()
DEFAULT_NNUNET_RESULTS = MODEL_ROOT / "weights"
NNUNET_RAW = Path(os.environ.get("nnUNet_raw", str(MODEL_ROOT / "work" / "nnUNet_raw")))
NNUNET_PREPROCESSED = Path(os.environ.get("nnUNet_preprocessed", str(MODEL_ROOT / "work" / "nnUNet_preprocessed")))

SEMANTIC_DATASET_ID = "121"
SEMANTIC_CONFIGURATION = "3d_fullres_resample_torch_256_bs8_ctnorm"
SEMANTIC_TRAINER = "nnUNetTrainer_onlyMirror01_DASegOrd0"
SEMANTIC_FOLD = "5"
SEMANTIC_CHECKPOINT = "checkpoint_final.pth"
SEMANTIC_CHECKPOINT_RELATIVE = (
    Path("Dataset121_ToothFairy2_Teeth")
    / f"{SEMANTIC_TRAINER}__nnUNetPlans__{SEMANTIC_CONFIGURATION}"
    / f"fold_{SEMANTIC_FOLD}"
    / SEMANTIC_CHECKPOINT
)
TOOTHSEG_TO_PROJECT_DENSE = {
    **{i: i for i in range(1, 17)},
    **{i: i + 32 for i in range(17, 33)},
}


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    result = []
    for path in paths:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _candidate_nnunet_results_roots() -> list[Path]:
    raw_candidates = []
    for env_name in ("nnUNet_results", "TOOTHSEG_NNUNET_RESULTS", "CBCT_NNUNET_RESULTS"):
        value = os.environ.get(env_name)
        if value:
            raw_candidates.append(Path(value))

    raw_candidates.extend([
        DEFAULT_NNUNET_RESULTS,
        TOOTHSEG_ROOT / "nnUNet_results",
        PROJECT_ROOT / "ToothSeg" / "nnUNet_results",
        PROJECT_ROOT.parent / "model_weights",
        PROJECT_ROOT.parent / "nnUNet_results",
        Path("D:/ToothSegWork/nnUNet_results"),
    ])

    expanded = []
    for candidate in raw_candidates:
        expanded.append(candidate)
        if candidate.name.lower() != "nnunet_results":
            expanded.append(candidate / "nnUNet_results")
    return _unique_paths(expanded)


def _resolve_nnunet_results() -> Path:
    candidates = _candidate_nnunet_results_roots()
    for candidate in candidates:
        if (candidate / SEMANTIC_CHECKPOINT_RELATIVE).exists():
            return candidate
    env_value = (
        os.environ.get("nnUNet_results")
        or os.environ.get("TOOTHSEG_NNUNET_RESULTS")
        or os.environ.get("CBCT_NNUNET_RESULTS")
    )
    if env_value:
        return Path(env_value)
    return DEFAULT_NNUNET_RESULTS


NNUNET_RESULTS = _resolve_nnunet_results()


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("._") or "case"


def _file_ending(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix.lower()


def _mode_spacing(mode: str, spacing_mm: float | None = None) -> float:
    """Conservative local defaults for an 8 GB laptop GPU."""
    if spacing_mm is not None:
        spacing = float(spacing_mm)
        if not 0.5 <= spacing <= 2.0:
            raise ValueError("降采样间距 spacing_mm 必须在 0.50 到 2.00 mm 之间。")
        return round(spacing, 2)
    mode = (mode or "balanced").lower()
    if mode == "fast":
        return 0.75
    return 0.5


def _log(message: str):
    print(f"[toothseg-semantic] {message}", flush=True)


def _nnunet_predict_exe() -> str:
    exe = shutil.which("nnUNetv2_predict")
    if exe:
        return exe
    exe = shutil.which("nnUNetv2_predict.exe")
    if exe:
        return exe
    fallback = Path("D:/Anaconda/envs/nnInteractive/Scripts/nnUNetv2_predict.exe")
    if fallback.exists():
        return str(fallback)
    raise RuntimeError("未找到 nnUNetv2_predict，请使用包含 nnU-Net v2 的 nnInteractive 环境启动服务。")


def toothseg_status() -> dict[str, Any]:
    nnunet_results = _resolve_nnunet_results()
    checkpoint = nnunet_results / SEMANTIC_CHECKPOINT_RELATIVE
    try:
        predict_exe = _nnunet_predict_exe()
    except Exception as exc:
        predict_exe = None
        exe_error = str(exc)
    else:
        exe_error = None

    return {
        "available": bool(checkpoint.exists() and predict_exe),
        "toothseg_root": str(TOOTHSEG_ROOT),
        "runtime_root": str(RUNTIME_ROOT),
        "nnunet_results": str(nnunet_results),
        "nnunet_results_candidates": [str(p) for p in _candidate_nnunet_results_roots()],
        "checkpoint_path": str(checkpoint),
        "checkpoint_exists": checkpoint.exists(),
        "predict_exe": predict_exe,
        "error": exe_error,
    }


def _copy_input_to_ascii_workspace(image_path: Path, case_dir: Path) -> Path:
    ending = _file_ending(image_path)
    if ending not in {".nii", ".nii.gz", ".nrrd", ".nhdr", ".mha", ".mhd"}:
        raise RuntimeError(f"暂不支持的图像格式: {ending}")

    dst = case_dir / "input" / f"original{ending}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, dst)
    return dst


def _resample_to_spacing(input_file: Path, output_file: Path, spacing_mm: float) -> dict[str, Any]:
    import SimpleITK as sitk

    img = sitk.ReadImage(str(input_file))
    old_spacing = img.GetSpacing()
    old_size = img.GetSize()
    new_spacing = (float(spacing_mm), float(spacing_mm), float(spacing_mm))
    new_size = [
        max(1, int(round(old_size[i] * old_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    out = resampler.Execute(img)
    sitk.WriteImage(out, str(output_file))

    return {
        "original_size": [int(x) for x in old_size],
        "original_spacing": [float(x) for x in old_spacing],
        "resampled_size": [int(x) for x in out.GetSize()],
        "resampled_spacing": [float(x) for x in out.GetSpacing()],
    }


def _inspect_mask(mask_path: Path) -> dict[str, Any]:
    import SimpleITK as sitk

    img = sitk.ReadImage(str(mask_path))
    arr = sitk.GetArrayFromImage(img)
    labels = [int(x) for x in np.unique(arr)]
    return {
        "size": [int(x) for x in img.GetSize()],
        "spacing": [float(x) for x in img.GetSpacing()],
        "labels": labels,
        "nonzero_voxels": int((arr != 0).sum()),
    }


def _project_code_from_dense(dense_label: int) -> int:
    if 1 <= dense_label <= 16:
        return 100 + dense_label
    if 49 <= dense_label <= 64:
        return 400 + (dense_label - 48)
    return dense_label


def _ensure_project_label_mask(raw_mask_path: Path, project_mask_path: Path) -> dict[str, Any]:
    """Map ToothSeg semantic labels 1-32 to this project's dense natural-tooth labels.

    ToothSeg Dataset121 uses 1-16 for upper teeth and 17-32 for lower teeth. In
    this project dense labels 17-32 mean upper pulp, so lower teeth must be
    remapped to 49-64 before the mask is shown, checked, or exported.
    """
    import SimpleITK as sitk

    img = sitk.ReadImage(str(raw_mask_path))
    raw = sitk.GetArrayFromImage(img).astype(np.uint8, copy=False)
    project = np.zeros_like(raw, dtype=np.uint8)
    for src, dst in TOOTHSEG_TO_PROJECT_DENSE.items():
        project[raw == src] = dst

    out = sitk.GetImageFromArray(project)
    out.CopyInformation(img)
    project_mask_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(project_mask_path))

    labels = [int(x) for x in np.unique(project) if int(x) != 0]
    locations = []
    for dense_label in labels:
        zyx = np.argwhere(project == dense_label)
        if zyx.size == 0:
            continue
        z_min, y_min, x_min = zyx.min(axis=0)
        z_max, y_max, x_max = zyx.max(axis=0)
        center_zyx = zyx.mean(axis=0)
        project_code = _project_code_from_dense(dense_label)
        locations.append({
            "dense_label": int(dense_label),
            "project_code": int(project_code),
            "category": "UpperTooth" if dense_label <= 16 else "LowerTooth",
            "position": int(dense_label if dense_label <= 16 else dense_label - 48),
            "center_xyz": [
                float(center_zyx[2]),
                float(center_zyx[1]),
                float(center_zyx[0]),
            ],
            "bbox_xyz": [
                [int(x_min), int(x_max) + 1],
                [int(y_min), int(y_max) + 1],
                [int(z_min), int(z_max) + 1],
            ],
            "voxel_count": int(zyx.shape[0]),
        })

    mapping_info = {
        "mapping": "toothseg_1_32_to_project_dense_natural_teeth",
        "raw_mask_path": str(raw_mask_path),
        "project_mask_path": str(project_mask_path),
        "labels": labels,
        "locations": locations,
    }
    (project_mask_path.parent / "label_mapping.json").write_text(
        json.dumps(mapping_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (project_mask_path.parent / "tooth_locations.json").write_text(
        json.dumps({"teeth": locations}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return mapping_info


def _inspect_image(path: Path, require_nonzero: bool = False) -> dict[str, Any]:
    import SimpleITK as sitk

    if not path.exists() or path.stat().st_size <= 0:
        return {"valid": False, "reason": "文件不存在或为空"}
    try:
        img = sitk.ReadImage(str(path))
        info = {
            "valid": True,
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "size": [int(x) for x in img.GetSize()],
            "spacing": [float(x) for x in img.GetSpacing()],
        }
        if require_nonzero:
            arr = sitk.GetArrayFromImage(img)
            labels = [int(x) for x in np.unique(arr)]
            nonzero = int((arr != 0).sum())
            info.update({"labels": labels, "nonzero_voxels": nonzero})
            if nonzero <= 0:
                return {**info, "valid": False, "reason": "分割文件没有前景标签"}
        return info
    except Exception as exc:
        return {"valid": False, "reason": f"文件无法读取: {exc}", "path": str(path)}


def _whole_file_sha256(path: Path) -> str:
    """按用户要求直接读取完整原始图像内容后计算 SHA256。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_stem(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def _reuse_package_root(source: Path) -> Path:
    source_id = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:12]
    return RUNTIME_ROOT / "reuse_packages" / f"{_safe_name(_source_stem(source))}_{source_id}"


def _prediction_base_dir(output_dir: str | None) -> Path:
    if output_dir and _path_is_ascii(output_dir):
        return Path(output_dir)
    if output_dir:
        _log(f"忽略非英文 output_dir，改用英文运行目录: {output_dir}")
    return RUNTIME_ROOT / "semantic_predictions"


def stage_image_for_reading(image_path: str, scope: str = "input_cache") -> str:
    source = Path(image_path)
    if not source.exists():
        raise FileNotFoundError(f"图像不存在: {image_path}")
    ending = _file_ending(source)
    cache_id = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:12]
    target = RUNTIME_ROOT / scope / f"{_safe_name(_source_stem(source))}_{cache_id}" / f"original{ending}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.stat().st_size != source.stat().st_size:
        shutil.copy2(source, target)
    return str(target)


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, data: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_summary(path: Path, message: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%F %T")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {message}\n")


def _relative_to(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except Exception:
        return str(path)


def _checkpoint_identity() -> dict[str, Any]:
    checkpoint = _resolve_nnunet_results() / SEMANTIC_CHECKPOINT_RELATIVE
    if not checkpoint.exists():
        return {"path": str(checkpoint), "exists": False}
    stat = checkpoint.stat()
    return {
        "path": str(checkpoint),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime": int(stat.st_mtime),
    }


def _task_key(source_sha256: str, mode: str, spacing: float) -> str:
    identity = {
        "source_sha256": source_sha256,
        "model_id": "toothseg-semantic-05mm",
        "dataset_id": SEMANTIC_DATASET_ID,
        "configuration": SEMANTIC_CONFIGURATION,
        "trainer": SEMANTIC_TRAINER,
        "fold": SEMANTIC_FOLD,
        "checkpoint": _checkpoint_identity(),
        "mode": mode,
        "spacing_mm": spacing,
    }
    raw = json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def _new_card(source: Path, source_sha256: str) -> dict[str, Any]:
    return {
        "reuse_card_version": "1.0",
        "source_image": str(source),
        "source_sha256": source_sha256,
        "created_at": time.strftime("%F %T"),
        "updated_at": time.strftime("%F %T"),
        "runs": {},
    }


def _update_reuse_card(
    card_path: Path,
    source: Path,
    source_sha256: str,
    task_key: str,
    run_data: dict[str, Any],
):
    card = _load_json(card_path, _new_card(source, source_sha256))
    if card.get("source_sha256") != source_sha256:
        history = card.setdefault("source_history", [])
        history.append({
            "source_image": card.get("source_image"),
            "source_sha256": card.get("source_sha256"),
            "replaced_at": time.strftime("%F %T"),
        })
        card["source_sha256"] = source_sha256
        card["source_image"] = str(source)
    card["updated_at"] = time.strftime("%F %T")
    card.setdefault("runs", {})[task_key] = run_data
    _write_json(card_path, card)


def inspect_reuse_package(
    image_path: str,
    model_id: str = "toothseg-semantic-05mm",
    mode: str = "balanced",
    spacing_mm: float | None = None,
) -> dict[str, Any]:
    source = Path(image_path)
    if not source.exists():
        raise FileNotFoundError(f"图像不存在: {image_path}")
    source_sha256 = _whole_file_sha256(source)
    spacing = _mode_spacing(mode, spacing_mm)
    key = _task_key(source_sha256, mode, spacing)
    root = _reuse_package_root(source)
    card_path = root / "reuse_card.json"
    card = _load_json(card_path, {})
    run = (card.get("runs") or {}).get(key, {})
    final_path = root / run.get("files", {}).get("final", "") if run else None
    preprocessed_path = root / run.get("files", {}).get("preprocessed", "") if run else None
    final_info = _inspect_image(final_path, True) if final_path else {"valid": False, "reason": "没有最终分割文件记录"}
    pre_info = _inspect_image(preprocessed_path, False) if preprocessed_path else {"valid": False, "reason": "没有预处理文件记录"}
    if final_info.get("valid"):
        resume_from = "final"
        message = "检测到完整最终分割文件，可直接复用。"
    elif pre_info.get("valid"):
        resume_from = "preprocess_done"
        message = "检测到可复用的预处理文件，下次可跳过降采样。"
    elif root.exists():
        resume_from = "none"
        message = "检测到复用包，但没有可直接复用的有效产物。"
    else:
        resume_from = "none"
        message = "尚未创建当前图像的复用包。"
    return {
        "source_image": str(source),
        "source_sha256": source_sha256,
        "reuse_dir": str(root),
        "reuse_card": str(card_path),
        "task_key": key,
        "exists": root.exists(),
        "can_reuse": resume_from != "none",
        "resume_from": resume_from,
        "message": message,
        "run": run,
        "final_info": final_info,
        "preprocessed_info": pre_info,
    }


def delete_reuse_package(image_path: str) -> dict[str, Any]:
    source = Path(image_path)
    root = _reuse_package_root(source)
    if root.exists():
        shutil.rmtree(root)
        return {"deleted": True, "reuse_dir": str(root), "message": "已删除当前图像的复用包文件夹。"}
    return {"deleted": False, "reuse_dir": str(root), "message": "当前图像没有复用包文件夹。"}


def run_toothseg_semantic(
    image_path: str,
    case_id: str,
    mode: str = "balanced",
    spacing_mm: float | None = None,
    output_dir: str | None = None,
    device: str = "cuda",
    overwrite: bool = False,
    keep_reuse: bool = True,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_checker: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    def progress(percent: int, stage: str, message: str, **details):
        if progress_callback is None:
            return
        try:
            progress_callback({
                "percent": percent,
                "stage": stage,
                "message": message,
                "details": details,
            })
        except Exception:
            pass

    def is_cancelled() -> bool:
        if cancel_checker is None:
            return False
        try:
            return bool(cancel_checker())
        except Exception:
            return False

    def ensure_not_cancelled(message: str = "用户已中止本次分割。"):
        if is_cancelled():
            progress(0, "cancelled", message)
            raise PredictionCancelled(message)

    source = Path(image_path)
    if not source.exists():
        raise FileNotFoundError(f"图像不存在: {image_path}")

    ensure_not_cancelled()
    progress(3, "received", "后端已收到分割请求。")
    _log(f"收到推理请求: case_id={case_id}, mode={mode}, spacing_mm={spacing_mm}, image={image_path}")
    progress(6, "check_runtime", "正在检查 ToothSeg 语义模型环境与权重。")
    status = toothseg_status()
    if not status["available"]:
        raise RuntimeError(f"ToothSeg 语义模型不可用: {status}")
    nnunet_results = _resolve_nnunet_results()

    ensure_not_cancelled()
    spacing = _mode_spacing(mode, spacing_mm)
    spacing_tag = str(spacing).replace(".", "")
    safe_case = _safe_name(_source_stem(source))
    prediction_id = f"pred-{int(time.time())}"
    progress(8, "hash_input", "正在计算输入影像指纹，用于判断是否可复用。", prediction_id=prediction_id)
    source_sha256 = _whole_file_sha256(source)
    task_key = _task_key(source_sha256, mode, spacing)

    ensure_not_cancelled()
    if keep_reuse:
        progress(10, "check_reuse", "正在检测当前影像是否已有可复用分割结果。", prediction_id=prediction_id)
        reuse_root = _reuse_package_root(source)
        card_path = reuse_root / "reuse_card.json"
        summary_path = reuse_root / "readable_summary.txt"
        case_dir = reuse_root / "semantic" / task_key
        copied_input = case_dir / "input" / f"original{_file_ending(source)}"
        images_dir = case_dir / "preprocessed" / f"imagesTs_{spacing_tag}mm"
        semantic_dir = case_dir / "final"
        log_path = case_dir / "logs" / "run_log.txt"
        result_path = case_dir / "result.json"
        _append_summary(summary_path, "开始检测当前图像的复用包。")
        _update_reuse_card(card_path, source, source_sha256, task_key, {
            "model_id": "toothseg-semantic-05mm",
            "mode": mode,
            "spacing_mm": spacing,
            "status": "created",
            "can_reuse": False,
            "resume_from": "none",
            "files": {},
            "last_message": "已创建或更新复用名片，准备开始推理。",
        })
    else:
        reuse_root = None
        card_path = None
        summary_path = None
        base_dir = _prediction_base_dir(output_dir)
        case_dir = base_dir / safe_case / prediction_id
        copied_input = case_dir / "input" / f"original{_file_ending(source)}"
        images_dir = case_dir / f"imagesTs_{spacing_tag}mm"
        semantic_dir = case_dir / f"semantic_output_{spacing_tag}mm"
        log_path = case_dir / "run_log.txt"
        result_path = case_dir / "result.json"

    expected_mask = semantic_dir / f"{safe_case}.nii.gz"
    project_mask = semantic_dir / f"{safe_case}_project_labels.nii.gz"
    ensure_not_cancelled()
    final_info = _inspect_image(expected_mask, require_nonzero=True)
    if final_info.get("valid") and not overwrite:
        progress(88, "map_labels", "检测到可复用结果，正在映射为项目标签编号。", prediction_id=prediction_id, task_key=task_key)
        mapping_info = _ensure_project_label_mask(expected_mask, project_mask)
        mask_info = _inspect_mask(project_mask)
        _log(f"复用已有最终结果并映射到项目标签: {project_mask}")
        if keep_reuse:
            _append_summary(summary_path, "检测到完整最终分割文件，本次无需重新推理。")
            _update_reuse_card(card_path, source, source_sha256, task_key, {
                "model_id": "toothseg-semantic-05mm",
                "mode": mode,
                "spacing_mm": spacing,
                "status": "export_done",
                "can_reuse": True,
                "resume_from": "final",
                "files": {
                    "raw_final": _relative_to(expected_mask, reuse_root),
                    "final": _relative_to(project_mask, reuse_root),
                    "log": _relative_to(log_path, reuse_root),
                    "result": _relative_to(result_path, reuse_root),
                    "label_mapping": _relative_to(project_mask.parent / "label_mapping.json", reuse_root),
                    "tooth_locations": _relative_to(project_mask.parent / "tooth_locations.json", reuse_root),
                },
                "last_message": "最终分割文件完整，可直接复用。",
                "final_info": _inspect_image(project_mask, True),
            })
        result = {
            "status": "success",
            "prediction_id": prediction_id,
            "mask_path": str(project_mask),
            "raw_mask_path": str(expected_mask),
            "work_dir": str(case_dir),
            "reuse_dir": str(reuse_root) if reuse_root else None,
            "reuse_card": str(card_path) if card_path else None,
            "spacing_mm": spacing,
            "mask_info": mask_info,
            "mapping_info": mapping_info,
            "reused": True,
        }
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        progress(100, "done", "已复用历史分割结果。", prediction_id=prediction_id, task_key=task_key)
        return result

    ensure_not_cancelled()
    progress(14, "stage_input", "正在准备 ToothSeg 工作目录并复制输入影像。", prediction_id=prediction_id, task_key=task_key)
    _log("准备工作目录，避免中文路径影响医学影像库读取")
    copied_input.parent.mkdir(parents=True, exist_ok=True)
    if not copied_input.exists() or copied_input.stat().st_size != source.stat().st_size:
        shutil.copy2(source, copied_input)
    _log(f"输入文件已复制: {copied_input}")
    nnunet_input = images_dir / f"{safe_case}_0000.nii.gz"
    pre_info = _inspect_image(nnunet_input, require_nonzero=False)
    if pre_info.get("valid") and not overwrite:
        resample_info = {"reused_preprocessed": True, **pre_info}
        progress(30, "preprocess_reused", "检测到可复用降采样输入，已跳过预处理。", prediction_id=prediction_id, task_key=task_key)
        _log(f"复用已有降采样输入: {nnunet_input}")
        if keep_reuse:
            _append_summary(summary_path, "检测到完整降采样输入，本次跳过预处理。")
    else:
        ensure_not_cancelled()
        progress(18, "preprocess", f"正在将影像降采样到 {spacing}mm，生成 nnU-Net 输入。", prediction_id=prediction_id, task_key=task_key)
        _log(f"开始降采样: target_spacing={spacing}mm")
        try:
            resample_info = _resample_to_spacing(copied_input, nnunet_input, spacing)
        except Exception:
            if keep_reuse:
                _append_summary(summary_path, "预处理失败：无法生成模型输入文件。")
                _update_reuse_card(card_path, source, source_sha256, task_key, {
                    "model_id": "toothseg-semantic-05mm",
                    "mode": mode,
                    "spacing_mm": spacing,
                    "status": "failed_unusable",
                    "can_reuse": False,
                    "resume_from": "none",
                    "files": {"input": _relative_to(copied_input, reuse_root)},
                    "last_message": "预处理失败，没有可复用的有效中间文件。",
                })
            raise
        _log(
            "降采样完成: "
            f"{resample_info['original_size']} @ {resample_info['original_spacing']} -> "
            f"{resample_info['resampled_size']} @ {resample_info['resampled_spacing']}"
        )
        progress(30, "preprocess_done", "降采样完成，nnU-Net 输入已准备好。", prediction_id=prediction_id, task_key=task_key, resample_info=resample_info)
        if keep_reuse:
            _append_summary(summary_path, "预处理完成：已生成可复用的降采样输入文件。")
            _update_reuse_card(card_path, source, source_sha256, task_key, {
                "model_id": "toothseg-semantic-05mm",
                "mode": mode,
                "spacing_mm": spacing,
                "status": "preprocess_done",
                "can_reuse": True,
                "resume_from": "preprocess",
                "files": {
                    "input": _relative_to(copied_input, reuse_root),
                    "preprocessed": _relative_to(nnunet_input, reuse_root),
                },
                "last_message": "预处理完成，下次可跳过降采样。",
                "preprocessed_info": _inspect_image(nnunet_input, False),
            })

    env = os.environ.copy()
    env.update(
        {
            "nnUNet_raw": str(NNUNET_RAW),
            "nnUNet_preprocessed": str(NNUNET_PREPROCESSED),
            "nnUNet_results": str(nnunet_results),
            "nnUNet_compile": "F",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    if os.name != "nt":
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cmd = [
        _nnunet_predict_exe(),
        "-i",
        str(images_dir),
        "-o",
        str(semantic_dir),
        "-d",
        SEMANTIC_DATASET_ID,
        "-c",
        SEMANTIC_CONFIGURATION,
        "-tr",
        SEMANTIC_TRAINER,
        "-f",
        SEMANTIC_FOLD,
        "-chk",
        SEMANTIC_CHECKPOINT,
        "-device",
        device,
        "--disable_tta",
        "-npp",
        "0",
        "-nps",
        "0",
    ]

    # npp/nps 传 0 让 nnUNetv2_predict 进入官方 sequential 模式(主进程内完成
    # 预处理与分割导出)。Windows 上 spawn Pool 需要把整卷 logits 通过匿名管道
    # 序列化传给导出 worker, 大体积数组会触发 OSError [WinError 87](管道
    # overlapped WriteFile 参数错误), 进程直接崩溃; 单图场景 sequential 与
    # 1-worker 并行耗时基本相同。

    semantic_dir.mkdir(parents=True, exist_ok=True)
    ensure_not_cancelled()
    progress(40, "start_nnunet", "正在启动 nnU-Net 语义分割进程。", prediction_id=prediction_id, task_key=task_key)
    _log("启动 nnUNetv2_predict，后端窗口会实时显示进度")
    _log("命令: " + " ".join(cmd))

    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            process = subprocess.Popen(
                cmd,
                cwd=str(TOOTHSEG_ROOT),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
            assert process.stdout is not None
            line_queue = queue.Queue()

            def read_stdout():
                try:
                    for output_line in process.stdout:
                        line_queue.put(output_line)
                finally:
                    line_queue.put(None)

            threading.Thread(target=read_stdout, daemon=True).start()
            while True:
                if is_cancelled():
                    progress(
                        0,
                        "cancelling",
                        "已收到中止请求，正在终止 nnU-Net 子进程。",
                        prediction_id=prediction_id,
                        task_key=task_key,
                    )
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                    if keep_reuse:
                        _append_summary(summary_path, "用户中止推理，已保留当前可复用中间文件。")
                        _update_reuse_card(card_path, source, source_sha256, task_key, {
                            "model_id": "toothseg-semantic-05mm",
                            "mode": mode,
                            "spacing_mm": spacing,
                            "status": "cancelled_reusable",
                            "can_reuse": True,
                            "resume_from": "preprocess",
                            "files": {
                                "input": _relative_to(copied_input, reuse_root),
                                "preprocessed": _relative_to(nnunet_input, reuse_root),
                                "log": _relative_to(log_path, reuse_root),
                            },
                            "last_message": "用户中止推理；预处理文件和日志已保留。",
                        })
                    raise PredictionCancelled("用户已中止本次分割。")
                try:
                    line = line_queue.get(timeout=0.5)
                except queue.Empty:
                    if process.poll() is not None:
                        continue
                    continue
                if line is None:
                    break
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
                line_text = line.strip()
                if line_text:
                    progress(
                        55,
                        "nnunet_running",
                        "nnU-Net 语义分割正在运行。",
                        prediction_id=prediction_id,
                        task_key=task_key,
                        last_log=line_text[-300:],
                    )
            return_code = process.wait()
    except PredictionCancelled:
        raise
    except Exception:
        if keep_reuse:
            _append_summary(summary_path, "模型推理被中断或启动失败，已保留可读取的预处理文件。")
            _update_reuse_card(card_path, source, source_sha256, task_key, {
                "model_id": "toothseg-semantic-05mm",
                "mode": mode,
                "spacing_mm": spacing,
                "status": "failed_reusable",
                "can_reuse": True,
                "resume_from": "preprocess",
                "files": {
                    "input": _relative_to(copied_input, reuse_root),
                    "preprocessed": _relative_to(nnunet_input, reuse_root),
                    "log": _relative_to(log_path, reuse_root),
                },
                "last_message": "模型推理未完成，但预处理文件可复用。",
            })
        raise

    if return_code != 0:
        if keep_reuse:
            _append_summary(summary_path, "模型推理失败，已保留可复用的预处理文件。")
            _update_reuse_card(card_path, source, source_sha256, task_key, {
                "model_id": "toothseg-semantic-05mm",
                "mode": mode,
                "spacing_mm": spacing,
                "status": "failed_reusable",
                "can_reuse": True,
                "resume_from": "preprocess",
                "files": {
                    "input": _relative_to(copied_input, reuse_root),
                    "preprocessed": _relative_to(nnunet_input, reuse_root),
                    "log": _relative_to(log_path, reuse_root),
                },
                "last_message": "模型推理失败，下次可跳过降采样后重新推理。",
            })
        raise RuntimeError(f"nnUNetv2_predict 运行失败，日志见: {log_path}")
    progress(80, "validate_output", "模型进程已结束，正在检查分割输出文件。", prediction_id=prediction_id, task_key=task_key)
    final_info = _inspect_image(expected_mask, require_nonzero=True)
    if not final_info.get("valid"):
        if expected_mask.exists():
            try:
                expected_mask.unlink()
            except Exception:
                pass
        if keep_reuse:
            _append_summary(summary_path, f"导出失败：最终分割文件无效。原因：{final_info.get('reason')}")
            _update_reuse_card(card_path, source, source_sha256, task_key, {
                "model_id": "toothseg-semantic-05mm",
                "mode": mode,
                "spacing_mm": spacing,
                "status": "failed_reusable",
                "can_reuse": True,
                "resume_from": "preprocess",
                "files": {
                    "input": _relative_to(copied_input, reuse_root),
                    "preprocessed": _relative_to(nnunet_input, reuse_root),
                    "log": _relative_to(log_path, reuse_root),
                },
                "last_message": "模型运行结束但最终分割文件无效；已删除损坏结果并保留预处理文件。",
                "final_info": final_info,
            })
        raise RuntimeError(f"ToothSeg 未生成预期输出文件: {expected_mask}，日志见: {log_path}")

    progress(88, "map_labels", "正在将 ToothSeg 标签映射为项目标签编号。", prediction_id=prediction_id, task_key=task_key)
    mapping_info = _ensure_project_label_mask(expected_mask, project_mask)
    progress(94, "inspect_mask", "正在统计最终分割结果。", prediction_id=prediction_id, task_key=task_key)
    mask_info = _inspect_mask(project_mask)
    _log(
        f"推理完成: mask={project_mask}, labels={mask_info['labels']}, "
        f"nonzero_voxels={mask_info['nonzero_voxels']}"
    )
    result = {
        "status": "success",
        "prediction_id": prediction_id,
        "model_id": "toothseg-semantic-05mm",
        "model_name": "ToothSeg Dataset121 semantic branch",
        "mask_path": str(project_mask),
        "raw_mask_path": str(expected_mask),
        "work_dir": str(case_dir),
        "log_path": str(log_path),
        "result_path": str(result_path),
        "reuse_dir": str(reuse_root) if reuse_root else None,
        "reuse_card": str(card_path) if card_path else None,
        "task_key": task_key,
        "spacing_mm": spacing,
        "device": device,
        "resample_info": resample_info,
        "mask_info": mask_info,
        "mapping_info": mapping_info,
        "command": cmd,
        "reused": False,
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if keep_reuse:
        _append_summary(summary_path, "推理完成：最终分割文件完整，可直接复用。")
        _update_reuse_card(card_path, source, source_sha256, task_key, {
            "model_id": "toothseg-semantic-05mm",
            "mode": mode,
            "spacing_mm": spacing,
            "status": "export_done",
            "can_reuse": True,
            "resume_from": "final",
            "files": {
                "input": _relative_to(copied_input, reuse_root),
                "preprocessed": _relative_to(nnunet_input, reuse_root),
                "raw_final": _relative_to(expected_mask, reuse_root),
                "final": _relative_to(project_mask, reuse_root),
                "log": _relative_to(log_path, reuse_root),
                "result": _relative_to(result_path, reuse_root),
                "label_mapping": _relative_to(project_mask.parent / "label_mapping.json", reuse_root),
                "tooth_locations": _relative_to(project_mask.parent / "tooth_locations.json", reuse_root),
            },
            "last_message": "推理与导出均已完成，最终分割文件可复用。",
            "preprocessed_info": _inspect_image(nnunet_input, False),
            "final_info": _inspect_image(project_mask, True),
        })
    progress(100, "done", "分割完成，最终标签文件已生成。", prediction_id=prediction_id, task_key=task_key)
    return result
