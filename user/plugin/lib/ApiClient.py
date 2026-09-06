"""HTTP client for the 3D Slicer CBCT Annotator plugin."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class ApiError(Exception):
    def __init__(self, error_code: str, message: str, details=None):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.details = details or {}


class ApiClient:
    def __init__(self, base_url: str, timeout: int = 3600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, payload=None, timeout=None):
        url = self.base_url + path
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout if timeout is None else timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ApiError("HTTP_ERROR", f"HTTP {exc.code}: {body}") from exc
        except Exception as exc:
            raise ApiError("CONNECTION_FAILED", f"无法连接服务端: {exc}") from exc

        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ApiError("INVALID_RESPONSE", f"服务端返回不是 JSON: {body[:200]}") from exc

        if result.get("status") == "error":
            raise ApiError(
                result.get("error_code", "SERVER_ERROR"),
                result.get("message", "服务端返回错误"),
                result.get("details", {}),
            )
        return result

    def status(self):
        return self._request("GET", "/status")

    def config(self):
        return self._request("GET", "/config")

    def set_model_path(self, model_path: str):
        return self._request(
            "POST",
            "/model/set_path",
            {"model_path": model_path},
            timeout=10,
        )

    def create_case(self, image_path, image_format, label_template_id, operator):
        return self._request(
            "POST",
            "/cases",
            {
                "image_path": image_path,
                "image_format": image_format,
                "label_template_id": label_template_id,
                "operator": operator,
            },
        )

    def inspect_image(self, case_id, image_path):
        return self._request(
            "POST",
            "/images/inspect",
            {"case_id": case_id, "image_path": image_path},
        )

    def recommend_mode(self, case_id, roi, target):
        return self._request(
            "POST",
            "/agent/recommend_mode",
            {"case_id": case_id, "roi": roi, "target": target},
        )

    def predict(
        self,
        case_id,
        image_path,
        roi,
        model_id,
        mode,
        targets=None,
        output_format="nii.gz",
        output_dir=None,
        keep_reuse=True,
        spacing_mm=None,
    ):
        payload = {
            "case_id": case_id,
            "image_path": image_path,
            "roi": roi,
            "model_id": model_id,
            "mode": mode,
            "targets": targets or ["teeth"],
            "output_format": output_format,
            "keep_reuse": bool(keep_reuse),
        }
        if spacing_mm is not None:
            payload["spacing_mm"] = float(spacing_mm)
        if output_dir:
            payload["output_dir"] = output_dir
        return self._request("POST", "/predict", payload)

    def predict_progress(self, case_id):
        return self._request("GET", f"/predict/progress/{case_id}", timeout=2)

    def cancel_predict(self, case_id, image_path=None, model_id=None, mode="balanced", keep_reuse=True, spacing_mm=None):
        payload = {
            "case_id": case_id,
            "image_path": image_path,
            "model_id": model_id or "toothseg-semantic-05mm",
            "mode": mode,
            "keep_reuse": bool(keep_reuse),
        }
        if spacing_mm is not None:
            payload["spacing_mm"] = float(spacing_mm)
        return self._request("POST", "/predict/cancel", payload, timeout=5)

    def reuse_status(self, image_path, model_id, mode, spacing_mm=None):
        payload = {"image_path": image_path, "model_id": model_id, "mode": mode}
        if spacing_mm is not None:
            payload["spacing_mm"] = float(spacing_mm)
        return self._request(
            "POST",
            "/reuse/status",
            payload,
        )

    def delete_reuse(self, image_path, model_id, mode, spacing_mm=None):
        payload = {"image_path": image_path, "model_id": model_id, "mode": mode}
        if spacing_mm is not None:
            payload["spacing_mm"] = float(spacing_mm)
        return self._request(
            "POST",
            "/reuse/delete",
            payload,
        )

    def check_label(self, case_id, label_path, label_template_id, checks=None):
        return self._request(
            "POST",
            "/check_label",
            {
                "case_id": case_id,
                "label_path": label_path,
                "label_template_id": label_template_id,
                "checks": checks or [],
            },
        )

    def export_result(
        self,
        case_id,
        image_path,
        label_path,
        export_format="nii.gz",
        include_report=True,
        output_dir=None,
    ):
        payload = {
            "case_id": case_id,
            "image_path": image_path,
            "label_path": label_path,
            "export_format": export_format,
            "include_report": include_report,
        }
        if output_dir:
            payload["output_dir"] = output_dir
        return self._request("POST", "/export", payload)

    def export(
        self,
        case_id,
        image_path,
        label_path,
        export_format="nii.gz",
        include_report=True,
        output_dir=None,
    ):
        return self.export_result(
            case_id,
            image_path,
            label_path,
            export_format,
            include_report,
            output_dir,
        )

    def log(self, case_id, event, operator, payload=None):
        return self._request(
            "POST",
            "/agent/log",
            {
                "case_id": case_id,
                "event": event,
                "operator": operator,
                "payload": payload or {},
            },
        )
