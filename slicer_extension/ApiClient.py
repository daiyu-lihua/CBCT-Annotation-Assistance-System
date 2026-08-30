"""本地推理服务端 HTTP 客户端（B 组 Slicer 前端接口层）。

封装仓库 README「统一接口协议」中的 9 个后端接口，
统一 Base URL 与错误解析，供 3D Slicer 插件面板调用。

用法示例：
    client = ApiClient("http://127.0.0.1:8000/api/v1")
    try:
        st = client.status()
    except ApiError as e:
        print(e.error_code, e.message)
"""

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8000/api/v1"


class ApiError(Exception):
    """服务端统一错误（status == 'error'）或连接失败。"""

    def __init__(self, error_code, message, details=None):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.details = details or {}


class ApiClient:
    def __init__(self, base_url=DEFAULT_BASE_URL, timeout=600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout  # /predict 等长任务可能耗时，超时放长

    # ---------- 内部工具 ----------

    def _req(self, method, path, payload=None):
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            resp = requests.request(
                method, url, json=payload, timeout=self.timeout
            )
        except requests.RequestException as e:
            raise ApiError("CONNECTION_FAILED", f"无法连接服务端 ({url}): {e}")

        try:
            data = resp.json()
        except ValueError:
            data = {}

        if data.get("status") == "error":
            raise ApiError(
                data.get("error_code", "UNKNOWN"),
                data.get("message", "未知错误"),
                data.get("details"),
            )
        return data

    # ---------- 1. 服务状态 ----------

    def status(self):
        """检查服务/模型/硬件是否就绪。"""
        return self._req("GET", "/status")

    # ---------- 2. 配置 ----------

    def config(self):
        """获取可用模型列表、推理模式、标签模板。"""
        return self._req("GET", "/config")

    # ---------- 3. 病例初始化 ----------

    def create_case(self, image_path, image_format, label_template_id, operator):
        """创建标注任务，返回 case_id / case_state_path。"""
        return self._req("POST", "/cases", {
            "image_path": image_path,
            "image_format": image_format,
            "label_template_id": label_template_id,
            "operator": operator,
        })

    # ---------- 4. 图像信息检查 ----------

    def inspect_image(self, case_id, image_path):
        """读取 CBCT 基础信息（shape/spacing）。"""
        return self._req("POST", "/images/inspect", {
            "case_id": case_id,
            "image_path": image_path,
        })

    # ---------- 5. 推理模式推荐 ----------

    def recommend_mode(self, case_id, roi, target):
        """Agent 推荐推理模式。roi={'start':[x,y,z],'size':[x,y,z]}。"""
        return self._req("POST", "/agent/recommend_mode", {
            "case_id": case_id,
            "roi": roi,
            "target": target,
        })

    # ---------- 6. AI 初分割 ----------

    def predict(self, case_id, image_path, roi, model_id, mode,
                targets, output_format="nii.gz", output_dir=None):
        """AI 初分割，返回 mask_path / confidence_path / prediction_id。"""
        return self._req("POST", "/predict", {
            "case_id": case_id,
            "image_path": image_path,
            "roi": roi,
            "model_id": model_id,
            "mode": mode,
            "targets": targets,
            "output_format": output_format,
            "output_dir": output_dir,
        })

    # ---------- 7. 标签质量检查 ----------

    def check_label(self, case_id, label_path, label_template_id,
                    checks=None):
        """检查人工修正后的标签。checks 缺省时用后端默认全开。"""
        if checks is None:
            checks = ["empty", "duplicate_id", "component", "volume",
                      "tiny_fragment", "format"]
        return self._req("POST", "/check_label", {
            "case_id": case_id,
            "label_path": label_path,
            "label_template_id": label_template_id,
            "checks": checks,
        })

    # ---------- 8. 结果导出 ----------

    def export(self, case_id, image_path, label_path, export_format,
               include_report=True, output_dir=None):
        """导出训练数据包，返回 export_dir / files。"""
        return self._req("POST", "/export", {
            "case_id": case_id,
            "image_path": image_path,
            "label_path": label_path,
            "export_format": export_format,
            "include_report": include_report,
            "output_dir": output_dir,
        })

    # ---------- 9. 任务日志 ----------

    def log(self, case_id, event, operator, payload=None):
        """记录一次标注事件，供 Agent 经验沉淀。"""
        return self._req("POST", "/agent/log", {
            "case_id": case_id,
            "event": event,
            "operator": operator,
            "payload": payload or {},
        })