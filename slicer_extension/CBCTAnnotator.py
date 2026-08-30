"""牙科 CBCT 交互式实例级辅助标注系统 —— 3D Slicer 前端插件（B 组）。

当前阶段：插件骨架 + 服务连接层。
已实现：服务地址配置、连接测试(/status)、配置加载(/config)填充下拉框、日志区。
后续任务（ROI/AI分割/修正/导出）将基于 ApiClient 在此面板上逐步扩展。

参考：docs/api/slicer_plugin_plan.md
"""

import os
import sys

# 保证能 import 同目录下的 ApiClient 模块
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import qt  # noqa: E402
import slicer  # noqa: E402
from slicer.ScriptedLoadableModule import (  # noqa: E402
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
)

from ApiClient import ApiClient, ApiError  # noqa: E402


class CBCTAnnotator(ScriptedLoadableModule):
    """5.1. 模块注册类：提供说明文本和图标。"""

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "CBCT Annotator"
        self.parent.categories = ["Dentistry"]
        self.parent.dependencies = []
        self.parent.contributors = ["B Group"]
        self.parent.helpText = (
            "牙科 CBCT 交互式实例级辅助标注系统前端插件。"
            "对接本地推理服务端（/api/v1）。"
        )
        self.parent.acknowledgementText = (
            "基于自进化 Agent 端侧推理的牙科 CBCT 交互式实例级辅助标注系统"
        )


class CBCTAnnotatorWidget(ScriptedLoadableModuleWidget):
    """5.2. 模块主面板。"""

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.api = None  # 连接测试成功后才创建 ApiClient

        layout = qt.QVBoxLayout(self.parent)

        # ---- 顶栏：服务地址 + 连接测试 ----
        connBox = qt.QGroupBox("服务连接")
        connLayout = qt.QVBoxLayout(connBox)

        addressRow = qt.QHBoxLayout()
        addressRow.addWidget(qt.QLabel("服务地址:"))
        self.addressEdit = qt.QLineEdit("http://127.0.0.1:8000/api/v1")
        addressRow.addWidget(self.addressEdit)
        connLayout.addLayout(addressRow)

        connBtns = qt.QHBoxLayout()
        self.connectBtn = qt.QPushButton("连接测试（/status）")
        self.loadConfigBtn = qt.QPushButton("加载配置（/config）")
        self.connectBtn.clicked.connect(self.on_connect)
        self.loadConfigBtn.clicked.connect(self.on_load_config)
        connBtns.addWidget(self.connectBtn)
        connBtns.addWidget(self.loadConfigBtn)
        connLayout.addLayout(connBtns)

        layout.addWidget(connBox)

        # ---- 配置区：模型/推理模式/标签模板下拉框 ----
        cfgBox = qt.QGroupBox("配置")
        cfgLayout = qt.QFormLayout(cfgBox)

        self.modelCombo = qt.QComboBox()
        self.modeCombo = qt.QComboBox()
        self.templateCombo = qt.QComboBox()
        cfgLayout.addRow("模型:", self.modelCombo)
        cfgLayout.addRow("推理模式:", self.modeCombo)
        cfgLayout.addRow("标签模板:", self.templateCombo)

        # 推理模式：按协议固定为三种
        self.modeCombo.addItem("fast")
        self.modeCombo.addItem("balanced")
        self.modeCombo.addItem("fine")
        self.modeCombo.setCurrentText("balanced")

        layout.addWidget(cfgBox)

        # ---- 状态栏 + 日志区 ----
        statusRow = qt.QHBoxLayout()
        self.statusLabel = qt.QLabel("未连接")
        self.statusLabel.setStyleSheet("color: gray;")
        statusRow.addWidget(self.statusLabel)
        statusRow.addStretch(1)
        layout.addLayout(statusRow)

        layout.addWidget(qt.QLabel("日志:"))
        self.logEdit = qt.QPlainTextEdit()
        self.logEdit.setReadOnly(True)
        layout.addWidget(self.logEdit)

        layout.addStretch(1)

    # ---------- 工具 ----------

    def _log(self, msg):
        self.logEdit.appendPlainText(msg)
        slicer.util.logInfo(msg)

    def _set_status(self, text, color):
        self.statusLabel.setText(text)
        self.statusLabel.setStyleSheet(f"color: {color};")

    # ---------- 槽函数 ----------

    def on_connect(self):
        """连接测试：调用 /status，成功后创建 ApiClient。"""
        url = self.addressEdit.text.strip()
        try:
            self.api = ApiClient(url)
            st = self.api.status()
        except ApiError as e:
            self.api = None
            self._set_status(f"连接失败: [{e.error_code}] {e.message}", "red")
            self._log(f"连接失败: {e.error_code} | {e.message}")
            return

        model_state = st["model"].get("loaded", False)
        device = st["device"].get("data") if "device" in st else "unknown"
        ready = "就绪" if st.get("service") == "running" and model_state else "模型未加载"
        self._set_status(
            f"已连接 | 设备 {device} | 模型 {'已加载' if model_state else '未加载'}",
            "green" if ready == "就绪" else "orange",
        )
        self._log(
            f"连接成功: service={st.get('service')}, "
            f"device={device}, model_loaded={model_state}"
        )

    def on_load_config(self):
        """加载配置：填充模型/模板下拉框。"""
        if self.api is None:
            self._set_status("请先连接测试", "orange")
            self._log("尚未连接，请先点击连接测试")
            return
        try:
            cfg = self.api.config()
        except ApiError as e:
            self._set_status(f"配置加载失败: [{e.error_code}]", "red")
            self._log(f"配置加载失败: {e.error_code} | {e.message}")
            return

        self.modelCombo.clear()
        for m in cfg.get("models", []):
            self.modelCombo.addItem(m.get("model_id", "?"))
        self.templateCombo.clear()
        for t in cfg.get("label_templates", []):
            self.templateCombo.addItem(t.get("template_id", "?"))

        modes = cfg.get("inference_modes", {}).keys()
        if modes:
            self._log(f"可用推理模式: {', '.join(modes)}")
        self._set_status(f"配置已加载: {self.modelCombo.count()} 个模型, "
                         f"{self.templateCombo.count()} 个标签模板", "green")
        self._log("配置加载完成")