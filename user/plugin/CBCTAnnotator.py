# -*- coding: utf-8 -*-
"""3D Slicer frontend for the CBCT tooth annotation assistant."""

import os
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = os.path.join(_HERE, "lib")
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)

import ctk  # noqa: E402
import qt  # noqa: E402
import slicer  # noqa: E402
from slicer.ScriptedLoadableModule import (  # noqa: E402
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
)

from ApiClient import ApiClient, ApiError  # noqa: E402


_BORDER = "#dce3ed"
_PRIMARY = "#276a9f"
_PRIMARY_DARK = "#1f547f"
_TEXT = "#253142"
_TEXT_DIM = "#6f7b8d"
_DANGER = "#d64545"
_OK = "#2e9e6b"
_WARN = "#c9861c"
_DEFAULT_TEMPLATE_ID = "teeth-dense-96"

_PANEL_QSS = f"""
#mainTitle {{
    font-size: 16px;
    font-weight: bold;
    padding: 0 0 2px 0;
}}
QLabel#hint {{ color: {_TEXT_DIM}; }}
QLabel#value {{ color: {_TEXT}; font-weight: bold; }}
#inlineStatus {{
    border: 1px solid {_BORDER};
}}
QPushButton {{
    font-size: 11px;
    min-height: 22px;
    padding: 2px 6px;
}}
QPushButton[promptButton="true"] {{
    min-width: 48px;
}}
QPushButton[promptButton="true"]:checked {{
    background-color: #3498db;
    color: #fff;
}}
QProgressBar {{
    text-align: center;
    font-size: 10px;
    min-height: 13px;
    max-height: 13px;
}}
QTableWidget {{ gridline-color: {_BORDER}; }}
#cbctScroll {{
    background: transparent;
    border: none;
}}
QToolButton#labelSetButton {{ text-align: left; padding: 2px 6px; }}
QPlainTextEdit {{ font-family: Consolas, monospace; }}
"""


class CBCTAnnotator(ScriptedLoadableModule):
    """Slicer module registration."""

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "CBCT Annotator"
        self.parent.categories = ["Dentistry"]
        self.parent.dependencies = []
        self.parent.contributors = ["B Group"]
        self.parent.helpText = "牙科 CBCT 交互式实例级辅助标注系统前端插件。"
        self.parent.acknowledgementText = "CBCT tooth annotation assistant"


def _combo_text(combo):
    value = combo.currentText
    if callable(value):
        value = value()
    return str(value or "")


def _line_text(line_edit):
    value = line_edit.text
    if callable(value):
        value = value()
    return str(value or "")


def _spin_value(spin_box):
    value = spin_box.value
    if callable(value):
        value = value()
    return float(value or 0.0)


def _settings_float(settings, key, default):
    try:
        return float(str(settings.value(key, str(default))))
    except Exception:
        return float(default)


def _collect_mode_ids(cfg):
    raw = cfg.get("modes") or cfg.get("inference_modes") or {}
    if isinstance(raw, dict):
        return list(raw.keys())
    if isinstance(raw, list):
        ids = []
        for item in raw:
            if isinstance(item, dict) and item.get("id"):
                ids.append(item["id"])
            elif isinstance(item, str):
                ids.append(item)
        return ids
    return []


def _button(text, kind="ghost", width=None, checkable=False):
    btn = qt.QPushButton(text)
    btn.setProperty("buttonKind", kind)
    if checkable:
        btn.setCheckable(True)
        btn.setProperty("promptButton", True)
    if width:
        btn.setFixedWidth(width)
    else:
        btn.setMaximumWidth(120)
    return btn


def _combo_count(combo):
    value = combo.count
    if callable(value):
        return value()
    return int(value or 0)


def _make_panel(title):
    panel = ctk.ctkCollapsibleButton()
    panel.text = title
    panel.collapsed = False
    layout = qt.QVBoxLayout(panel)
    layout.setContentsMargins(8, 6, 8, 6)
    layout.setSpacing(5)
    return panel, layout


class CBCTAnnotatorWidget(ScriptedLoadableModuleWidget):
    """Main Slicer module widget."""

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.api = None
        self.image_path = None
        self.volume_node = None
        self.roi_node = None
        self.roi_ijk_start = None
        self.roi_ijk_size = None
        self.point_node = None
        self.curve_node = None
        self.case_id = None
        self.mask_node = None
        self.correct_seg_node = None
        self.corrected_node = None
        self.last_mask_path = None
        self.label_rules = []
        self.label_collections = {}
        self.active_label_collection = None
        self.point_prompts = []
        self._predict_started_at = None
        self._predict_model_id = None
        self._predict_mode = None
        self._predict_spacing_mm = None
        self._last_progress_logged = -1
        self._last_backend_progress_key = None
        self._progress_polling = False
        self._cancel_requested = False
        self._pending_progress = None
        self._pending_progress_error = None
        self._pending_predict_result = None
        self._pending_predict_error = None
        self._pending_cancel_result = None
        self._pending_cancel_error = None
        self._last_auto_volume_id = None
        self._last_auto_label_id = None
        self._last_auto_segmentation_id = None
        self._settings = qt.QSettings("CBCTAnnotationAssistant", "SlicerPlugin")

        self.parent.setStyleSheet(_PANEL_QSS)
        root = qt.QFrame()
        root.setObjectName("cbctPanel")
        self.parent.layout().addWidget(root)
        root_layout = qt.QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        scroll = qt.QScrollArea()
        scroll.setObjectName("cbctScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(qt.QFrame.NoFrame)
        scroll.setAutoFillBackground(False)
        root_layout.addWidget(scroll)

        content = qt.QFrame()
        main = qt.QVBoxLayout(content)
        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(8)
        scroll.setWidget(content)

        title = qt.QLabel("牙科 CBCT 辅助标注系统")
        title.setObjectName("mainTitle")
        main.addWidget(title)

        self.statusLabel = qt.QLabel("正在检测本地服务与模型配置...")
        self.statusLabel.setObjectName("hint")
        self.statusLabel.setWordWrap(True)
        main.addWidget(self.statusLabel)

        panels = [
            self._build_service_panel(),
            self._build_case_panel(),
            self._build_predict_panel(),
            self._build_correction_panel(),
            self._build_label_panel(),
            self._build_quality_export_panel(),
            self._build_log_panel(),
        ]
        for panel in panels:
            main.addWidget(panel)
        main.addStretch(1)

        self.predictTimer = qt.QTimer()
        self.predictTimer.setInterval(1000)
        self.predictTimer.timeout.connect(self._on_predict_tick)

        self.sceneAutoTimer = qt.QTimer()
        self.sceneAutoTimer.setInterval(1500)
        self.sceneAutoTimer.timeout.connect(self._auto_detect_scene_content)
        self.sceneAutoTimer.start()

        qt.QTimer.singleShot(250, self._auto_initialize)

    # ---------- UI builders ----------

    def _build_service_panel(self):
        panel, layout = _make_panel("服务与模型")

        address_row = qt.QHBoxLayout()
        address_row.setSpacing(6)
        address_row.addWidget(qt.QLabel("地址"))
        default_url = self._settings.value(
            "base_url", "http://127.0.0.1:8000/api/v1")
        self.addressEdit = qt.QLineEdit(str(default_url))
        address_row.addWidget(self.addressEdit, 1)
        self.refreshBtn = _button("刷新", "primary", 58)
        self.refreshBtn.clicked.connect(self.on_refresh_service)
        address_row.addWidget(self.refreshBtn)
        layout.addLayout(address_row)

        state_box = qt.QFrame()
        state_box.setObjectName("inlineStatus")
        state_layout = qt.QVBoxLayout(state_box)
        state_layout.setContentsMargins(8, 5, 8, 5)
        state_layout.setSpacing(3)
        self.connStatusLabel = qt.QLabel("服务：检测中")
        self.connStatusLabel.setObjectName("value")
        self.modelStatusLabel = qt.QLabel("模型：等待配置")
        self.modelStatusLabel.setObjectName("hint")
        state_layout.addWidget(self.connStatusLabel)
        state_layout.addWidget(self.modelStatusLabel)
        layout.addWidget(state_box)

        form = qt.QFormLayout()
        form.setSpacing(5)
        self.modelCombo = qt.QComboBox()
        self.modeCombo = qt.QComboBox()
        self.templateCombo = qt.QComboBox()
        self.modeCombo.addItem("balanced")
        self.modeCombo.setCurrentText("balanced")
        self.modeCombo.setEnabled(False)
        self.modeCombo.setToolTip("当前 ToothSeg 接入固定使用 balanced 处理方式")
        form.addRow("模型", self.modelCombo)
        form.addRow("模式", self.modeCombo)
        form.addRow("标签", self.templateCombo)
        layout.addLayout(form)

        self.modelSupportLabel = qt.QLabel("等待服务返回模型配置")
        self.modelSupportLabel.setObjectName("hint")
        self.modelSupportLabel.setWordWrap(True)
        layout.addWidget(self.modelSupportLabel)

        try:
            self.modelCombo.currentIndexChanged.connect(self._on_model_changed)
        except Exception:
            pass
        return panel

    def _build_case_panel(self):
        panel, layout = _make_panel("当前病例")
        self.imageInfoLabel = qt.QLabel(
            "请使用 Slicer 原生功能打开 CBCT 影像；插件会自动检测、检查可用性并绑定当前病例。"
        )
        self.imageInfoLabel.setObjectName("hint")
        self.imageInfoLabel.setWordWrap(True)
        layout.addWidget(self.imageInfoLabel)
        return panel

    def _build_predict_panel(self):
        panel, layout = _make_panel("AI 分割")
        row = qt.QHBoxLayout()
        row.setSpacing(6)
        self.predictBtn = _button("开始分割", "primary", 82)
        self.predictBtn.clicked.connect(self.on_predict)
        self.cancelPredictBtn = _button("中止分割", "danger", 82)
        self.cancelPredictBtn.setEnabled(False)
        self.cancelPredictBtn.clicked.connect(self.on_cancel_predict)
        row.addWidget(self.predictBtn)
        row.addWidget(self.cancelPredictBtn)
        row.addStretch(1)
        layout.addLayout(row)

        self.processModeLabel = qt.QLabel("处理方式：balanced（固定）")
        self.processModeLabel.setObjectName("hint")
        self.processModeLabel.setWordWrap(True)
        layout.addWidget(self.processModeLabel)

        spacing_row = qt.QHBoxLayout()
        spacing_row.setSpacing(6)
        spacing_row.addWidget(qt.QLabel("降采样间距"))
        self.spacingSpin = qt.QDoubleSpinBox()
        self.spacingSpin.setRange(0.50, 2.00)
        self.spacingSpin.setDecimals(2)
        self.spacingSpin.setSingleStep(0.05)
        self.spacingSpin.setSuffix(" mm")
        self.spacingSpin.setValue(_settings_float(self._settings, "spacing_mm", 0.75))
        self.spacingSpin.setToolTip("数值越大，推理占用显存越少，但分割细节会减少；本机 RTX 4060 建议从 0.75 mm 开始。")
        self.spacingSpin.valueChanged.connect(self._on_spacing_changed)
        spacing_row.addWidget(self.spacingSpin)
        spacing_row.addStretch(1)
        layout.addLayout(spacing_row)

        self.spacingHintLabel = qt.QLabel("")
        self.spacingHintLabel.setObjectName("hint")
        self.spacingHintLabel.setWordWrap(True)
        layout.addWidget(self.spacingHintLabel)
        self._on_spacing_changed()

        self.predictProgress = qt.QProgressBar()
        self.predictProgress.setRange(0, 100)
        self.predictProgress.setValue(0)
        self.predictProgress.setFormat("等待开始")
        layout.addWidget(self.predictProgress)

        reuse_row = qt.QHBoxLayout()
        reuse_row.setSpacing(6)
        self.keepReuseCheck = qt.QCheckBox("保留复用包")
        self.keepReuseCheck.setChecked(
            str(self._settings.value("keep_reuse", "true")).lower() != "false")
        self.keepReuseCheck.stateChanged.connect(lambda *_: self._save_ui_settings())
        self.checkReuseBtn = _button("检测复用", "ghost", 74)
        self.checkReuseBtn.clicked.connect(self.on_check_reuse)
        self.deleteReuseBtn = _button("删除复用包", "danger", 86)
        self.deleteReuseBtn.clicked.connect(self.on_delete_reuse)
        reuse_row.addWidget(self.keepReuseCheck)
        reuse_row.addWidget(self.checkReuseBtn)
        reuse_row.addWidget(self.deleteReuseBtn)
        reuse_row.addStretch(1)
        layout.addLayout(reuse_row)

        self.reuseInfoLabel = qt.QLabel("复用包：尚未检测")
        self.reuseInfoLabel.setObjectName("hint")
        self.reuseInfoLabel.setWordWrap(True)
        layout.addWidget(self.reuseInfoLabel)

        self.maskLabel = qt.QLabel("尚未产生分割结果")
        self.maskLabel.setObjectName("hint")
        self.maskLabel.setWordWrap(True)
        layout.addWidget(self.maskLabel)
        return panel

    def _build_correction_panel(self):
        panel, layout = _make_panel("人工修正")
        hint = qt.QLabel(
            "这些按钮会调用 Slicer 自带的 Segment Editor。画笔用于涂抹，擦除用于删去误分割，"
            "剪刀/闭合曲线适合圈画式修边。"
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        row1 = qt.QHBoxLayout()
        row1.setSpacing(6)
        self.openEditorBtn = _button("打开编辑", "ghost", 74)
        self.openEditorBtn.clicked.connect(self.on_open_editor)
        self.paintBtn = _button("涂抹", "ghost", 54)
        self.paintBtn.clicked.connect(lambda: self.on_activate_segment_effect("Paint"))
        self.eraseBtn = _button("擦除", "ghost", 54)
        self.eraseBtn.clicked.connect(lambda: self.on_activate_segment_effect("Erase"))
        self.drawBtn = _button("圈画", "ghost", 54)
        self.drawBtn.clicked.connect(lambda: self.on_activate_segment_effect("Draw"))
        for btn in (self.openEditorBtn, self.paintBtn, self.eraseBtn, self.drawBtn):
            row1.addWidget(btn)
        row1.addStretch(1)
        layout.addLayout(row1)

        row2 = qt.QHBoxLayout()
        row2.setSpacing(6)
        self.scissorsBtn = _button("剪刀", "ghost", 58)
        self.scissorsBtn.clicked.connect(lambda: self.on_activate_segment_effect("Scissors"))
        self.smoothBtn = _button("平滑", "ghost", 58)
        self.smoothBtn.clicked.connect(lambda: self.on_activate_segment_effect("Smoothing"))
        self.islandsBtn = _button("连通域", "ghost", 66)
        self.islandsBtn.clicked.connect(lambda: self.on_activate_segment_effect("Islands"))
        self.readCorrectBtn = _button("读取修正", "accent", 82)
        self.readCorrectBtn.clicked.connect(self.on_read_result)
        for btn in (self.scissorsBtn, self.smoothBtn, self.islandsBtn, self.readCorrectBtn):
            row2.addWidget(btn)
        row2.addStretch(1)
        layout.addLayout(row2)

        self.correctLabel = qt.QLabel("尚未进入人工修正")
        self.correctLabel.setObjectName("hint")
        self.correctLabel.setWordWrap(True)
        layout.addWidget(self.correctLabel)
        return panel

    def _build_label_panel(self):
        panel, layout = _make_panel("标签集合与标签列表")
        row = qt.QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(qt.QLabel("标签集合"))
        self.labelSetMenu = qt.QMenu()
        self.labelSetButton = qt.QToolButton()
        self.labelSetButton.setObjectName("labelSetButton")
        self.labelSetButton.setText("暂无标签集合")
        self.labelSetButton.setPopupMode(qt.QToolButton.InstantPopup)
        self.labelSetButton.setMenu(self.labelSetMenu)
        self.createLabelSetBtn = _button("新建", "ghost", 54)
        self.createLabelSetBtn.clicked.connect(self.on_create_label_collection)
        row.addWidget(self.labelSetButton, 1)
        row.addWidget(self.createLabelSetBtn)
        layout.addLayout(row)

        self.labelTable = qt.QTableWidget()
        self.labelTable.setColumnCount(4)
        self.labelTable.setHorizontalHeaderLabels(["显示", "标签名称", "颜色", "状态"])
        self.labelTable.setMinimumHeight(210)
        self.labelTable.setMaximumHeight(260)
        try:
            self.labelTable.horizontalHeader().setStretchLastSection(True)
            self.labelTable.verticalHeader().setVisible(False)
        except Exception:
            pass
        layout.addWidget(self.labelTable)
        self.labelSummaryLabel = qt.QLabel("当前没有标签集合。检测到分割结果或完成模型分割后才会生成列表。")
        self.labelSummaryLabel.setObjectName("hint")
        self.labelSummaryLabel.setWordWrap(True)
        layout.addWidget(self.labelSummaryLabel)
        return panel

    def _build_quality_export_panel(self):
        panel, layout = _make_panel("质检与导出")
        row = qt.QHBoxLayout()
        row.setSpacing(6)
        self.qualityBtn = _button("标签质检", "primary", 82)
        self.qualityBtn.clicked.connect(self.on_quality_check)
        self.exportBtn = _button("导出训练数据", "accent", 104)
        self.exportBtn.clicked.connect(self.on_export)
        row.addWidget(self.qualityBtn)
        row.addWidget(self.exportBtn)
        row.addStretch(1)
        layout.addLayout(row)

        self.qualityLabel = qt.QLabel("尚未质检")
        self.qualityLabel.setObjectName("hint")
        self.exportLabel = qt.QLabel("尚未导出")
        self.exportLabel.setObjectName("hint")
        layout.addWidget(self.qualityLabel)
        layout.addWidget(self.exportLabel)
        return panel

    def _build_log_panel(self):
        panel, layout = _make_panel("日志")
        row = qt.QHBoxLayout()
        row.addStretch(1)
        self.clearLogBtn = _button("清空", "ghost", 58)
        self.clearLogBtn.clicked.connect(self._on_clear_log)
        row.addWidget(self.clearLogBtn)
        layout.addLayout(row)

        self.logEdit = qt.QPlainTextEdit()
        self.logEdit.setReadOnly(True)
        self.logEdit.setMaximumHeight(155)
        layout.addWidget(self.logEdit)
        return panel

    # ---------- common helpers ----------

    def _script_dir(self):
        return os.path.dirname(os.path.abspath(__file__))

    def _project_root(self):
        return os.path.dirname(os.path.dirname(self._script_dir()))

    def _log(self, msg):
        stamp = time.strftime("%H:%M:%S")
        self.logEdit.appendPlainText(f"[{stamp}] {msg}")
        sb = self.logEdit.verticalScrollBar()
        sb.setValue(sb.maximum)
        import logging
        logging.getLogger("CBCTAnnotator").info(msg)

    def _on_clear_log(self):
        self.logEdit.clear()

    def _set_status(self, text, color=None):
        self.statusLabel.setText(text)
        color = color or _TEXT_DIM
        self.statusLabel.setStyleSheet(f"QLabel#hint{{color:{color};}}")

    def _set_busy(self, busy, text=None):
        if text:
            self._set_status(text, None)

    def _save_ui_settings(self):
        self._settings.setValue("base_url", _line_text(self.addressEdit).strip())
        self._settings.setValue("model_id", _combo_text(self.modelCombo))
        self._settings.setValue("mode", "balanced")
        if hasattr(self, "spacingSpin"):
            self._settings.setValue("spacing_mm", f"{self._current_spacing_mm():.2f}")
        self._settings.setValue("template_id", _combo_text(self.templateCombo))
        if hasattr(self, "keepReuseCheck"):
            self._settings.setValue(
                "keep_reuse", "true" if self.keepReuseCheck.isChecked() else "false")

    def _current_spacing_mm(self):
        if not hasattr(self, "spacingSpin"):
            return 0.75
        value = _spin_value(self.spacingSpin)
        return max(0.5, min(2.0, round(value, 2)))

    def _on_spacing_changed(self, *args):
        spacing = self._current_spacing_mm()
        if spacing <= 0.55:
            hint = "当前为高细节设置，对显存压力较大；8GB 显卡可能失败。"
        elif spacing < 0.9:
            hint = "当前为推荐设置，速度、显存与细节相对均衡。"
        else:
            hint = "当前为保守设置，占用更低，适合先跑通流程或显存不足时使用。"
        if hasattr(self, "spacingHintLabel"):
            self.spacingHintLabel.setText(f"本次推理将先把图像重采样到 {spacing:.2f} mm。{hint}")
        if hasattr(self, "modelSupportLabel"):
            self._on_model_changed()
        self._save_ui_settings()

    def _auto_initialize(self):
        self._log("自动检测服务连接与模型配置")
        if self.on_connect(auto=True):
            self.on_load_config(auto=True)
        self._auto_detect_scene_content(force=True)

    def _format_progress_bar(self, percent):
        blocks = 20
        filled = int(round(blocks * percent / 100.0))
        return "[" + "#" * filled + "-" * (blocks - filled) + f"] {percent:3d}%"

    def _format_elapsed(self, seconds):
        seconds = max(0, int(seconds or 0))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _set_predict_progress(self, percent, stage):
        percent = max(0, min(100, int(percent)))
        self.predictProgress.setValue(percent)
        self.predictProgress.setFormat(f"{percent}%  {stage}")

    def _current_volume_from_scene(self):
        try:
            selection_node = slicer.app.applicationLogic().GetSelectionNode()
            volume_id = selection_node.GetActiveVolumeID()
            if volume_id:
                node = slicer.mrmlScene.GetNodeByID(volume_id)
                if node is not None:
                    return node
        except Exception:
            pass
        nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        if nodes:
            return nodes[-1]
        return None

    def _volume_file_path(self, node):
        storage = node.GetStorageNode() if node else None
        if storage:
            path = storage.GetFileName()
            if path:
                return path
        return None

    def _latest_labelmap_from_scene(self):
        nodes = slicer.util.getNodesByClass("vtkMRMLLabelMapVolumeNode")
        return nodes[-1] if nodes else None

    def _latest_segmentation_from_scene(self):
        nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        return nodes[-1] if nodes else None

    def _bind_volume_node(self, node, auto=False):
        old_path = self.image_path
        self.volume_node = node
        self.image_path = self._volume_file_path(node)
        if old_path != self.image_path:
            self.case_id = None
            self.roi_ijk_start = None
            self.roi_ijk_size = None
        dims = node.GetImageData().GetDimensions() if node.GetImageData() else None
        spacing = node.GetSpacing()
        path_text = self.image_path or "当前影像没有磁盘路径，需先保存为 .nii.gz 后才能调用后端"
        self.imageInfoLabel.setText(
            f"影像：{node.GetName()} | 尺寸：{dims} | 间距："
            f"{spacing[0]:.3g}/{spacing[1]:.3g}/{spacing[2]:.3g} mm\n路径：{path_text}"
        )
        if self.image_path:
            if self.api is None:
                self._set_status("已自动绑定当前 Slicer 影像；后端服务连接后会继续检查可用性。", _WARN)
            elif not self.image_path.lower().endswith((".nii", ".nii.gz")):
                self._set_status("已自动绑定当前影像，但 ToothSeg 推理需要 .nii 或 .nii.gz 文件。", _WARN)
            else:
                self._set_status("已自动绑定当前 Slicer 影像，可点击开始分割。", _OK)
            self._log(f"{'自动' if auto else '手动'}绑定当前影像: {self.image_path}")
        else:
            self._set_status("已检测到当前影像，但后端需要可读取的磁盘文件路径。", _WARN)
            self._log("当前影像缺少磁盘路径，建议另存为 .nii.gz 后再分割")

    def _auto_inspect_current_image(self):
        if self.api is None or not self.image_path:
            return
        try:
            case_id = self._ensure_case_id()
            info = self.api.inspect_image(case_id, self.image_path)
        except ApiError as e:
            self._set_status(f"影像自动检查失败: [{e.error_code}] {e.message}", _DANGER)
            self._log(f"影像自动检查失败: {e.error_code} | {e.message}")
            return
        self._log(f"影像自动检查通过: shape={info.get('shape')}, spacing={info.get('spacing')}")
        self._set_status("当前影像已通过后端可用性检查，可点击开始分割。", _OK)

    def _auto_detect_scene_content(self, force=False):
        volume = self._current_volume_from_scene()
        volume_id = volume.GetID() if volume is not None else None
        if volume is not None and (force or volume_id != self._last_auto_volume_id):
            self._last_auto_volume_id = volume_id
            self._bind_volume_node(volume, auto=True)
            self._auto_inspect_current_image()
        elif force and volume is None:
            self._set_status("尚未检测到 Slicer 当前影像，请先用 Slicer 原生功能打开 CBCT。", _WARN)

        label = self._latest_labelmap_from_scene()
        label_id = label.GetID() if label is not None else None
        if (
            label is not None
            and label_id != self._last_auto_label_id
            and label is not getattr(self, "mask_node", None)
            and label is not getattr(self, "corrected_node", None)
        ):
            self._last_auto_label_id = label_id
            self.mask_node = label
            self.last_mask_path = self._volume_file_path(label)
            self._create_label_collection_from_labelmap(
                label, f"当前标签-{time.strftime('%H%M%S')}")
            self._set_status("已自动检测到分割标签，并生成标签集合。", _OK)

        seg = self._latest_segmentation_from_scene()
        seg_id = seg.GetID() if seg is not None else None
        if (
            seg is not None
            and seg_id != self._last_auto_segmentation_id
            and seg is not getattr(self, "correct_seg_node", None)
        ):
            self._last_auto_segmentation_id = seg_id
            self.correct_seg_node = seg
            self._create_label_collection_from_segmentation(
                seg, f"当前分割-{time.strftime('%H%M%S')}")
            self._set_status("已自动检测到 Segmentation，并生成标签集合。", _OK)

    def _ras_to_ijk(self, ras):
        import vtk
        matrix = vtk.vtkMatrix4x4()
        self.volume_node.GetRASToIJKMatrix(matrix)
        p = matrix.MultiplyPoint([ras[0], ras[1], ras[2], 1.0])
        return [int(round(p[0])), int(round(p[1])), int(round(p[2]))]

    # ---------- service and config ----------

    def on_refresh_service(self):
        if self.on_connect(auto=False):
            self.on_load_config(auto=False)

    def on_connect(self, auto=False):
        url = _line_text(self.addressEdit).strip()
        try:
            self.api = ApiClient(url)
            st = self.api.status()
        except ApiError as e:
            self.api = None
            self.connStatusLabel.setText("服务：不可用")
            self.connStatusLabel.setStyleSheet(f"color:{_DANGER}; font-weight:bold;")
            self.modelStatusLabel.setText("模型：未检测")
            self._set_status("未连接到本地服务，请先启动后端服务。", _DANGER)
            self._log(f"服务检测失败: {e.error_code} | {e.message}")
            return False

        model = st.get("model", {})
        device = st.get("device", {})
        loaded = bool(model.get("loaded"))
        device_text = device.get("data") or device.get("name") or device.get("type") or "unknown"
        service_name = st.get("service", {}).get("name", "local-service")
        self.connStatusLabel.setText(f"服务：可用 ({service_name})")
        self.connStatusLabel.setStyleSheet(f"color:{_OK}; font-weight:bold;")
        self.modelStatusLabel.setText(
            f"模型：{'已加载' if loaded else '未完整加载'} · 设备：{device_text}"
        )
        self.modelStatusLabel.setStyleSheet(
            f"color:{_OK if loaded else _WARN};")
        self._set_status("服务已自动连接，模型配置将自动同步。", _OK)
        self._log(f"服务连接成功: service={service_name}, device={device_text}, model_loaded={loaded}")
        self._save_ui_settings()
        return True

    def on_load_config(self, auto=False):
        if self.api is None:
            if not auto:
                self._set_status("服务不可用，无法加载模型配置。", _DANGER)
            return False
        try:
            cfg = self.api.config()
        except ApiError as e:
            self._set_status(f"配置加载失败: [{e.error_code}]", _DANGER)
            self._log(f"配置加载失败: {e.error_code} | {e.message}")
            return False

        last_model = str(self._settings.value("model_id", "toothseg-semantic-05mm"))
        last_template = str(self._settings.value("template_id", _DEFAULT_TEMPLATE_ID))

        self.modelCombo.clear()
        for model in cfg.get("models", []):
            self.modelCombo.addItem(model.get("model_id", "?"))
        if self.modelCombo.findText(last_model) >= 0:
            self.modelCombo.setCurrentText(last_model)
        elif self.modelCombo.findText("toothseg-semantic-05mm") >= 0:
            self.modelCombo.setCurrentText("toothseg-semantic-05mm")

        self.modeCombo.clear()
        self.modeCombo.addItem("balanced")
        self.modeCombo.setCurrentText("balanced")
        self.modeCombo.setEnabled(False)

        downsample = cfg.get("downsample") or {}
        if hasattr(self, "spacingSpin") and downsample:
            self.spacingSpin.blockSignals(True)
            self.spacingSpin.setRange(
                float(downsample.get("min", 0.5)),
                float(downsample.get("max", 2.0)),
            )
            self.spacingSpin.setSingleStep(float(downsample.get("step", 0.05)))
            saved_spacing = _settings_float(
                self._settings, "spacing_mm", float(downsample.get("default", 0.75)))
            self.spacingSpin.setValue(saved_spacing)
            self.spacingSpin.blockSignals(False)
            self._on_spacing_changed()

        self.templateCombo.clear()
        self.label_rules = []
        for template in cfg.get("label_templates", []):
            template_id = template.get("template_id", "?")
            self.templateCombo.addItem(template_id)
            if template_id == last_template or not self.label_rules:
                self.label_rules = template.get("labels", []) or []
        if self.templateCombo.findText(last_template) >= 0:
            self.templateCombo.setCurrentText(last_template)

        if not self.active_label_collection:
            self._populate_label_table([])
            self.labelSummaryLabel.setText(
                f"已加载标签命名规则 {len(self.label_rules)} 条；当前病例还没有标签集合。")
        self._on_model_changed()
        self._set_status("模型与标签配置已自动加载。", _OK)
        self._log(
            f"配置加载完成: models={_combo_count(self.modelCombo)}, "
            f"modes={_combo_count(self.modeCombo)}, label_rules={len(self.label_rules)}"
        )
        self._save_ui_settings()
        return True

    def _on_model_changed(self, *args):
        model_id = _combo_text(self.modelCombo)
        spacing = self._current_spacing_mm()
        if model_id == "mock-simple-cube":
            text = "当前模型用于流程联调，会生成测试 mask。"
        elif model_id == "toothseg-semantic-05mm":
            text = f"当前模型按整张 CBCT 做语义分割，处理方式固定为 balanced，降采样间距 {spacing:.2f}mm。"
        elif model_id == "toothseg-full":
            text = f"当前模型按整张 CBCT 做 ToothSeg 双分支实例分割，处理方式固定为 balanced，降采样间距 {spacing:.2f}mm。"
        else:
            text = "当前模型按服务端配置执行，处理方式固定为 balanced。"
        self.modelSupportLabel.setText(text)
        self._save_ui_settings()

    def on_check_reuse(self):
        self._auto_detect_scene_content(force=True)
        if self.api is None:
            self._set_status("服务不可用，无法检测复用包。", _DANGER)
            return
        if not self.image_path:
            self._set_status("当前影像没有文件路径，无法检测复用包。", _WARN)
            return
        spacing_mm = self._current_spacing_mm()
        try:
            info = self.api.reuse_status(
                self.image_path, _combo_text(self.modelCombo), _combo_text(self.modeCombo), spacing_mm)
        except ApiError as e:
            self.reuseInfoLabel.setText(f"复用包检测失败：{e.message}")
            self._set_status(f"复用包检测失败: [{e.error_code}]", _DANGER)
            self._log(f"复用包检测失败: {e.error_code} | {e.message}")
            return
        self.reuseInfoLabel.setText(
            f"{info.get('message')} 路径：{info.get('reuse_dir')}")
        self._set_status("复用包检测完成。", _OK if info.get("can_reuse") else _WARN)
        self._log(
            f"复用包检测: can_reuse={info.get('can_reuse')}, "
            f"resume_from={info.get('resume_from')}, spacing={spacing_mm:.2f}mm, task_key={info.get('task_key')}"
        )

    def on_delete_reuse(self):
        self._auto_detect_scene_content(force=True)
        if self.api is None:
            self._set_status("服务不可用，无法删除复用包。", _DANGER)
            return
        if not self.image_path:
            self._set_status("当前影像没有文件路径，无法删除复用包。", _WARN)
            return
        spacing_mm = self._current_spacing_mm()
        try:
            result = self.api.delete_reuse(
                self.image_path, _combo_text(self.modelCombo), _combo_text(self.modeCombo), spacing_mm)
        except ApiError as e:
            self._set_status(f"删除复用包失败: [{e.error_code}]", _DANGER)
            self._log(f"删除复用包失败: {e.error_code} | {e.message}")
            return
        self.reuseInfoLabel.setText(result.get("message", "复用包删除操作完成"))
        self._set_status("复用包删除操作完成。", _OK)
        self._log(f"复用包删除: {result}")

    # ---------- case and selection ----------

    def on_use_current_volume(self, silent=False):
        node = self._current_volume_from_scene()
        if node is None:
            if not silent:
                self._set_status("Slicer 场景中没有可用影像。", _DANGER)
                self._log("未找到当前影像，请先用 Slicer 打开 CBCT 文件")
            return
        self._bind_volume_node(node, auto=silent)

    def on_inspect_current_image(self):
        self.on_use_current_volume(silent=True)
        if self.api is None:
            self._set_status("服务不可用，无法调用 /images/inspect。", _DANGER)
            return
        if not self.image_path:
            self._set_status("当前影像没有可传给后端的文件路径。", _WARN)
            return
        try:
            case_id = self._ensure_case_id()
            info = self.api.inspect_image(case_id, self.image_path)
        except ApiError as e:
            self._set_status(f"影像检查失败: [{e.error_code}]", _DANGER)
            self._log(f"影像检查失败: {e.error_code} | {e.message}")
            return
        self._log(f"后端影像检查: shape={info.get('shape')}, spacing={info.get('spacing')}")
        self._set_status("后端已确认可以读取当前影像。", _OK)

    def on_import_volume(self):
        path = qt.QFileDialog.getOpenFileName(
            self.parent,
            "导入原始 CBCT Volume",
            "",
            "Volume (*.nii *.nii.gz *.nrrd *.mha *.mhd);;所有文件 (*)",
        )
        if not path:
            return
        path = path if isinstance(path, str) else path[0]
        try:
            node = slicer.util.loadVolume(path, {
                "name": os.path.basename(path),
                "autoWindowLevel": True,
            })
        except Exception as e:
            self._set_status(f"导入 Volume 失败: {e}", _DANGER)
            self._log(f"导入 Volume 失败: {e}")
            return
        self.volume_node = node
        self.image_path = path
        self.case_id = None
        slicer.util.resetSliceViews()
        self.on_use_current_volume(silent=True)
        self._log(f"已导入 Volume: {path}")

    def on_import_segmentation(self):
        path = qt.QFileDialog.getOpenFileName(
            self.parent,
            "导入分割文件 Segmentation / LabelMap",
            "",
            "Segmentation or LabelMap (*.seg.nrrd *.nii *.nii.gz *.nrrd);;所有文件 (*)",
        )
        if not path:
            return
        path = path if isinstance(path, str) else path[0]
        try:
            if path.lower().endswith(".seg.nrrd"):
                loaded = slicer.util.loadSegmentation(path)
                if isinstance(loaded, (list, tuple)) and len(loaded) >= 2:
                    node = loaded[1]
                else:
                    node = loaded
                if node is None or not hasattr(node, "GetSegmentation"):
                    raise RuntimeError("Slicer 未返回有效的 Segmentation 节点")
                self.correct_seg_node = node
                self._create_label_collection_from_segmentation(
                    node, f"导入分割-{time.strftime('%H%M%S')}")
            else:
                node = slicer.util.loadLabelVolume(path)
                node.SetName("Imported_LabelMap")
                self.mask_node = node
                self.last_mask_path = path
                self._create_label_collection_from_labelmap(
                    node, f"导入标签-{time.strftime('%H%M%S')}")
        except Exception as e:
            self._set_status(f"导入分割失败: {e}", _DANGER)
            self._log(f"导入分割失败: {e}")
            return
        self._set_status("分割文件已导入，标签集合已生成。", _OK)
        self._log(f"已导入分割文件: {path}")

    def on_place_point(self):
        self._set_prompt_tool_button(self.pointBtn)
        if self.volume_node is None:
            self.on_use_current_volume(silent=True)
        if self.volume_node is None:
            self._set_status("请先在 Slicer 中打开并绑定影像。", _DANGER)
            return
        try:
            if self.point_node is None:
                self.point_node = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLMarkupsFiducialNode", "CBCT_Points")
                self.point_node.CreateDefaultDisplayNodes()
            self._activate_markups_placement(self.point_node)
            self._set_status("点选模式已开启，在视图中点击牙齿中心点。", _OK)
            self._log("已开启点选工具：当前模型仅记录点坐标，暂不作为 ToothSeg 输入")
        except Exception as e:
            self._set_status(f"点选工具启动失败: {e}", _DANGER)
            self._log(f"点选工具异常: {e}")

    def on_place_curve(self):
        self._set_prompt_tool_button(self.curveBtn)
        if self.volume_node is None:
            self.on_use_current_volume(silent=True)
        if self.volume_node is None:
            self._set_status("请先在 Slicer 中打开并绑定影像。", _DANGER)
            return
        try:
            if self.curve_node is None:
                self.curve_node = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLMarkupsClosedCurveNode", "CBCT_ClosedCurve")
                self.curve_node.CreateDefaultDisplayNodes()
            self._activate_markups_placement(self.curve_node)
            self._set_status("圈画模式已开启，可围绕目标区域放置闭合曲线点。", _OK)
            self._log("已开启圈画工具：当前模型仅记录圈画提示，暂不作为 ToothSeg 输入")
        except Exception as e:
            self._set_status(f"圈画工具启动失败: {e}", _DANGER)
            self._log(f"圈画工具异常: {e}")

    def _activate_markups_placement(self, node):
        app_logic = slicer.app.applicationLogic()
        selection_node = app_logic.GetSelectionNode()
        if hasattr(selection_node, "SetReferenceActivePlaceNodeClassName"):
            selection_node.SetReferenceActivePlaceNodeClassName(node.GetClassName())
        if hasattr(selection_node, "SetActivePlaceNodeID"):
            selection_node.SetActivePlaceNodeID(node.GetID())
        elif hasattr(selection_node, "SetReferenceActivePlaceNodeID"):
            selection_node.SetReferenceActivePlaceNodeID(node.GetID())
        interaction_node = app_logic.GetInteractionNode()
        interaction_node.SetCurrentInteractionMode(interaction_node.Place)

    def on_new_roi(self):
        self._set_prompt_tool_button(self.newRoiBtn)
        if self.volume_node is None:
            self.on_use_current_volume(silent=True)
        if self.volume_node is None:
            self._set_status("请先在 Slicer 中打开并绑定影像。", _DANGER)
            self._log("尚未绑定影像，无法创建 ROI")
            return
        try:
            if self.roi_node is not None and self.roi_node.IsA("vtkMRMLMarkupsROINode"):
                slicer.mrmlScene.RemoveNode(self.roi_node)
            roi = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", "CBCT_ROI")
            bounds = [0] * 6
            self.volume_node.GetBounds(bounds)
            center = [
                0.5 * (bounds[0] + bounds[1]),
                0.5 * (bounds[2] + bounds[3]),
                0.5 * (bounds[4] + bounds[5]),
            ]
            size = [
                0.85 * (bounds[1] - bounds[0]),
                0.85 * (bounds[3] - bounds[2]),
                0.85 * (bounds[5] - bounds[4]),
            ]
            roi.SetCenter(center)
            roi.SetSize(size)
            roi.CreateDefaultDisplayNodes()
            display = roi.GetDisplayNode()
            if display:
                display.SetVisibility(True)
                display.SetSelected(True)
            self.roi_node = roi
            slicer.util.resetSliceViews()
            self._set_status("ROI 框已创建，可在视图中拖拽调整后读取选择。", _OK)
            self._log("已创建 ROI 框")
        except Exception as e:
            self._set_status(f"创建 ROI 失败: {e}", _DANGER)
            self._log(f"创建 ROI 异常: {e}")

    def on_clear_selection(self):
        for node in (self.roi_node, self.point_node, self.curve_node):
            try:
                if node is not None:
                    slicer.mrmlScene.RemoveNode(node)
            except Exception:
                pass
        self.roi_node = None
        self.point_node = None
        self.curve_node = None
        self.roi_ijk_start = None
        self.roi_ijk_size = None
        self.point_prompts = []
        self._set_prompt_tool_button(None)
        self.selectionInfoLabel.setText("选择信息已清空")
        self._set_status("点选、框选、圈画信息已清空。", None)
        self._log("已清空选择提示")

    def _set_prompt_tool_button(self, active_button):
        for btn in (
            getattr(self, "pointBtn", None),
            getattr(self, "newRoiBtn", None),
            getattr(self, "curveBtn", None),
        ):
            if btn is not None:
                btn.setChecked(btn is active_button)

    def on_read_selection(self):
        parts = []
        if self.roi_node is not None:
            start, size = self._read_roi_ijk()
            if start is not None:
                self.roi_ijk_start = start
                self.roi_ijk_size = size
                parts.append(f"ROI start={start}, size={size}")
        if self.point_node is not None:
            self.point_prompts = self._read_points_ijk(self.point_node)
            parts.append(f"点选 {len(self.point_prompts)} 个")
        if self.curve_node is not None:
            parts.append(f"圈画控制点 {self.curve_node.GetNumberOfControlPoints()} 个")
        if not parts:
            self.selectionInfoLabel.setText("尚未创建点选、ROI 或圈画提示")
            self._set_status("没有可读取的选择提示。", _WARN)
            return
        text = " | ".join(parts)
        self.selectionInfoLabel.setText(text)
        self._set_status("选择提示已读取。", _OK)
        self._log("选择提示: " + text)

    def _read_points_ijk(self, node):
        points = []
        for i in range(node.GetNumberOfControlPoints()):
            ras = [0.0, 0.0, 0.0]
            node.GetNthControlPointPositionWorld(i, ras)
            points.append(self._ras_to_ijk(ras))
        return points

    def _read_roi_ijk(self):
        if self.roi_node is None or self.volume_node is None:
            return None, None
        center = list(self.roi_node.GetCenter())
        size = list(self.roi_node.GetSize())
        mins = [center[i] - size[i] / 2.0 for i in range(3)]
        maxs = [center[i] + size[i] / 2.0 for i in range(3)]
        corners = []
        for x in (mins[0], maxs[0]):
            for y in (mins[1], maxs[1]):
                for z in (mins[2], maxs[2]):
                    corners.append(self._ras_to_ijk([x, y, z]))
        lower = [min(p[i] for p in corners) for i in range(3)]
        upper = [max(p[i] for p in corners) for i in range(3)]
        dims = self.volume_node.GetImageData().GetDimensions()
        lower = [max(0, min(lower[i], dims[i] - 1)) for i in range(3)]
        upper = [max(0, min(upper[i], dims[i])) for i in range(3)]
        size_ijk = [max(1, upper[i] - lower[i]) for i in range(3)]
        return lower, size_ijk

    # ---------- predict ----------

    def _ensure_case_id(self):
        if not self.case_id:
            template_id = _combo_text(self.templateCombo) or _DEFAULT_TEMPLATE_ID
            case = self.api.create_case(
                self.image_path, "nii", template_id, "operator-b")
            self.case_id = case.get("case_id")
        return self.case_id

    def on_predict(self):
        if self.api is None:
            self._set_status("服务不可用，请先启动后端并刷新。", _DANGER)
            self._log("尚未连接服务端")
            return
        self._auto_detect_scene_content(force=True)
        if self.volume_node is None or not self.image_path:
            self._set_status("当前影像缺少可供后端读取的文件路径。", _DANGER)
            self._log("无法分割：请先在 Slicer 打开并保存 .nii.gz 影像")
            return
        dims = self.volume_node.GetImageData().GetDimensions()
        roi = {
            "start": [0, 0, 0],
            "size": [int(dims[0]), int(dims[1]), int(dims[2])],
        }
        model_id = _combo_text(self.modelCombo)
        mode = "balanced"
        spacing_mm = self._current_spacing_mm()
        if not model_id:
            self._set_status("尚未加载模型配置，无法开始分割。", _DANGER)
            return
        try:
            case_id = self._ensure_case_id()
        except ApiError as e:
            self._set_status(f"病例创建失败: [{e.error_code}]", _DANGER)
            self._log(f"病例创建失败: {e.error_code} | {e.message}")
            return

        self._save_ui_settings()
        self.predictBtn.setEnabled(False)
        self._predict_started_at = time.time()
        self._predict_model_id = model_id
        self._predict_mode = mode
        self._predict_spacing_mm = spacing_mm
        self._last_progress_logged = -1
        self._last_backend_progress_key = None
        self._cancel_requested = False
        self._pending_progress = None
        self._pending_progress_error = None
        self._pending_predict_result = None
        self._pending_predict_error = None
        self._pending_cancel_result = None
        self._pending_cancel_error = None
        self._set_predict_progress(1, "已提交请求")
        self.cancelPredictBtn.setEnabled(True)
        self.maskLabel.setText(
            f"正在分割，已运行 00:00:00 · 模型 {model_id} · 模式 {mode} · "
            f"降采样 {spacing_mm:.2f}mm · 已提交请求"
        )
        self._set_status("AI 分割已开始，进度将按后端真实处理阶段更新。", _OK)
        self._log(
            f"提交分割: model={model_id}, mode={mode}, spacing={spacing_mm:.2f}mm, case_id={case_id}, "
            f"roi={roi}"
        )
        self._log("推理进度 " + self._format_progress_bar(1) + " 已提交请求")
        self.predictTimer.start()

        def worker():
            try:
                result = self.api.predict(
                    case_id,
                    self.image_path,
                    roi,
                    model_id,
                    mode,
                    targets=["teeth"],
                    output_format="nii.gz",
                    keep_reuse=self.keepReuseCheck.isChecked(),
                    spacing_mm=spacing_mm,
                )
            except ApiError as e:
                self._pending_predict_error = e
            except Exception as e:
                self._pending_predict_error = ApiError("PREDICTION_FAILED", str(e))
            else:
                self._pending_predict_result = result

        threading.Thread(target=worker, daemon=True).start()

    def on_cancel_predict(self):
        if self.api is None or not self.case_id:
            self._set_status("当前没有可中止的后端分割任务。", _WARN)
            return
        keep_reuse = self.keepReuseCheck.isChecked()
        self._cancel_requested = True
        self.cancelPredictBtn.setEnabled(False)
        action = "保留并检测复用包" if keep_reuse else "删除复用包"
        self._set_status(f"正在请求中止分割；终止后将{action}。", _WARN)
        self._log(f"请求中止分割: case_id={self.case_id}, keep_reuse={keep_reuse}")

        def worker():
            try:
                result = self.api.cancel_predict(
                    self.case_id,
                    self.image_path,
                    self._predict_model_id or _combo_text(self.modelCombo),
                    self._predict_mode or "balanced",
                    keep_reuse,
                    self._predict_spacing_mm or self._current_spacing_mm(),
                )
            except ApiError as e:
                self._pending_cancel_error = e
            else:
                self._pending_cancel_result = result

        threading.Thread(target=worker, daemon=True).start()

    def _on_cancel_ok(self, result):
        self._log(f"中止请求返回: {result}")
        if result.get("cancel_requested"):
            self._set_status("后端已收到中止请求，正在停止模型进程。", _WARN)
            self.maskLabel.setText("正在中止分割，请等待后端释放 GPU 与文件句柄。")
        else:
            self._set_status(result.get("message", "当前没有正在运行的后端推理任务。"), _WARN)
            if self._predict_started_at:
                self.cancelPredictBtn.setEnabled(True)

    def _on_cancel_error(self, error):
        self._log(f"中止请求失败: {error.error_code} | {error.message}")
        self._set_status(f"中止请求失败: [{error.error_code}]", _DANGER)
        if self._predict_started_at:
            self.cancelPredictBtn.setEnabled(True)

    def _on_predict_tick(self):
        if not self._predict_started_at:
            return
        elapsed = int(time.time() - self._predict_started_at)
        elapsed_text = self._format_elapsed(elapsed)
        if self._pending_cancel_result is not None:
            result = self._pending_cancel_result
            self._pending_cancel_result = None
            self._on_cancel_ok(result)
        if self._pending_cancel_error is not None:
            error = self._pending_cancel_error
            self._pending_cancel_error = None
            self._on_cancel_error(error)
        if self._pending_predict_result is not None:
            result = self._pending_predict_result
            self._pending_predict_result = None
            self._on_predict_ok(result)
            return
        if self._pending_predict_error is not None:
            error = self._pending_predict_error
            self._pending_predict_error = None
            self._on_predict_error(error)
            return
        if self._pending_progress is not None:
            progress, progress_elapsed = self._pending_progress
            self._pending_progress = None
            self._apply_backend_progress(progress, progress_elapsed)
        if self._pending_progress_error is not None:
            error, progress_elapsed = self._pending_progress_error
            self._pending_progress_error = None
            self._on_backend_progress_error(error, progress_elapsed)

        if self._last_backend_progress_key is None:
            self.maskLabel.setText(
                f"正在分割，已运行 {elapsed_text} · 已提交请求，等待后端进度状态。"
            )
        if self.api is None or not self.case_id:
            return
        if self._progress_polling:
            return
        self._progress_polling = True

        def poller():
            try:
                progress = self.api.predict_progress(self.case_id)
            except ApiError as e:
                self._pending_progress_error = (e, elapsed)
            else:
                self._pending_progress = (progress, elapsed)

        threading.Thread(target=poller, daemon=True).start()

    def _on_backend_progress_error(self, error, elapsed):
        self._progress_polling = False
        if not self._predict_started_at:
            return
        self.maskLabel.setText(
            f"正在分割，已运行 {self._format_elapsed(elapsed)} · 暂时无法读取后端进度：{error.message}"
        )

    def _apply_backend_progress(self, progress, elapsed):
        self._progress_polling = False
        if not self._predict_started_at:
            return
        try:
            percent = int(progress.get("percent", 0))
        except Exception:
            percent = 0
        stage = progress.get("stage", "unknown")
        message = progress.get("message", "后端正在处理。")
        run_status = progress.get("run_status", "running")
        spacing_text = f"{(self._predict_spacing_mm or self._current_spacing_mm()):.2f}mm"
        self._set_predict_progress(percent, stage)
        text = (
            f"正在分割，已运行 {self._format_elapsed(elapsed)} · 模型 {self._predict_model_id} · "
            f"模式 {self._predict_mode} · 降采样 {spacing_text} · "
            f"后端阶段：{stage} · {message}"
        )
        self.maskLabel.setText(text)
        key = (percent, stage, message, run_status)
        if key != self._last_backend_progress_key:
            self._last_backend_progress_key = key
            self._last_progress_logged = percent
            self._log("后端进度 " + self._format_progress_bar(percent) + f" {stage}: {message}")
            last_log = (progress.get("details") or {}).get("last_log")
            if last_log:
                self._log(f"后端日志: {last_log}")
        if run_status == "error":
            self._set_status(f"后端报告分割异常：{message}", _DANGER)
            self._on_predict_error(
                ApiError("BACKEND_PROGRESS_ERROR", message, progress.get("details", {}))
            )
        elif run_status in {"cancelling", "cancelled"}:
            self._set_status(message, _WARN)

    def _on_predict_ok(self, result):
        self.predictTimer.stop()
        self._predict_started_at = None
        self._progress_polling = False
        self.predictBtn.setEnabled(True)
        self.cancelPredictBtn.setEnabled(False)
        self._set_predict_progress(100, "完成")
        self._log(f"predict 返回: {result}")
        mask_path = result.get("mask_path")
        self.last_mask_path = mask_path
        if result.get("log_path"):
            self._log(f"后端日志: {result.get('log_path')}")
        if result.get("work_dir"):
            self._log(f"工作目录: {result.get('work_dir')}")
        if result.get("elapsed_sec"):
            self._log(f"后端耗时: {result.get('elapsed_sec')} 秒")
        if not mask_path or not os.path.exists(mask_path):
            self._set_status("分割结束，但结果文件不存在。", _DANGER)
            self.maskLabel.setText("分割失败：结果文件不存在")
            return
        try:
            self._set_status("正在把分割结果加载到 Slicer。", _OK)
            label_node = slicer.util.loadLabelVolume(mask_path)
            label_node.SetName("AI_Seg_Result")
            self.mask_node = label_node
            self._create_label_collection_from_labelmap(
                label_node, f"模型结果-{time.strftime('%H%M%S')}")
            self.maskLabel.setText(f"AI 分割结果已载入: {os.path.basename(mask_path)}")
            self._set_status("AI 分割完成，结果已加载。", _OK)
            self._log(f"结果已加载: {mask_path}")
        except Exception as e:
            self._set_status("分割文件加载失败。", _DANGER)
            self.maskLabel.setText(f"加载 mask 失败: {e}")
            self._log(f"加载 mask 异常: {e}")

    def _on_predict_error(self, error):
        self.predictTimer.stop()
        self._predict_started_at = None
        self._progress_polling = False
        self.predictBtn.setEnabled(True)
        self.cancelPredictBtn.setEnabled(False)
        if error.error_code == "PREDICTION_CANCELLED":
            self._set_predict_progress(0, "已中止")
            details = error.details or {}
            keep_reuse = bool(details.get("keep_reuse"))
            if keep_reuse:
                reuse = details.get("reuse_status") or {}
                message = reuse.get("message") or "已保留可复用中间文件。"
                self.reuseInfoLabel.setText(message)
                self._set_status("分割已中止，复用包已保留并完成检测。", _WARN)
                self._log(f"分割已中止；复用包检测: {reuse or details}")
            else:
                delete_result = details.get("delete_result") or {}
                message = delete_result.get("message") or "已按当前设置删除复用包。"
                self.reuseInfoLabel.setText(message)
                self._set_status("分割已中止，复用包已按设置删除。", _WARN)
                self._log(f"分割已中止；复用包删除: {delete_result or details}")
            self.maskLabel.setText("分割已中止，未生成新的分割结果。")
            return
        self._set_predict_progress(0, "失败")
        self._set_status(f"分割失败: [{error.error_code}]", _DANGER)
        self.maskLabel.setText(f"分割失败：{error.message}")
        self._log(f"分割失败: {error.error_code} | {error.message}")

    # ---------- correction ----------

    def on_open_editor(self):
        source = getattr(self, "mask_node", None)
        if source is None:
            self._set_status("请先完成 AI 分割。", _WARN)
            self._log("尚无 AI 分割结果，无法打开编辑器")
            return
        try:
            seg = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode", "AI_Seg_Edit")
            seg.CreateDefaultDisplayNodes()
            seg.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(source, seg)
            self.correct_seg_node = seg
            self._select_segment_editor_nodes(seg)
        except Exception as e:
            self._set_status(f"导入编辑器失败: {e}", _DANGER)
            self._log(f"导入编辑器异常: {e}")
            return
        slicer.util.resetSliceViews()
        self.correctLabel.setText("已打开 Segment Editor，可使用下方工具进行人工修正。")
        self._set_status("人工修正编辑器已打开。", _OK)
        self._log("已打开 Segment Editor，并绑定 AI_Seg_Edit")

    def _select_segment_editor_nodes(self, seg):
        slicer.util.mainWindow().moduleSelector().selectModule("SegmentEditor")
        widget = slicer.modules.segmenteditor.widgetRepresentation().self()
        editor = getattr(widget, "editor", widget)
        if hasattr(editor, "setSegmentationNode"):
            editor.setSegmentationNode(seg)
        if self.volume_node is not None and hasattr(editor, "setSourceVolumeNode"):
            editor.setSourceVolumeNode(self.volume_node)
        return editor

    def on_activate_segment_effect(self, effect_name):
        seg = getattr(self, "correct_seg_node", None)
        if seg is None:
            self.on_open_editor()
            seg = getattr(self, "correct_seg_node", None)
            if seg is None:
                return
        try:
            editor = self._select_segment_editor_nodes(seg)
            if hasattr(editor, "setActiveEffectByName"):
                editor.setActiveEffectByName(effect_name)
            self.correctLabel.setText(f"已切换工具：{effect_name}")
            self._set_status(f"已启用人工修正工具：{effect_name}", _OK)
            self._log(f"Segment Editor 工具: {effect_name}")
        except Exception as e:
            self._set_status(f"切换工具失败: {e}", _DANGER)
            self._log(f"切换 Segment Editor 工具失败: {e}")

    def on_read_result(self):
        seg = getattr(self, "correct_seg_node", None)
        if seg is None:
            self._set_status("请先打开人工修正编辑器。", _WARN)
            return
        try:
            result = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLabelMapVolumeNode", "AI_Seg_Corrected")
            logic = slicer.modules.segmentations.logic()
            try:
                logic.ExportAllSegmentsToLabelmapNode(seg, result, self.volume_node)
            except TypeError:
                logic.ExportAllSegmentsToLabelmapNode(seg, result)
        except Exception as e:
            self._set_status(f"读取修正结果失败: {e}", _DANGER)
            self._log(f"读取修正结果异常: {e}")
            return
        self.corrected_node = result
        self._create_label_collection_from_labelmap(
            result, f"修正结果-{time.strftime('%H%M%S')}")
        self.correctLabel.setText("修正结果已导出为 AI_Seg_Corrected，可用于质检/导出。")
        self._set_status("人工修正结果已读取。", _OK)
        self._log("人工修正结果已导出为 AI_Seg_Corrected")

    # ---------- labels ----------

    def _create_label_collection_from_labelmap(self, label_node, name=None):
        try:
            import numpy as np
            arr = slicer.util.arrayFromVolume(label_node)
            values = [int(v) for v in np.unique(arr) if int(v) > 0]
        except Exception as e:
            self._set_status(f"读取标签值失败: {e}", _DANGER)
            self._log(f"读取 LabelMap 标签值失败: {e}")
            return
        labels = [self._label_row_from_value(v) for v in values]
        if not name:
            name = f"标签集合-{time.strftime('%H%M%S')}"
        self._set_label_collection(name, labels)
        self._log(f"已从 LabelMap 生成标签集合: {name}, labels={values}")

    def _create_label_collection_from_segmentation(self, seg_node, name=None):
        labels = []
        try:
            segmentation = seg_node.GetSegmentation()
            for i in range(segmentation.GetNumberOfSegments()):
                segment_id = segmentation.GetNthSegmentID(i)
                segment = segmentation.GetSegment(segment_id)
                color = [int(c * 255) for c in segment.GetColor()]
                labels.append({
                    "value": i + 1,
                    "name": segment.GetName(),
                    "color": color,
                    "source": "segmentation",
                })
        except Exception as e:
            self._set_status(f"读取 Segmentation 失败: {e}", _DANGER)
            self._log(f"读取 Segmentation 标签失败: {e}")
            return
        if not name:
            name = f"标签集合-{time.strftime('%H%M%S')}"
        self._set_label_collection(name, labels)
        self._log(f"已从 Segmentation 生成标签集合: {name}, count={len(labels)}")

    def _label_row_from_value(self, value):
        for rule in self.label_rules:
            if int(rule.get("id", -1)) == int(value):
                return {
                    "value": int(value),
                    "name": rule.get("name", f"Label_{value:03d}"),
                    "color": rule.get("color", [180, 180, 180]),
                    "source": "template_id",
                }
        if 11 <= int(value) <= 48:
            return {
                "value": int(value),
                "name": f"FDI_{int(value)}_Tooth",
                "color": [120, 180, 230],
                "source": "fdi",
            }
        return {
            "value": int(value),
            "name": f"Label_{int(value):03d}",
            "color": [180, 180, 180],
            "source": "raw",
        }

    def _set_label_collection(self, name, labels):
        base_name = name
        suffix = 2
        while name in self.label_collections:
            name = f"{base_name}-{suffix}"
            suffix += 1
        self.label_collections[name] = labels
        self.active_label_collection = name
        self._refresh_label_collection_menu()
        self._select_label_collection(name)

    def _refresh_label_collection_menu(self):
        if not hasattr(self, "labelSetMenu"):
            return
        self.labelSetMenu.clear()
        if not self.label_collections:
            self.labelSetButton.setText("暂无标签集合")
            empty_action = self.labelSetMenu.addAction("暂无标签集合")
            empty_action.setEnabled(False)
        else:
            for name in self.label_collections.keys():
                self._add_label_collection_menu_row(name)
        self.labelSetMenu.addSeparator()
        new_action = self.labelSetMenu.addAction("新建空集合")
        new_action.triggered.connect(self.on_create_label_collection)

    def _add_label_collection_menu_row(self, name):
        try:
            row_widget = qt.QWidget()
            row = qt.QHBoxLayout(row_widget)
            row.setContentsMargins(4, 2, 4, 2)
            row.setSpacing(4)

            name_button = qt.QPushButton(name)
            name_button.setFlat(True)
            name_button.setMinimumWidth(160)
            name_button.clicked.connect(
                lambda checked=False, n=name: self._select_label_collection(n))

            rename_button = qt.QPushButton("重命名")
            rename_button.setFixedWidth(56)
            rename_button.clicked.connect(
                lambda checked=False, n=name: self.on_rename_label_collection(n))

            delete_button = qt.QPushButton("删除")
            delete_button.setFixedWidth(44)
            delete_button.clicked.connect(
                lambda checked=False, n=name: self.on_delete_label_collection(n))

            row.addWidget(name_button, 1)
            row.addWidget(rename_button)
            row.addWidget(delete_button)

            action = qt.QWidgetAction(self.labelSetMenu)
            action.setDefaultWidget(row_widget)
            self.labelSetMenu.addAction(action)
        except Exception:
            action = self.labelSetMenu.addAction(name)
            action.triggered.connect(
                lambda checked=False, n=name: self._select_label_collection(n))

    def _select_label_collection(self, name):
        self.active_label_collection = name or None
        if name:
            self.labelSetButton.setText(name)
        else:
            self.labelSetButton.setText("暂无标签集合")
        self._populate_label_table(self.label_collections.get(name, []))
        try:
            self.labelSetMenu.hide()
        except Exception:
            pass

    def on_create_label_collection(self):
        if getattr(self, "mask_node", None) is not None:
            self._create_label_collection_from_labelmap(
                self.mask_node, f"当前分割-{time.strftime('%H%M%S')}")
            return
        if getattr(self, "correct_seg_node", None) is not None:
            self._create_label_collection_from_segmentation(
                self.correct_seg_node, f"当前修正-{time.strftime('%H%M%S')}")
            return
        self._set_label_collection(f"空集合-{time.strftime('%H%M%S')}", [])
        self._set_status("已创建空标签集合，等待后续分割实例。", _OK)

    def on_rename_label_collection(self, name=None):
        name = name or self.active_label_collection
        if not name:
            self._set_status("当前没有可重命名的标签集合。", _WARN)
            return
        try:
            result = qt.QInputDialog.getText(
                self.parent,
                "重命名标签集合",
                "新的集合名称",
                qt.QLineEdit.Normal,
                name,
            )
            if isinstance(result, tuple):
                new_name, ok = result
            else:
                new_name, ok = result, bool(result)
        except Exception as e:
            self._set_status(f"打开重命名窗口失败: {e}", _DANGER)
            self._log(f"标签集合重命名窗口异常: {e}")
            return
        if not ok:
            return
        new_name = str(new_name or "").strip()
        if not new_name:
            self._set_status("标签集合名称不能为空。", _WARN)
            return
        if new_name != name and new_name in self.label_collections:
            self._set_status("已有同名标签集合，请换一个名称。", _WARN)
            return
        self.label_collections[new_name] = self.label_collections.pop(name, [])
        if self.active_label_collection == name:
            self.active_label_collection = new_name
        self._refresh_label_collection_menu()
        self._select_label_collection(new_name)
        self._set_status(f"已重命名标签集合：{name} -> {new_name}", _OK)
        self._log(f"已重命名标签集合: {name} -> {new_name}")

    def on_delete_label_collection(self, name=None):
        name = name or self.active_label_collection
        if not name:
            self._set_status("当前没有可删除的标签集合。", _WARN)
            return
        self.label_collections.pop(name, None)
        next_name = next(iter(self.label_collections.keys()), "")
        self.active_label_collection = next_name or None
        self._refresh_label_collection_menu()
        if next_name:
            self._select_label_collection(next_name)
        else:
            self._select_label_collection(None)
        self._populate_label_table(self.label_collections.get(next_name, []))
        self._set_status(f"已删除标签集合：{name}", _OK)
        self._log(f"已删除标签集合: {name}")

    def on_label_collection_changed(self, *args):
        self._select_label_collection(self.active_label_collection)

    def _populate_label_table(self, labels):
        self.labelTable.setRowCount(0)
        if not labels:
            self.labelSummaryLabel.setText("当前标签集合为空。导入或生成分割结果后才会出现标签。")
            return
        for row, label in enumerate(labels):
            self.labelTable.insertRow(row)
            check = qt.QCheckBox()
            check.setChecked(True)
            self.labelTable.setCellWidget(row, 0, check)

            value = label.get("value")
            name = label.get("name") or f"Label_{int(value):03d}"
            if value is not None:
                name = f"{int(value):03d}  {name}"
            self.labelTable.setItem(row, 1, qt.QTableWidgetItem(name))

            color = label.get("color") or [180, 180, 180]
            swatch = qt.QLabel(" ")
            swatch.setFixedSize(34, 14)
            swatch.setStyleSheet(
                "QLabel {"
                f"background: rgb({int(color[0])}, {int(color[1])}, {int(color[2])});"
                f"border: 1px solid {_BORDER}; border-radius: 3px;"
                "}"
            )
            self.labelTable.setCellWidget(row, 2, swatch)

            state = qt.QComboBox()
            state.addItems(["待审核", "已确认", "需修正", "忽略"])
            self.labelTable.setCellWidget(row, 3, state)
        collection = self.active_label_collection or "当前集合"
        self.labelSummaryLabel.setText(
            f"{collection}：共 {len(labels)} 个实际标签，可记录显示与审核状态。")

    # ---------- quality and export ----------

    def _resolve_label_path(self):
        corrected = getattr(self, "corrected_node", None)
        if corrected is not None:
            out_dir = os.path.join(self._project_root(), "data", "outputs", "labels")
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, "corrected_label.nii.gz")
            try:
                slicer.util.exportNodeToFile(corrected, path)
                return path, True
            except Exception as e:
                self._log(f"导出修正节点失败，回退 AI 结果: {e}")
        return getattr(self, "last_mask_path", None), False

    def on_quality_check(self):
        if self.api is None:
            self._set_status("服务不可用，无法质检。", _DANGER)
            return
        label_path, _ = self._resolve_label_path()
        if not label_path or not os.path.exists(label_path):
            self._set_status("没有可质检的标签。", _WARN)
            return
        self.qualityBtn.setEnabled(False)
        self._log(f"质检标签: {label_path}")
        try:
            result = self.api.check_label(
                self.case_id or "case-manual",
                label_path,
                _combo_text(self.templateCombo) or _DEFAULT_TEMPLATE_ID,
            )
        except ApiError as e:
            self.qualityBtn.setEnabled(True)
            self._set_status(f"质检失败: [{e.error_code}]", _DANGER)
            self._log(f"质检失败: {e.error_code} | {e.message}")
            return
        self.qualityBtn.setEnabled(True)
        issues = result.get("issues", [])
        passed = result.get("passed", len(issues) == 0)
        self.qualityLabel.setText("质检通过" if passed else f"发现问题 {len(issues)} 项")
        self.qualityLabel.setStyleSheet(f"color:{_OK if passed else _DANGER};")
        self._set_status("质检完成。", _OK if passed else _WARN)
        for issue in issues[:20]:
            self._log(f"质检: {issue}")

    def on_export(self):
        if self.api is None:
            self._set_status("服务不可用，无法导出。", _DANGER)
            return
        if not self.image_path:
            self._set_status("没有可导出的原始影像路径。", _WARN)
            return
        label_path, _ = self._resolve_label_path()
        if not label_path or not os.path.exists(label_path):
            self._set_status("没有可导出的标签。", _WARN)
            return
        self.exportBtn.setEnabled(False)
        self._log(f"导出: image={self.image_path}, label={label_path}")
        try:
            result = self.api.export(
                self.case_id or "case-manual",
                self.image_path,
                label_path,
                "nnunet",
                True,
            )
        except ApiError as e:
            self.exportBtn.setEnabled(True)
            self._set_status(f"导出失败: [{e.error_code}]", _DANGER)
            self._log(f"导出失败: {e.error_code} | {e.message}")
            return
        self.exportBtn.setEnabled(True)
        export_dir = result.get("export_dir", "?")
        files = result.get("files", [])
        self.exportLabel.setText(f"已导出 {len(files)} 个文件 -> {export_dir}")
        self.exportLabel.setStyleSheet(f"color:{_OK};")
        self._set_status("导出完成。", _OK)
        for file_path in files:
            self._log(f"已生成: {file_path}")

    def _report_event(self, event, payload=None):
        if self.api is None or not self.case_id:
            return
        try:
            self.api.log(self.case_id, event, "operator-b", payload or {})
        except Exception as e:
            import logging
            logging.getLogger("CBCTAnnotator").warning(
                f"agent/log {event} 上报失败: {e}")
