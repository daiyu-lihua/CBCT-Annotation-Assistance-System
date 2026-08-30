"""牙科 CBCT 交互式实例级辅助标注系统 —— 3D Slicer 前端插件（B 组）。

当前阶段：插件骨架 + 服务连接层 + 数据导入。
已实现：服务连接(/status)、配置加载(/config)、CBCT 影像图形化导入并显示 + 信息卡。
已实现美化：现代风格卡片式 UI，面向普通用户的图形交互。

参考：docs/接口与开发/前端插件任务清单与接口方案.md
"""

import os
import sys

# 保证能 import lib 子目录下的 ApiClient（放子目录避免被 Slicer 误当脚本模块扫描）
_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = os.path.join(_HERE, "lib")
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)

import qt  # noqa: E402
import slicer  # noqa: E402
from slicer.ScriptedLoadableModule import (  # noqa: E402
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
)

from ApiClient import ApiClient, ApiError  # noqa: E402

# ---------- 配色与样式 ----------
_BG = "#eef2f7"
_CARD = "#ffffff"
_BORDER = "#e5eaf1"
_PRIMARY = "#2a6fa9"
_PRIMARY_DARK = "#1f5886"
_ACCENT = "#16a2a8"
_TEXT = "#24303f"
_TEXT_DIM = "#7a8699"
_DANGER = "#d64545"
_OK = "#2e9e6b"

_PANEL_QSS = f"""
#cbctPanel {{ background-color: {_BG}; }}

#tbTitle {{ font-size: 16px; font-weight: bold; color: {_PRIMARY_DARK}; }}

#card {{
    background-color: {_CARD};
    border: 1px solid {_BORDER};
    border-radius: 12px;
}}
#cardTitle {{
    font-size: 13px; font-weight: bold; color: {_PRIMARY_DARK};
    padding: 2px 0 6px 0;
}}
QLabel#hint {{ color: {_TEXT_DIM}; }}
QLabel#value {{ color: {_TEXT}; font-weight: bold; }}

QPushButton {{
    border-radius: 8px; padding: 6px 14px; font-size: 12px;
}}
QPushButton#primary {{
    background-color: {_PRIMARY}; color: #fff; border: none; font-weight: bold;
}}
QPushButton#primary:hover {{ background-color: {_PRIMARY_DARK}; }}
QPushButton#accent {{
    background-color: {_ACCENT}; color: #fff; border: none; font-weight: bold;
}}
QPushButton#accent:hover {{ background-color: {_PRIMARY_DARK}; }}
QPushButton#danger {{
    background-color: {_DANGER}; color: #fff; border: none; font-weight: bold;
}}
QPushButton#danger:hover {{ background-color: #b23a3a; }}
QPushButton#ghost {{
    background-color: {_CARD}; color: {_PRIMARY}; border: 1px solid {_PRIMARY};
}}
QPushButton#ghost:hover {{ background-color: #eaf1f8; }}
QPushButton:disabled {{ background-color: #dfe5ec; color: #98a2b3; border: none; }}

QLineEdit, QPlainTextEdit, QComboBox {{
    background: {_CARD}; border: 1px solid {_BORDER}; border-radius: 8px;
    padding: 5px 8px; color: {_TEXT};
}}
QLineEdit:focus, QComboBox:focus {{ border: 1px solid {_PRIMARY}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QGroupBox {{ border: none; }}

QProgressBar {{
    background: {_BORDER}; border: none; border-radius: 4px;
    text-align: center; color: transparent;
}}
QProgressBar::chunk {{ background: {_PRIMARY}; border-radius: 4px; }}
"""


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


def _make_card(title):
    """构造一个白色卡片，返回 (card, bodyLayout)。"""
    card = qt.QFrame()
    card.setObjectName("card")
    v = qt.QVBoxLayout(card)
    v.setContentsMargins(14, 12, 14, 14)
    v.setSpacing(8)
    t = qt.QLabel(title)
    t.setObjectName("cardTitle")
    v.addWidget(t)
    return card, v


def _collect_mode_ids(cfg):
    """从 config['modes'] 提取模式 id。兼容数组[{id,name}]或 dict{id:name} 两种结构。"""
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


class CBCTAnnotatorWidget(ScriptedLoadableModuleWidget):
    """5.2. 模块主面板。"""

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.api = None          # 连接测试成功后才创建 ApiClient
        self.image_path = None   # 当前选中的 CBCT 路径
        self.volume_node = None  # 已加载到 Slicer 的关键点 node
        self.roi_node = None     # 当前 ROI 节点
        self.case_id = None      # 由服务端 /cases 返回（后续接入）

        self.parent.setStyleSheet(_PANEL_QSS)
        root = qt.QFrame()
        root.setObjectName("cbctPanel")
        self.parent.layout().addWidget(root)
        m = qt.QVBoxLayout(root)
        m.setContentsMargins(12, 12, 12, 12)
        m.setSpacing(10)

        # ---- 顶部标题 ----
        title = qt.QLabel("牙科 CBCT 辅助标注系统")
        title.setObjectName("tbTitle")
        hint = qt.QLabel("面向实例级牙齿分割的本地化人机协同标注工具")
        hint.setObjectName("hint")
        m.addWidget(title)
        m.addWidget(hint)
        m.addSpacing(2)

        m.addWidget(self._build_status_card())      # 状态 + 日志（置顶）
        m.addWidget(self._build_import_card())      # 数据导入
        m.addWidget(self._build_roi_card())         # ROI 选择
        m.addWidget(self._build_connect_card())     # 服务连接
        m.addWidget(self._build_config_card())      # 模型/配置
        m.addWidget(self._build_predict_card())     # AI 分割
        m.addWidget(self._build_correct_card())     # 人工修正
        m.addWidget(self._build_quality_card())     # 标签质检
        m.addWidget(self._build_export_card())      # 结果导出

        m.addStretch(1)

    # ---------- 各卡片 ----------

    def _build_import_card(self):
        card, v = _make_card("数据导入")
        line = qt.QHBoxLayout()
        self.pathEdit = qt.QLineEdit()
        self.pathEdit.setPlaceholderText("选择牙科 CBCT 影像（.nii / .nii.gz / .nrrd）")
        self.pathEdit.setEnabled(False)
        self.chooseBtn = qt.QPushButton("选择影像")
        self.chooseBtn.setObjectName("primary")
        self.loadBtn = qt.QPushButton("加载并显示")
        self.loadBtn.setObjectName("accent")
        self.loadBtn.setEnabled(False)
        self.chooseBtn.clicked.connect(self.on_choose_image)
        self.loadBtn.clicked.connect(self.on_load_image)
        line.addWidget(self.pathEdit, 1)
        line.addWidget(self.chooseBtn)
        line.addWidget(self.loadBtn)
        v.addLayout(line)

        # 信息卡（加载后填充）
        self.infoLabel = qt.QLabel("尚未加载影像")
        self.infoLabel.setObjectName("hint")
        v.addWidget(self.infoLabel)
        return card

    def _build_roi_card(self):
        card, v = _make_card("ROI 选择")
        hint = qt.QLabel("新建 ROI 后，在切片视图拖拽框体框住要分割的牙弓区域；再点\"读取 ROI\"。")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        v.addWidget(hint)

        btns = qt.QHBoxLayout()
        self.newRoiBtn = qt.QPushButton("新建 ROI 框")
        self.newRoiBtn.setObjectName("primary")
        self.readRoiBtn = qt.QPushButton("读取 ROI 坐标")
        self.readRoiBtn.setObjectName("accent")
        self.delRoiBtn = qt.QPushButton("删除 ROI 框")
        self.delRoiBtn.setObjectName("danger")
        self.newRoiBtn.clicked.connect(self.on_new_roi)
        self.readRoiBtn.clicked.connect(self.on_read_roi)
        self.delRoiBtn.clicked.connect(self.on_delete_roi)
        btns.addWidget(self.newRoiBtn)
        btns.addWidget(self.readRoiBtn)
        btns.addWidget(self.delRoiBtn)
        v.addLayout(btns)

        self.roiInfoLabel = qt.QLabel("尚未框选 ROI")
        self.roiInfoLabel.setObjectName("hint")
        v.addWidget(self.roiInfoLabel)
        return card

    def _build_connect_card(self):
        card, v = _make_card("服务连接")
        row = qt.QHBoxLayout()
        row.addWidget(qt.QLabel("服务地址"))
        self.addressEdit = qt.QLineEdit("http://127.0.0.1:8000/api/v1")
        row.addWidget(self.addressEdit, 1)
        v.addLayout(row)

        btns = qt.QHBoxLayout()
        self.connectBtn = qt.QPushButton("连接测试")
        self.connectBtn.setObjectName("primary")
        self.connectBtn.clicked.connect(self.on_connect)
        btns.addWidget(self.connectBtn)
        self.connStatusLabel = qt.QLabel("● 未连接")
        self.connStatusLabel.setStyleSheet(
            f"color:{_TEXT_DIM}; font-weight:bold;")
        btns.addStretch(1)
        btns.addWidget(self.connStatusLabel)
        v.addLayout(btns)
        return card

    def _build_config_card(self):
        card, v = _make_card("模型与配置")
        form = qt.QFormLayout()
        self.modelCombo = qt.QComboBox()
        self.modeCombo = qt.QComboBox()
        self.templateCombo = qt.QComboBox()
        self.modeCombo.addItems(["fast", "balanced", "fine"])
        self.modeCombo.setCurrentText("balanced")
        form.addRow("模型", self.modelCombo)
        form.addRow("推理模式", self.modeCombo)
        form.addRow("标签模板", self.templateCombo)
        v.addLayout(form)

        self.loadConfigBtn = qt.QPushButton("加载可用模型与模板")
        self.loadConfigBtn.setObjectName("ghost")
        self.loadConfigBtn.clicked.connect(self.on_load_config)
        v.addWidget(self.loadConfigBtn)
        return card

    def _build_predict_card(self):
        card, v = _make_card("AI 分割")
        hint = qt.QLabel("读取 ROI 坐标后，可先让 Agent 推荐推理模式，"
                         "再点击\"开始分割\"进行 AI 初分割。")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        v.addWidget(hint)
        self.recommendBtn = qt.QPushButton("智能推荐推理模式")
        self.recommendBtn.setObjectName("ghost")
        self.recommendBtn.clicked.connect(self.on_recommend_mode)
        v.addWidget(self.recommendBtn)
        self.recommendLabel = qt.QLabel("尚未推荐")
        self.recommendLabel.setObjectName("hint")
        v.addWidget(self.recommendLabel)
        self.predictBtn = qt.QPushButton("开始分割")
        self.predictBtn.setObjectName("primary")
        self.predictBtn.clicked.connect(self.on_predict)
        v.addWidget(self.predictBtn)
        self.maskLabel = qt.QLabel("尚未分割")
        self.maskLabel.setObjectName("hint")
        v.addWidget(self.maskLabel)
        return card

    def _build_correct_card(self):
        card, v = _make_card("人工修正")
        hint = qt.QLabel("\"打开编辑器\"将 AI 结果转成可编辑的分割；"
                         "用 Slicer 画笔工具修正牙齿边界后，点\"读取修正结果\"。")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        v.addWidget(hint)

        btns = qt.QHBoxLayout()
        self.openEditorBtn = qt.QPushButton("打开编辑器")
        self.openEditorBtn.setObjectName("primary")
        self.readCorrectBtn = qt.QPushButton("读取修正结果")
        self.readCorrectBtn.setObjectName("accent")
        self.openEditorBtn.clicked.connect(self.on_open_editor)
        self.readCorrectBtn.clicked.connect(self.on_read_result)
        btns.addWidget(self.openEditorBtn)
        btns.addWidget(self.readCorrectBtn)
        v.addLayout(btns)

        self.correctLabel = qt.QLabel("尚未修正")
        self.correctLabel.setObjectName("hint")
        v.addWidget(self.correctLabel)
        return card

    def _build_quality_card(self):
        card, v = _make_card("标签质检")
        hint = qt.QLabel("对当前结果（AI 分割或人工修正）做质量检查，"
                         "检查空标签、重复编号、连通域等。")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        v.addWidget(hint)
        self.qualityBtn = qt.QPushButton("开始质检")
        self.qualityBtn.setObjectName("primary")
        self.qualityBtn.clicked.connect(self.on_quality_check)
        v.addWidget(self.qualityBtn)
        self.qualityLabel = qt.QLabel("尚未质检")
        self.qualityLabel.setObjectName("hint")
        v.addWidget(self.qualityLabel)
        return card

    def _build_export_card(self):
        card, v = _make_card("结果导出")
        hint = qt.QLabel("把当前标注结果导出为可供模型训练使用的数据包。")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        v.addWidget(hint)
        self.exportBtn = qt.QPushButton("导出训练数据")
        self.exportBtn.setObjectName("primary")
        self.exportBtn.clicked.connect(self.on_export)
        v.addWidget(self.exportBtn)
        self.exportLabel = qt.QLabel("尚未导出")
        self.exportLabel.setObjectName("hint")
        v.addWidget(self.exportLabel)
        return card

    def _build_status_card(self):
        card, v = _make_card("运行状态")
        self.statusLabel = qt.QLabel("尚未连接服务端")
        self.statusLabel.setObjectName("hint")
        self.statusLabel.setWordWrap(True)
        v.addWidget(self.statusLabel)
        self.progressBar = qt.QProgressBar()
        self.progressBar.setRange(0, 0)   # busy 动画模式
        self.progressBar.setVisible(False)
        self.progressBar.setMaximumHeight(8)
        v.addWidget(self.progressBar)

        logHeader = qt.QHBoxLayout()
        logTitle = qt.QLabel("日志")
        logTitle.setObjectName("cardTitle")
        self.clearLogBtn = qt.QPushButton("清空日志")
        self.clearLogBtn.setObjectName("ghost")
        self.clearLogBtn.clicked.connect(self._on_clear_log)
        logHeader.addWidget(logTitle)
        logHeader.addStretch(1)
        logHeader.addWidget(self.clearLogBtn)
        v.addLayout(logHeader)

        self.logEdit = qt.QPlainTextEdit()
        self.logEdit.setReadOnly(True)
        self.logEdit.setMaximumHeight(120)
        v.addWidget(self.logEdit)
        return card

    # ---------- 工具 ----------

    def _script_dir(self):
        return os.path.dirname(os.path.abspath(__file__))

    def _log(self, msg):
        self.logEdit.appendPlainText(msg)
        sb = self.logEdit.verticalScrollBar()
        sb.setValue(sb.maximum)
        import logging
        logging.getLogger("CBCTAnnotator").info(msg)

    def _on_clear_log(self):
        self.logEdit.clear()

    def _set_status(self, text, color=None):
        if color == _DANGER:
            text = "⚠ " + text
        self.statusLabel.setText(text)
        self.statusLabel.setStyleSheet(
            f"color: {color};" if color else f"QLabel{{color:{_TEXT_DIM};}}"
        )

    def _set_busy(self, busy, text=None):
        """显示/隐藏忙碌进度条；busy=True 时可同时更新状态文字。"""
        if busy:
            if text:
                self.statusLabel.setText(text)
            self.progressBar.setVisible(True)
        else:
            self.progressBar.setVisible(False)

    # ---------- 槽函数：ROI ----------

    def on_new_roi(self):
        """在 Slicer 中新建一个 ROI 框，便于框选分割区域。"""
        if self.volume_node is None:
            self._set_status("请先加载影像", None)
            self._log("尚未加载影像，无法创建 ROI")
            return
        try:
            # 清除已存在的旧 ROI，避免堆叠
            if self.roi_node is not None and self.roi_node.IsA("vtkMRMLMarkupsROINode"):
                slicer.mrmlScene.RemoveNode(self.roi_node)
            roi = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsROINode", "CBCT_ROI")

            bounds = [0]*6
            self.volume_node.GetBounds(bounds)
            center = [0.5*(bounds[0]+bounds[1]),
                      0.5*(bounds[2]+bounds[3]),
                      0.5*(bounds[4]+bounds[5])]
            size = [bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]]
            roi.SetCenter(center)
            roi.SetSize(size)

            # 确保创建显示节点并强制可见
            roi.CreateDefaultDisplayNodes()
            dn = roi.GetDisplayNode()
            if dn is None:
                roi.CreateDefaultDisplayNodes()
                dn = roi.GetDisplayNode()
            dn.SetVisibility(True)
            dn.SetSelected(True)

            self.roi_node = roi
            slicer.util.resetSliceViews()
            self._set_status(
                "ROI 框已创建。在 3D 视图或切片视图中拖拽/调整框体框住牙弓，"
                "然后点\"读取 ROI\"")
            self._log("已新建 ROI 框，请框住目标牙弓区域")
        except Exception as e:
            import logging
            logging.getLogger("CBCTAnnotator").error(f"on_new_roi: {e}")
            self._set_status(f"创建 ROI 失败: {e}", "#d64545")
            self._log(f"创建 ROI 异常: {e}")

    def on_delete_roi(self):
        """删除当前 ROI 框并清空已读取的 ROI 坐标。"""
        if self.roi_node is not None and self.roi_node.IsA("vtkMRMLMarkupsROINode"):
            slicer.mrmlScene.RemoveNode(self.roi_node)
            self.roi_node = None
        self.roi_ijk_start = None
        self.roi_ijk_size = None
        self.roiInfoLabel.setText("尚未框选 ROI")
        self._set_status("ROI 框已删除", None)
        self._log("已删除 ROI 框，坐标已清空")

    def on_read_roi(self):
        """读取 ROI 的 IJK 起点与尺寸，显示并暂存供后续接口使用。"""
        if self.roi_node is None or self.volume_node is None:
            self._set_status("请先新建 ROI 框", None)
            self._log("尚无 ROI，请先点击\"新建 ROI 框\"")
            return
        start, size = self._read_roi_ijk()
        if start is None:
            self._set_status("ROI 坐标读取失败", "#d64545")
            return
        self.roi_ijk_start = start
        self.roi_ijk_size = size
        self.roiInfoLabel.setStyleSheet(f"QLabel{{color:{_TEXT};}}")
        self.roiInfoLabel.setText(
            f"ROI → 起点 ({start[0]}, {start[1]}, {start[2]})  ·  尺寸 "
            f"({size[0]}, {size[1]}, {size[2]}) 体素"
        )
        self._set_status("ROI 已读取，坐标格式为 IJK（供后端 /predict 使用）", _OK)
        self._log(f"ROI IJK: start={start}, size={size}")

    def _read_roi_ijk(self):
        """把 ROI 框转换为体素坐标 start/size（RAS→IJK）。"""
        import vtk
        center = list(self.roi_node.GetCenter())
        size = list(self.roi_node.GetSize())
        rasToIjk = vtk.vtkMatrix4x4()
        self.volume_node.GetRASToIJKMatrix(rasToIjk)
        ijk = list(rasToIjk.MultiplyPoint(
            [center[0], center[1], center[2], 1.0]))
        spacing = self.volume_node.GetSpacing()
        ijkSize = [max(1, int(round(size[k] / spacing[k]))) for k in range(3)]
        start = [int(round(ijk[0])), int(round(ijk[1])), int(round(ijk[2]))]
        return start, ijkSize

    # ---------- 槽函数：数据导入 ----------

    def on_choose_image(self):
        """图形化选择 CBCT 文件。"""
        f = qt.QFileDialog.getOpenFileName(
            self.parent, "选择牙科 CBCT 影像",
            "", "医学影像 (*.nii *.nii.gz *.nrrd *.dcm);;所有文件 (*)",
        )
        if not f:
            return
        path = f if isinstance(f, str) else f[0]
        self.image_path = path
        self.pathEdit.setText(path)
        self.loadBtn.setEnabled(True)
        self._log(f"已选择影像: {path}")
        self._set_status("已选择影像，点击\"加载并显示\"")

    def on_load_image(self):
        """加载影像到 Slicer 并显示，读取信息。"""
        if not self.image_path:
            return
        try:
            name = os.path.splitext(os.path.basename(self.image_path))[0]
            node = slicer.util.loadVolume(self.image_path, {
                "name": name, "autoWindowLevel": True,
            })
        except Exception as e:
            self._set_status(f"加载失败: {e}", _DANGER)
            self._log(f"影像加载失败: {e}")
            return

        self.volume_node = node
        dims = node.GetImageData().GetDimensions()
        spacing = node.GetSpacing()
        size_mb = os.path.getsize(self.image_path) / 1024.0 / 1024.0

        self.infoLabel.setStyleSheet(f"QLabel{{color:{_TEXT};}}")
        self.infoLabel.setText(
            f"尺寸 {dims[0]} × {dims[1]} × {dims[2]}  ·  间距 "
            f"{spacing[0]:.3g} / {spacing[1]:.3g} / {spacing[2]:.3g} mm  ·  "
            f"{size_mb:.0f} MB"
        )
        slicer.util.resetSliceViews()
        self._set_status(
            f"已加载 {os.path.basename(self.image_path)}，可在切片视图查看",
            _OK,
        )
        self._log(f"影像加载成功: dims={dims}, spacing={spacing}")

    # ---------- 槽函数：服务 ----------

    def on_connect(self):
        url = self.addressEdit.text.strip()
        try:
            self.api = ApiClient(url)
            st = self.api.status()
        except ApiError as e:
            self.api = None
            self.connStatusLabel.setText("● 连接失败")
            self.connStatusLabel.setStyleSheet(
                f"color:{_DANGER}; font-weight:bold;")
            self._set_status(f"连接失败: [{e.error_code}]", _DANGER)
            self._log(f"连接失败: {e.error_code} | {e.message}")
            return

        model_state = st["model"].get("loaded", False)
        device = st["device"].get("data") if "device" in st else "unknown"
        self.connStatusLabel.setText("● 已连接")
        self.connStatusLabel.setStyleSheet(
            f"color:{_OK}; font-weight:bold;")
        self._set_status(
            f"已连接 · 设备 {device} · 模型 {'已加载' if model_state else '未加载'}",
            _OK if model_state else None,
        )
        self._log(f"连接成功: service={st.get('service')}, device={device}")

    def on_load_config(self):
        if self.api is None:
            self._set_status("请先连接测试", None)
            self._log("尚未连接，请先点击连接测试")
            return
        try:
            cfg = self.api.config()
        except ApiError as e:
            self._set_status(f"配置加载失败: [{e.error_code}]", _DANGER)
            self._log(f"配置加载失败: {e.error_code} | {e.message}")
            return
        model_count = 0
        self.modelCombo.clear()
        for m in cfg.get("models", []):
            self.modelCombo.addItem(m.get("model_id", "?"))
            model_count += 1
        template_count = 0
        self.templateCombo.clear()
        for t in cfg.get("label_templates", []):
            self.templateCombo.addItem(t.get("template_id", "?"))
            template_count += 1
        mode_ids = _collect_mode_ids(cfg)
        if mode_ids:
            self.modeCombo.clear()
            self.modeCombo.addItems(mode_ids)
            if "balanced" in mode_ids:
                self.modeCombo.setCurrentText("balanced")
            self._log(f"可用推理模式: {', '.join(mode_ids)}")
        self._set_status(
            f"配置已加载: {model_count} 个模型, "
            f"{template_count} 个标签模板", _OK,
        )

    # ---------- 槽函数：AI 分割 ----------

    def _ensure_case_id(self):
        """确保已有 case_id，没有再调用 /cases 创建。"""
        if not self.case_id:
            case = self.api.create_case(
                self.image_path, "nii", "teeth-16", "operator-b")
            self.case_id = case.get("case_id")
        return self.case_id

    def on_recommend_mode(self):
        """让后端 Agent 按 ROI 大小推荐推理模式，并自动选中。"""
        if self.api is None:
            self._set_status("请先连接服务端", None)
            self._log("尚未连接服务端，请先连接测试")
            return
        if not getattr(self, "roi_ijk_start", None):
            self._set_status("请先读取 ROI 坐标", None)
            self._log("尚未读取 ROI 坐标，请先框选并读取 ROI")
            return
        roi = {"start": self.roi_ijk_start, "size": self.roi_ijk_size}
        target = self.templateCombo.currentText or "teeth"
        self._log(f"请求推荐模式: roi_size={roi['size']}, target={target}")
        try:
            self._ensure_case_id()
            result = self.api.recommend_mode(self.case_id, roi, target)
        except ApiError as e:
            self._set_status(f"推荐失败: [{e.error_code}]", "#d64545")
            self._log(f"推荐失败: {e.error_code} | {e.message}")
            return
        mode = result.get("mode")
        reason = result.get("reason", "")
        if mode and self.modeCombo.findText(mode) >= 0:
            self.modeCombo.setCurrentText(mode)
        self.recommendLabel.setStyleSheet(f"QLabel{{color:{_OK};}}")
        self.recommendLabel.setText(
            f"推荐 {mode} 模式：{reason}")
        self._set_status(f"Agent 已推荐 {mode} 模式", _OK)
        self._log(f"Agent 推荐 {mode}: {reason}")

    def on_predict(self):
        """校验状态后，同步调用 /predict 并加载返回的 mask。"""
        if self.api is None:
            self._set_status("请先连接服务端", None)
            self._log("尚未连接服务端，请先连接测试")
            return
        if self.volume_node is None or not self.image_path:
            self._set_status("请先加载影像", None)
            self._log("尚未加载影像，请先选择并加载 CBCT")
            return
        if not getattr(self, "roi_ijk_start", None):
            self._set_status("请先读取 ROI 坐标", None)
            self._log("尚未读取 ROI 坐标，请先框选并读取 ROI")
            return

        model_id = self.modelCombo.currentText
        mode = self.modeCombo.currentText
        roi = {"start": self.roi_ijk_start, "size": self.roi_ijk_size}
        self._log(f"提交分割: model={model_id}, mode={mode}, "
                  f"roi_start={roi['start']}")
        self.predictBtn.setEnabled(False)
        self._set_busy(True, "正在进行 AI 分割...")
        try:
            result = self._run_predict(roi, model_id, mode)
        except ApiError as e:
            self.predictBtn.setEnabled(True)
            self._set_busy(False)
            self._set_status(f"分割失败: [{e.error_code}]", _DANGER)
            self._log(f"分割失败: {e.error_code} | {e.message}")
            return
        self.predictBtn.setEnabled(True)
        self._set_busy(False)
        self._on_predict_ok(result)

    def _run_predict(self, roi, model_id, mode):
        """在子线程执行，负责补齐 case_id 并调用 predict 接口。"""
        case_id = self._ensure_case_id()
        return self.api.predict(
            case_id, self.image_path, roi, model_id, mode,
            targets=["teeth"], output_format="nii.gz")

    def _on_predict_ok(self, result):
        self.predictBtn.setEnabled(True)
        self._set_busy(False)
        self._log(f"predict 原始返回: {result}")
        mask_path = result.get("mask_path")
        self._log(f"分割完成: {mask_path}")
        self.last_mask_path = mask_path
        if not mask_path or not os.path.exists(mask_path):
            self._set_status("分割结果文件不存在或已丢失", _DANGER)
            self.maskLabel.setText("分割失败：结果文件不存在")
            self._log("mask 文件不存在，跳过加载")
            return
        try:
            labelNode = slicer.util.loadLabelVolume(mask_path)
            labelNode.SetName("AI_Seg_Result")
            self.mask_node = labelNode
            self.maskLabel.setStyleSheet(f"QLabel{{color:{_OK};}}")
            self.maskLabel.setText(
                f"AI 分割结果已载入: {os.path.basename(mask_path)}")
            self._set_status("AI 分割完成，结果已加载显示", _OK)
        except Exception as e:
            self.maskLabel.setText(f"加载 mask 失败: {e}")
            self._set_status("加载 mask 失败", "#d64545")
            self._log(f"加载 mask 异常: {e}")

    # ---------- 槽函数：人工修正 ----------

    def on_open_editor(self):
        """把 AI 分割结果转成 Segmentation 并打开 Slicer 的 Segment Editor。"""
        source = getattr(self, "mask_node", None)
        if source is None:
            self._set_status("请先完成 AI 分割", None)
            self._log("尚无 AI 分割结果，请先点击\"开始分割\"")
            return
        try:
            seg = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode", "AI_Seg_Edit")
            seg.CreateDefaultDisplayNodes()
            seg.SetReferenceImageGeometryParameterFromVolumeNode(
                self.volume_node)
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                source, seg)
        except Exception as e:
            self._set_status(f"导入编辑器失败: {e}", "#d64545")
            self._log(f"导入编辑器异常: {e}")
            return
        self.correct_seg_node = seg

        # 切换到 Segment Editor 模块并激活该分割
        slicer.util.mainWindow().moduleSelector().selectModule("SegmentEditor")
        try:
            editor = slicer.modules.segmenteditor.widgetRepresentation().self()
            if hasattr(editor, "setSegmentationNode"):
                editor.setSegmentationNode(seg)
            if self.volume_node is not None and hasattr(
                    editor, "setSourceVolumeNode"):
                editor.setSourceVolumeNode(self.volume_node)
        except Exception as e:
            self._log(f"提示：请手动在 Segment Editor 中选中 AI_Seg_Edit ({e})")
        slicer.util.resetSliceViews()
        self.correctLabel.setStyleSheet(f"QLabel{{color:{_OK};}}")
        self.correctLabel.setText(
            "编辑器已打开（AI_Seg_Edit）。用画笔工具修正牙齿边界，"
            "完成后点\"读取修正结果\"")
        self._set_status("已打开分割编辑器，请手工修正", _OK)

    def on_read_result(self):
        """把修正后的 Segmentation 导出回 LabelMap，供后续质检/导出。"""
        seg = getattr(self, "correct_seg_node", None)
        if seg is None:
            self._set_status("请先打开编辑器", None)
            self._log("尚未打开编辑器，请先点击\"打开编辑器\"")
            return
        try:
            result = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLabelMapVolumeNode", "AI_Seg_Corrected")
            seg.GetSegmentation().CreateLabelmapVolumeFromOrientedImage(result)
            if result.GetImageData() is None:
                # 兼容较旧/不同实现的导出接口
                seg.GetSegmentation().ExportSegmentsToLabelmapNode(result)
                self._log("已通过 ExportSegmentsToLabelmapNode 导出")
        except Exception as e:
            self._set_status(f"读取修正结果失败: {e}", "#d64545")
            self._log(f"读取修正结果异常: {e}")
            return
        self.corrected_node = result
        self.correctLabel.setStyleSheet(f"QLabel{{color:{_OK};}}")
        self.correctLabel.setText(
            "修正结果已导出为节点 AI_Seg_Corrected，可用于质检/导出")
        self._set_status("人工修正结果已读取", _OK)
        self._log("人工修正结果已导出为 AI_Seg_Corrected")

    # ---------- 槽函数：标签质检 / 结果导出 ----------

    def _resolve_label_path(self):
        """确定要质检/导出的标签磁盘路径。

        若已读取人工修正节点，则先把它导出为文件；否则退回 AI 分割结果。
        返回 (path, is_exported) 或 (None, False)。
        """
        corrected = getattr(self, "corrected_node", None)
        if corrected is not None:
            out_dir = os.path.join(
            os.path.dirname(os.path.dirname(self._script_dir())), "data", "outputs", "labels")
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, "corrected_label.nii.gz")
            try:
                slicer.util.exportNodeToFile(corrected, path)
            except Exception as e:
                self._log(f"导出修正节点失败，回退 AI 结果: {e}")
                return getattr(self, "last_mask_path", None), False
            return path, True
        return getattr(self, "last_mask_path", None), False

    def on_quality_check(self):
        if self.api is None:
            self._set_status("请先连接服务端", None)
            self._log("尚未连接服务端")
            return
        label_path, _ = self._resolve_label_path()
        if not label_path or not os.path.exists(label_path):
            self._set_status("没有可质检的标签，请先分割或修正", None)
            self._log("缺少标签文件，请先开始分割或读取修正结果")
            return
        case_id = self.case_id or "case-manual"
        self.qualityBtn.setEnabled(False)
        self._set_busy(True, "正在进行标签质检...")
        self._log(f"质检标签: {label_path}")
        try:
            result = self.api.check_label(case_id, label_path, "teeth-16")
        except ApiError as e:
            self.qualityBtn.setEnabled(True)
            self._set_busy(False)
            self._set_status(f"质检失败: [{e.error_code}]", _DANGER)
            self._log(f"质检失败: {e.error_code} | {e.message}")
            return
        self.qualityBtn.setEnabled(True)
        self._set_busy(False)
        self._on_quality_done(result, case_id)

    def _on_quality_done(self, result, case_id=None):
        self.qualityBtn.setEnabled(True)
        self._set_busy(False)
        if case_id is None:
            self._set_status(f"质检失败: {result}", _DANGER)
            self._log(f"质检失败: {result}")
            return
        issues = result.get("issues", [])
        passed = result.get("passed", len(issues) == 0)
        self.qualityLabel.setStyleSheet(
            f"QLabel{{color:{_OK if passed else _DANGER};}}")
        if passed:
            self.qualityLabel.setText(
                f"质检通过：共检查 {len(result.get('checked', []))} 项，无问题")
            self._set_status("质检通过，标签合格", _OK)
        else:
            self.qualityLabel.setText(
                f"发现问题 {len(issues)} 项，请查看日志")
            self._set_status(f"质检发现 {len(issues)} 项问题", _DANGER)
            for it in issues[:20]:
                self._log(f"问题: {it.get('type', '?')} -> {it.get('detail', '')}")

    def on_export(self):
        if self.api is None:
            self._set_status("请先连接服务端", None)
            self._log("尚未连接服务端")
            return
        if not self.image_path:
            self._set_status("请先加载影像", None)
            self._log("尚未加载影像")
            return
        label_path, _ = self._resolve_label_path()
        if not label_path or not os.path.exists(label_path):
            self._set_status("没有可导出的标签，请先分割或修正", None)
            self._log("缺少标签文件，请先开始分割或读取修正结果")
            return
        case_id = self.case_id or "case-manual"
        self.exportBtn.setEnabled(False)
        self._set_busy(True, "正在导出训练数据...")
        self._log(f"导出: image={self.image_path}, label={label_path}")
        try:
            result = self.api.export(case_id, self.image_path, label_path,
                                     "nnunet", True)
        except ApiError as e:
            self.exportBtn.setEnabled(True)
            self._set_busy(False)
            self._set_status(f"导出失败: [{e.error_code}]", _DANGER)
            self._log(f"导出失败: {e.error_code} | {e.message}")
            return
        self.exportBtn.setEnabled(True)
        self._set_busy(False)
        self._on_export_done(result, case_id)

    def _on_export_done(self, result, case_id=None):
        self.exportBtn.setEnabled(True)
        self._set_busy(False)
        if case_id is None:
            self._set_status(f"导出失败: {result}", _DANGER)
            self._log(f"导出失败: {result}")
            return
        export_dir = result.get("export_dir", "?")
        files = result.get("files", [])
        self.exportLabel.setStyleSheet(f"QLabel{{color:{_OK};}}")
        self.exportLabel.setText(
            f"已导出 {len(files)} 个文件 → {export_dir}")
        self._set_status("导出完成", _OK)
        self.last_export_dir = export_dir
        for f in files:
            self._log(f"已生成: {f}")
        self._report_event("case_exported", {"export_dir": export_dir})

    def _report_event(self, event, payload=None):
        """静默上报标注事件给 Agent（失败不阻断主流程）。"""
        if self.api is None or not self.case_id:
            return
        try:
            self.api.log(self.case_id, event, "operator-b", payload or {})
        except Exception as e:
            import logging
            logging.getLogger("CBCTAnnotator").warning(
                f"agent/log {event} 上报失败: {e}")