# ToothSeg 语义分割 nnU-Net 导出阶段 WinError 87 故障修复报告

- **日期**：2026-09-05

- **影响链路**：3D Slicer（CBCT Annotator 插件）→ toothseg-server（FastAPI）→ nnUNetv2\_predict（Dataset121 语义分割）

- **故障等级**：推理 100% 完成后导出崩溃，分割结果全部丢失，任务以 `error` 收场

- **修复状态**：已修复并通过 3D Slicer 端到端验证

***

## 1. 故障现象

用户在 3D Slicer 中点击"开始分割"后，后端进度长期停在 55%（`nnunet_running`），约 50 分钟后任务报错终止：

```text
[toothseg-server] progress: {'stage': 'error', 'message': 'ToothSeg 语义分割失败: nnUNetv2_predict 运行失败，日志见: D:\ToothSegWork\_runtime\reuse_packages\...\logs\run_log.txt', 'run_status': 'error'}
```

`run_log.txt` 关键片段（完整证据链）：

```text
100%|██████████| 32/32 [44:41<00:00, 83.79s/it]     ← 滑窗推理 32 个 tile 全部完成
Process SpawnPoolWorker-13:                          ← 导出 Pool worker 崩溃
  File "...\multiprocessing\connection.py", line 337, in _get_more_data
    assert left > 0
AssertionError
Traceback (most recent call last):                   ← 主进程随后崩溃
  File "...\nnunetv2\inference\predict_from_raw_data.py", line 412, in predict_from_data_iterator
    ret = [i.get()[0] for i in r]
  File "...\multiprocessing\pool.py", line 540, in _handle_tasks
    put(task)
  File "...\multiprocessing\connection.py", line 280, in _send_bytes
    ov, err = _winapi.WriteFile(self._handle, buf, overlapped=True)
OSError: [WinError 87] 参数错误。
sending off prediction to background worker for resampling and export
done with STS24_Train_Unlabeled_0001
```

## 2. 根因分析

**崩溃点不在模型推理，而在"分割结果导出"阶段。** 推理 32/32 tile 已 100% 完成，nnU-Net 随后要把整卷预测 logits 通过多进程交给导出 worker 做重采样与 NIfTI 落盘，这一步在 Windows 上崩了。

机制链条：

1. `nnUNetPredictor.predict_from_data_iterator()`（`predict_from_raw_data.py` L358）**无条件**创建 spawn Pool：

   ```python
   with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
   ```

   即使服务端传 `-nps 1`（1 个导出 worker），也依然走 Pool + 进程间管道。
2. 主进程把整卷 logits（数亿体素 × 33 类）pickle 后通过 Windows 匿名管道一次性 `WriteFile` 发给 worker。Python `multiprocessing` 在 Windows 上对超大负载的 overlapped 管道写入偶发 `OSError: [WinError 87] 参数错误`（ERROR\_INVALID\_PARAMETER），这是 CPython multiprocessing 在 Windows 的已知问题；使用 nnU-Net 的 TotalSegmentator 在 Windows 上有完全相同的崩溃报告（预测完成后导出阶段 `AssertionError` + `WinError 87`）。
3. 管道写入失败后，读端 worker 收到残缺数据，`connection.py` 的 `assert left > 0` 触发 `AssertionError`（即日志中 SpawnPoolWorker-13 的崩溃）；随后主进程 `starmap_async` 的结果 `i.get()` 重新抛出异常，`nnUNetv2_predict` 以非零退出码结束，`toothseg_semantic.py` 抛出 `RuntimeError`，任务标记为 `error`。
4. 崩溃与数据体积/系统时序相关，属于**偶发**问题——同样的命令此前成功过，本次图像（640×640×400 @0.25mm 降采样到 0.75mm）触发了大数组管道传输失败。

## 3. 修复方案

nnUNet 官方 CLI 本身提供全顺序模式：当 `-npp 0` 且 `-nps 0` 同时传入时，入口函数走 `predict_from_files_sequential()` 分支（源码注释："Just like predict\_from\_files but doesn't use any multiprocessing"）——**预处理、滑窗推理、导出全部在主进程内完成，预测数组零跨进程传输**，从机制上根除该类管道崩溃，且不修改任何第三方库源码。

对单图推理场景没有性能代价：原先 `-npp 1 -nps 1` 也只有 1 个预处理/导出 worker，顺序执行与"1 个后台 worker"耗时基本相同；反而省去了大数组序列化（pickle 副本）的内存峰值。

## 4. 变更清单

仅改动 1 个文件：`implementation/model/toothseg_semantic.py`（服务端调用 nnUNetv2\_predict 的适配层）。

| 位置                                   | 修改前                                                      | 修改后                        | 目的                                                                     |
| ------------------------------------ | -------------------------------------------------------- | -------------------------- | ---------------------------------------------------------------------- |
| `run_toothseg_semantic()` 内 `cmd` 列表 | `"-npp", "1", "-nps", "1"`                               | `"-npp", "0", "-nps", "0"` | 进入 nnUNet 官方 sequential 模式，导出在主进程执行，绕过 Windows 管道大数组传输（根除 WinError 87） |
| `env` 组装                             | 无条件设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 仅 `os.name != "nt"` 时设置    | Windows PyTorch 不支持该分配器配置，每次运行产生 UserWarning 噪音                        |

cmd 修改处新增注释说明原因（Windows spawn Pool 管道传输整卷 logits 触发 WinError 87）。

**不需要改动的部分（已核实）**：

- `implementation/model/toothseg/run_toothseg.py`（手动批量双分支脚本）：使用 `MemSafePredictor`，其已覆写 `predict_from_data_iterator()`（`memsafe_inference.py` L203-206），导出本来就在主进程流式执行，无同类风险。

- 环境补丁 `E:\miniconda3\envs\nninteractive\lib\site-packages\sitecustomize.py`（SimpleITK→nibabel 兜底）：与本故障无关，保持不动。

- 3D Slicer 插件（`user/plugin/`）与服务端其余代码：无改动。

## 5. 验证测试（3D Slicer 端到端）

**测试环境**：RTX 4060 Laptop GPU（8GB）、nninteractive conda 环境、3D Slicer 5.12.3 + CBCT Annotator 插件、输入图像 `STS24_Train_Unlabeled_0001.nii`（640×640×400 @0.25mm，中文路径）。

**服务启动要点**：必须把 `E:\miniconda3\envs\nninteractive\Scripts` 加入 PATH 后启动（一键启动脚本 `user\launcher\一键启动服务.bat` 已包含此逻辑），否则 `shutil.which("nnUNetv2_predict")` 找不到可执行文件，`/status` 返回 `available: false`。

**测试过程与结果**（2026-09-05 13:41–14:35 实测）：

1. 通过 3D Slicer 的 CBCT Annotator 面板发起语义分割（模式 balanced、降采样 0.75mm）。
2. 复用包机制生效：自动复用上次失败的 0.75mm 降采样输入，跳过预处理直接推理。
3. 服务端日志确认实际命令行已带 `-npp 0 -nps 0`，nnUNet 进入 sequential 模式（日志出现 "Running in non-multiprocessing mode"）。
4. 滑窗推理 32/32 tile 全部完成（53 分钟，首个 tile 约 21 分钟为 GPU/CUDA 初始化开销，其余约 60 秒/个），**导出阶段在主进程顺利完成**（此前正是在此崩溃），全程未再出现 `WinError 87` / `AssertionError`。
5. 退出码 0，服务端完成 ToothSeg 标签 → 项目 dense 标签映射（1–16 上牙、17–32→49–64 下牙），生成最终 mask 与 `label_mapping.json` / `tooth_locations.json`。
6. 3D Slicer 端进度走完 100%（`POST /predict` 返回 200，`run_status: success`，`elapsed_sec: 3213.2`），界面状态由"分割失败: \[PREDICTION\_FAILED]"变为"**AI 分割完成，结果已加载。**"。
7. 分割质量抽样：共分割出 26 颗牙（上颌 13 颗 101–115、下颌 13 颗 401–415，缺位为智齿/缺失牙，符合该病例牙列）。
8. 附带改善：日志中的 `nifti_image_write_engine ... _probe.nii.gz` 探测错误从上一轮 17 条（每个 spawn worker 各探测一次）降为 1 条（仅主进程），`expandable_segments not supported on this platform` 警告消失。

**结论：修复有效，故障链路闭环。**

## 6. 遗留事项与建议

1. **SimpleITK 中文路径问题（历史遗留）**：本机用户名为中文（`C:\Users\王\...`），SimpleITK 原生 NIfTI 引擎写含非 ASCII 的路径会失败，目前靠 `sitecustomize.py` 的 nibabel 兜底补丁解决（每个进程启动时探测一次，探测失败会打印一条 `_probe.nii.gz` 错误，属无害噪音）。若将来升级 SimpleITK 或重装环境，需重新验证该补丁是否仍必要。
2. **推理耗时**：0.75mm 全卷滑窗推理在本机约 45 分钟（首个 tile 因 GPU/CUDA 初始化常达 20 分钟以上，属环境特性）。如需提速可评估 fast 模式更高降采样间距或更小 tile 步长的取舍。
3. **运行** **`run_toothseg.py`** **双分支批量脚本**不受本次改动影响，也无需修改（见第 4 节）。
4. 服务端启动务必使用一键启动脚本或手动补 PATH（见第 5 节），否则模型探测失败。

