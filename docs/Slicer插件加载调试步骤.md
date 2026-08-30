# 3D Slicer 加载与调试本插件 —— 操作步骤（B 组）

> 适用范围：本机已装 3D Slicer 5.12.3（`D:\person_download_old\Slicer\3D Slicer 5.12.3\Slicer.exe`）。
> 目标：把 `slicer_extension/` 里的 `CBCTAnnotator.py` + `ApiClient.py` 加载进 3D Slicer，
> 在扩展模块面板中打开并点"连接测试"验证通信。

---

## 0. 先说结论（如何加载）

3D Slicer 的脚本模块（Scripted Module）就放在某文件夹里，让 Slicer **知道这个文件夹路径**，它就会在模块列表里出现。所以核心就一步：**把 `slicer_extension/` 文件夹加入 Slicer 的"模块加载路径"**。

> 术语说明：
> - **Scripted Module**：用 Python 写的 Slicer 扩展，`CBCTAnnotator.py` 就是这种。
> - **模块加载路径（Module path）**：Slicer 搜索可加载模块的文件夹列表。把自己的代码文件夹加进去，插件就会出现。

---

## 1. 一次性配置：添加模块路径

1. 打开 3D Slicer（双击 `Slicer.exe`）。
2. 菜单栏点 **Edit**（编辑）→ **Application Settings**（应用设置）。
   - 中文界面则为：**编辑 → 应用设置**。
3. 左侧分类选 **Modules**（模块）。
4. 右侧找到 **Additional module paths**（附加模块路径）区域。
5. 点 **+**（添加）按钮，选择文件夹：
   ```
   D:\study\Competition\Dentistry\CBCT_Annotation_Assistance_System\slicer_extension
   ```
   （注意选到 `slicer_extension` 这一层，里面才是 `CBCTAnnotator.py`）
6. 点 **OK** 保存，重启 3D Slicer。

> 重启后才能被识别（Slicer 在启动时扫描模块路径）。
> 以后如果改了 `CBCTAnnotator.py`，不用重启整个 Slicer，用第 3 节的"重新载入"更快。

---

## 2. 运行插件

1. 顶部模块搜索框输入 `CBCT`，应该会出现 **CBCT Annotator**（或"标注系统"）。
2. 点击它，左侧会出现我们的面板：
   - 服务地址：默认 `http://127.0.0.1:8000/api/v1`
   - 三个按钮：连接测试、加载配置
   - 模型 / 推理模式 / 标签模板三个下拉框
   - 日志区
3. 先点 **连接测试**：
   - 若服务端没启动/没模型 → 状态栏变红，日志显示 `[CONNECTION_FAILED]` 或某错误码。此时只需确认"连接报错能正常提示"，也算调试通过（说明 UI 和 API 层是通的）。
   - 若服务端已就绪 → 状态栏变绿 "已连接"。
4. 再点 **加载配置**：三个下拉框应被填入模型/模板内容，日志显示可用推理模式。

> 即使找不到真实服务端，只要点击后日志有中文提示、界面不卡死，就证明**插件本身加载成功**——这就是本阶段目标的验证标准。

---

## 3. 开发时的"重新载入"（改代码后不用重启 Slicer）

1. 修改 `CBCTAnnotator.py` 保存。
2. 在 Slicer 右下角的 **Python 交互控制台（Python Console）**里依次执行：
   ```
   slicer.util.reloadScriptedModule('CBCTAnnotator')
   ```
3. 回到模块，重新点开 **CBCT Annotator**。面板会刷新成新代码。

> 若加了对面板的修改有时不够彻底，可用开发者工具：**Developer（开发者）→ Reload & Test（重新载入并测试）**，选择本项目。

---

## 4. 如何查看 Python 报错（排查用）

- 打开 **View（视图）→ Python Interactor（Python 交互控制台）**，以及 **View → Error Log（错误日志）** 两个面板。
- 插件一加载失败，错误堆栈会打印在这里。把红色堆栈发我即可。

---

## 5. 常见的三个坑（提前预防）

| 现象 | 原因 | 解决 |
|------|------|------|
| 模块列表里搜不到 CBCT | 路径没加对 / 没重启 | 重做第 1 节，确保重启 |
| 点按钮提示"无法连接服务端(请求前失败)" | 服务端没启动，或 `requests` 库缺失 | 先用假说明验证；若缺库，在第 3 节控制台 `pip_install('requests')` |
| 面板打开了但下拉框空 | 服务端 `/config` 返回空或连不上 | 属正常（前端空是因为后端没数据），确认日志有错误码即可 |

> 提示：Slicer 自带 `requests`，一般无需装；若报 `No module named 'requests'`，在 Python Console 执行：
> `slicer.util.pip_install('requests')`