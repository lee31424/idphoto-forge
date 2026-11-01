# ID 证件照制作工具 · 开发者交接文档（DEV_GUIDE）

> 适用对象：接手本工程继续开发的开发者 / 另一套 AI。
> 最后更新：2026-08-19。覆盖到：顶部导航双页（证件照处理 / 排版打印）、统一右侧单预览窗、处理只产「通用照」（尺寸下放到排版页）、排版覆盖裁切不变形、背景线程刷新打印机、标题栏模型状态显示、成品证件照自动导入/去尺寸、按规格分组排版、出血线、空白区切割线开关、批量处理完自动跳转排版页、删除「一键排版」、窗口不撑大、缩略图统一 100×100 / 右侧列宽 120。
> 阅读完本文即可独立运行、测试、修改本工程，无需回溯历史对话。

---

## 1. 项目概述

一个**纯本地离线**的证件照制作桌面工具（Windows），基于 [HivisionIDPhotos](https://github.com/Zeyi-Lin/HivisionIDPhotos) 的推理思路自行实现：

- 输入：用户照片（支持常见格式批量导入）
- 处理：AI 抠图（MODNet / Hivision-MODNet / RMBG-1.4 / BiRefNet-lite 任选）→ 人脸检测（RetinaFace 默认 / 可选 MTCNN）→ 裁剪对齐 → 换底色 / 渐变背景 / HEX 自由底色 → 美颜 / 水印 / 横向翻转
- 输出：**通用照 + 高清照**（处理阶段不分尺寸，一次处理产一张固定构图的「通用照」1400×1000；尺寸在排版页按规格设置，排版时**覆盖式居中裁切**，只裁溢出、绝不拉伸变形）；单一尺寸排版 / 混合尺寸排版（按规格分组、货架式铺排、可带裁剪线）；导出 PNG（带 DPI）/ JPEG（按 KB 限制）
- 技术栈（**2026-11 现状**）：`PySide6` + `PySide6-Fluent-Widgets` 做 GUI；`onnxruntime` 跑 ONNX 模型；`opencv-python` / `numpy` / `Pillow` 做图像处理。除「模型下载」外无网络请求。
- ⚠️ 本文档 §2 的目录结构（`app.py` / `pages.py` / `ui_widgets.py`、tkinter）是 **2026-08 的历史版本**；当前工程已迁到 `ui/` 包 + PySide6，模块清单以 README「项目结构」为准。

**用户铁律（必须遵守，来自需求方，优先级高于一切美观诉求）：**
1. **业务逻辑不动**，只做 UI 优化 / 样式修复 / 布局调整。
2. **深色主题**：全局背景 `#18181B`，橙色强调色 `#F97316`。
3. **不使用纯白边框**——边框统一用灰白（`#3F3F46` 等），避免刺眼。
4. **按钮要有 4 态**（默认 / 悬停 / 按下 / 禁用），禁用态不能用「黑字深底」这种看不清的组合。
5. **沙箱无 DISPLAY**：GUI 无法在 WorkBuddy 沙箱里直接 `python app.py` 跑，必须在用户真机（Python 3.10 + tkinter）验证视觉效果。

---

## 2. 目录结构与各文件职责

```
idphoto_app/
├── app.py                  # 【核心】主程序入口。IDPhotoApp 类：建窗、配样式、页面路由、事件分发、
│                           #         并统一持有「右侧单预览窗 + 视图切换」(right_pane) 及其刷新逻辑。
├── pages.py                # 四个页面视图类 + 缩略图队列控件 + 通用辅助函数。
├── ui_widgets.py           # 自定义控件：TopNav / SidebarNav / ColorSwatch / ColorSwatchGroup /
│                           #         draw_empty_placeholder；高清图标（PIL 4× 超采样）绘制。
├── styles.py               # 【样式唯一来源】所有颜色 / 字体 / 间距 / 尺寸常量集中在此。
│                           #         禁止在其它文件散写样式数值。
├── inference.py            # 推理层：MattingEngine（4 种抠图模型）、FaceDetector（RetinaFace）、
│                           #         EngineManager（懒加载 + 会话管理）。
├── photo_processor.py      # 业务逻辑层：PhotoProcessor.process_id_photo / process_matting_only。
├── layout_engine.py        # 排版引擎：LayoutItem + LayoutEngine（单一/混合尺寸排版，货架式铺排）。
├── image_utils.py          # 图像处理工具：加背景、渐变、美颜、水印、DPI/KB 导出。
├── test_all.py             # 7 项综合测试（见 §7）。注意：模型/测试图路径硬编码在桌面，需改成本机路径。
├── requirements.txt        # numpy / opencv-python / onnxruntime / Pillow
├── run.bat                 # Windows 启动脚本，自动按优先级定位 Python 解释器后跑 app.py。
├── idphoto_workflow.svg    # 工作原理思维图（人像→抠图→检测→裁剪→换底→排版→导出）。
├── model/                  # 【必须随工程一起】5 个 ONNX 模型（见 §6）。
└── test_output/            # 测试生成的预览/排版图（可忽略 / 重新生成）。
```

### 2.1 各模块关键符号（接手时直接按名定位）

**app.py —— `class IDPhotoApp`**（重点：右侧预览由 app 统一管理，页面只调刷新）
- 窗口/样式：`_setup_window`、`_setup_styles`、`_build_titlebar`（含右上角模型状态）、`_build_titlebar_divider`
- 导航：`_build_top_nav`（顶部水平导航，**2 个入口**：证件照处理 / 排版打印，右侧「v1.0·本地离线」）、`_show_page(page_id)`
- 内容区：`_build_content_area`（拆成 `main_pane` 左/中功能面板 + `right_pane` 右侧单预览窗 + 视图切换）
- 预览核心：`_build_right_previews(parent)`、`_update_preview_canvas(canvas, image, placeholder)`、`_refresh_right_previews(page_id=None)`
- 状态/队列：`original_image` / `original_file_path` 属性、`on_add_images`、`_load_image`、`on_select_queue_image`、`_select_image_at`
- 处理：`on_process`（处理选中）、`on_batch_process`（批量全部）、`_batch_process_worker`、`_process_one`、`_on_process_done`
- 排版/导出：`on_generate_mixed_layout`（基于排版页「规格/宽/高/数量」配置 + `ready_photos` 成品列表覆盖裁切生成）、`_rebuild_layout_from_sources`、`_generate_layout_from_queue`、`on_export`
- 模型状态：`_MODEL_DISPLAY_NAMES`、`_current_matting_model_display()`、`_update_model_status(in_use=False)`（标题栏「模型: MODNet 已就绪/使用中/未找到」，绿/橙/红点）
- 打印机：`_list_printers`（Windows 调 `win32print.EnumPrinters`，枚举本地+连接/网络打印机）、`on_printer_properties`；刷新入口在 `pages.py` 的 `_refresh_printer_list` + `_apply_printer_list`（**后台线程枚举，避免网络打印机阻塞 UI**，见 §4.7）
- 成品证件照：`ready_photos` 状态列表、`on_add_ready_photos`、`on_remove_ready_photo`、`on_clear_ready_photos`、`_center_crop_to_size`（覆盖+居中裁切，无损不变形）；处理好的通用照自动导入成品列表，尺寸完全由排版页控制

**pages.py**
- 辅助：`_ratio_fit`、`_make_thumbnail`、`_normalize_queue_items`、`add_slider_row`（抽出的滑块行，全工程复用）
- `class ThumbnailList(ttk.Frame)`：左侧缩略图队列（单击选中 / 双击放大）
- `class ProcessPage`：`_build_left_column` / `_build_middle_column`；
  `show_result_gallery` / `_hide_result_gallery`（中列底部缩略图画廊，双击放大调 `app._refresh_right_previews`）；
  `update_previews(std, hd)`（→ `app._refresh_right_previews`）；`get_*` 系列取参数
- `class LayoutPage`：`_build_left_column` / `_build_middle_column`（混合尺寸排版 + 成品证件照面板 + 导出）；
  `set_layout_pages`、`_refresh_page`、`update_preview`（均 → `app._refresh_right_previews`）；
  `get_mixed_items`、`get_paper_px`；成品面板：`refresh_ready_list`、`get_selected_ready_idx`
  （成品尺寸已移除：尺寸统一由排版页规格控制，处理好的通用照自动进成品列表后由排版覆盖裁切）

**ui_widgets.py**
- `TopNav(tk.Frame)`：顶部水平导航，API 与 `SidebarNav` 一致（`set_active` / `_hover` / `_draw_icon`），选中项底部橙色强调条 + 暖色高亮背景。
- `SidebarNav(tk.Frame)`：早期左侧导航（现已不用，保留为可复用组件）。
- `ColorSwatch` / `ColorSwatchGroup`：圆形色板（高清绘制）。
- `draw_empty_placeholder(canvas, text, icon)`：预览窗空状态占位图。
- 图标：`_create_icon_photo` / `create_color_swatch_photo`（PIL 4× 超采样 + LANCZOS，保证清晰）。

**styles.py**：常量全在这。常用：`COLOR_BG=#18181B`、`COLOR_PRIMARY=#F97316`、`COLOR_SURFACE=#27272A`、`COLOR_BORDER=#3F3F46`；
尺寸：`TOP_NAV_HEIGHT=56`、`PREVIEW_COL_WIDTH=470`、`THUMB_COL_WIDTH=324`、`THUMB_IMG_WIDTH=140`、`THUMB_IMG_WIDTH_RIGHT=280`（最右缩略图列专用更大边长）、`PREVIEW_A4_WIDTH=440/HEIGHT=622`（右侧预览画布，A4 竖版比例）；
预设：`PRESET_SIZES`（一寸/二寸…）、`PRESET_COLORS`、`MATTING_MODELS`、`FACE_MODELS`、`PAPER_PRESETS`。

**inference.py**
- `MattingEngine`：`process(input_image_bgr) -> BGRA`；支持 4 种模型的前/后处理；`_hollow_out_fix` 修复发丝空洞。
- `FaceDetector`（RetinaFace）：`detect(image_bgr, confidence_threshold) -> list[[x1,y1,x2,y2,score, lm×10], ...]`。
- `MTCNNFaceDetector`（可选，需 `pip install mtcnn-runtime`）：基于 mtcnn-runtime，输出格式与 `FaceDetector` 完全一致；无人脸时返回 `[]`（上层按"无人脸"报错）。
- `EngineManager`：`set_matting_model`、`set_face_model`、`get_matting_engine`、`get_face_detector`（按 `face_model_name` 构建对应检测器）、`preload`（懒加载，不阻塞启动）。`face_model_var` 选项已接通 command，切换即生效。

**photo_processor.py**：`PhotoProcessor.process_id_photo(image, size=None)` → `PhotoResult(standard, hd, matting, face_info)`；
`size=None` 时产出固定构图的「通用照」`UNIVERSAL_PHOTO_SIZE=(1400,1000)`，人像居中、头顶留安全边距，**不绑定任何规格**；传入 `(h,w)` 仍可用（兼容显式尺寸，`bg_only` 模式只换底不裁剪）。
`process_matting_only(image) -> BGRA`；`FaceError`、人脸距离/倾斜校正等内部方法。

**layout_engine.py**：`LayoutItem(image, width, height, label)`；
`LayoutEngine.generate_single_size(img, h, w, crop_line)` / `generate_photos(photos, w, h, crop_line)` /
`generate_mixed_size(items, crop_line)` → `list[ndarray]`（多页）。混合排版按 `(w,h)` 分组、组间强制分页；
尺寸超过画布的组自动跳过并记入 `engine.last_skipped_sizes`（全部过大时抛 ValueError）。

**color_utils.py**：零依赖的共享颜色转换工具。`hex_to_rgb_tuple` 由 `image_utils` 和 `ui_widgets` 共用，避免两处重复实现。

**image_utils.py**：`add_background_to_image`（纯色/渐变）、`apply_beauty`、`apply_watermark`、
`save_image_with_dpi`、`save_image_to_jpeg_kb`、`resize_image_esp`、`numpy_to_photo`。

---

## 3. 架构与数据流

```
app.py (IDPhotoApp)
  ├─ 持有全部业务状态：batch_images / current_image_idx / standard_image / hd_image /
  │                    standard_matting / layout_pages / processed_results / layout_queue ...
  ├─ 持有 UI 容器：root → titlebar → top_nav → [ main_pane | right_pane ]
  ├─ right_pane 内含 1 个固定预览画布 + 顶部视图切换（自动/原图/证件照/排版），由 app 统一管理：
  │     _build_right_previews() 只建一次；_refresh_right_previews(page_id) 按 self.preview_view 刷新内容，
  │     切换页面时只更新内容、不重命名、不重排（用户硬要求）。
  └─ 页面（pages.py）是「视图」：构建左/中功能控件，需要更新预览时统一调 app._refresh_right_previews()。

引擎层（与 UI 解耦）：
  app.engine_manager → MattingEngine / FaceDetector
  app.photo_processor (PhotoProcessor)  ← 业务裁剪/换底
  app.layout_engine   (LayoutEngine)    ← 排版
  image_utils         ← 纯函数图像处理

事件流（典型）：
  选图 → on_select_queue_image → _select_image_at → 刷新右侧「原图预览」
  处理 → on_process → 后台线程 _batch_process_worker → _process_one（调 photo_processor）
        → _on_process_done → 存 standard_image/hd_image → app._refresh_right_previews（更新右侧预览）
  排版 → on_generate_*_layout → layout_engine → set_layout_pages → app._refresh_right_previews（更新右侧预览）
```

**关键约定（影响所有页面修改）：**
- 页面**不再自建**大预览画布。`pages.py` 里 `ProcessPage._build_right_column` / `LayoutPage._build_right_column` 已是**空实现**，
  `UploadPage` 也删掉了原图 Canvas。任何「在右侧看预览」的需求都走 `app._refresh_right_previews()`。
- 切页面时只换 `main_pane` 内容；`right_pane` 三个窗口始终在，只刷新内容。

---

## 4. 关键设计决策（为什么是这样）

1. **顶部导航 + 右侧单预览窗 + 视图切换**（2026-08-16 重构，2026-08-17 三窗合并为一）
   - 需求：导航从左侧栏改到**顶部**；右侧建立**单个固定预览画布**，顶部用分段按钮切换要看哪类图像
     （自动/原图/证件照/排版），默认「自动」跟随当前页面。切换页面时只更新内容、不重命名、不重排。
   - 实现：app 层统一持有 `right_pane`、`preview_canvas` 与视图状态 `preview_view`；页面层调用 `app._refresh_right_previews()`。
   - 风险点：若有人在页面里又 new 一个预览 Canvas，会破坏「固定不变」的约束，请勿这样做。

2. **深色主题 + 橙色强调**：所有颜色集中在 `styles.py`，改色只动这里。

3. **模型路径相对化**（为「回家/换机继续开发」铺路）
   ```python
   _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
   DEFAULT_MODEL_DIR = os.path.join(_BASE_DIR, "model")
   ```
   模型目录直接使用工程内 `model/`（相对路径），无硬编码回退。**换机只需保证 `idphoto_app/model/` 下 5 个 onnx 齐全。**

4. **懒加载 + 后台线程 + `_safe_after` 守卫**：模型不阻塞启动；耗时处理放线程，结果用
   `self._safe_after(ms, func, *args)`（主线程安全调度）回写 UI，避免 Tk 跨线程崩溃。

5. **自定义控件 + 高清图标**：`ui_widgets.py` 统一封装导航/色板/占位；图标用 PIL 4× 超采样 + LANCZOS，解决缩放锯齿。

6. **处理只产「通用照」，尺寸下放排版页（2026-08-19 重构）**
   - 动机：原来处理页要选「一寸/二寸/自定义」具体尺寸，`standard_image` 是精确尺寸照；排版页再用 `_resize_photo_for_layout` **直接拉伸**到规格尺寸，长宽比一不同（一寸 295×413≈1.4 vs 二寸 354×472≈1.29）就**变形**。
   - 新流程：
     1. 处理页**移除尺寸框**（预设下拉 + 宽/高输入、`get_size`/`_on_size_changed` 全删），只留底色/美颜/水水印/翻转/高级头部参数——这些只影响通用照构图，不影响尺寸。
     2. `process_id_photo(size=None)` 默认产固定构图「通用照」`UNIVERSAL_PHOTO_SIZE=(1400,1000)`，人像居中、头顶留安全边距，一次处理可切任意尺寸。
     3. 排版页统一用 `_center_crop_to_size`（等比放大→居中取目标比例区域→精确缩放）**覆盖裁切**，只裁溢出、绝不拉伸；`_resize_photo_for_layout` 已删除。
     4. 「一键排版」「排版」均基于排版页「规格/宽/高/数量」配置，尺寸唯一入口在排版页。
   - 风险点：不要在处理页重新加回尺寸框；不要在排版主流程恢复拉伸式缩放（会变回变形）。

7. **启动期打印机枚举放后台线程（2026-08-19 修复）**
   - 现象：8/18 加了「启动自动刷新打印机列表」后，启动变慢/窗口卡顿。根因：`pages.py` 在 UI 线程调 `win32print.EnumPrinters(PRINTER_ENUM_CONNECTIONS)` 枚举**网络/连接打印机**，有离线/慢速网络打印机时阻塞数秒。
   - 修复：`_refresh_printer_list` 把枚举放进 daemon 线程，结果经 `self.after(0, self._apply_printer_list, printers)` 回主线程写下拉框（加 `winfo_exists()` 守卫）。窗口立即可操作，打印机列表后台就绪后自动填充。
   - 风险点：枚举结果回写 UI 必须走 `after(0, …)`（主线程），切勿在子线程直接 `config()` 控件。

---

## 5. 已踩过的坑（别再踩）

| 坑 | 现象 | 修复 |
|---|---|---|
| 模型路径写死桌面 | 换机必报「模型未找到」 | 改为相对 `model/` + 桌面回退（见 §4.3） |
| BGRA 通道判断语法 | `if image.shape[2]==4 if len(shape)==3 else False` 非法嵌套 | 改为 `if len(image.shape)==3 and image.shape[2]==4:` |
| `ImageTk` 未导入 | `_update_preview_canvas` 用 `PhotoImage` 报错 | 顶部 `from PIL import Image, ImageTk` |
| `scrollbar` 引用错 | `show_result_gallery` 里 `ttk.Scrollbar(self.result_gallery,...)` | 应为 `self.batch_gallery` |
| 窗口最小宽度不足 | 压窄后标签页/按钮重叠裁切 | `_setup_window` 设 `minsize(PREVIEW_COL_WIDTH + THUMB_COL_WIDTH + 380 + 120, 680)` |
| 测试脚本硬编码桌面 | ~~旧版 `test_all.py`~~ 已修复：`MODEL_DIR` 相对化、`TEST_IMAGE` 由 `find_test_image()` 自动定位 `test_images/` | 已修复，仅需在 `test_images/` 放一张含人脸图 |
| 沙箱无 DISPLAY | 沙箱里 `python app.py` 起不来 | 视觉验证必须在真机 Python 3.10 跑 |
| 启动卡顿（打印机枚举阻塞 UI） | 8/18 加「启动自动刷新打印机」后窗口卡数秒 | 枚举放 daemon 线程，`after(0,…)` 回主线程写下拉框（见 §4.7） |
| 排版拉伸变形 | 用 `_resize_photo_for_layout` 直接拉伸到规格尺寸，长宽比不同即变形 | 改用 `_center_crop_to_size` 覆盖式居中裁切（见 §4.6） |
| 一键排版与排版页配置冲突 | 旧流程存在「一键排版」按钮，尺寸来源混乱 | 已删除「一键排版」按钮，统一走 `on_generate_mixed_layout`：读排版页 `get_mixed_items()` 规格/数量 + `ready_photos` 成品列表（见 §2.1 app.py） |
| 移除中间照片后结果错位 | `on_remove_queue_image` pop 后 `processed_results` 索引未前移 | pop 后统一重排：被移除结果丢弃、后续索引前移 1（8/18） |
| 缩略图滚轮互相抢全局绑定 | `bind_all("<MouseWheel>")` 多实例覆盖，后创建抢走 handler | 改控件级绑定 + `return "break"`（8/18） |
| **Fluent SpinBox 数字看不见** | `SpinBox` 内部编辑区只有 **1px** 宽，设成 72/150px 都一样，数字完全不显示（qfluentwidgets 1.11 + PySide6 6.12）。根源是它自带 QSS 的 `padding: 0 80px 0 10px` 与右侧两个 31px 按钮被 `QStyleSheetStyle` 重复扣减 | 数值输入一律改用 `widgets.NumberField`（LineEdit + `QIntValidator`，接口兼容 `value/setValue/setRange`）；原生 `QSpinBox` 不受影响（`model_download_dialog` 仍在用，实测编辑区 48px 正常） |

---

## 6. 模型文件（务必随工程一起打包）

`idphoto_app/model/` 下需 5 个 ONNX：
- `modnet_photographic_portrait_matting.onnx`（推荐默认，速度快）
- `hivision_modnet.onnx`（兼容性好，修复发丝空洞）
- `rmbg-1.4.onnx`（精度高但慢）
- `birefnet-v1-lite.onnx`（精度最高但最慢）
- `retinaface-resnet50.onnx`（人脸检测，必选；默认）
- MTCNN 为可选人脸检测器：选「MTCNN」时需 `pip install mtcnn-runtime`（首次运行下载权重，非仓库内置）

注意：模型共约 **536 MB**，会使打包 zip 体积很大（≈540 MB+）。打包时对模型用 `ZIP_STORED`（已压缩过，再压无意义且慢）。

---

## 7. 如何运行与测试

### 运行
```bash
# 1) 安装依赖（一次性）
pip install -r requirements.txt
# 2) 启动（确保 model/ 下 5 个 onnx 就位）
python app.py
#    或直接双击 run.bat（会自动定位 Python 解释器）
```
- **必须在有显示器的机器 + tkinter 可用的 Python 上跑**（本机为 Python 3.10）。
- 沙箱（无 DISPLAY）只能做「语法编译 / import 冒烟 / 无 GUI 的引擎测试」，不能看界面。

### 语法校验（可在沙箱做）
```bash
python -m py_compile app.py pages.py ui_widgets.py styles.py inference.py photo_processor.py layout_engine.py image_utils.py
# 期望输出：ALL_COMPILE_OK（无报错即通过）
```

### 综合测试（test_all.py，7 项）
> 注意：脚本里 `MODEL_DIR`、`TEST_IMAGE` 当前硬编码为桌面路径，换机后**先改成本机路径**再跑：
```python
MODEL_DIR = r"<你的工程>/idphoto_app/model"
TEST_IMAGE = r"<一张含人脸的测试图绝对路径>"
```
```bash
python test_all.py
# 期望：通过 7/7（模型加载 / 人脸检测 / 抠图 / 证件照处理 / 单一排版 / 混合排版 / 导出）
```

### import 冒烟（确认无循环依赖 / 缺模块）
```bash
python -c "import sys; sys.path.insert(0,'.'); import app; print('IMPORT_APP_OK')"
```

---

## 8. 已知限制

1. `test_all.py` 的 `TEST_IMAGE` 需自行准备含人脸的测试图（放 `idphoto_app/test_images/` 即可，已支持自动定位）；`run.bat` 已改为 `py`/`python` 自动定位解释器，无硬编码路径。
   换机接手需改 `MODEL_DIR` / `TEST_IMAGE` / `run.bat` 里的提示路径。
2. GUI 无法在 WorkBuddy 沙箱（无 DISPLAY）直接运行，视觉改动必须真机验证。
3. 打包体积大（模型 536MB），传输慢；如只需改代码可只传源码，到目标机再放 `model/`。
4. `SidebarNav` 已弃用（被 `TopNav` 取代），仍保留在 `ui_widgets.py` 中作为可复用组件，勿删以免误伤。

---

## 9. 接手后建议的检查点

- 想改**导航/顶部栏**：`app.py` 的 `_build_top_nav` + `ui_widgets.TopNav`。
- 想改**右侧预览**：只动 `app.py` 的 `_build_right_previews` / `_refresh_right_previews`；页面侧不要新建 Canvas。
- 想改**颜色/字体/间距**：只动 `styles.py`。
- 想改**处理算法/排版规则**：动 `inference.py` / `photo_processor.py` / `layout_engine.py` / `image_utils.py`，UI 层只做调用。
- 想加**新页面**：在 `pages.py` 加 Page 类 → 在 `app.py` 的 `_build_content_area` 注册 → 在 `TopNav` 的 `items` 加入口。
- **尺寸约定（2026-08-19 起）**：处理页**不再有尺寸框**，尺寸唯一入口在排版页（规格/宽/高/数量）。处理只产「通用照」，排版阶段按规格覆盖裁切——不要在处理页重新加尺寸、不要恢复拉伸式缩放（会变回变形）。
- **缩略图约定**：三处缩略图统一用 `THUMB_IMG_WIDTH=140`（左侧队列子列较窄，受此值限制）；最右侧「缩略图」列专用 `THUMB_IMG_WIDTH_RIGHT=280` 填充加宽后的列（约 27% 窗口宽、列宽 `THUMB_COL_WIDTH=324`），占位提示 `wraplength` 同步保持 ≤ 列宽，避免文字请求额外宽度。
- 任何视觉改动完成后：跑 §7 的编译 + 真机 `python app.py` 肉眼确认，再跑 `test_all.py` 防回归。

---

## 10. 一句话总结

本工程是「本地离线证件照工具」，UI 用 tkinter、推理用 ONNX。最新状态：**顶部双页导航（证件照处理 / 排版打印）+ 右侧单预览窗（视图切换）**已就位；
处理只产**通用照**（不分尺寸），尺寸在**排版页**按规格设置并**覆盖裁切不变形**；打印机枚举放后台线程、标题栏显示当前模型；**工程已打包含全部模型**（≈540MB zip），**模型路径已相对化可换机运行**。
接手时遵守 §1 用户铁律，改样式只动 `styles.py`，预览只走 `app._refresh_right_previews()`，尺寸只动排版页，改完务必真机验证。

---

## 11. 2026-11-01 修复记录（内存 / 状态一致性）

> 本轮只改源码，不动界面布局与交互。回归脚本：`python _regression_test.py`
> （offscreen，不加载模型，8 项，全过才返回 0）。

| # | 问题 | 修法 | 位置 |
|---|---|---|---|
| 1 | 实时预览可无限并发，ORT 内存池被撑大且不归还（实测 6 路并发 1.16GB → 4.9GB） | 单飞锁 + pending 补跑；`SessionOptions.enable_cpu_mem_arena=False`；`Run()` 串行化 | `app.on_live_preview`、`inference._session_options` / `MattingEngine` / `FaceDetector` |
| 2 | 原图全分辨率永久驻留（100 张 ≈ 2.9GB） | `_get_full_image` 改 LRU（6 张），删除/清空时同步清理 | `app._get_full_image`、`on_remove_queue_images`、`on_clear_all_images` |
| 3 | 删除中间图片后结果索引不前移（缩略图错位、结果丢失） | 删除后重建 `processed_results` 索引，并同步 `_gallery_selected_idx` | `app.on_remove_queue_images` |
| 4 | 删除后重新导入同一文件被判重复 | 删除时从 `_imported_paths` discard | 同上 |
| 5 | 美颜在换底之后执行，把用户选的底色一起改掉（#438EDB → 发白） | 美颜只作用于 BGRA 的人像 RGB 通道，再合成背景 | `app._process_one` / `_apply_beauty_bgra` |
| 6 | 翻转在水印之后，水印文字被镜像 | 顺序调整为 美颜 → 换底 → 翻转 → 水印 | 同上 |
| 7 | 批量导出 JPG + KB=0 写出 PNG 数据 | `target_kb<=0` 时也写 JPEG（q95） | `image_utils.save_image_to_jpeg_kb` |
| 8 | 导出用「最后处理的那张」而非「当前选中」 | 新增 `_selected_result()`，导出按画廊/队列选中项取图 | `app.on_export` |
| 9 | 中心渐变逐圈 ellipse，1400×1000 要 511ms | numpy 向量化，同图 <10ms；与旧实现逐像素误差 <4 | `image_utils._generate_gradient` |
| 10 | 换纸张后排版缩略图不刷新 | 清空数据时同时刷新「第 N 页」缩略图 | `layout_view._on_paper_changed` |
| 11 | MTCNN 永远被视为「可用」（不在 manifest 里） | 用 `find_spec("mtcnnruntime")` 探测，纳入缺失集合 | `inference.mtcnn_available`、`app._check_models`、`process_view._detect_missing` |
| 12 | 两个模块各写一份配置，互相丢字段 | 写回前先读磁盘，只覆盖自己负责的键；改原子写 | `core/mirror_manager._save_public`、`model_downloader.save_config` |
| 13 | `mdlCard_rmbg-1.4` 的 ID 选择器含 `.` 导致 QSS 整条失效 | objectName 非字母数字替换为 `_` | `settings_dialog._refresh_model_list` |
| 14 | 模型目录兜底路径写成 `models`（真实为 `model`） | 改为单数，与 `_resolve_data_dir` 一致 | `process_view._detect_missing`、`settings_dialog` |
| 15 | 下载不校验长度，截断的模型会静默落盘 | 有 `Content-Length` 时必须收满，否则判失败并回退下一源 | `model_downloader._download_one` |
| 16 | 测试假绿/假红 | 拖拽用例等事件循环；`test_all` 人脸检测失败改为抛异常 | `_smoke_qt.py`、`test_all.py` |

### 新增的硬约束（后续改动请遵守）

1. **任何推理都不许在无上限并发下跑**：`on_live_preview` 已单飞，新增入口（如新预览、批量预取）必须走同一把 `_live_lock` 或自己限流。
2. **`processed_results` 的键是「队列下标」**，任何改变队列顺序/长度的操作都必须重排它（本轮已修删除路径；若将来支持拖拽排序，务必调用同一段重排逻辑，或改成稳定 uid）。
3. **美颜只能作用在人像上**（`_apply_beauty_bgra`），不要对已合成背景的图调 `apply_beauty`。
4. **水印必须在翻转之后画**。
5. **全分辨率原图只经 `_get_full_image` 取**，不要往 `item["image"]` 里写（那是 LRU 之外的旁路，会让内存重新失控）。

### 修复前后实测（同一台机器，3840×2560 人像）

| 场景 | 修复前 | 修复后 |
|---|---|---|
| 6 路并发完整流水线（GPU） | 1.16GB → 4.9GB，gc 后不回落 | 1.13GB → 1.18GB（peak 1.48GB），耗时 8.4s → 1.6s |
| 单张处理后的常驻内存（CPU） | 1055 MB | **240 MB** |
| 3 路并发后（CPU） | 4981 MB | 242 MB |
| 顺序推理耗时（CPU，5 次均值） | 1.38 s | 1.42 s（+3%，换 800MB 内存，划算） |
| 中心渐变 1400×1000 | 511 ms | < 10 ms |

### 2026-11-01（二）界面与规格调整

| # | 需求 | 实现 |
|---|---|---|
| 17 | 人脸旋转校正**默认开启** | `process_view._tab_basic` 里 `rotation_switch.setChecked(True)`；`face_alignment` 仍由该开关决定，用户可关 |
| 18 | 调整窗口大小时控件不重排 / 位置偏移 | ① `TabStrip` 每个页签、排版页整页都包进 `widgets.wrap_scrollable()`（内容高于窗口时滚动，而不是被压到 `minimumSizeHint` 以下导致卡片重叠）；② 右栏宽度 300~420 → **360~480**（360 是"目标尺寸"那一行的实际最小宽度）；③ 新增 `IDPhotoApp.resizeEvent` + `_apply_split_layout()`，按 左16% : 中60% : 右24% 重排三栏，用户拖过分隔条后按他拖出的比例走 |
| 19 | 排版规格：删护照/签证/驾照/社保卡，加自用1/自用2 | 只改 `styles.PRESET_SIZES`（删 4 项，加 `("自用1",319,425)`、`("自用2",378,496)`，即 27×36mm / 32×42mm @300DPI）与 `DEFAULT_COUNT_BY_SIZE`；处理页删掉原先硬编码的自用1/2 循环，与排版页共用同一份清单 |
| 20 | 排版规格「数量」框看不到数字 | 不是宽度问题：`qfluentwidgets.SpinBox` 的编辑区恒为 1px（见 §5 坑表）。新增 `widgets.NumberField`（LineEdit + `QIntValidator`，接口兼容 SpinBox），替换排版页数量框、设置里 5 个数值框（KB/留白/横距/纵距/超时）以及规格/纸张对话框里的宽·高·张数框；删除已无用的 `theme.spinbox_qss`。实测数量框 72×33、文字区完整，截图内深色像素 34（空框为 2），数字确实画出来了 |
| 21 | 「选项」默认值：出血 50 / 横距 20 / 纵距 20 | 三个默认值收敛到 `styles.LAYOUT_DEFAULT_BLEED / _GAP_H / _GAP_V`，`layout_view` 与 `settings_dialog` 共用，`DEFAULT_SETTINGS["layout"]` 同步（新装即 20/20/50）。注意 `settings.json` 在 `.gitignore` 里，本机已手动把旧的 1/1 改成 20/20 |
| 22 | 排版固定用「标准照」，小规格处理后再排大规格会放大发虚 | 成品列表同时记 `hd`（仅当 hd 确实比 std 大才保留；通用构图下两者同尺寸，自动丢弃避免白占内存）。排版改用 `_pick_layout_base(src, w, h)`：标准照不需要放大就用标准照，需要放大才用高清照。实测「一寸 295×413 处理 → 大二寸 413×626 排版」锐度 16.8 → **74.5** |
| 23 | 成品列表看不出哪张带高清；多页排版只能一页页导出 | 含高清版的条目在名称后加 `· HD`（悬停有说明）；导出卡片新增「导出全部排版图」，按页序写 `排版图_01/02/…`，复用当前输出格式 / DPI / KB 限制，遇到同名文件先弹一次覆盖确认。回归用例覆盖 PNG 与 JPEG 两种输出 |
| 24 | 排版纸张默认是 A4 | 新增 `styles.DEFAULT_PAPER = "6寸相纸"`（152×102mm，冲印最常用），`layout_view._load_custom_paper_presets` 用它预选；默认画布 1795×1205。方向单选补了提示：「纵向 = 按纸张定义方向，横向 = 旋转 90°」——6 寸本身就是横向，所以默认「纵向」得到的正是常规 6 寸画布 |
| 25 | 生成排版 / 导出全部排版图 时软件假死；导出的文件夹里没有文件 | 两个操作都改到**后台线程**：状态栏显示「正在生成/导出 i/N」、等待光标、`_layout_busy` / `_export_busy` 防重入；导出**开始前先探测文件夹可写**，逐页**先写 `.part` 再原子改名**，`.part` 校验失败会自动回退为直接写目标文件，两条路都失败才报错；失败信息带真实异常，并写入 `%LOCALAPPDATA%\证件照制作工具\export_debug.log`；同名文件自动加 `_2/_3` 后缀。根因：真实照片 PNG 编码约 **190ms/页**，105 页 ≈ 20s，之前全跑在 UI 线程上，用户以为死机直接关进程，文件自然没写出来 |
| 26 | 成品证件照只能一张张导入；排版规格每次都要手选规格/背景/数量 | ① 成品卡片改两行按钮：**导入图片 / 导入文件夹**（后台线程加载、跳过已存在、忽略非图片，`_ready_import_busy` 防重入）；② 排版规格新增快捷行：**自用2 ×8**、**自用1 ×9**（背景默认），一键入清单（`_quick_add_mixed`）；③ 清单文字缩短 + 省略号 + tooltip，去掉窄栏里的横向滚动条 |
| 27 | 排好的多页分不清哪页是哪张成品 | 每页右下角画来源标注 `最近两级目录\文件名`（如 `导出测试\批次A\_F8A4960.jpg`，只有一级目录时自动降为 `导出测试\_F8A4960.jpg`）：`LayoutItem.source` 由 `_source_display_label(levels=2)` 从成品路径生成 → `LayoutEngine.last_page_sources` 记录每页来源 → `_compute_layout_pages` 调 `image_utils.draw_bottom_right_label()` 逐页绘制（半透明白底 + 深灰字，msyh 字体，只处理右下角 ROI，超长文本从头部截断保留文件名）。一页混多张时列前 2 个，超过 3 个用「首个 等N个」。标注烘焙进页面，预览/导出/打印一致 |
| 28 | 美颜只有调色（亮度/对比/饱和/锐化），没有真正的磨皮 | 重写 `image_utils.apply_beauty`：**肤色遮罩**（YCrCb 133~173 / 77~127 + 形态学 + 羽化）→ **半分辨率双边滤波磨皮**（缩小一半做 d=15/sigma=100，覆盖范围等效翻倍、更快；回加 25% 高频细节避免塑料脸；强度曲线 `(s/10)^0.85*0.95`）→ **LAB 美白**（L 提亮 + b 轻微去黄）。只处理「皮肤包围盒」，`_apply_beauty_bgra` 传入 alpha，遮罩外（含透明区）逐位不变。UI 砍掉 4 个调色滑块，只留「磨皮 / 美白」0–10（磨皮默认 3）。实测 1400×1000：磨皮 3 纹理 −41%、6 −67%、10 −87%，单张约 66ms；合成后底色与所选 HEX 完全一致 |
| 29 | 原图偏黄 / 偏绿 / 偏蓝，没有肤色矫正 | 新增 `_apply_skin_tone`：用皮肤区域估计 **B/G、R/G 比例**（参考 0.657 / 1.390，与亮度无关）→ 算 RGB 增益（夹在 0.80~1.40）→ 按强度混合 → `cv2.LUT` 校正**整张人像**（脸/发/衣服颜色关系一致；背景后期合成，不受影响）。顺序：颜色均衡 → 肤色矫正 → 磨皮/美白。UI 增加「肤色矫正」0–10 默认 3。实测（同一遮罩量）：偏黄 B/G 0.473→0.640、偏绿 0.467→0.652（目标 0.657）；单张 +10ms。同时修掉 `_process_one` 只对标准/高清照美颜、**抠图未美颜**的问题（排版页换底色会变回素颜） |
| 30 | 脸上红一块白一块（大块色斑），磨皮压不住 | 新增 `_apply_skin_even`：LAB 的 a/b 通道做**大尺度高斯**（缩小 4 倍计算，sigma = 短边 ×0.02，约脸宽 2%，遮罩加权模糊）→ 按强度把低频相对皮肤均值的偏离压回（k = strength/10 × 0.8），高频纹理保留。UI 增加「颜色均衡」0–10 默认 3。实测低频 a std 5.60→4.41(3)/1.64(10)，b std 6.17→4.88/2.01；四项默认全开约 127ms/张（1400×1000） |
| 31 | 额头/鼻尖反光油光；美颜参数多了不好一键对比；默认界面是浅色 | ① `apply_beauty` 新增 `highlight`：以皮肤 L 的「均值 + 0.8σ」为高光起点（夹在 110~225），`L -= k·w·(L-cap)`，w 用 30 灰阶平滑过渡，只压反光不动正常肤色；实测合成高光 L 231→218(3)/188(10)，正常肤色 113 不变；② 美颜页顶部加**一键开关**（关闭时 `get_beauty_params()` 全 0、滑块置灰），重置时自动打开；③ `DEFAULT_SETTINGS.appearance.theme_mode` 改 **dark**（`#18181B` 黑背景，设置里仍可切浅色），并同步本机 `settings.json`。回归新增「高光压制」「美颜开关」（28/28） |
| 32 | 高光压制把衣服也压暗了；顶栏没有主题切换 | ① 高光原来只按肤色 YCrCb 选区域，**暖白衣服（白平衡偏暖）也会落进肤色区间**；改为只用「最上方皮肤连通块」（`_top_skin_component`，脸+脖子，衣服在下面且通常被衣领断开），实测脸油光 231→198、衣服 239 不变；② 顶栏「设置」左侧新增主题按钮（`BRIGHTNESS` 图标 + 当前模式文字），点击 `_toggle_theme()`：翻转 `appearance.theme_mode` → `settings.save()` → `theme.apply_theme()`，17 个主题监听控件自动重染。回归新增「高光不碰衣服」「右上角主题切换」（30/30） |
| 33 | 浅色衣领仍被磨皮/颜色均衡误处理 | 在肤色遮罩之外再加两道筛选：① **几何**：只保留最上方皮肤连通块（`_top_skin_component`，脸+脖子）；② **颜色**：低饱和（S<35）且比脸更亮的像素判为衣物（`_refine_skin_by_color`，参考亮度取连通块上半部分中位数，避免衣领占比大时把参考值拉高）。磨皮/均衡/美白/矫正/高光统一用筛选后的遮罩；红斑是高饱和，不会被误杀（颜色均衡仍能处理）。实测合成场景（脸+脖子+**相接的**暖白衣领）：衣领权重 0.00、磨皮变化 0.0000，脸正常；真实照片皮肤 10.8%→9.9%，纹理 547→70，五项默认 149ms |
| 34 | 单张处理 1.4s 偏慢；拖美颜滑块每次重跑整条流水线 | ① `_process_one` 新增 `result` / `portrait_key` 参数：检测+抠图+裁剪的结果按「图片指纹 + 结构参数」缓存（`_portrait_key` / `_portrait_cache`，最多 2 条、约 44MB），实时预览命中缓存时只重算「美颜+换底+水印」——实测首次 2.15s（含模型加载）→ 再拖美颜 **0.66s**（关美颜 0.08s），推理次数保持 1；② 高清照与标准照尺寸相同且平均差 <2 灰阶时复用标准照的美颜结果（实测平均差 0.35），省 ~250ms。完整单张 1.41s → **1.07s**。回归新增 1 项（31/31） |
| 35 | 3000 张批量处理必崩（结果全留内存） | 每张 std+hd+matting ≈ 18MB，3000 张 ≈ **54GB**。改成**结果落盘**：`core/result_store.py` 把三张图写成 PNG（实测 4.77MB/张，3000 张 ≈ 14GB 临时盘），`processed_results` 只存 `StoredResult`（路径 + 240px 缩略图，`.get(kind)` 按需读盘、LRU 3 张）；批量 worker 每处理完一张**立刻落盘并释放数组**，`ready_photos` 存 `LazyImage`（路径引用，列表刷新用预存 shape 不读图），生成排版时才 `_resolve_ready_image` 读回。目录默认 `%TEMP%/idphoto_results`，可用环境变量 `IDPHOTO_RESULT_CACHE` 改到别的盘；退出时清理，启动时清 24h 前残留。回归新增「结果落盘 + 缩略图」（32/32） |
| 36 | 生成排版时状态栏没反应，像卡死 | 两个原因：① `_collect_layout_source_photos()` 在主线程逐张读盘（3000 张要几十秒），现在**移到后台线程**，每 10 张回报「正在读取成品照 i/N…」；② 排版引擎生成页没有进度回调，现在 `generate_photos` / `generate_mixed_size` 支持 `on_page`，`_compute_layout_pages` 透传，状态栏显示「正在排版：已生成 N 页…」（0.25s 节流）。另加**页数预估 + 内存提醒**（`_estimate_layout_pages`，预计 >1.5GB 时弹窗建议分批），页面缩略图只渲染前 200 页（`_refresh_layout_thumbnails`），避免 3000 页控件把主线程卡死。实测 200 张：读取 → 排版 N 页 → 排版完成 25 页，状态栏全程变化 |

> 附：`布局页` 之前把整列最小高度顶到 766px，导致 700 高的窗口把主窗口内容区挤出屏幕（底部被裁）。现在 `LayoutView.minimumSizeHint()` 降到 68px，窗口 1000×700 时不再裁切。

### 再补两条硬约束

6. **参数面板类页面必须放在 `widgets.wrap_scrollable()` 里**，否则内容高度会顶高整窗最小高度、或在窗口变矮时压出重叠。
7. **规格只改 `styles.PRESET_SIZES`**（元组顺序是 `(名称, 宽, 高)`）。处理页与排版页都从这一份派生，不要在页面里再硬编码尺寸表。
8. **数值输入用 `widgets.NumberField`，不要用 `qfluentwidgets.SpinBox`**（编辑区 1px，数字看不见）；原生 `QSpinBox` 可以用。
9. **排版取图必须走 `_pick_layout_base(src, w, h)`**，不要直接读 `src["image"]`——否则会绕过「按目标规格自动选标准照/高清照」的逻辑，小规格处理再排大规格时又会放大发虚。
10. **长耗时操作不要跑在 UI 线程**：生成排版、批量导出排版图已经改为后台线程（`threading.Thread` + `_safe_after` 回主线程）。新增类似功能时照此办理——同步写盘/批量图像处理会让窗口"未响应"，用户关进程就会丢结果。
11. **排版页角标注统一在 `_compute_layout_pages` 里画**（`_draw_page_labels`），不要放到导出阶段单独画——否则预览、导出、打印三处会不一致。
12. **美颜只走 `apply_beauty`（磨皮 + 颜色均衡 + 肤色矫正 + 高光压制 + 美白）**，不要再往回加「亮度/对比/饱和/锐化」这类全局调色滑块——它们会把底色和衣服一起改掉；美颜总开关在 `get_beauty_params()` 里统一返回全 0，不要在每个调用点单独判断。
