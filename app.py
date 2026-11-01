# -*- coding: utf-8 -*-
"""
证件照制作工具 —— PyQt-Fluent-Widgets 主控制器
============================================================
职责：持有全部业务状态、构建窗口 / 导航 / 内容区 / 状态栏，并把用户操作
分发给后端模块（inference / photo_processor / layout_engine / image_utils）。

UI 与代码分离：
  * 本文件只做「状态 + 逻辑 + 信号接线」，不直接画控件；
  * 所有控件长相 / 布局都封装在 ui/ 包（views + widgets + theme）；
  * 后端模块（推理 / 处理 / 排版）保持原样复用，不改动。
"""
import os
import sys
import time
import threading
import queue
import functools
import traceback
from collections import OrderedDict

# 说明：numpy / cv2 保留在顶层导入。实测即便改为函数内按需导入也省不掉——
# photo_processor / image_utils 在顶层就要 import 它们，本文件终究会被
# 间接拖入，延迟只是把开销换个位置、还平添散落的 import 语句。
# 真正值得延迟的是 onnxruntime（见 inference.py），它确实能省下约 46ms。
import numpy as np
import cv2

# 确保能 import 同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSplitter,
    QStackedWidget, QMessageBox, QFileDialog, QApplication, QSizePolicy, QDialog,
)
from PySide6.QtCore import Qt, QTimer, QObject, QEvent, QUrl
from PySide6.QtGui import QIcon, QPixmap, QImage, QPainter, QDesktopServices
from PySide6.QtPrintSupport import QPrinter, QPrintDialog, QPrinterInfo

from qfluentwidgets import Pivot, FluentIcon

from styles import WINDOW_WIDTH, WINDOW_HEIGHT, IMAGE_EXTENSIONS  # 复用尺寸/扩展名常量
from inference import EngineManager
from photo_processor import PhotoProcessor, FaceError
from layout_engine import LayoutEngine, LayoutItem
from color_utils import hex_to_rgb_tuple
from image_utils import (
    add_background_to_image, save_image_with_dpi, save_image_to_jpeg_kb,
)

from ui import theme, widgets
from ui.theme import make_logo_pixmap
from ui.preview_pane import PreviewPane
from ui.process_view import ProcessView
from ui.layout_view import LayoutView
from ui.queue_pane import ProcessQueuePane
from ui.layout_left_pane import LayoutLeftPane
from ui.model_download_dialog import ModelDownloadDialog
from ui.mirror_dialog import MirrorManagerDialog

from core.settings_store import SettingsStore, DEFAULT_SETTINGS, get_default_settings_path
from core.mirror_manager import MirrorManager

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_data_dir() -> str:
    """「用户可写数据」的根目录：设置 JSON 与模型下载目录。

    源码运行时就是项目根目录，行为与改动前完全一致。

    打包运行（PyInstaller frozen）时切到 %LOCALAPPDATA%\\证件照制作工具：
    因为打包后 _BASE_DIR 指向 dist\\...\\_internal，有两个硬伤——
      1. 装到 Program Files 时该目录不可写，设置存不下、模型也下不了；
      2. 每次重新打包都会清空 _internal，用户已下载的模型跟着丢失。
    只有读写用户数据的地方走这里；随程序分发的只读配置
    （如 model_download_config.json）仍用 _BASE_DIR。
    """
    if getattr(sys, "frozen", False):
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        d = os.path.join(base, "证件照制作工具")
        try:
            os.makedirs(d, exist_ok=True)
            return d
        except OSError:
            return _BASE_DIR
    return _BASE_DIR


_DATA_DIR = _resolve_data_dir()
DEFAULT_MODEL_DIR = os.path.join(_DATA_DIR, "model")

# ============================================================
#  主线程回调调度器
# ------------------------------------------------------------
#  后台 worker 线程不能直接用 QTimer.singleShot 调度 UI 回调：
#  QTimer.singleShot 的 timer 依附于"调用它的线程"，而 worker 线程
#  没有运行事件循环，回调永远不会触发（实测会静默失效，导致处理完成后
#  UI 不更新、看起来"卡住没反应"）。
#  这里用一个在主线程运行的 QTimer 轮询线程安全队列，worker 线程只往
#  队列 put 回调，由主线程取出执行，彻底规避该陷阱。
# ============================================================
class MainThreadInvoker(QObject):
    def __init__(self, interval_ms=20):
        super().__init__()
        self._queue = queue.Queue()
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._drain)
        self._timer.start()

    def invoke_later(self, func):
        self._queue.put(func)

    def stop(self):
        """窗口关闭时调用：停掉轮询定时器并清空待执行回调，避免线程/定时器残留。"""
        self._timer.stop()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _drain(self):
        while True:
            try:
                func = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                func()
            except Exception:
                traceback.print_exc()

IMAGE_FILE_FILTER = " ".join(ext for _label, ext in IMAGE_EXTENSIONS if ext.startswith("*."))


# ============================================================
#  右侧「缩略图」区：结果缩略图 / 排版页缩略图 切换
# ============================================================
class RightThumbArea(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.c = controller
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        card = theme.SectionCard("缩略图")
        self.result_list = widgets.ThumbnailList(
            thumb_size=80, horizontal=True, name_max_len=14)
        self.page_list = widgets.ThumbnailList(
            thumb_size=80, horizontal=True, name_max_len=14)
        self.result_list.itemSelected.connect(lambda i: self.c.on_result_select(i))
        self.result_list.itemDoubleClickedSig.connect(lambda i: self.c.on_result_zoom(i))
        self.page_list.itemSelected.connect(lambda i: self.c.on_layout_page_select(i))
        self.page_list.itemDoubleClickedSig.connect(lambda i: self.c.on_layout_page_select(i))

        self.result_ph = theme.label("tip", "（批量 / 单张处理后，此处显示处理后的证件照缩略图）")
        self.result_ph.setAlignment(Qt.AlignCenter)
        self.page_ph = theme.label("tip", "（排版后，此处显示页面缩略图）")
        self.page_ph.setAlignment(Qt.AlignCenter)

        self.stack = QStackedWidget()
        rp = QWidget(); rpv = QVBoxLayout(rp); rpv.setContentsMargins(0, 0, 0, 0)
        rpv.addWidget(self.result_list, 1); rpv.addWidget(self.result_ph, 1)
        pp = QWidget(); ppv = QVBoxLayout(pp); ppv.setContentsMargins(0, 0, 0, 0)
        ppv.addWidget(self.page_list, 1); ppv.addWidget(self.page_ph, 1)
        self.stack.addWidget(rp); self.stack.addWidget(pp)
        card.addWidget(self.stack, 1)
        v.addWidget(card)

    def set_mode(self, mode):
        if mode == "result":
            self.stack.setCurrentIndex(0)
        else:
            self.stack.setCurrentIndex(1)

    def refresh_results(self, items):
        """items: [(name, image), ...]"""
        if not items:
            self.result_list.hide()
            self.result_ph.show()
            return
        self.result_ph.hide()
        self.result_list.show()
        self.result_list.set_items(items)

    def refresh_pages(self, pages):
        if not pages:
            self.page_list.hide()
            self.page_ph.show()
            return
        self.page_ph.hide()
        self.page_list.show()
        items = [(f"第 {i + 1} 页", img) for i, img in enumerate(pages)]
        self.page_list.set_items(items)


# ============================================================
#  主控制器
# ============================================================
class IDPhotoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("证件照制作工具")

        # ===== 状态数据 =====
        self.batch_images = []
        self.current_image_idx = -1
        self.matting_image = None
        self.standard_image = None
        self.hd_image = None
        self.standard_matting = None
        self.layout_pages = []
        self.layout_queue = []
        self.processed_results = {}
        self.ready_photos = []
        self.selected_ready_idx = -1
        self.current_page = None
        self.processing = False
        self._layout_busy = False        # 排版生成中（后台线程），防重入
        self._export_busy = False        # 排版图导出中（后台线程），防重入
        self._ready_import_busy = False  # 成品文件夹导入中（后台线程），防重入
        self._stop_batch = False
        self._pause_evt = threading.Event()  # 批量暂停/继续控制（set=运行，clear=暂停）
        self._pause_evt.set()
        self.preview_view = "auto"
        self._gallery_selected_idx = None
        self._result_thumb_indices = []
        self._live_gen = 0               # 实时预览代际号：只应用最新一次的过期结果丢弃
        self._live_lock = threading.Lock()   # 实时预览单飞锁（见 on_live_preview）
        self._live_pending = False           # 预览运行中又有新请求 → 结束后补跑一次
        self._imported_paths = set()    # 已导入路径（规范化），O(1) 去重
        self._import_thread = None      # 后台导入线程
        # 原图 LRU 缓存：只保留最近用到的几张全分辨率原图。
        # 旧实现把 item["image"] 永久挂住，3840×2560 一张约 29.5MB，
        # 100 张就是 2.9GB —— 是内存涨到十几 GB 的主要来源之一。
        self._image_cache = OrderedDict()
        self._IMAGE_CACHE_SIZE = 6

        # 主线程回调调度器（必须在主线程构造，其 QTimer 才能驱动 UI 回调）
        self._invoker = MainThreadInvoker()

        # ===== 引擎 =====
        self.model_dir = DEFAULT_MODEL_DIR
        self.model_available = self._check_models(self.model_dir)
        self.engine_manager = EngineManager(self.model_dir)
        self.photo_processor = PhotoProcessor(self.engine_manager)

        # ===== 设置 + 镜像源（本地 JSON） =====
        # 设置写用户数据目录；镜像配置是随程序分发的只读文件，仍从 _BASE_DIR 读
        self._settings = SettingsStore(get_default_settings_path(_DATA_DIR), DEFAULT_SETTINGS)
        self._mirror_mgr = MirrorManager(_BASE_DIR)

        # 用持久化设置（主题模式 / 强调色 / 默认模型）点亮界面，必须在构建控件前调用
        theme.apply_theme(self._settings)
        # 同步 settings 里的默认模型到 engine_manager（设置里改的抠图/人脸模型生效）
        try:
            self.engine_manager.set_matting_model(
                self._settings.get("processing.default_matting_model",
                                   "modnet_photographic_portrait_matting"))
        except Exception:
            traceback.print_exc()
        try:
            self.engine_manager.set_face_model(
                self._settings.get("processing.default_face_model",
                                   "retinaface-resnet50"))
        except Exception:
            traceback.print_exc()

        # ===== 构建 UI =====
        self._build_header()   # 顶部：logo + 标题 + 模型状态
        self._apply_app_identity()  # 用持久化名称/图标点亮顶栏与窗口
        self._build_nav()      # 顶部下方：证件照处理 / 排版打印（独立一行）
        self._build_content()
        self._sync_engine_to_available_models()  # 让引擎加载“本地存在”的默认模型
        self._build_statusbar()
        self._setup_drop_targets()   # 支持拖拽图片到软件任意位置

        # 注册主题变更监听：设置里切浅/深或主题色后，重染顶部 chrome
        # （Fluent 原生控件 + 继承调色板的文本标签由 apply_theme 自动跟随）
        theme.add_theme_listener(self._on_theme_changed)

        self._show_page("process")
        # 启动只刷新「内嵌状态」(红/绿点 + 处理页缺失红字)，不再弹缺失询问框。
        QTimer.singleShot(150, self._update_model_status)
        QTimer.singleShot(200, self.start_refresh_printers)
        self._update_model_status()

    # ----------------------------------------------------------
    #  模型检查
    # ----------------------------------------------------------
    def _check_models(self, model_dir):
        """模型是否可用：每个类别（抠图 / 人脸）至少存在一个本地模型即可使用，
        不要求下载全部模型。具体哪几个缺失由 UI 禁用对应选项呈现。

        注意：MTCNN 不在模型 manifest 里（权重由 mtcnnruntime 自己管理），
        所以必须额外探测运行库是否可导入——否则「没装 mtcnnruntime」时
        它会被当成可用，UI 显示就绪，点处理才抛 ImportError。
        """
        try:
            from model_downloader import list_missing_models, BUILTIN_MANIFEST
            from inference import mtcnn_available
            miss = {m["name"] for m in list_missing_models(model_dir, BUILTIN_MANIFEST)}
            if not mtcnn_available():
                miss.add("mtcnn")
        except Exception:
            miss = set()
        try:
            from styles import MATTING_MODELS, FACE_MODELS
            matting_present = any(k not in miss for k, _ in MATTING_MODELS)
            face_present = any(k not in miss for k, _ in FACE_MODELS)
        except Exception:
            matting_present = face_present = False
        return matting_present and face_present

    def _sync_engine_to_available_models(self):
        """让引擎加载“本地存在”的默认模型，而不是可能缺失的 settings 配置项。

        settings 里若记录了某个缺失模型作为默认，引擎加载会失败；这里回退到
        该类别第一个“存在”的模型，并写回 settings，保证 引擎 / UI 选择 / 配置 三者一致。
        即使全部缺失（两类别都无可用模型）也安全降级：引擎保持空、UI 全部禁用、
        处理时由 _prepare_process 的 model_available 拦截。
        """
        try:
            from model_downloader import list_missing_models, BUILTIN_MANIFEST
            from styles import MATTING_MODELS, FACE_MODELS
            from inference import mtcnn_available
            miss = {m["name"] for m in list_missing_models(self.model_dir, BUILTIN_MANIFEST)}
            if not mtcnn_available():
                miss.add("mtcnn")   # 运行库缺失时不要把默认人脸模型回退到它
        except Exception:
            return

        def pick(cfg_key, models):
            cfg = self._settings.get(cfg_key, models[0][0] if models else None)
            if cfg and cfg not in miss and any(cfg == k for k, _ in models):
                return cfg
            for k, _ in models:
                if k not in miss:
                    return k
            return cfg

        mdl = pick("processing.default_matting_model", MATTING_MODELS)
        fdl = pick("processing.default_face_model", FACE_MODELS)
        try:
            self.engine_manager.set_matting_model(mdl)
        except Exception:
            traceback.print_exc()
        try:
            self.engine_manager.set_face_model(fdl)
        except Exception:
            traceback.print_exc()
        self._settings.set("processing.default_matting_model", mdl)
        self._settings.set("processing.default_face_model", fdl)

    # ----------------------------------------------------------
    #  顶部头部（独占一行）：logo + 标题双行 + 状态卡 + 版本号
    # ----------------------------------------------------------
    def _build_header(self):
        from PySide6.QtWidgets import QFrame
        bar = QWidget()
        bar.setObjectName("headerBar")   # 供 ID 选择器限定背景只染顶栏自身
        h = QHBoxLayout(bar)
        h.setContentsMargins(14, 8, 14, 8)
        h.setSpacing(12)

        # ============ 左侧：logo + 标题双行 ============
        left = QWidget()
        lv = QHBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(10)

        # 新版证件照图标
        self._hdr_logo = QLabel()
        self._hdr_logo.setPixmap(make_logo_pixmap(36))
        self._hdr_logo.setFixedSize(36, 36)
        lv.addWidget(self._hdr_logo, 0, Qt.AlignVCenter)

        # 双行标题区(应用名 + 副标题)
        title_box = QWidget()
        tb = QVBoxLayout(title_box)
        tb.setContentsMargins(0, 0, 0, 0)
        tb.setSpacing(0)

        # 走统一 label("title", ...) 工厂：14px Bold，背景 transparent，
        # 跟 SectionCard 标题、tab 标签同档的视觉权重。
        # 字号用 15px（顶栏稍大些），通过 lbl.style() 不被 QSS 覆盖即可。
        self._hdr_app_name = theme.label("title", "证件照制作工具")
        # 顶栏标题字号比 SectionCard 标题（12px）略大：直接调整 QSS size。
        self._hdr_app_name.setStyleSheet(self._hdr_app_name.styleSheet().replace(
            "font-size:12px", "font-size:15px").replace(
            "font-weight:500", "font-weight:700"))
        tb.addWidget(self._hdr_app_name)

        # 副标题走 label("chrome", ...) 工厂：11px Medium + pal('text2')。
        self._hdr_subtitle = theme.label("chrome", "ID Photo Studio  ·  本地离线处理")
        self._hdr_subtitle.setStyleSheet(
            self._hdr_subtitle.styleSheet()
            + "letter-spacing:0.3px;margin-top:2px;"
        )
        tb.addWidget(self._hdr_subtitle)

        lv.addWidget(title_box, 0, Qt.AlignVCenter)
        left.setFixedHeight(40)
        h.addWidget(left, 0, Qt.AlignVCenter)

        h.addSpacing(16)

        # ============ 中间分隔细线 ============
        self._hdr_sep = QFrame()
        self._hdr_sep.setObjectName("hdrSep")
        self._hdr_sep.setFrameShape(QFrame.VLine)
        self._hdr_sep.setFrameShadow(QFrame.Plain)
        # 分隔线只染自身（ID 选择器限定，无选择器声明会渗给子孙）
        self._hdr_sep.setStyleSheet(
            f"#hdrSep{{color:{theme.pal('border')};"
            f"background-color:{theme.pal('border')};}}")
        self._hdr_sep.setFixedHeight(28)
        h.addWidget(self._hdr_sep, 0, Qt.AlignVCenter)

        h.addStretch(1)

        # ============ 右侧：状态卡(带圆角背景) ============
        self._hdr_status_card = QWidget()
        self._hdr_status_card.setObjectName("hdrStatusCard")
        # 状态卡只染自身（ID 选择器限定；无选择器声明会渗给卡内状态文字）
        self._hdr_status_card.setStyleSheet(
            f"#hdrStatusCard{{background-color:{theme.pal('surface2')};"
            f"border:1px solid {theme.pal('border')};border-radius:6px;}}"
        )
        sc = QHBoxLayout(self._hdr_status_card)
        sc.setContentsMargins(10, 5, 12, 5)
        sc.setSpacing(8)

        self.model_dot = widgets.StatusDot("#22C55E", 9)
        sc.addWidget(self.model_dot, 0, Qt.AlignVCenter)

        # 状态文本两行(模型名 + 状态)
        st_box = QWidget()
        sb = QVBoxLayout(st_box)
        sb.setContentsMargins(0, 0, 0, 0)
        sb.setSpacing(0)

        # 模型名走 label("chrome", ...) 工厂
        self._hdr_st_label = theme.label("chrome", "MODNet")
        self._hdr_st_label.setStyleSheet(
            self._hdr_st_label.styleSheet() + "letter-spacing:0.3px;"
        )
        sb.addWidget(self._hdr_st_label)

        # 模型状态文字（成功/失败色由本控件单独决定，不走 chrome 调色板）
        self.model_status_label = theme.label("chrome", "已就绪")
        self.model_status_label.setStyleSheet(
            self.model_status_label.styleSheet()
            + "color:#22C55E;"
        )
        sb.addWidget(self.model_status_label)

        sc.addWidget(st_box, 0, Qt.AlignVCenter)
        h.addWidget(self._hdr_status_card, 0, Qt.AlignVCenter)

        h.addSpacing(8)

        # 版本号小角标（chrome 工厂 + 容器色背景）
        self._hdr_ver = theme.label("chrome", "v1.0")
        self._hdr_ver.setStyleSheet(
            self._hdr_ver.styleSheet()
            + f"background-color:{theme.pal('bg')};"
              f"border:1px solid {theme.pal('border')};"
              f"border-radius:4px;padding:2px 6px;"
        )
        h.addWidget(self._hdr_ver, 0, Qt.AlignVCenter)

        self._header = bar
        # 顶栏背景跟随主题。
        # ⚠️ 必须用 objectName（ID）选择器限定：Qt 里不带选择器的声明
        #（如 "background-color:X;"）会作用于自身**及全部子孙控件**，
        # 把顶栏底色渗给所有子 QLabel（表现就是「标签下出现一块底色」）。
        # 实测复现见 2026-09-03 诊断（_diag_leak.py）。
        self._header.setStyleSheet(
            f"#headerBar{{background-color:{theme.pal('surface')};}}")

    # ----------------------------------------------------------
    #  软件名称 / 图标（来自设置，保存即生效，并成为下次启动默认值）
    # ----------------------------------------------------------
    def _apply_app_identity(self):
        """应用「软件名称 + 图标」：窗口标题、顶栏名称、窗口/顶栏图标。

        自定义图标路径失效（文件被删/移动）时自动回退到默认 logo。
        """
        name = (self._settings.get("appearance.app_name", "证件照制作工具")
                or "证件照制作工具")
        self.setWindowTitle(name)
        try:
            self._hdr_app_name.setText(name)
        except Exception:
            pass

        icon_path = self._settings.get("appearance.app_icon", "") or ""
        pm = QPixmap()
        if icon_path and os.path.isfile(icon_path):
            pm = QPixmap(icon_path)
        if pm.isNull():
            pm = make_logo_pixmap(36)
        try:
            self._hdr_logo.setPixmap(
                pm.scaled(36, 36, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception:
            pass
        try:
            self.setWindowIcon(QIcon(pm))
        except Exception:
            pass

    # ----------------------------------------------------------
    #  主题样式热刷新（设置里改 浅/深 或 主题色 后即时反映到顶部 chrome）
    # ----------------------------------------------------------
    def _refresh_theme_styles(self):
        """主题切换后刷新顶部 chrome 的自定义 QSS。

        顶栏的标题/副标题/状态/版本号都来自统一 label(kind, ...) 工厂，
        它们已自动注册主题监听（弱引用），切浅/深时会被 _notify_theme_changed
        自动重染。这里只需要重设那些"由顶栏/中央容器自己决定"的非 label 样式：
        顶栏背景、nav 背景、central 背景、分隔线、状态卡容器、版本号容器的
        background（label 工厂只控制 label 内部色，控制不到 label 的容器）。

        注意：所有 setStyleSheet 都只能写 self 自身样式（不能写 "QWidget{...}" 宽选择器），
        否则会把 surface 底色传染给所有子 QWidget（QLabel），造成「标签下方出现一块
        surface 底色」的视觉污染。这是上一次迭代被截图明确指出的 bug。
        """
        p = theme.pal()
        # 容器背景：一律用 objectName（ID）选择器限定自身。
        # ⚠️ Qt 规则：不带选择器的声明作用于自身及全部子孙控件，
        # 会把底色渗给子 QLabel（=「标签下面有一块底色」的真正根因）。
        self._hdr_sep.setStyleSheet(
            f"#hdrSep{{color:{p['border']};background-color:{p['border']};}}")
        self._hdr_status_card.setStyleSheet(
            f"#hdrStatusCard{{background-color:{p['surface2']};"
            f"border:1px solid {p['border']};border-radius:6px;}}")
        self._hdr_ver.setStyleSheet(
            self._hdr_ver.styleSheet().split(";background-color")[0]
            + f";background-color:{p['bg']};"
              f"border:1px solid {p['border']};"
              f"border-radius:4px;padding:2px 6px;"
        )
        # 主窗口背景跟随浅/深模式
        try:
            self._central.setStyleSheet(
                f"#centralArea{{background-color:{p['bg']};}}")
            self._header.setStyleSheet(
                f"#headerBar{{background-color:{p['surface']};}}")
            self._nav_widget.setStyleSheet(
                f"#navBar{{background-color:{p['surface']};}}")
        except Exception:
            pass

    def _on_theme_changed(self):
        """apply_theme 触发：重染顶部 chrome 的自定义 QSS。

        Fluent 原生控件、以及继承 Fluent 调色板文字色的文本标签
        （small_label / tip_label / SectionCard 标题）已由 apply_theme
        自动跟随浅/深，无需逐个处理；这里只补刀标题栏等手写 QSS 部分。
        """
        try:
            self._refresh_theme_styles()
        except Exception:
            pass

    # ----------------------------------------------------------
    #  顶部导航（独占一行，独立于 header 下方）
    # ----------------------------------------------------------
    def _build_nav(self):
        self._nav_widget = QWidget()
        self._nav_widget.setObjectName("navBar")
        # nav 自身背景：必须用 ID 选择器限定（无选择器声明会渗给全部子孙）
        self._nav_widget.setStyleSheet(
            f"#navBar{{background-color:{theme.pal('surface')};}}")
        h = QHBoxLayout(self._nav_widget)
        h.setContentsMargins(12, 4, 12, 6)
        h.setSpacing(8)
        self.top_nav = Pivot()
        for key, text in [("process", "证件照处理"), ("layout", "排版打印")]:
            self.top_nav.addItem(key, text)
        # Pivot 的 currentItemChanged 信号才携带 routeKey；onClick 只收到 bool，故不能用。
        self.top_nav.currentItemChanged.connect(self._show_page)
        h.addWidget(self.top_nav, 0, Qt.AlignVCenter)
        h.addStretch(1)

        # 设置按钮（右侧固定位置，紧凑风）
        from qfluentwidgets import PushButton as _PB, FluentIcon as _FI
        self.settings_btn = _PB(_FI.SETTING.qicon(), "设置")
        self.settings_btn.setToolTip("主题、外观与各项默认设置")
        self.settings_btn.clicked.connect(self._open_settings_dialog)
        h.addWidget(self.settings_btn, 0, Qt.AlignVCenter)

        # 开源按钮（位于「设置」右侧，点击跳转本项目开源仓库）
        self.open_source_btn = _PB(_FI.GITHUB.qicon(), "开源")
        self.open_source_btn.setToolTip("查看本项目开源地址")
        self.open_source_btn.clicked.connect(self._open_source_link)
        h.addWidget(self.open_source_btn, 0, Qt.AlignVCenter)

    # ----------------------------------------------------------
    #  内容区 (3 列): 左 = 队列 / 成品+排版 ｜ 中 = 预览画布 (最大) ｜ 右 = 参数面板
    # ----------------------------------------------------------
    def _build_content(self):
        self._central = QWidget()
        self._central.setObjectName("centralArea")
        # central 自身背景：必须用 ID 选择器限定（无选择器声明会渗给
        # 全部子孙 QLabel，造成卡片内标签出现一块 bg 底色 —— 2026-09-03 实测）
        self._central.setStyleSheet(
            f"#centralArea{{background-color:{theme.pal('bg')};}}")
        root = QVBoxLayout(self._central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._header)    # 行 1：logo + 标题 + 模型状态
        root.addWidget(self._nav_widget)  # 行 2：顶部 Pivot 导航

        splitter = QSplitter(Qt.Horizontal)

        # ===== 左列：根据当前页面在「处理队列 / 成品 + 排版」之间切换 =====
        self.left_stack = QStackedWidget()
        self.queue_pane = ProcessQueuePane(self)
        self.layout_pane = LayoutLeftPane(self)
        self.left_stack.addWidget(self.queue_pane)    # idx 0: 处理队列
        self.left_stack.addWidget(self.layout_pane)   # idx 1: 成品 + 排版
        # 左栏 220~300：窗口变宽时允许轻微变宽，但不抢预览区
        self.left_stack.setMinimumWidth(220)
        self.left_stack.setMaximumWidth(300)
        splitter.addWidget(self.left_stack)

        # ===== 中列：预览画布 (最大) + 下方缩略图条 =====
        self.preview_pane = PreviewPane()
        self.preview_pane.set_view_changed_callback(self._on_view_changed)
        self.right_thumb = RightThumbArea(self)

        middle = QWidget()
        mv = QVBoxLayout(middle)
        mv.setContentsMargins(8, 8, 8, 8)
        mv.setSpacing(8)
        mv.addWidget(self.preview_pane, 1)
        # 缩略图条固定 120-160px, 仅显示一行
        self.right_thumb.setMaximumHeight(160)
        self.right_thumb.setMinimumHeight(120)
        mv.addWidget(self.right_thumb, 0)
        splitter.addWidget(middle)

        # ===== 右列：根据当前页面在「处理参数面板 / 排版参数面板」之间切换 =====
        self.page_stack = QStackedWidget()
        self.pages = {
            "process": ProcessView(self),
            "layout": LayoutView(self),
        }
        self.page_stack.addWidget(self.pages["process"])
        self.page_stack.addWidget(self.pages["layout"])
        # 右栏 360~480：
        #  · 下限 360 来自参数面板里最宽的一行（"目标尺寸"下拉 200px + 标签 + 边距，
        #    实测页面 minimumSizeHint 约 358px）。窗口只有 320 宽时控件会被裁切/重叠，
        #    这也是"调整窗口大小后目标尺寸下拉框位置偏移"的直接原因。
        #  · 上限 480 让窗口变宽时参数面板按比例变宽一些，而不是永远钉在 320px。
        self.page_stack.setMinimumWidth(360)
        self.page_stack.setMaximumWidth(480)
        splitter.addWidget(self.page_stack)

        # 拉伸因子：左 0（固定）｜ 中 1 ｜ 右 1（两栏按比例分剩余空间）
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)
        # 初始宽度：左 240 / 中自适应 / 右 360（右栏取内容下限，保证不裁切）
        splitter.setSizes([240, 900, 360])
        # 记下 splitter：窗口缩放时按比例重排三栏（否则右栏永远钉在初始宽度，
        # 窗口变宽只有预览区变宽，参数面板里的控件位置相对"跑偏"）。
        self._splitter = splitter
        # 用户手动拖过分隔条后，按他拖出来的比例走
        self._split_ratios = None
        splitter.splitterMoved.connect(self._on_splitter_moved)
        # 防止左/右被拖到完全 0
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setCollapsible(2, False)

        root.addWidget(splitter, 1)

        self.setCentralWidget(self._central)

        # 屏幕自适应
        screen = QApplication.primaryScreen()
        sw, sh = (screen.availableGeometry().width(),
                  screen.availableGeometry().height()) if screen else (WINDOW_WIDTH, WINDOW_HEIGHT)
        w = min(WINDOW_WIDTH, max(1050, sw - 40))
        h = min(WINDOW_HEIGHT, max(700, sh - 80))
        self.resize(w, h)
        self.setMinimumSize(1000, 700)

    # ----------------------------------------------------------
    #  三栏按比例重排（窗口缩放时同步）
    # ----------------------------------------------------------
    def _on_splitter_moved(self, pos, index):
        """用户拖过分隔条：记住他设定的比例，后续窗口缩放沿用。"""
        sizes = self._splitter.sizes()
        total = sum(sizes)
        if total > 0:
            self._split_ratios = [s / total for s in sizes]

    def _apply_split_layout(self):
        """按当前宽度重新分配三栏宽度。

        默认比例 左:中:右 ≈ 16% : 60% : 24%，并夹在各自 min/max 之间：
         · 左栏 220~300（队列/成品列表，太宽会抢预览）
         · 右栏 360~480（参数面板；下限来自"目标尺寸"那一行的实际最小宽度）
        剩余空间全给中栏（预览画布）。
        不做这一步的话，窗口变宽/变窄时只有预览区在变，参数面板固定 320px，
        面板里的控件相对位置就"跑偏"了。
        """
        sp = getattr(self, "_splitter", None)
        if sp is None:
            return
        width = sp.width()
        if width <= 0:
            return
        ratios = getattr(self, "_split_ratios", None) or [0.16, 0.60, 0.24]
        left = int(width * ratios[0])
        right = int(width * ratios[2])
        left = max(220, min(300, left))
        right = max(360, min(480, right))
        middle = max(1, width - left - right)
        sp.setSizes([left, middle, right])

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_split_layout()

    # ----------------------------------------------------------
    #  状态栏
    # ----------------------------------------------------------
    def _build_statusbar(self):
        bar = self.statusBar()
        # 状态栏文字走统一 label("field", ...) 工厂，背景 transparent、字色 pal('text')
        self.status_label = theme.label("field", "")
        self.status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.addWidget(self.status_label, 1)
        self.progress_label = theme.label("field", "")
        bar.addPermanentWidget(self.progress_label)

    # ----------------------------------------------------------
    #  页面路由 (3 列布局: 同时切换 左stack / 右stack / 顶部 Pivot / 缩略图模式)
    # ----------------------------------------------------------
    def _show_page(self, page_id):
        if page_id not in self.pages:
            return
        self.current_page = page_id
        # 右列: 切换处理参数 / 排版参数
        self.page_stack.setCurrentWidget(self.pages[page_id])
        # 左列: 切换队列 / 成品+排版
        self.left_stack.setCurrentIndex(0 if page_id == "process" else 1)
        self.left_stack.setVisible(True)
        # 顶部 Pivot 高亮
        self.top_nav.setCurrentItem(page_id)
        # 下方缩略图条切换模式
        self.right_thumb.set_mode("result" if page_id == "process" else "layout")
        if page_id == "process":
            self._refresh_result_thumbnails()
        else:
            self.right_thumb.refresh_pages(self.layout_pages)
        # 切页时把预览视图重置为「自动」，让预览跟随当前页面
        # （处理页→证件照，排版页→成品照/排版图），
        # 否则会停在上一页手动选的视图上，看起来像"切不动"。
        self.preview_pane.set_view("auto")
        # 同步刷新预览
        self._refresh_right_previews()

    def _on_view_changed(self, view):
        self.preview_view = view
        self._refresh_right_previews()

    # ----------------------------------------------------------
    #  状态栏辅助
    # ----------------------------------------------------------
    def _set_status(self, text):
        max_chars = 72
        s = str(text or "")
        if len(s) > max_chars:
            cut = max(s.rfind("\\"), s.rfind("/"))
            s = s[:max_chars] + ("…" + s[cut:] if cut > 0 else "")
        self.status_label.setText(s)

    def _set_progress(self, text):
        self.progress_label.setText(text)

    def _update_model_status(self, in_use=False):
        color = "#22C55E" if self.model_available else "#EF4444"
        self.model_dot.set_color(color)
        if not self.model_available:
            text, text_color = "无可用模型", "#EF4444"
        elif in_use:
            text, text_color = "使用中…", "#F97316"
        else:
            text, text_color = "已就绪（可用现有模型）", "#22C55E"
        self.model_status_label.setText(text)
        # 只改色（不再重设 font-size/weight，由 label 工厂保证；也保留 background:transparent）
        cur = self.model_status_label.styleSheet()
        # 用正则就地改 color: 段（保持其它样式不动）
        import re as _re
        cur = _re.sub(r"color:\s*#[0-9A-Fa-f]+", f"color:{text_color}", cur)
        if "color:" not in cur:
            cur += f"color:{text_color};"
        self.model_status_label.setStyleSheet(cur)
        # 同步模型页内嵌状态；把「缺失列表」返回给调用方复用。
        # 原实现让启动阶段对模型目录扫描了 3 次（init 一次 + 延迟任务里两次），
        # 这里改为扫一次、结果共享。
        miss = []
        try:
            from model_downloader import list_missing_models, BUILTIN_MANIFEST
            from inference import mtcnn_available
            miss = list_missing_models(self.model_dir, BUILTIN_MANIFEST)
            if not mtcnn_available():
                # 合成一条"缺失"项，让状态文字与选项禁用保持一致
                miss.append({"name": "mtcnn", "path": "",
                             "info": {"filename": "MTCNN（未安装 mtcnn-runtime）"}})
            inline = ("✓ 所有模型已就绪" if not miss
                      else "⚠ 部分模型缺失，仍可使用现有模型：" + ", ".join(
                          m["info"].get("filename", m["name"]) for m in miss[:3])
                      + (f" 等 {len(miss)} 个" if len(miss) > 3 else ""))
            # 「去设置下载」按钮仅在“真正无法使用”（无可用模型）时出现，
            # 部分缺失时不强制弹下载，符合“不必下载全部即可使用”。
            self.pages["process"].set_model_status(
                inline, missing=not self.model_available)
        except Exception:
            pass
        return miss

    def _safe_after(self, ms, func, *args):
        """主线程安全调度。

        注意：不能在此直接调 QTimer.singleShot —— 本方法会被后台 worker 线程调用，
        而 QTimer.singleShot 的定时器依附于调用线程，worker 线程无事件循环会导致
        回调永不触发。统一经主线程的 MainThreadInvoker 队列调度。
        """
        if args:
            func = functools.partial(func, *args)
        self._invoker.invoke_later(func)

    # ----------------------------------------------------------
    #  属性
    # ----------------------------------------------------------
    @property
    def original_image(self):
        if 0 <= self.current_image_idx < len(self.batch_images):
            return self._get_full_image(self.current_image_idx)
        return None

    @property
    def original_file_path(self):
        if 0 <= self.current_image_idx < len(self.batch_images):
            return self.batch_images[self.current_image_idx].get("path")
        return None

    # ----------------------------------------------------------
    #  队列 UI
    # ----------------------------------------------------------
    def _rebuild_queue_ui(self):
        items = [(it["name"],
                  it["thumb"] if it.get("thumb") is not None else it.get("image"))
                 for it in self.batch_images]
        self.queue_pane.update_queue(items)

    def _name_at(self, idx):
        if 0 <= idx < len(self.batch_images):
            return self.batch_images[idx]["name"]
        return f"结果 {idx + 1}"

    # ----------------------------------------------------------
    #  图片导入 / 选择
    # ----------------------------------------------------------
    def on_add_images(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择图片（可多选）", "", f"图片文件 ({IMAGE_FILE_FILTER})")
        if not file_paths:
            return
        self._enqueue_import(file_paths, auto_select_new=False)

    def on_add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not folder:
            return
        exts = set()
        for _label, e in IMAGE_EXTENSIONS:
            e = e.lower().lstrip("*")
            if e:
                exts.add(e)
        paths = []
        for root, _dirs, files in os.walk(folder):
            for fn in sorted(files):
                if os.path.splitext(fn)[1].lower() in exts:
                    paths.append(os.path.join(root, fn))
        if not paths:
            theme.info(self, "warning", "提示", "所选文件夹中没有找到支持的图片。")
            return
        self._enqueue_import(paths, auto_select_new=False)

    def _enqueue_import(self, file_paths, auto_select_new=False):
        """后台线程逐张解码缩略图并增量加入队列，UI 不阻塞（解决大量导入假死）。

        相比原实现：
        - 去重改为 O(1) 集合（原为 O(n²) 全表扫描）
        - 只解码小缩略图用于队列显示；全图按需懒加载（见 _get_full_image）
        - 逐张经 MainThreadInvoker 回到主线程增量追加，300+ 张也不卡界面
        """
        # 去重（O(1)）
        new_paths = []
        for fp in file_paths:
            norm = os.path.normcase(os.path.normpath(fp))
            if norm in self._imported_paths:
                continue
            self._imported_paths.add(norm)
            new_paths.append(fp)
        if not new_paths:
            self._set_status(f"没有新增图片（当前队列 {len(self.batch_images)} 张）")
            return

        self._set_status(f"正在导入 {len(new_paths)} 张图片...")
        first_new_index = len(self.batch_images)

        def _worker():
            try:
                for fp in new_paths:
                    thumb = self._load_thumb(fp)
                    self._invoker.invoke_later(
                        lambda fp=fp, t=thumb: self._commit_imported_item(
                            fp, t, first_new_index, auto_select_new))
                self._invoker.invoke_later(
                    lambda: self._set_status(f"已导入 {len(self.batch_images)} 张图片"))
            except Exception:
                traceback.print_exc()

        self._import_thread = threading.Thread(target=_worker, daemon=True)
        self._import_thread.start()

    def _commit_imported_item(self, fp, thumb, first_new_index, auto_select_new):
        """主线程：把后台解好的缩略图增量追加进队列（不整体重建）。"""
        item = {
            "name": os.path.basename(fp),
            "path": fp,
            "image": None,      # 全图懒加载
            "thumb": thumb,
        }
        self.batch_images.append(item)
        self.queue_pane.append_queue_items([(item["name"], thumb)])
        if auto_select_new and self.current_image_idx < 0:
            self.on_select_queue_image(first_new_index)
        elif self.current_image_idx < 0:
            self.on_select_queue_image(0)

    def _load_thumb(self, file_path, max_side=200):
        """只解码小缩略图（快速），供队列显示；失败返回 None。"""
        try:
            img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            if img is None:
                img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img is None:
                return None
            h, w = img.shape[:2]
            scale = max_side / max(h, w)
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
            return img
        except Exception:
            return None

    def _load_image(self, file_path):
        """读取完整图片，失败返回 None（懒加载时调用）。"""
        try:
            image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
            if image is None:
                image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            return image
        except Exception:
            return None

    def _get_full_image(self, idx):
        """按需解码完整原图（懒加载 + LRU 缓存），用于预览 / 批量处理。

        与旧实现的区别：不再把整张原图永久挂在 item["image"] 上，改为只保留
        最近 6 张。批量处理 100 张 × 3840×2560 时，旧实现常驻约 2.9GB，
        新实现上限约 180MB。仍兼容外部直接注入 item["image"] 的场景（测试用）。
        """
        if not (0 <= idx < len(self.batch_images)):
            return None
        item = self.batch_images[idx]
        # 兼容：调用方/测试直接塞进来的内存图，优先返回
        if item.get("image") is not None:
            return item["image"]
        path = item.get("path")
        if not path:
            return None

        key = item.get("path_key") or os.path.normcase(os.path.normpath(path))
        cached = self._image_cache.get(key)
        if cached is not None:
            self._image_cache.move_to_end(key)
            return cached

        image = self._load_image(path)
        if image is not None:
            self._image_cache[key] = image
            while len(self._image_cache) > self._IMAGE_CACHE_SIZE:
                self._image_cache.popitem(last=False)
        return image

    # ----------------------------------------------------------
    #  拖拽图片到窗口（任意位置可接收）
    # ----------------------------------------------------------
    def _setup_drop_targets(self):
        """让整个软件窗口都能接收图片拖放，且对其它控件零侵入。

        采用「在 QApplication 层装一个全局事件过滤器」的方案：
        - 应用级过滤器能在事件到达目标控件之前截获它，因此**不需要**给每个子控件
          setAcceptDrops(True)（那样会改到预览标签 / 缩略图列表等的事件行为，
          曾导致预览不自动切换的回归）。
        - 只处理落在主窗口范围内的拖放（独立对话框不拦截），对非图片拖拽一律放行。
        """
        self.setAcceptDrops(True)
        QApplication.instance().installEventFilter(self)

    def _extract_image_paths(self, mime):
        """从拖放的 QMimeData 中提取本地图片文件路径（按扩展名过滤）。"""
        # IMAGE_EXTENSIONS 形如 ("PNG 图片", "*.png")，统一去掉可能的 "*" 前缀，
        # 与 os.path.splitext 产出的 ".png" 对齐。
        exts = set()
        for _, e in IMAGE_EXTENSIONS:
            e = e.lower().lstrip("*")
            if e:
                exts.add(e)
        out = []
        for u in mime.urls():
            if not u.isLocalFile():
                continue
            p = u.toLocalFile()
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() in exts:
                out.append(p)
        return out

    def eventFilter(self, obj, event):
        et = event.type()
        if et in (QEvent.DragEnter, QEvent.DragMove, QEvent.Drop):
            # 仅处理主窗口范围内的拖放（独立对话框不拦截，避免误伤其原有行为）
            if not (obj is self or self.isAncestorOf(obj)):
                return super().eventFilter(obj, event)
            mime = event.mimeData()
            if mime is not None and mime.hasUrls():
                paths = self._extract_image_paths(mime)
                if paths:
                    event.acceptProposedAction()
                    if et == QEvent.Drop:
                        self._enqueue_import(paths, auto_select_new=True)
                    return True
            # 非图片拖拽（如文本）→ 不拦截
            return False
        return super().eventFilter(obj, event)

    def on_remove_queue_image(self, idx):
        """删除单张（兼容旧调用：拖拽清理、smoke 测试）。自动选中下一张见 on_remove_queue_images。"""
        self.on_remove_queue_images([idx])

    def on_remove_queue_images(self, rows):
        """批量删除队列中选中的多张；删除后自动选中合适的一张（原位置或末尾），保持预览同步。

        ⚠️ 结果字典 processed_results 以「队列下标」为键，删除后必须整体重排：
        否则删掉中间一张，其后的结果会留在旧下标上 —— 表现为结果缩略图错位、
        选中结果时预览空白（历史回归 bug，见 DEV_GUIDE 的坑表）。
        """
        rows = sorted({int(r) for r in rows if 0 <= int(r) < len(self.batch_images)})
        if not rows:
            return
        names = [self.batch_images[r]["name"] for r in rows]
        anchor = rows[0]  # 删除后从该位置重新选中（被删图顶上的下一张）
        removed = set(rows)

        # 1) 移出队列，同时把这些路径从去重集合里放出来：
        #    否则「删掉一张 → 重新导入同一张」会被当成重复直接忽略。
        for r in reversed(rows):
            item = self.batch_images.pop(r)
            p = item.get("path")
            if p:
                norm = os.path.normcase(os.path.normpath(p))
                self._imported_paths.discard(norm)
                self._image_cache.pop(norm, None)

        # 2) 重排结果下标：被删项丢弃，其后的索引前移（保持与队列一一对应）
        if self.processed_results:
            shifted = {}
            for old_idx, res in self.processed_results.items():
                if old_idx in removed:
                    continue
                shift = sum(1 for r in rows if r < old_idx)
                shifted[old_idx - shift] = res
            self.processed_results = shifted
        # 画廊选中项同样要跟着前移，否则会指到别的图（或越界）上
        if self._gallery_selected_idx is not None:
            gi = self._gallery_selected_idx
            if gi in removed:
                self._gallery_selected_idx = None
            else:
                self._gallery_selected_idx = gi - sum(1 for r in rows if r < gi)

        self._rebuild_queue_ui()
        self._refresh_result_thumbnails()
        if not self.batch_images:
            self.current_image_idx = -1
            self._gallery_selected_idx = None
            self._refresh_right_previews()
        else:
            new_idx = min(anchor, len(self.batch_images) - 1)
            self.current_image_idx = new_idx
            self._gallery_selected_idx = new_idx
            self._select_image_at(new_idx)
        self._set_status(f"已删除 {len(names)} 张，剩 {len(self.batch_images)} 张")

    def on_clear_all_images(self):
        if not self.batch_images:
            return
        self.batch_images.clear()
        self._imported_paths.clear()
        self._image_cache.clear()   # 原图 LRU 一并释放
        self.processed_results.clear()
        self.current_image_idx = -1
        self._rebuild_queue_ui()
        self._refresh_result_thumbnails()
        self._refresh_right_previews()
        self._set_status("已清空图片队列")

    def _clear_results(self):
        self.processed_results.clear()
        self.matting_image = self.standard_image = self.hd_image = None

    def on_select_queue_image(self, idx):
        if not (0 <= idx < len(self.batch_images)):
            return
        self.current_image_idx = idx
        # 同步预览目标：_gallery_selected_idx 决定"证件照"视图显示哪张结果，
        # 不在这里同步的话，点队列缩略图时预览会一直停在上一次处理完那张。
        self._gallery_selected_idx = idx
        self._select_image_at(idx)

    def _select_image_at(self, idx):
        self.queue_pane.select_queue_index(idx)
        self.queue_pane.set_item_active(idx)
        self._refresh_right_previews()

    def on_queue_zoom(self, idx):
        if not (0 <= idx < len(self.batch_images)):
            return
        self.current_image_idx = idx
        self._gallery_selected_idx = idx
        self._refresh_right_previews()

    # ----------------------------------------------------------
    #  模型切换
    # ----------------------------------------------------------
    def on_model_changed(self, model_name):
        try:
            self.engine_manager.set_matting_model(model_name)
        except Exception:
            traceback.print_exc()
        # 记住用户选择，保证下次启动引擎/UI/配置一致
        try:
            self._settings.set("processing.default_matting_model", model_name)
        except Exception:
            pass
        self._update_model_status()

    def on_face_model_changed(self, model_name):
        try:
            self.engine_manager.set_face_model(model_name)
        except Exception:
            traceback.print_exc()
        try:
            self._settings.set("processing.default_face_model", model_name)
        except Exception:
            pass

    # ----------------------------------------------------------
    #  处理参数收集
    # ----------------------------------------------------------
    def _get_process_params(self):
        params = self.pages["process"].get_advanced_params()
        if params is None:
            return None
        params["process_mode"] = self.pages["process"].get_process_mode()
        params["beauty"] = self.pages["process"].get_beauty_params()
        params["watermark"] = self.pages["process"].get_watermark_params()
        return params

    # ----------------------------------------------------------
    #  处理（单张 / 批量）
    # ----------------------------------------------------------
    def _prepare_process(self, require_selection):
        """单张 / 批量处理共用的前置校验：返回参数 dict，失败返回 None。

        require_selection=True  需要当前选中一张图（单张处理）
        require_selection=False 只要队列非空即可（批量处理）
        """
        if (self.original_image is None) if require_selection \
                else (not self.batch_images):
            theme.info(self, "warning", "提示", "请先上传图片。")
            self._show_page("process")
            return None
        if not self.model_available:
            theme.info(self, "error", "模型未找到",
                       f"请将模型文件放在以下目录:\n{DEFAULT_MODEL_DIR}")
            return None
        return self._get_process_params()

    def on_process(self):
        params = self._prepare_process(require_selection=True)
        if params is None:
            return
        self._show_page("process")
        self._start_processing("处理中...")
        self.queue_pane.set_progress(10, "正在加载模型并推理...")
        self.queue_pane.set_item_active(self.current_image_idx)
        self.queue_pane.scroll_to_item(self.current_image_idx)
        # 必须在主线程读取 UI 参数，再传入子线程（FluentWidgets 控件跨线程访问会死锁）
        bgr_color = self.pages["process"].get_background_color_bgr()
        render_mode = self.pages["process"].get_render_mode()
        threading.Thread(target=self._process_worker,
                         args=(self.original_image, params,
                               self.current_image_idx, bgr_color, render_mode),
                         daemon=True).start()

    def on_batch_process(self):
        params = self._prepare_process(require_selection=False)
        if params is None:
            return
        self._show_page("process")
        self._start_processing("批量处理中...")
        self._stop_batch = False
        self._pause_evt.set()  # 初始为运行态
        self.pages["process"].set_batch_controls(True)
        # 必须在主线程读取 UI 参数，再传入子线程
        bgr_color = self.pages["process"].get_background_color_bgr()
        render_mode = self.pages["process"].get_render_mode()
        threading.Thread(target=self._batch_process_worker,
                         args=(params, bgr_color, render_mode), daemon=True).start()

    def on_batch_pause(self):
        """暂停批量处理：worker 在每张前等待 continue 事件。"""
        self._pause_evt.clear()
        self._set_status("批量处理已暂停")
        self.pages["process"].set_batch_controls(True, paused=True)

    def on_batch_resume(self):
        """继续批量处理。"""
        self._pause_evt.set()
        self._set_status("批量处理继续")
        self.pages["process"].set_batch_controls(True, paused=False)

    def on_batch_stop(self):
        """停止批量处理：置停止标志，并放行暂停中的 worker 以便走到 break。"""
        self._stop_batch = True
        self._pause_evt.set()
        self._set_status("正在停止批量处理…")

    def on_live_preview(self):
        """高级参数变化（滑块/留白/目标尺寸）时，对当前图片重新处理并刷新预览。

        防抖已由 ProcessView 的 350ms 定时器完成；本方法在主线程读取 UI 参数后
        交给子线程推理，避免跨线程访问 Fluent 控件。用代际号丢弃过期结果，
        防止快速连续拖动滑块时旧结果覆盖新预览。

        单飞（single-flight）：同一时刻只允许一条预览流水线在跑。
        350ms 防抖只限制「触发频率」，不限制「同时在跑几条」——拖一次滑块
        就能叠出好几条推理。每条都会让 ONNX Runtime 把内存池扩到峰值且不归还
        （实测 6 路并发 1.16GB → 4.9GB，gc 后不回落），这是内存涨到十几 GB
        的主要原因。跑到一半又有新请求时：作废当前结果、记 pending，
        等这条结束再补跑一次最新参数（保证松手后画面是最新的）。
        """
        if self.processing or not self.model_available:
            return
        idx = self.current_image_idx
        if not (0 <= idx < len(self.batch_images)):
            return
        if not self._live_lock.acquire(blocking=False):
            self._live_gen += 1        # 让在飞的那条结果作废
            self._live_pending = True  # 结束后补跑最新参数
            return
        try:
            params = self._get_process_params()
            img = self._get_full_image(idx) if params is not None else None
            if params is None or img is None:
                self._live_lock.release()
                return
            bgr = self.pages["process"].get_background_color_bgr()
            render = self.pages["process"].get_render_mode()
        except Exception:
            self._live_lock.release()
            raise

        self._live_gen += 1
        gen = self._live_gen

        def _worker():
            std = hd = mat = None
            try:
                std, hd, mat = self._process_one(img, params, bgr, render)
                if gen == self._live_gen:
                    self._safe_after(0, lambda: self._apply_live_result(std, hd, mat, idx))
            except Exception:
                # 实时预览属辅助功能，失败不影响主流程
                pass
            finally:
                self._live_lock.release()
                if self._live_pending:
                    self._live_pending = False
                    self._safe_after(0, self.on_live_preview)

        threading.Thread(target=_worker, daemon=True).start()

    def _apply_live_result(self, std, hd, mat, idx):
        self.standard_image, self.hd_image, self.matting_image = std, hd, mat
        if 0 <= idx < len(self.batch_images):
            self.processed_results[idx] = {"std": std, "hd": hd, "matting": mat}
        self._refresh_right_previews()

    def _start_processing(self, status_text):
        self.processing = True
        # 作废可能还在飞的实时预览结果，避免它随后覆盖正式处理结果
        self._live_gen += 1
        self._live_pending = False
        self.queue_pane.set_processing_state(True)
        self.queue_pane.clear_queue_status()
        self._set_status(status_text)
        self._update_model_status(in_use=True)

    def _stop_processing(self):
        self.processing = False
        self.queue_pane.set_processing_state(False)
        self.queue_pane.set_progress(0, "")
        self._set_progress("")
        self._update_model_status(in_use=False)

    def _batch_process_worker(self, params, bgr_color, render_mode):
        """注意：bgr_color / render_mode 必须由主线程读取后传入。

        绝不能在本线程内访问 self.pages["process"] 等 Qt 控件——
        跨线程操作 QWidget 会死锁或崩溃。
        """
        total = len(self.batch_images)
        results = {}
        failures = []  # 收集失败项，处理完统一弹窗
        for idx, item in enumerate(self.batch_images):
            if self._stop_batch:
                break
            # 暂停门：运行中 Event 已 set 直接通过；暂停时阻塞等待，停止时放行以便 break
            if not self._pause_evt.wait(0.05):
                self._safe_after(0, lambda: self.queue_pane.set_progress(-1, "已暂停"))
                while not self._pause_evt.wait(0.1):
                    if self._stop_batch:
                        break
                if self._stop_batch:
                    break
            pct = int((idx / total) * 100)
            self._safe_after(0, lambda p=pct, name=item["name"]:
                             self.queue_pane.set_progress(
                                 p, f"{idx + 1}/{total} {name}"))
            self._safe_after(0, lambda i=idx: self.queue_pane.select_queue_index(i))
            self._safe_after(0, lambda i=idx: self.queue_pane.set_item_active(i))
            self._safe_after(0, lambda i=idx: self.queue_pane.scroll_to_item(i))
            self._safe_after(0, lambda i=idx: self.queue_pane.set_item_progress(i, 0))
            # 阶段进度回调：推理→换底→美颜/水印，经 _safe_after 回主线程刷进度条
            def _item_progress(f, i=idx):
                self._safe_after(0, lambda: self.queue_pane.set_item_progress(
                    i, int(f * 100)))
            try:
                std, hd, matting = self._process_one(
                    self._get_full_image(idx), params, bgr_color, render_mode,
                    progress=_item_progress)
                results[idx] = (std, hd, matting)
                self._safe_after(0, lambda i=idx: self.queue_pane.set_queue_status(i, "✓"))
                self._safe_after(0, lambda i=idx: self.queue_pane.set_item_progress_done(i))
                # 路径在这里取好再传出去：lambda 里的 i 等到主线程执行时队列可能
                # 已被用户删改，届时 self.batch_images[i] 可能越界或指向别的图。
                src_path = (self.batch_images[idx].get("path")
                            if 0 <= idx < len(self.batch_images) else None)
                self._safe_after(0, lambda s=std, m=matting, p=src_path, h=hd:
                                 self._auto_import_ready(s, p, matting=m, hd=h))
            except Exception as e:
                failures.append((item.get("name", "?"), str(e)))
                self._safe_after(0, lambda i=idx: self.queue_pane.reset_item_progress(i))
        if results:
            for i, (std, hd, matting) in results.items():
                self.processed_results[i] = {"std": std, "hd": hd, "matting": matting}
            self._safe_after(0, lambda: self._refresh_result_thumbnails())
            # 批量处理完后自动选中第一个结果并预览（受设置开关控制）
            if self._settings.get("ui.auto_select_first_after_batch", True):
                first_idx = min(results.keys())
                self._safe_after(0, lambda i=first_idx: self._on_gallery_select(i))
        stopped = self._stop_batch
        self._safe_after(0, lambda: self._on_batch_done(failures, stopped))

    def _on_batch_done(self, failures=None, stopped=False):
        self._stop_processing()
        self.pages["process"].set_batch_controls(False)
        if stopped:
            self._set_status("批量处理已停止")
            self._show_page("process")
        else:
            self._set_status("批量处理完成，已自动切换到排版页")
            self._show_page("layout")
        if failures:
            self._write_failures_txt(failures)

    def _write_failures_txt(self, failures):
        """把失败项写入临时 txt 并用系统默认程序打开，完整列出不受长度限制。"""
        import tempfile
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(tempfile.gettempdir(), f"证件照_处理失败_{ts}.txt")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"批量处理失败 {len(failures)} 张\n")
                f.write(f"时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                for n, e in failures:
                    f.write(f"· {n}: {e}\n")
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            self._set_status(f"批量处理完成，{len(failures)} 张失败，详情已写入临时文件")
        except Exception as ex:
            theme.info(self, "error", "无法打开失败清单", str(ex))

    def _process_worker(self, image, params, image_idx, bgr_color, render_mode):
        """bgr_color / render_mode 由主线程读取后传入（禁止本线程访问 Qt 控件）。"""
        def _item_progress(f, i=image_idx):
            self._safe_after(0, lambda: self.queue_pane.set_item_progress(
                i, int(f * 100)))
        try:
            std, hd, matting = self._process_one(
                image, params, bgr_color, render_mode, progress=_item_progress)
            self.standard_matting = matting
            self._safe_after(0, lambda: self._on_process_done(std, hd, matting, image_idx))
            self._safe_after(0, lambda i=image_idx:
                             self.queue_pane.set_queue_status(i, "✓"))
            self._safe_after(0, lambda i=image_idx:
                             self.queue_pane.set_item_progress_done(i))
        except FaceError as e:
            msg = f"人脸检测: {e}"
            self._safe_after(0, lambda i=image_idx:
                             self.queue_pane.reset_item_progress(i))
            self._safe_after(0, lambda m=msg: self._on_process_error(
                m, "检测到多张人脸或无人脸，请上传仅包含单张人脸的照片。"))
        except Exception as e:
            msg = f"处理出错: {e}"
            detail = str(e)
            self._safe_after(0, lambda i=image_idx:
                             self.queue_pane.reset_item_progress(i))
            self._safe_after(0, lambda m=msg, d=detail: self._on_process_error(m, d))

    def _process_one(self, image, params, bgr_color, render_mode, progress=None):
        """progress: 可选回调 progress(frac)，frac∈0~1，用于单项进度条。

        阶段划分：推理 0.1~0.5 → 美颜(人像) 0.55 → 换底 0.7 → 翻转 0.8
                  → 水印 0.9 → 完成 1.0

        顺序有两个必须遵守的约束：
        ① **美颜必须在换底之前、且只作用在人像上**。旧顺序是「先换底再美颜」，
           美白/亮度/对比会把用户选的底色一起改掉（实测 #438EDB 被拉成
           (255,211,131)），导出的底片颜色和界面上选的色不一致。
        ② **翻转必须在加文字水印之前**。旧顺序先画字后整图翻转，水印文字会
           左右反写，看起来像乱码。
        """
        _report = (lambda f: progress(f)) if progress else (lambda f: None)
        _report(0.1)
        result = self.photo_processor.process_id_photo(
            image,
            size=params.get("size"),
            head_measure_ratio=params["head_measure_ratio"],
            head_height_ratio=params["head_height_ratio"],
            head_top_range=params["head_top_range"],
            face_confidence_threshold=params["face_confidence_threshold"],
            face_alignment=params.get("face_alignment", False),
            process_mode=params["process_mode"],
        )
        _report(0.5)
        beauty = params.get("beauty", {})
        std_portrait = self._apply_beauty_bgra(result.standard, beauty)
        hd_portrait = self._apply_beauty_bgra(result.hd, beauty)
        _report(0.6)
        standard_with_bg = add_background_to_image(std_portrait, bgr_color, render_mode)
        hd_with_bg = add_background_to_image(hd_portrait, bgr_color, render_mode)
        _report(0.75)
        if params.get("flip"):
            standard_with_bg = cv2.flip(standard_with_bg, 1)
            hd_with_bg = cv2.flip(hd_with_bg, 1)
        _report(0.85)
        standard_with_bg = self._apply_watermark(standard_with_bg, params.get("watermark", {}))
        hd_with_bg = self._apply_watermark(hd_with_bg, params.get("watermark", {}))
        _report(1.0)
        return standard_with_bg, hd_with_bg, result.matting

    @staticmethod
    def _beauty_active(beauty):
        """美颜参数是否有任何一项非 0（全 0 时跳过，省掉整图 float 运算）。"""
        if not beauty:
            return False
        for k in ("whiten", "brightness", "contrast", "saturation", "sharpen"):
            try:
                if float(beauty.get(k, 0) or 0) != 0:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    @classmethod
    def _apply_beauty_bgra(cls, image, beauty):
        """对 BGRA 抠图的 RGB 通道做美颜，alpha 原样保留。

        只在人像像素上做，合成背景时由 alpha 自然过渡；背景色不受影响。
        """
        if image is None or getattr(image, "ndim", 0) != 3 or image.shape[2] != 4:
            return image
        if not cls._beauty_active(beauty):
            return image
        from image_utils import apply_beauty
        rgb_out = apply_beauty(image[:, :, :3], beauty)
        return np.dstack((rgb_out, image[:, :, 3]))

    @staticmethod
    def _apply_watermark(image, watermark):
        from image_utils import apply_watermark
        return apply_watermark(image, watermark)

    def _on_process_done(self, std_img, hd_img, matting_img, idx=None):
        self.standard_image = std_img
        self.hd_image = hd_img
        self.matting_image = matting_img
        if idx is None:
            idx = self.current_image_idx
        if 0 <= idx < len(self.batch_images):
            self.processed_results[idx] = {
                "std": std_img, "hd": hd_img, "matting": matting_img}
        self._auto_import_ready(
            std_img,
            self.batch_images[idx].get("path") if 0 <= idx < len(self.batch_images) else None,
            matting=matting_img, hd=hd_img)
        self._refresh_result_thumbnails()
        # 不再强制切换视图，保持用户当前选择的模式（默认"自动"）
        self._refresh_right_previews()
        if not self.processing:
            return
        self._stop_processing()
        self._set_status("证件照处理完成")

    def _auto_import_ready(self, image, src_path, matting=None, hd=None):
        """把处理结果写进「成品证件照」列表（排版的数据源）。

        image 是标准照；hd 是高清照，只有**确实比标准照大**时才保留——
        默认「通用构图」下 hd 与 std 同尺寸，存下来只会白占内存（每张约 5MB），
        此时按 None 处理；排版需要放大时才会用到它（见 _pick_layout_base）。
        """
        if image is None:
            return
        if hd is not None:
            try:
                if hd.shape[0] <= image.shape[0] and hd.shape[1] <= image.shape[1]:
                    hd = None
            except Exception:
                hd = None
        if src_path:
            norm = os.path.normcase(os.path.normpath(src_path))
            for it in self.ready_photos:
                if it.get("path") and os.path.normcase(os.path.normpath(it["path"])) == norm:
                    it["image"] = image
                    it["hd"] = hd
                    if matting is not None:
                        it["matting"] = matting
                    break
            else:
                self.ready_photos.append({
                    "name": os.path.basename(src_path), "path": src_path,
                    "image": image, "matting": matting, "hd": hd})
        else:
            name = f"照片_{len(self.ready_photos) + 1}"
            self.ready_photos.append({
                "name": name, "path": "", "image": image,
                "matting": matting, "hd": hd})
        self.layout_pane.refresh_ready_list()

    # ----------------------------------------------------------
    #  结果缩略图 / 画廊
    # ----------------------------------------------------------
    def _refresh_result_thumbnails(self):
        keys = sorted(self.processed_results.keys())
        self._result_thumb_indices = keys
        items = [{
            "name": self._name_at(idx),
            "image": self.processed_results[idx].get("std"),
        } for idx in keys]
        self.right_thumb.refresh_results(
            [(it["name"], it["image"]) for it in items])

    def on_result_select(self, item_idx):
        keys = self._result_thumb_indices
        if 0 <= item_idx < len(keys):
            self._on_gallery_select(keys[item_idx])

    def on_result_zoom(self, item_idx):
        keys = self._result_thumb_indices
        if 0 <= item_idx < len(keys):
            self._on_gallery_zoom(keys[item_idx])

    def _on_gallery_select(self, idx):
        if not (0 <= idx < len(self.batch_images)):
            return
        self.current_image_idx = idx
        self.queue_pane.select_queue_index(idx)
        self._gallery_selected_idx = idx
        self._refresh_right_previews()

    def _on_gallery_zoom(self, idx):
        res = self.processed_results.get(idx)
        if not res or res.get("std") is None:
            return
        self._gallery_selected_idx = idx
        # 不再强制切换视图，保持用户当前选择的模式（默认"自动"）
        self._refresh_right_previews()

    # ----------------------------------------------------------
    #  排版
    # ----------------------------------------------------------
    def _collect_layout_source_photos(self):
        """排版数据源：每张成品带上标准照 / 高清照 / 抠图三份。"""
        photos = []
        for it in self.ready_photos:
            if it.get("image") is not None:
                photos.append({"image": it["image"],
                               "hd": it.get("hd"),
                               "matting": it.get("matting")})
        return photos

    @staticmethod
    def _pick_layout_base(src, target_w, target_h):
        """按目标规格选图：标准照够大就用标准照，需要放大时才用高清照。

        排版会先把源图裁到目标比例、再缩放到 (w, h)。若标准照本身比目标小，
        这一步就是「放大」，画面会虚；此时换用同一张照片的高清版，
        缩放方向变成「缩小」，清晰度明显更好（实测锐度 16.8 → 74.5）。

        只判断"会不会放大"，不做更复杂的评分：规则简单、可预测，
        且源图更大时缩小的结果不会比标准照差。
        """
        std = src.get("image")
        hd = src.get("hd")
        if std is None:
            return hd
        if hd is None:
            return std
        try:
            sh, sw = std.shape[:2]
            need = max(float(target_w) / max(int(sw), 1),
                       float(target_h) / max(int(sh), 1))
        except Exception:
            return std
        return hd if need > 1.0 else std

    # ----------------------------------------------------------
    #  排版：纯计算（不碰 Qt，可放后台线程）+ 结果落地
    # ----------------------------------------------------------
    def _layout_params(self):
        """在主线程读取排版参数（worker 线程禁止访问 Qt 控件）。"""
        lv = self.pages["layout"]
        return {
            "paper": lv.get_paper_px(),
            "gaps": lv.get_photo_interval(),
            "bleed": lv.get_bleed_margin(),
            "crop_line": lv.get_crop_line(),
            "crop_full": lv.get_crop_line_full_page(),
        }

    def _build_layout_items(self, mixed_items_config, source_photos):
        """配置 + 源图 → LayoutItem 列表（纯计算）。

        每张成品会按「规格 × 源图」组合一次：先按目标尺寸挑标准照/高清照，
        需要换底时用 2000px 抠图重新铺底，再居中裁切到规格像素。
        """
        from image_utils import add_background_to_image
        items = []
        for name, w, h, count, bg_hex in mixed_items_config:
            w, h, count = int(w), int(h), int(count)
            if w <= 0 or h <= 0 or count <= 0:
                continue
            for src in source_photos:
                base_img = self._pick_layout_base(src, w, h)
                if bg_hex:
                    matting = src.get("matting")
                    if matting is not None:
                        rgb = hex_to_rgb_tuple(bg_hex)
                        bgr_color = (rgb[2], rgb[1], rgb[0])
                        base_img = add_background_to_image(matting, bgr_color, render_mode=0)
                resized = self._center_crop_to_size(base_img, w, h)
                for _ in range(count):
                    items.append(LayoutItem(image=resized.copy(),
                                            width=w, height=h, label=name))
        return items

    def _compute_layout_pages(self, items, params):
        """LayoutItem 列表 → 画布页（纯计算）。返回 (pages, skipped, labels)。"""
        pw, ph = params["paper"]
        gap_h, gap_v = params["gaps"]
        bleed = params["bleed"]
        engine = LayoutEngine(
            canvas_width=pw, canvas_height=ph,
            photo_interval_h=gap_h, photo_interval_v=gap_v,
            side_interval_w=bleed, side_interval_h=bleed,
        )
        unique_sizes = {(int(it.width), int(it.height)) for it in items}
        if len(unique_sizes) == 1:
            w, h = int(items[0].width), int(items[0].height)
            pages = engine.generate_photos(
                [it.image for it in items], w, h,
                crop_line=params["crop_line"],
                crop_line_full_page=params["crop_full"])
            return pages, [], []

        pages = engine.generate_mixed_size(
            items, crop_line=params["crop_line"],
            crop_line_full_page=params["crop_full"])
        skipped = list(getattr(engine, "last_skipped_sizes", []))
        label_by_size = {}
        for it in items:
            label_by_size.setdefault((int(it.width), int(it.height)),
                                     getattr(it, "label", "") or "")
        labels = [(label_by_size.get(k, ""), v)
                  for k, v in getattr(engine, "last_size_colors", {}).items()]
        return pages, skipped, labels

    def _apply_layout_pages(self, pages, labels, skipped, show):
        """把算好的页交给界面（必须在主线程调用）。"""
        if not pages:
            theme.info(self, "error", "排版失败", "没有可排版的有效照片。")
            return
        self.layout_pages = pages
        self.pages["layout"].set_layout_pages(pages, labels)
        if show:
            self._show_page("layout")
        msg = f"生成排版：共 {len(pages)} 页"
        if skipped:
            msg += "（过大尺寸已跳过：" + "、".join(
                f"{w}x{h}" for w, h in skipped) + "）"
        self._set_status(msg)

    def _rebuild_layout_from_sources(self, show=True, source_photos=None):
        """同步生成排版（测试 / 单张兜底用；界面按钮走异步版本）。"""
        mixed_items_config = self.layout_pane.get_mixed_items()
        if not mixed_items_config:
            theme.info(self, "warning", "提示", "排版页没有配置规格，请先添加至少一个尺寸条目。")
            return False
        if source_photos is None:
            source_photos = self._collect_layout_source_photos()
        if not source_photos:
            theme.info(self, "warning", "提示",
                       "没有可排版的照片。请先在「证件照处理」页面处理照片，"
                       "或在排版页导入成品证件照。")
            return False
        items = self._build_layout_items(mixed_items_config, source_photos)
        if not items:
            theme.info(self, "warning", "提示", "没有有效的排版条目。")
            return False
        self.layout_queue = items
        self._generate_layout_from_queue(show=show)
        return True

    def _generate_layout_from_queue(self, show=False):
        """用已构建的 layout_queue 同步出图（兼容旧调用）。"""
        if not self.layout_queue:
            return
        try:
            pages, skipped, labels = self._compute_layout_pages(
                list(self.layout_queue), self._layout_params())
        except ValueError as e:
            theme.info(self, "error", "排版失败", str(e))
            return
        self._apply_layout_pages(pages, labels, skipped, show)

    def _on_process_error(self, short_msg, detail_msg):
        self._stop_processing()
        self._set_status(f"错误: {short_msg}")
        theme.info(self, "error", "处理失败", detail_msg)

    def on_generate_mixed_layout(self):
        """生成排版（后台线程）：页数多时不再卡死界面。"""
        if getattr(self, "_layout_busy", False):
            return
        mixed_items_config = self.layout_pane.get_mixed_items()
        if not mixed_items_config:
            theme.info(self, "warning", "提示", "请先添加至少一个尺寸条目。")
            return
        if not self.ready_photos:
            if self.standard_image is not None or self.standard_matting is not None:
                src = {
                    "image": self.standard_image if self.standard_image is not None
                    else self.standard_matting,
                    "hd": self.hd_image,
                    "matting": self.standard_matting,
                }
                # 单张兜底：只有一张，同步即可
                self._rebuild_layout_from_sources(show=True, source_photos=[src])
            else:
                theme.info(self, "warning", "提示",
                           "请先在「证件照处理」页面完成证件照制作，"
                           "或在排版页导入成品证件照。")
                self._show_page("process")
            return

        source_photos = self._collect_layout_source_photos()
        params = self._layout_params()   # 必须在主线程读
        self._layout_busy = True
        self._set_status("正在生成排版…")
        QApplication.setOverrideCursor(Qt.WaitCursor)

        def _worker():
            try:
                items = self._build_layout_items(mixed_items_config, source_photos)
                if not items:
                    raise ValueError("没有有效的排版条目。")
                pages, skipped, labels = self._compute_layout_pages(items, params)
            except Exception as e:  # noqa: BLE001 - 统一回到主线程报错
                self._safe_after(0, lambda err=e: self._on_layout_failed(err))
                return
            self._safe_after(0, lambda: self._on_layout_ready(
                items, pages, skipped, labels))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_layout_failed(self, err):
        self._layout_busy = False
        QApplication.restoreOverrideCursor()
        self._set_status("排版失败")
        theme.info(self, "error", "排版失败", str(err))

    def _on_layout_ready(self, items, pages, skipped, labels):
        self._layout_busy = False
        QApplication.restoreOverrideCursor()
        self.layout_queue = items
        self._apply_layout_pages(pages, labels, skipped, show=True)

    def _refresh_layout_thumbnails(self, pages):
        self.right_thumb.refresh_pages(pages)

    def on_layout_page_select(self, idx):
        if 0 <= idx < len(self.layout_pages):
            self.pages["layout"].current_page_idx = idx
            self._refresh_right_previews()

    # ----------------------------------------------------------
    #  成品证件照
    # ----------------------------------------------------------
    def on_add_ready_photos(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择成品证件照（可多选）", "", f"图片文件 ({IMAGE_FILE_FILTER})")
        if not file_paths:
            return
        existing = {os.path.normcase(os.path.normpath(it["path"]))
                    for it in self.ready_photos if it.get("path")}
        added = skipped = 0
        failed = []
        for fp in file_paths:
            norm = os.path.normcase(os.path.normpath(fp))
            if norm in existing:
                skipped += 1
                continue
            try:
                image = self._load_image(fp)
                if image is None:
                    failed.append(os.path.basename(fp))
                    continue
                self.ready_photos.append({
                    "name": os.path.basename(fp), "path": fp,
                    "image": image, "matting": None, "hd": None})
                existing.add(norm)
                added += 1
            except Exception as e:
                failed.append(f"{os.path.basename(fp)}: {e}")
        if added:
            self.layout_pane.refresh_ready_list()
            status = f"已导入 {added} 张成品证件照，共 {len(self.ready_photos)} 张"
            if skipped:
                status += f"（跳过重复 {skipped} 张）"
            self._set_status(status)
        if failed:
            theme.info(self, "warning", "部分图片加载失败",
                       "无法读取以下文件：\n" + "\n".join(failed[:5]))

    def on_add_ready_folder(self):
        """从文件夹批量导入成品证件照（后台加载，界面不卡）。"""
        if getattr(self, "_ready_import_busy", False):
            return
        folder = QFileDialog.getExistingDirectory(self, "选择成品证件照文件夹")
        if not folder:
            return

        exts = set()
        for _label, e in IMAGE_EXTENSIONS:
            e = str(e).lower().lstrip("*")
            if e:
                exts.add(e)
        paths = []
        for root, _dirs, files in os.walk(folder):
            for fn in sorted(files):
                if os.path.splitext(fn)[1].lower() in exts:
                    paths.append(os.path.join(root, fn))
        if not paths:
            theme.info(self, "warning", "提示", "所选文件夹中没有找到支持的图片。")
            return

        existing = {os.path.normcase(os.path.normpath(it["path"]))
                    for it in self.ready_photos if it.get("path")}
        todo = [p for p in paths
                if os.path.normcase(os.path.normpath(p)) not in existing]
        skipped = len(paths) - len(todo)
        if not todo:
            theme.info(self, "info", "提示",
                       f"文件夹里的 {len(paths)} 张图都已在列表中，没有新增。")
            return

        self._ready_import_busy = True
        total = len(todo)
        self._set_status(f"正在导入成品证件照 0/{total}…")
        QApplication.setOverrideCursor(Qt.WaitCursor)

        def _worker():
            loaded, failed = [], []
            for i, fp in enumerate(todo, 1):
                try:
                    img = self._load_image(fp)
                    if img is None:
                        failed.append(os.path.basename(fp))
                    else:
                        loaded.append((fp, img))
                except Exception as e:  # noqa: BLE001
                    failed.append(f"{os.path.basename(fp)}: {e}")
                if i % 5 == 0 or i == total:
                    self._safe_after(0, lambda i=i: self._set_status(
                        f"正在导入成品证件照 {i}/{total}…"))
            self._safe_after(0, lambda: self._on_ready_folder_loaded(
                loaded, failed, skipped))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_ready_folder_loaded(self, loaded, failed, skipped):
        self._ready_import_busy = False
        QApplication.restoreOverrideCursor()
        for fp, img in loaded:
            self.ready_photos.append({
                "name": os.path.basename(fp), "path": fp,
                "image": img, "matting": None, "hd": None})
        if loaded:
            self.layout_pane.refresh_ready_list()
        msg = f"已从文件夹导入 {len(loaded)} 张成品证件照"
        if skipped:
            msg += f"（跳过重复 {skipped} 张）"
        self._set_status(msg if loaded else "没有导入任何图片")
        if failed:
            theme.info(self, "warning", "部分图片加载失败",
                       "无法读取以下文件：\n" + "\n".join(failed[:5]))
        elif not loaded:
            theme.info(self, "warning", "提示", "没有成功导入任何图片。")

    def on_remove_ready_photo(self, idx):
        if 0 <= idx < len(self.ready_photos):
            name = self.ready_photos[idx]["name"]
            self.ready_photos.pop(idx)
            self.layout_pane.refresh_ready_list()
            self._set_status(f"已移除成品照片「{name}」，剩 {len(self.ready_photos)} 张")

    def on_clear_ready_photos(self):
        if not self.ready_photos:
            return
        self.ready_photos.clear()
        self.selected_ready_idx = -1
        self.layout_pane.refresh_ready_list()
        self._set_status("已清空成品证件照列表")

    def on_select_ready_photo(self, idx):
        if 0 <= idx < len(self.ready_photos):
            self.selected_ready_idx = idx
            # 点成品照时把视图切回「自动」（自动模式下排版页优先展示成品照），
            # 这样 Pivot 高亮跟随、且用户随后仍能自由切到 原图/证件照/排版。
            # set_view 若已处于 auto 不会发信号，下面再兜底刷新一次。
            self.preview_pane.set_view("auto")
            self._refresh_right_previews()

    @staticmethod
    def _center_crop_to_size(image, target_w, target_h):
        import cv2
        if image is None or image.size == 0:
            return None
        ih, iw = image.shape[:2]
        if iw < target_w or ih < target_h:
            scale = max(target_w / iw, target_h / ih)
            image = cv2.resize(image, (int(round(iw * scale)), int(round(ih * scale))),
                               interpolation=cv2.INTER_CUBIC)
            ih, iw = image.shape[:2]
        ratio = target_w / target_h
        img_ratio = iw / ih
        if img_ratio > ratio:
            crop_w = int(round(ih * ratio))
            x0 = (iw - crop_w) // 2
            cropped = image[:, x0:x0 + crop_w]
        else:
            crop_h = int(round(iw / ratio))
            y0 = (ih - crop_h) // 2
            cropped = image[y0:y0 + crop_h, :]
        return cv2.resize(cropped, (int(target_w), int(target_h)),
                          interpolation=cv2.INTER_AREA)

    # ----------------------------------------------------------
    #  打印
    # ----------------------------------------------------------
    def _list_printers(self):
        try:
            import win32print
            printers = win32print.EnumPrinters(
                win32print.PRINTER_ENUM_LOCAL | win32print.PRINTER_ENUM_CONNECTIONS)
            names = [p[2] for p in printers if p[2]]
            if names:
                return ["系统默认"] + names
        except Exception:
            pass
        return ["系统默认"]

    def start_refresh_printers(self):
        def _worker():
            try:
                printers = self._list_printers()
            except Exception:
                printers = ["系统默认"]
            self._safe_after(0, lambda: self.pages["layout"].set_printer_list(printers))
        threading.Thread(target=_worker, daemon=True).start()

    def on_printer_properties(self):
        try:
            import win32print
            name = self.pages["layout"].get_printer_name()
            if name is None:
                name = win32print.GetDefaultPrinter()
            hprinter = win32print.OpenPrinter(name)
            try:
                DM_IN_PROMPT = getattr(win32print, "DM_IN_PROMPT", 0x00000004)
                DM_OUT_DEFAULT = getattr(win32print, "DM_OUT_DEFAULT", 0x00000002)
                # 关键修复：devmode 不能传 None —— pywin32 的 DocumentProperties
                # 在 devmode=None + DM_IN_PROMPT 时多数驱动直接报 87(参数错误)。
                # 正确做法：先用 GetPrinter(level=2) 拿驱动当前 DEVMODE，
                # 再把它同时作为输入/输出传进去弹出属性对话框。
                devmode = win32print.GetPrinter(hprinter, 2).get("pDevMode")
                if devmode is None:
                    raise RuntimeError("驱动未返回 DEVMODE，无法打开属性对话框")
                win32print.DocumentProperties(
                    int(self.winId()), hprinter, name,
                    devmode, devmode, DM_IN_PROMPT | DM_OUT_DEFAULT)
            finally:
                win32print.ClosePrinter(hprinter)
        except Exception as e:
            theme.info(self, "warning", "打印机属性",
                       f"无法打开打印机属性：{e}\n\n请在打印排版时通过系统对话框设置纸张/打印机。")

    # ----------------------------------------------------------
    #  导出
    # ----------------------------------------------------------
    def _selected_result(self) -> dict:
        """当前选中项的处理结果（优先结果画廊选中项，其次队列选中项）。

        导出必须按"用户当前看的那张"取图：旧实现直接用 self.standard_image
        等全局状态，批量处理完再点缩略图切换选择时，预览会跟着变、
        导出却仍然是最后处理的那张。
        """
        idx = self._gallery_selected_idx
        if idx is None or idx not in self.processed_results:
            idx = self.current_image_idx
        return self.processed_results.get(idx) or {}

    def on_export(self, export_type):
        layout_page_idx = self.pages["layout"].current_page_idx if self.layout_pages else 0
        if self.layout_pages and 0 <= layout_page_idx < len(self.layout_pages):
            layout_image = self.layout_pages[layout_page_idx]
        else:
            layout_image = None

        res = self._selected_result()

        def _pick(key, fallback):
            v = res.get(key)
            return v if v is not None else fallback

        export_map = {
            "standard": ("通用证件照", _pick("std", self.standard_image)),
            "hd": ("高清证件照", _pick("hd", self.hd_image)),
            "matting": ("透明底抠图", _pick("matting", self.matting_image)),
            "layout": ("排版图", layout_image),
        }
        label, image = export_map.get(export_type, (None, None))
        if image is None:
            display = label or export_type or "图片"
            theme.info(self, "warning", "提示", f"没有可导出的{display}，请先完成相应操作。")
            return

        try:
            dpi = int(self.pages["process"].dpi_combo.currentText())
        except Exception:
            dpi = 300
        try:
            kb_limit = int(self.pages["process"].kb_limit_edit.text() or "0")
        except Exception:
            kb_limit = 0
        fmt = self.pages["layout"].get_format()

        is_rgba = (len(image.shape) == 3 and image.shape[2] == 4)
        if is_rgba:
            ext = "png"
            filetypes = "PNG 图片 (*.png);;JPEG 图片 (*.jpg)"
        else:
            ext = "jpg" if fmt == "jpg" or kb_limit > 0 else "png"
            filetypes = ("JPEG 图片 (*.jpg);;PNG 图片 (*.png)"
                         if ext == "jpg" else "PNG 图片 (*.png);;JPEG 图片 (*.jpg)")

        file_path, _ = QFileDialog.getSaveFileName(
            self, f"保存{label}", f"{label}.{ext}", filetypes)
        if not file_path:
            return

        actual_ext = os.path.splitext(file_path)[1].lower()
        if actual_ext in (".jpg", ".jpeg"):
            force_jpg = True
        elif actual_ext == ".png":
            force_jpg = False
        else:
            force_jpg = (ext == "jpg")

        try:
            if is_rgba or not force_jpg:
                save_image_with_dpi(image, file_path, dpi)
            else:
                save_image_to_jpeg_kb(image, file_path,
                                      kb_limit if kb_limit > 0 else 200, dpi)
            self._set_status(f"已保存: {os.path.basename(file_path)}")
            theme.info(self, "success", "保存成功", f"文件已保存至:\n{file_path}")
        except Exception as e:
            theme.info(self, "error", "保存失败", str(e))

    def on_export_all_layouts(self):
        """把所有排版页一次性导出到所选文件夹（后台线程，逐页回报进度）。

        与「保存排版图」的区别：那个只保存预览中的那一页，多页排版时要一页页点；
        这个按页序导出全部，命名 `排版图_01 / _02 / …`。
        输出格式 / DPI / KB 限制与单张导出共用同一套设置。

        之前是同步写盘：真实照片的 PNG 编码比纯色图慢得多，几十页会让窗口
        完全无响应（用户以为死机、关掉进程，结果一个文件都没留下）。
        现在放到后台线程，并在写完后逐个校验文件是否真的存在。
        同名文件不再弹阻塞确认框，自动加 `_2/_3` 后缀。
        """
        if getattr(self, "_export_busy", False):
            return
        if not self.layout_pages:
            theme.info(self, "warning", "提示",
                       "还没有排版图，请先在「排版规格」里点「排版」生成。")
            return
        folder = QFileDialog.getExistingDirectory(self, "选择排版图导出文件夹")
        if not folder:
            return

        # 预检：先把"目录能不能写"查清楚，避免跑到一半才发现权限/磁盘问题
        probe = os.path.join(folder, ".idphoto_write_test")
        try:
            with open(probe, "wb") as f:
                f.write(b"ok")
            os.remove(probe)
        except Exception as e:  # noqa: BLE001
            self._log_export(f"预检失败 folder={folder!r} {type(e).__name__}: {e}")
            theme.info(self, "error", "文件夹不可写",
                       f"{folder}\n\n{type(e).__name__}: {e}\n\n"
                       "请换一个有写入权限的文件夹（或检查磁盘是否只读/已满）。")
            return

        try:
            dpi = int(self.pages["process"].dpi_combo.currentText())
        except Exception:
            dpi = 300
        try:
            kb_limit = int(self.pages["process"].kb_limit_edit.text() or "0")
        except Exception:
            kb_limit = 0
        fmt = self.pages["layout"].get_format()
        # KB 限制 > 0 时强制 jpg 并压缩到该体积；否则按所选输出格式
        ext = "jpg" if fmt == "jpg" or kb_limit > 0 else "png"

        pages = list(self.layout_pages)
        digits = max(2, len(str(len(pages))))
        targets = [self._unique_export_path(folder, f"排版图_{i:0{digits}d}", ext)
                   for i in range(1, len(pages) + 1)]

        self._export_busy = True
        total = len(pages)
        self._set_status(f"正在导出排版图 0/{total}…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._log_export(f"开始导出 {total} 页 → {folder!r} ext={ext} dpi={dpi} kb={kb_limit}")

        def _worker():
            saved, failed = 0, []
            for i, (out, page) in enumerate(zip(targets, pages), 1):
                try:
                    self._save_layout_page(page, out, ext, kb_limit, dpi)
                    saved += 1
                except Exception as e:  # noqa: BLE001 - 逐页收集，不中断整批
                    failed.append(f"{os.path.basename(out)}: {e}")
                    self._log_export(
                        f"第 {i} 页失败 out={out!r} "
                        f"exists={os.path.exists(out)} "
                        f"folder_isdir={os.path.isdir(folder)} "
                        f"{type(e).__name__}: {e}")
                self._safe_after(0, lambda i=i: self._set_status(
                    f"正在导出排版图 {i}/{total}…"))
            self._log_export(f"导出结束 saved={saved} failed={len(failed)}")
            self._safe_after(0, lambda: self._on_export_layouts_done(
                saved, failed, folder))

        threading.Thread(target=_worker, daemon=True).start()

    @staticmethod
    def _save_layout_page(page, out, ext, kb_limit, dpi):
        """写一页排版图。

        先用 `out.part` 写再原子改名（中途关进程不会留半张图）；
        如果 ".part 写入后校验失败"（个别安全软件对未知扩展名会静默拦截），
        再退一步直接写目标文件。两条路都失败才抛错，错误信息带真实原因。
        """
        tmp = out + ".part"
        errors = []
        for target in (tmp, out):
            try:
                if ext == "jpg":
                    save_image_to_jpeg_kb(page, target, kb_limit, dpi)
                else:
                    save_image_with_dpi(page, target, dpi)
                if not os.path.isfile(target) or os.path.getsize(target) <= 0:
                    raise IOError("写入后文件不存在或为空")
                if target == tmp:
                    os.replace(tmp, out)
                return
            except Exception as e:  # noqa: BLE001 - 记录后尝试下一种写法
                errors.append(f"{os.path.basename(target)}: {type(e).__name__}: {e}")
                try:
                    if target != out and os.path.exists(target):
                        os.remove(target)
                except OSError:
                    pass
        raise IOError("；".join(errors))

    @staticmethod
    def _export_log_path():
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "证件照制作工具", "export_debug.log")

    @classmethod
    def _log_export(cls, msg):
        try:
            path = cls._export_log_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
        except Exception:
            pass

    @staticmethod
    def _unique_export_path(folder, stem, ext):
        """同名时自动加 _2/_3 后缀，避免弹阻塞式覆盖确认。"""
        path = os.path.join(folder, f"{stem}.{ext}")
        n = 2
        while os.path.exists(path) and n <= 999:
            path = os.path.join(folder, f"{stem}_{n}.{ext}")
            n += 1
        return path

    def _on_export_layouts_done(self, saved, failed, folder):
        self._export_busy = False
        QApplication.restoreOverrideCursor()
        log_path = self._export_log_path()
        if saved == 0:
            self._set_status("排版图导出失败")
            detail = "\n".join(failed[:5]) if failed else "未知原因"
            theme.info(self, "error", "导出失败",
                       f"没有写出任何文件：\n{detail}\n\n目标文件夹：{folder}\n"
                       f"排查日志：{log_path}")
            return
        self._set_status(f"已导出排版图 {saved} 张 → {folder}")
        if failed:
            theme.info(self, "warning", "部分导出失败",
                       f"成功 {saved} 张，失败 {len(failed)} 张：\n"
                       + "\n".join(failed[:5]) + f"\n\n文件夹：{folder}\n"
                       f"排查日志：{log_path}")
        else:
            theme.info(self, "success", "导出完成",
                       f"已导出 {saved} 张排版图到:\n{folder}")

    def on_export_batch(self):
        """把批量处理后的「标准证件照」全部导出。

        默认导出到用户选择的文件夹；勾选「覆盖原图」时直接写回源文件路径
        （破坏性操作，覆盖前弹确认框）。
        """
        if not self.processed_results:
            theme.info(self, "warning", "提示",
                       "还没有批量处理结果，请先完成批量处理。")
            return
        overwrite = self.pages["layout"].get_overwrite()
        if overwrite:
            # 覆盖原图：目标路径为源文件本身（不可恢复，需用户二次确认）
            srcs = [self.batch_images[idx].get("path")
                    for idx in sorted(self.processed_results)
                    if 0 <= idx < len(self.batch_images)]
            srcs = [p for p in srcs if p]
            if not srcs:
                theme.info(self, "warning", "提示", "没有可覆盖的源文件路径。")
                return
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("覆盖原图")
            box.setText(
                f"将用处理后的证件照覆盖 {len(srcs)} 个源文件，"
                "原文件会被替换且无法恢复。确定继续？")
            box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            box.button(QMessageBox.Yes).setText("确定覆盖")
            box.button(QMessageBox.No).setText("取消")
            if box.exec() != QMessageBox.Yes:
                return
        else:
            folder = QFileDialog.getExistingDirectory(self, "选择导出文件夹")
            if not folder:
                return
        try:
            dpi = int(self.pages["process"].dpi_combo.currentText())
        except Exception:
            dpi = 300
        try:
            kb_limit = int(self.pages["process"].kb_limit_edit.text() or "0")
        except Exception:
            kb_limit = 0
        fmt = self.pages["layout"].get_format()
        # KB 限制 >0 时强制 jpg 并压缩到该体积；否则按所选输出格式
        ext = "jpg" if fmt == "jpg" or kb_limit > 0 else "png"
        saved = 0
        for idx, res in sorted(self.processed_results.items()):
            img = res.get("std")
            if img is None:
                continue
            if overwrite:
                src = (self.batch_images[idx].get("path")
                       if 0 <= idx < len(self.batch_images) else "")
                if not src:
                    continue
                out = src  # 覆盖原图：直接写回源路径
                is_jpg = os.path.splitext(out)[1].lower() in (".jpg", ".jpeg")
            else:
                base = self._name_at(idx) or f"照片_{idx + 1}"
                name = os.path.splitext(base)[0]
                out = os.path.join(folder, f"{name}.{ext}")
                is_jpg = (ext == "jpg")
            try:
                if is_jpg:
                    # kb_limit>0 压缩到该 KB；kb=0 走默认高质量保存
                    save_image_to_jpeg_kb(img, out, kb_limit, dpi)
                else:
                    save_image_with_dpi(img, out, dpi)
                saved += 1
            except Exception as e:
                theme.info(self, "error", "导出失败", f"{os.path.basename(out)}: {e}")
        target = "原图" if overwrite else folder
        self._set_status(f"批量导出完成：{saved} 张 → {target}")
        theme.info(self, "success", "批量导出完成",
                   f"已{'覆盖原图' if overwrite else '导出到文件夹'} {saved} 张")

    def on_print(self):
        """打印全部排版页（调用系统打印对话框，默认带入已选打印机）。

        多页排版时一次输出所有有效页，每页用 printer.newPage() 分页。
        """
        layout_pages = getattr(self, "layout_pages", None)
        if not layout_pages:
            theme.info(self, "warning", "提示",
                       "没有可打印的排版图，请先在排版页生成排版。")
            return
        # 收集有效（非 None）的页，保留原始序号用于提示
        pages_to_print = [(i, img) for i, img in enumerate(layout_pages)
                          if img is not None]
        if not pages_to_print:
            theme.info(self, "warning", "提示", "没有可打印的图像（所有页均为空）。")
            return

        # 用 QPrinterInfo 把 QPrinter 绑定到「下拉里选中的那台」，而不是 setPrinterName：
        # setPrinterName 拿 win32print 的名字去设，常和 Qt 打印后端内部名不匹配，
        # 导致 QPrinter 处于无效状态 —— pageRect 返回 (0,0) 图像被缩放成 0 尺寸 -> 空白，
        # 同时打印任务落到默认打印机。这就是「选了打印机却打空白/打错机」的根因。
        name = self.pages["layout"].get_printer_name()
        info = QPrinterInfo.printerInfo(name) if name else None
        if info is None or not info.printerName():
            info = QPrinterInfo.defaultPrinter()
        if info is not None and info.printerName():
            printer = QPrinter(info)
        else:
            printer = QPrinter(QPrinter.HighResolution)
        printer.setResolution(300)

        # 是否「直接打印（跳过选择对话框）」：开启且已能确定目标打印机时，
        # 不弹系统对话框，直接发送任务；否则走 QPrintDialog。
        skip = False
        try:
            skip = bool(self._settings.get("print.skip_dialog", False))
        except Exception:
            skip = False

        if not skip:
            dlg = QPrintDialog(printer, self)
            dlg.setWindowTitle("打印排版图")
            # 告诉对话框总页数，用户可在对话框里选范围（选「全部」则 fromPage()/toPage() 均为 0）
            dlg.setMinMax(1, len(pages_to_print))
            if dlg.exec() != QDialog.Accepted:
                return

            # 尊重用户在对话框里设置的页码范围（1-based；都为 0 表示全部）
            fp, tp = dlg.fromPage(), dlg.toPage()
            if fp > 0 and tp > 0 and tp >= fp:
                pages_to_print = [p for p in pages_to_print
                                  if fp - 1 <= p[0] <= tp - 1]

        # 记录本次实际使用的打印机，下次自动选中（与排版页下拉联动）
        try:
            used = printer.printerName()
            if used:
                self._settings.set("print.last_printer", used)
                self._settings.save()
        except Exception:
            pass

        painter = QPainter(printer)
        try:
            for i, (orig_idx, image) in enumerate(pages_to_print):
                qimg = widgets._cv2_to_qimage(image)
                if qimg is None:
                    continue
                if i > 0:
                    printer.newPage()  # 后续页换页，避免叠在同一张纸上
                # 先铺白底：排版图若有透明/alpha 区域，避免打印成黑块或空白
                painter.fillRect(printer.paperRect(QPrinter.DevicePixel), Qt.white)
                page_rect = printer.pageRect(QPrinter.DevicePixel)
                # ⚠️ pageRect 返回 QRectF（浮点尺寸），但 QImage.scaled 只接受 QSize（整数）
                # 必须显式 toSize()，否则类型不匹配导致缩放失败 → 空白输出
                target_size = page_rect.size().toSize()
                # 用 QPixmap 中转打印比 QImage 更可靠（避免 bytes 缓冲区生命周期问题）
                src = QPixmap.fromImage(qimg.copy())
                target = src.scaled(target_size,
                                    Qt.KeepAspectRatio, Qt.SmoothTransformation)
                x = (target_size.width() - target.width()) / 2
                y = (target_size.height() - target.height()) / 2
                painter.drawPixmap(int(x), int(y), target)
        finally:
            painter.end()
        self._set_status(f"已发送打印任务：{printer.printerName()}（{len(pages_to_print)} 页）")

    # ----------------------------------------------------------
    #  模型下载
    # ----------------------------------------------------------
    def open_model_download_dialog(self):
        self._open_download_dialog()

    def open_highspeed_download_dialog(self):
        self._open_download_dialog()

    def open_settings_dialog(self):
        """供页面跳转设置菜单（如模型页「去设置下载」按钮）。"""
        self._open_settings_dialog()

    def refresh_paper_presets(self):
        """设置菜单里增删自定义纸张后，刷新排版页下拉（保持向后兼容）。"""
        try:
            self.pages["layout"]._load_custom_paper_presets()
        except Exception:
            traceback.print_exc()

    def _open_download_dialog(self):
        """两个入口共用同一实现；exec() 结束后显式销毁，避免弹窗对象驻留内存。"""
        dlg = ModelDownloadDialog(self.model_dir, self,
                                  mirror_manager=self._mirror_mgr,
                                  on_finished=self._post_download_refresh)
        try:
            dlg.exec()
        finally:
            dlg.deleteLater()

    def _open_settings_dialog(self):
        """打开「设置」对话框。保存后写回 controller._settings + 实时主题刷新。"""
        from ui.settings_dialog import SettingsDialog
        dlg = SettingsDialog(self, self)
        try:
            dlg.exec()
        finally:
            dlg.deleteLater()

    def _open_source_link(self):
        """打开本项目开源仓库（默认 GitHub 地址，可在设置中覆盖）。"""
        url = self._settings.get("appearance.open_source_url",
                                 "https://github.com/jintianjie/idphoto_app")
        if not url:
            url = "https://github.com/jintianjie/idphoto_app"
        QDesktopServices.openUrl(QUrl(url))

    def _open_mirror_dialog(self):
        """打开「镜像源管理」对话框。"""
        dlg = MirrorManagerDialog(self._mirror_mgr, self)
        try:
            dlg.exec()
        finally:
            dlg.deleteLater()

    def _post_download_refresh(self):
        self.model_available = self._check_models(self.model_dir)
        self._update_model_status()
        # 下载完成后，之前禁用的模型变为可选（不改变当前选择）
        try:
            self.pages["process"].refresh_model_states()
        except Exception:
            pass

    # ----------------------------------------------------------
    #  右侧预览刷新
    # ----------------------------------------------------------
    def _resolve_idphoto_preview(self):
        """解析「证件照」视图该显示哪张图 —— 预览跟随当前选中项。

        选中项优先取结果画廊选中项（点结果缩略图时设置），否则跟随队列选中项。
        已出结果显示 std，未处理回退原图，都没有才用 standard_image。

        这样点任意一张缩略图预览都会跟着走，不会停在上一次处理完那张图上
        （旧实现先拿 standard_image 再被 _gallery_selected_idx 覆盖，
        点队列缩略图不更新 _gallery_selected_idx，预览就不动）。
        """
        idx = self._gallery_selected_idx
        if idx is None or idx not in self.processed_results:
            idx = self.current_image_idx
        res = self.processed_results.get(idx)
        if res and res.get("std") is not None:
            return res.get("std")
        if 0 <= idx < len(self.batch_images):
            return self._get_full_image(idx)
        return self.standard_image

    def _refresh_right_previews(self, page_id=None):
        if page_id is None:
            page_id = self.current_page

        view = self.preview_view
        if view == "auto":
            view = {"process": "idphoto", "layout": "layout"}.get(page_id, "idphoto")

        image = None
        placeholder = "请选择图片并处理"
        # 成品照预览只在排版页 +「自动」视图下生效。
        # 之前这分支是无条件最高优先级，一旦在排版页点过成品照，
        # Pivot（原图/证件照/排版）和下方排版页缩略图就全都切不动了
        # （永远被那张成品照锁死）。现在仅当用户未显式指定视图时才走它。
        if (page_id == "layout" and self.preview_view == "auto"
                and 0 <= self.selected_ready_idx < len(self.ready_photos)):
            placeholder = "成品证件照"
            image = self.ready_photos[self.selected_ready_idx].get("image")
        elif view == "original":
            placeholder = "导入后显示原图"
            if 0 <= self.current_image_idx < len(self.batch_images):
                image = self._get_full_image(self.current_image_idx)
        elif view == "idphoto":
            placeholder = "处理后显示证件照"
            image = self._resolve_idphoto_preview()
        elif view == "layout":
            placeholder = "排版后显示效果图"
            layout_page = self.pages["layout"]
            if self.layout_pages:
                idx = layout_page.current_page_idx if page_id == "layout" else 0
                if 0 <= idx < len(self.layout_pages):
                    image = self.layout_pages[idx]
            else:
                # 无排版结果时显示空白纸张底图
                try:
                    pw, ph = layout_page.get_paper_px()
                    image = np.full((ph, pw, 3), 255, dtype=np.uint8)
                except Exception:
                    image = None

        self.preview_pane.show_image(image, placeholder)

    # ----------------------------------------------------------
    #  窗口关闭：停止后台调度、置停批量处理标志，确保进程干净退出
    # ----------------------------------------------------------
    def closeEvent(self, event):
        self._stop_batch = True
        self.processing = False
        try:
            self._invoker.stop()
        except Exception:
            pass
        # 落盘设置
        try:
            self._settings.save()
        except Exception:
            pass
        super().closeEvent(event)


# ============================================================
#  入口
# ============================================================
def main():
    app = QApplication(sys.argv)
    theme.apply_theme()
    win = IDPhotoApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
