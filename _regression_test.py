# -*- coding: utf-8 -*-
"""回归测试：覆盖本轮修复的缺陷（offscreen 运行，不加载 ONNX 模型，约 5~8 秒）。

运行：
    set QT_QPA_PLATFORM=offscreen
    python _regression_test.py

每个用例对应一个曾经的真实缺陷：
  1. 删除中间图片后，结果字典的索引没有前移（结果缩略图错位）
  2. 删除图片后重新导入同一文件被当成重复拒绝
  3. 美颜在换底之后执行，把用户选的底色一起改掉
  4. 批量导出 JPG + KB=0 时写出的是 PNG 数据
  5. 中心渐变用逐圈 cv2.ellipse 填充（1400×1000 要 511ms）
  6. 实时预览没有并发上限（内存涨到十几 GB 的主因）
  7. 导出取"最后处理的那张"而不是"当前选中的那张"
  8. 高清照比标准照还小 1px
  9. 窗口缩小时参数面板被压到最小高度以下 → 卡片重叠 / 控件错位
 10. 规格清单：删除护照/签证/驾照/社保卡，加入自用1/自用2
 11. 人脸旋转校正默认开启
"""
import os
import sys
import time
import tempfile
import threading
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QScrollArea

RESULT_PASS = []
RESULT_FAIL = []


def check(name, fn):
    try:
        fn()
        RESULT_PASS.append(name)
        print(f"[PASS] {name}")
    except Exception as e:  # noqa
        RESULT_FAIL.append((name, e))
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()


def reset_queue(win):
    """把控制器恢复成"干净队列"，避免用例之间互相污染。"""
    win.batch_images = []
    win.processed_results = {}
    win._imported_paths.clear()
    win._image_cache.clear()
    win.current_image_idx = -1
    win._gallery_selected_idx = None
    win.processing = False


def make_item(name, path=None):
    img = np.full((120, 90, 3), 100, dtype=np.uint8)
    return {"name": name, "path": path, "image": img if path is None else None,
            "thumb": img}


def pump(app, until=None, timeout=3.0):
    """跑一小段事件循环；until 为真值函数时提前退出。"""
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        time.sleep(0.005)
        if until is not None and until():
            break


# ---------------------------------------------------------------- 用例
def test_remove_reindex(win):
    reset_queue(win)
    img = np.full((60, 40, 3), 128, dtype=np.uint8)
    win.batch_images = [make_item(f"p{i}.png") for i in range(3)]
    win.processed_results = {i: {"std": img, "hd": img, "matting": None}
                             for i in range(3)}
    win.current_image_idx = 0
    win._gallery_selected_idx = 0

    win.on_remove_queue_images([1])          # 删中间那张

    names = [b["name"] for b in win.batch_images]
    assert names == ["p0.png", "p2.png"], names
    assert sorted(win.processed_results) == [0, 1], win.processed_results
    # p2 的结果应该落到新下标 1 上，且内容没串
    assert win.processed_results[1]["std"] is img
    assert win._result_thumb_indices == [0, 1], win._result_thumb_indices

    # 再删第 0 张：结果下标要跟着前移，画廊选中项也不能指错人
    win._gallery_selected_idx = 1        # 当前选中 p2
    win.on_remove_queue_images([0])
    assert [b["name"] for b in win.batch_images] == ["p2.png"]
    assert sorted(win.processed_results) == [0]
    assert win._gallery_selected_idx == 0, win._gallery_selected_idx


def test_readd_after_delete(win, app):
    reset_queue(win)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    cv2.imwrite(tmp.name, np.full((80, 60, 3), 60, dtype=np.uint8))
    try:
        win._enqueue_import([tmp.name])
        pump(app, until=lambda: len(win.batch_images) == 1)
        assert len(win.batch_images) == 1, "首次导入失败"

        win.on_remove_queue_images([0])
        assert len(win.batch_images) == 0
        assert len(win._imported_paths) == 0, "_imported_paths 未清理，重加会被判重复"

        win._enqueue_import([tmp.name])      # 同一文件再导入
        pump(app, until=lambda: len(win.batch_images) == 1)
        assert len(win.batch_images) == 1, "删除后重新导入同一文件被拒绝"
    finally:
        reset_queue(win)
        os.unlink(tmp.name)


def test_beauty_keeps_background():
    from app import IDPhotoApp
    from image_utils import add_background_to_image

    BLUE = (219, 142, 67)                    # #438EDB 的 BGR
    bgra = np.zeros((80, 60, 4), dtype=np.uint8)
    bgra[..., :3] = 200
    bgra[..., 3] = 0                         # 默认全透明
    bgra[10:70, 10:50, 3] = 255              # 中间一块"人像"

    beauty = {"whiten": 15, "brightness": 10, "contrast": 10,
              "saturation": 10, "sharpen": 0}
    out = IDPhotoApp._apply_beauty_bgra(bgra, beauty)
    assert np.array_equal(out[..., 3], bgra[..., 3]), "alpha 被改动"
    assert not np.array_equal(out[10:70, 10:50, :3], bgra[10:70, 10:50, :3]), \
        "人像区域没有被美颜"

    composed = add_background_to_image(out, BLUE, 0)
    corner = tuple(int(v) for v in composed[0, 0])
    assert corner == BLUE, f"背景被美颜污染: {corner} != {BLUE}"


def test_batch_jpeg_format():
    from image_utils import save_image_to_jpeg_kb

    img = np.full((80, 60, 3), 120, dtype=np.uint8)
    for kb in (0, 20):
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()
        try:
            save_image_to_jpeg_kb(img, tmp.name, kb, 300)
            with open(tmp.name, "rb") as f:
                head = f.read(3)
            assert head == b"\xff\xd8\xff", \
                f"target_kb={kb} 写出的不是 JPEG（头={head.hex()}）"
        finally:
            os.unlink(tmp.name)


def _old_center_gradient(start, width, height):
    """旧实现（逐圈 ellipse），仅用于一致性对照。"""
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    center = (width // 2, height // 2)
    max_axis = max(height, width)
    for radius in range(max_axis, 0, -1):
        ratio = (max_axis - radius) / max_axis
        val = tuple(start[c] * (1 - ratio) + 255.0 * ratio for c in range(3))
        cv2.ellipse(canvas, center, (radius, radius), 0, 0, 360, val, -1)
    return cv2.split(canvas)


def test_gradient_equivalence_and_speed():
    from image_utils import _generate_gradient

    start = (219, 142, 67)
    w, h = 90, 120
    new = _generate_gradient(start, w, h, "center")
    old = _old_center_gradient(start, w, h)
    for c in range(3):
        diff = float(np.max(np.abs(new[c].astype(np.float32) - old[c])))
        assert diff < 4.0, f"通道 {c} 与旧实现差异过大: {diff}"

    t0 = time.time()
    _generate_gradient(start, 1400, 1000, "center")
    cost = (time.time() - t0) * 1000
    assert cost < 100, f"中心渐变仍太慢: {cost:.0f}ms（旧实现约 511ms）"


def test_live_preview_single_flight(win, app):
    reset_queue(win)
    win.batch_images = [make_item("a.png")]
    win.current_image_idx = 0
    win.model_available = True

    state = {"running": 0, "max": 0, "calls": 0}
    lock = threading.Lock()

    def fake_process(image, params, bgr, render, progress=None):
        with lock:
            state["running"] += 1
            state["calls"] += 1
            state["max"] = max(state["max"], state["running"])
        time.sleep(0.25)
        img = np.full((40, 30, 3), 200, dtype=np.uint8)
        with lock:
            state["running"] -= 1
        return img, img, None

    win._process_one = fake_process
    try:
        for _ in range(5):                  # 模拟快速拖 5 次滑块
            win.on_live_preview()
            time.sleep(0.01)
        pump(app, until=lambda: state["calls"] >= 2 and state["running"] == 0
             and not win._live_pending, timeout=6.0)
        assert state["max"] == 1, f"并发度应恒为 1，实际 {state['max']}"
        assert state["calls"] >= 2, "运行中收到的新请求没有被补跑"
    finally:
        try:
            del win._process_one          # 去掉实例覆盖，恢复类方法
        except AttributeError:
            pass
        pump(app, timeout=0.3)


def test_export_follows_selection(win, app):
    import app as appmod

    reset_queue(win)
    green = np.full((60, 40, 3), (0, 255, 0), dtype=np.uint8)
    red = np.full((60, 40, 3), (0, 0, 255), dtype=np.uint8)
    win.batch_images = [make_item("a.png"), make_item("b.png")]
    win.processed_results = {
        0: {"std": green, "hd": green, "matting": None},
        1: {"std": red, "hd": red, "matting": None},
    }
    win.current_image_idx = 1          # 最后处理的是 b（红）
    win._gallery_selected_idx = 0      # 但用户点选的是 a（绿）

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    orig = appmod.QFileDialog.getSaveFileName
    appmod.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (tmp.name, ""))
    try:
        win.on_export("standard")
        saved = cv2.imread(tmp.name)
        assert saved is not None, "没有写出文件"
        b, g, r = [int(v) for v in saved[30, 20]]
        assert g > 200 and r < 60, f"导出的不是选中项（应绿，实际 BGR=({b},{g},{r})）"
    finally:
        appmod.QFileDialog.getSaveFileName = orig
        os.unlink(tmp.name)


def test_hd_not_smaller_than_std():
    from photo_processor import PhotoProcessor

    # 构造一张"人像"：上半身不透明块 + 人脸框，交给真实裁剪函数
    bgra = np.zeros((1200, 900, 4), dtype=np.uint8)
    bgra[150:1200, 200:700, 3] = 255
    bgra[150:1200, 200:700, :3] = 180
    face_rect = (330, 250, 300, 340)
    proc = PhotoProcessor(None)          # 该用例不触碰 engine_manager
    std, hd = proc._adjust_and_crop(
        bgra, face_rect, (1400, 1000), 0.2, 0.45, (0.12, 0.10))
    assert std.shape[:2] == (1400, 1000), std.shape
    assert hd.shape[0] >= std.shape[0] and hd.shape[1] >= std.shape[1], \
        f"高清照比标准照小: hd={hd.shape} std={std.shape}"


def test_resize_does_not_compress(win, app):
    """窗口缩小到 1000×700 时，内容必须滚动而不是被压到最小高度以下。"""
    import ui.widgets as widgets

    win.resize(1000, 700)
    pump(app, timeout=0.3)

    central = win.centralWidget()
    need = central.minimumSizeHint().height()
    assert need <= central.height(), \
        f"内容最小高度 {need} > 可用高度 {central.height()}，底部会被裁掉"

    assert win.page_stack.minimumWidth() >= 360, \
        f"右栏最小宽度 {win.page_stack.minimumWidth()} 不足 360，控件会被裁切"

    # 处理页的标签页必须包在滚动容器里，且内容没被压缩
    pv = win.pages["process"]
    strip = pv.findChild(widgets.TabStrip)
    assert strip is not None, "找不到 TabStrip"
    scroll = strip.stack.currentWidget()
    assert isinstance(scroll, QScrollArea), "标签页没有包滚动容器（会重叠）"
    inner = scroll.widget()
    assert inner.height() >= inner.minimumSizeHint().height() - 1, \
        f"页面被压缩：{inner.height()} < {inner.minimumSizeHint().height()}"

    # 排版页同样要在滚动容器里
    lv = win.pages["layout"]
    assert lv.findChild(QScrollArea, "layoutPageScroll") is not None, \
        "排版页没有包滚动容器"

    # 窗口变宽 → 右栏按比例变宽（不是永远钉死 360）
    w_before = win.page_stack.width()
    win.resize(1800, 1000)
    pump(app, timeout=0.3)
    w_after = win.page_stack.width()
    assert w_after > w_before, f"窗口变宽后右栏没有跟着变宽：{w_before} -> {w_after}"
    assert w_after <= 480, f"右栏超过上限：{w_after}"

    win.resize(1000, 700)
    pump(app, timeout=0.3)


def test_count_field_visible(win):
    """数量输入框必须能看见数字（SpinBox 的编辑区会被样式表算成 1px）。"""
    import ui.widgets as widgets

    f = win.layout_pane.mixed_count_edit
    assert isinstance(f, widgets.NumberField), f"数量框类型不对: {type(f).__name__}"
    assert f.width() >= 40, f"数量框太窄，数字显示不下: {f.width()}px"
    assert f.text().strip() != "", "数量框没有显示数字"
    assert 1 <= f.value() <= 99, f"数量默认值越界: {f.value()}"

    # 越界输入要被夹回范围内（不能留下 1000 这种值）
    f.setValue(1000)
    assert f.text() == "99", f"越界未夹紧: {f.text()}"
    f.setValue(8)
    assert f.value() == 8

    # 设置对话框里的数值框同样不能用 SpinBox
    from ui.settings_dialog import SettingsDialog
    dlg = SettingsDialog(win, win)
    try:
        for name in ("kb_spin", "layout_bleed_spin", "layout_gap_h_spin",
                     "layout_gap_v_spin", "timeout_spin"):
            sp = getattr(dlg, name, None)
            assert sp is not None, f"设置项 {name} 不存在"
            assert isinstance(sp, widgets.NumberField), \
                f"{name} 仍是 {type(sp).__name__}（数字会看不见）"
            assert sp.width() >= 40, f"{name} 宽度 {sp.width()}px"
            assert sp.text().strip() != "", f"{name} 没有显示数字"
    finally:
        dlg.deleteLater()


def test_preset_sizes():
    """规格清单：删掉护照/签证/驾照/社保卡，加入自用1/自用2（27×36mm、32×42mm @300DPI）。"""
    from styles import PRESET_SIZES, DEFAULT_COUNT_BY_SIZE

    table = {name: (w, h) for name, w, h in PRESET_SIZES}
    for bad in ("护照", "签证", "驾照", "社保卡"):
        assert bad not in table, f"规格里仍然有「{bad}」"

    assert table.get("自用1") == (319, 425), f"自用1 = {table.get('自用1')}"
    assert table.get("自用2") == (378, 496), f"自用2 = {table.get('自用2')}"
    for name in ("自用1", "自用2"):
        assert name in DEFAULT_COUNT_BY_SIZE, f"{name} 缺少默认数量"


def test_layout_option_defaults(win):
    """选项默认值：出血 50、横距 20、纵距 20（新装 / 无配置时）。"""
    import styles
    from core.settings_store import SettingsStore, DEFAULT_SETTINGS

    assert styles.LAYOUT_DEFAULT_BLEED == 50, styles.LAYOUT_DEFAULT_BLEED
    assert styles.LAYOUT_DEFAULT_GAP_H == 20, styles.LAYOUT_DEFAULT_GAP_H
    assert styles.LAYOUT_DEFAULT_GAP_V == 20, styles.LAYOUT_DEFAULT_GAP_V

    defaults = DEFAULT_SETTINGS["layout"]
    assert defaults["default_bleed"] == 50, defaults
    assert defaults["default_gap_h"] == 20, defaults
    assert defaults["default_gap_v"] == 20, defaults

    # 空配置（相当于首次启动）应该拿到 50 / 20 / 20
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    try:
        store = SettingsStore(tmp.name, DEFAULT_SETTINGS)
        assert store.get("layout.default_bleed") == 50
        assert store.get("layout.default_gap_h") == 20
        assert store.get("layout.default_gap_v") == 20
    finally:
        os.unlink(tmp.name)

    # 界面上三个输入框显示的就是配置里的值
    lv = win.pages["layout"]
    s = win._settings
    assert int(lv.bleed_edit.text()) == int(
        s.get("layout.default_bleed", styles.LAYOUT_DEFAULT_BLEED))
    assert int(lv.gap_h_edit.text()) == int(
        s.get("layout.default_gap_h", styles.LAYOUT_DEFAULT_GAP_H))
    assert int(lv.gap_v_edit.text()) == int(
        s.get("layout.default_gap_v", styles.LAYOUT_DEFAULT_GAP_V))


def test_layout_picks_hd_by_target(win):
    """排版按目标规格选图：标准照够大用标准照，需要放大才用高清照。"""
    std = np.full((413, 295, 4), 120, dtype=np.uint8)    # 一寸（小）
    hd = np.full((1037, 741, 4), 200, dtype=np.uint8)    # 高清（大）
    src = {"image": std, "hd": hd, "matting": None}

    # 目标 = 一寸：标准照正好，不需要放大 → 用 std
    assert win._pick_layout_base(src, 295, 413) is std, "不需要放大时应该用标准照"
    # 目标 = 大二寸 413×626：标准照要放大 1.5× → 用 hd
    assert win._pick_layout_base(src, 413, 626) is hd, "需要放大时应该用高清照"
    # 没有高清图 → 退回标准照，不能崩
    assert win._pick_layout_base({"image": std, "hd": None}, 413, 626) is std

    # 高清图不比标准照大时不应占内存（通用构图下两者同尺寸）
    win.ready_photos = []
    same = np.full((100, 80, 4), 90, dtype=np.uint8)
    win._auto_import_ready(same, "", hd=same.copy())
    assert win.ready_photos[-1]["hd"] is None, "hd 不大于 std 时不应保存"
    win._auto_import_ready(std, "", hd=hd)          # std 小、hd 大 → 保留 hd
    assert win.ready_photos[-1]["hd"] is hd, "更大的 hd 应该被保存"

    # 端到端：用大二寸排版，取到的应该是 hd（填充值 200 而不是 std 的 120）
    win.ready_photos = [{"name": "t.png", "path": "", "image": std,
                         "matting": None, "hd": hd}]
    win.layout_pane.mixed_items = [("大二寸", 413, 626, 1, "")]
    assert win._rebuild_layout_from_sources(show=False), "排版生成失败"
    mean = float(win.layout_queue[0].image.mean())
    assert abs(mean - 200) < 2, f"排版没用到高清图（均值 {mean:.0f}，应为 200）"

    win.ready_photos = []
    win.layout_pane.mixed_items = []


def test_ready_list_hd_marker(win):
    """成品列表：有高清版的条目带 HD 标记，没有的不带。"""
    small = np.full((100, 80, 4), 90, dtype=np.uint8)
    big = np.full((200, 160, 4), 90, dtype=np.uint8)
    win.ready_photos = [
        {"name": "plain.png", "path": "", "image": small, "matting": None, "hd": None},
        {"name": "sharp.png", "path": "", "image": small, "matting": None, "hd": big},
    ]
    win.layout_pane.refresh_ready_list()
    lb = win.layout_pane.ready_listbox
    assert lb.count() == 2, lb.count()
    assert "HD" not in lb.item(0).text(), lb.item(0).text()
    assert "HD" in lb.item(1).text(), lb.item(1).text()
    assert lb.item(1).toolTip(), "HD 条目应有说明 tooltip"
    win.ready_photos = []
    win.layout_pane.refresh_ready_list()


def test_export_all_layouts(win, app):
    """批量导出排版图：所有页各写一个文件，命名 排版图_01/02/…。"""
    import app as appmod

    page = np.full((300, 200, 3), 180, dtype=np.uint8)
    win.layout_pages = [page.copy() for _ in range(3)]
    with tempfile.TemporaryDirectory() as tmpdir:
        orig = appmod.QFileDialog.getExistingDirectory
        appmod.QFileDialog.getExistingDirectory = staticmethod(
            lambda *a, **k: tmpdir)
        try:
            win.on_export_all_layouts()
            # 导出已改为后台线程：等它真正结束（_export_busy 复位）
            pump(app, until=lambda: not win._export_busy, timeout=30)
        finally:
            appmod.QFileDialog.getExistingDirectory = orig

        files = sorted(os.listdir(tmpdir))
        assert files == ["排版图_01.png", "排版图_02.png", "排版图_03.png"], files
        for f in files:
            # 文件名含中文：cv2.imread 在 Windows 上读不了非 ASCII 路径，
            # 用 imdecode（和 app._load_image 的做法一致）来校验
            buf = np.fromfile(os.path.join(tmpdir, f), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            assert img is not None, f"{f} 读不出来"
            assert img.shape == (300, 200, 3), f"{f} 尺寸 {img.shape}"

        # 再导一次到同一目录：不能覆盖，也不能因为同名而卡住/不写
        orig = appmod.QFileDialog.getExistingDirectory
        appmod.QFileDialog.getExistingDirectory = staticmethod(
            lambda *a, **k: tmpdir)
        try:
            win.on_export_all_layouts()
            pump(app, until=lambda: not win._export_busy, timeout=30)
        finally:
            appmod.QFileDialog.getExistingDirectory = orig
        files2 = sorted(os.listdir(tmpdir))
        assert len(files2) == 6, f"重复导出应自动加后缀，实际 {files2}"
        assert "排版图_01_2.png" in files2, files2

    # 输出格式切到 JPEG 时，导出应为真 JPEG
    lv = win.pages["layout"]
    old_fmt = lv.get_format()
    lv.export_format_group.set("jpg")
    try:
        with tempfile.TemporaryDirectory() as jdir:
            orig = appmod.QFileDialog.getExistingDirectory
            appmod.QFileDialog.getExistingDirectory = staticmethod(
                lambda *a, **k: jdir)
            try:
                win.on_export_all_layouts()
                pump(app, until=lambda: not win._export_busy, timeout=30)
            finally:
                appmod.QFileDialog.getExistingDirectory = orig
            files = sorted(os.listdir(jdir))
            assert files == ["排版图_01.jpg", "排版图_02.jpg", "排版图_03.jpg"], files
            with open(os.path.join(jdir, files[0]), "rb") as fh:
                assert fh.read(3) == b"\xff\xd8\xff", "JPEG 格式不对"
    finally:
        lv.export_format_group.set(old_fmt)
    win.layout_pages = []


def test_export_part_fallback(win):
    """若 .part 写入被拦截，应自动回退为直接写目标文件。"""
    import app as appmod

    page = np.full((120, 80, 3), 100, dtype=np.uint8)
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "排版图_01.png")
        orig = appmod.save_image_with_dpi

        def flaky(img, path, dpi=300):
            if str(path).endswith(".part"):
                raise PermissionError("模拟安全软件拦截 .part")
            return orig(img, path, dpi)

        appmod.save_image_with_dpi = flaky
        try:
            win._save_layout_page(page, out, "png", 0, 300)
        finally:
            appmod.save_image_with_dpi = orig
        assert os.path.isfile(out) and os.path.getsize(out) > 0, "回退写盘失败"

        # 两条路都失败时必须抛出带原因的异常
        appmod.save_image_with_dpi = lambda *a, **k: (_ for _ in ()).throw(
            PermissionError("全部被拦截"))
        try:
            try:
                win._save_layout_page(page, out, "png", 0, 300)
                raise AssertionError("应当抛异常")
            except IOError as e:
                assert "全部被拦截" in str(e), str(e)
        finally:
            appmod.save_image_with_dpi = orig


def test_quick_add_shortcuts(win):
    """排版规格的两个快捷项：自用2×8、自用1×9，背景默认。"""
    from PySide6.QtWidgets import QPushButton

    llp = win.layout_pane
    llp.mixed_items = []
    llp.mixed_listbox.clear()

    base = llp.mixed_listbox.count()
    llp._quick_add_mixed("自用2", 8)
    assert llp.mixed_items[-1] == ("自用2", 378, 496, 8, ""), llp.mixed_items[-1]
    llp._quick_add_mixed("自用1", 9)
    assert llp.mixed_items[-1] == ("自用1", 319, 425, 9, ""), llp.mixed_items[-1]
    assert llp.mixed_listbox.count() == base + 2

    # 两个快捷按钮要真的在界面上
    texts = [b.text() for b in llp.findChildren(QPushButton)]
    assert any("自用2 ×8" in t for t in texts), texts
    assert any("自用1 ×9" in t for t in texts), texts
    assert any("导入文件夹" in t for t in texts), "成品卡片缺少文件夹导入按钮"

    llp.mixed_items = []
    llp.mixed_listbox.clear()


def test_ready_folder_import(win, app):
    """成品证件照：从文件夹导入，跳过重复，忽略非图片。"""
    import app as appmod

    with tempfile.TemporaryDirectory() as td:
        for name, color in (("a.png", 80), ("b.png", 140)):
            cv2.imwrite(os.path.join(td, name),
                        np.full((120, 90, 3), color, dtype=np.uint8))
        with open(os.path.join(td, "note.txt"), "w", encoding="utf-8") as f:
            f.write("not an image")

        win.ready_photos = []
        orig = appmod.QFileDialog.getExistingDirectory
        appmod.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: td)
        try:
            win.on_add_ready_folder()
            pump(app, until=lambda: not win._ready_import_busy, timeout=20)
            names = {it["name"] for it in win.ready_photos}
            assert names == {"a.png", "b.png"}, names
            assert all(it.get("path") for it in win.ready_photos)
            assert all(it.get("hd") is None for it in win.ready_photos)

            # 再导入一次：全部重复，不应新增
            before = len(win.ready_photos)
            win.on_add_ready_folder()
            pump(app, timeout=0.5)
            assert len(win.ready_photos) == before, "重复文件被再次导入"
        finally:
            appmod.QFileDialog.getExistingDirectory = orig
            win.ready_photos = []
            win.layout_pane.refresh_ready_list()


def test_generate_layout_async(win, app):
    """「排版」按钮走后台线程：结束后 busy 复位、页面已交给界面。"""
    std = np.full((413, 295, 4), 120, dtype=np.uint8)
    win.ready_photos = [{"name": "a.png", "path": "", "image": std,
                         "matting": None, "hd": None}]
    win.layout_pane.mixed_items = [("一寸", 295, 413, 4, "")]
    win.layout_pages = []

    win.on_generate_mixed_layout()
    assert win._layout_busy, "应立刻进入生成中状态（后台线程）"
    pump(app, until=lambda: not win._layout_busy, timeout=30)
    assert not win._layout_busy, "排版线程没有结束"
    assert win.layout_pages, "后台生成没有产出页面"
    assert win.pages["layout"].layout_pages, "界面没有收到排版结果"

    win.ready_photos = []
    win.layout_pane.mixed_items = []
    win.layout_pages = []


def test_default_paper(win):
    """排版默认纸张是 6 寸相纸（152×102mm），画布 1795×1205 @300DPI。"""
    import styles

    assert styles.DEFAULT_PAPER == "6寸相纸", styles.DEFAULT_PAPER
    assert styles.PAPER_PRESETS[0][0] == styles.DEFAULT_PAPER, \
        "DEFAULT_PAPER 必须是 PAPER_PRESETS 里存在的项"

    lv = win.pages["layout"]
    assert lv.paper_combo.currentText() == "6寸相纸", lv.paper_combo.currentText()
    assert lv.get_paper_px() == (1795, 1205), lv.get_paper_px()


def test_defaults_and_dropdowns(win):
    """人脸旋转校正默认开启；两个页面下拉里的规格一致。"""
    pv = win.pages["process"]
    assert pv.rotation_switch.isChecked(), "人脸旋转校正默认应为开启"

    texts = [pv.size_combo.itemText(i) for i in range(pv.size_combo.count())]
    assert any(t.startswith("自用1（") for t in texts), f"处理页缺少自用1: {texts}"
    assert any(t.startswith("自用2（") for t in texts), f"处理页缺少自用2: {texts}"
    for bad in ("护照", "签证", "驾照", "社保卡"):
        assert not any(t.startswith(bad) for t in texts), f"处理页仍有「{bad}」"

    llp = win.layout_pane
    llp._reload_size_combo()
    items = [llp.mixed_size_combo.itemText(i)
             for i in range(llp.mixed_size_combo.count())]
    assert "自用1" in items and "自用2" in items, f"排版页缺少自用规格: {items}"
    for bad in ("护照", "签证", "驾照", "社保卡"):
        assert bad not in items, f"排版页仍有「{bad}」"


def main():
    app = QApplication.instance() or QApplication([])

    # 这些用例需要真实控制器（控制器自身会读设置、建界面，但不加载模型）
    import app as appmod
    win = appmod.IDPhotoApp()
    win.show()                     # 让 Qt 真正走一遍布局（缩放类用例依赖它）
    win.model_available = False    # 避免任何后台预加载
    try:
        check("删除中间图片后结果索引重排", lambda: test_remove_reindex(win))
        check("删除后可重新导入同一文件", lambda: test_readd_after_delete(win, app))
        check("美颜不污染背景色", test_beauty_keeps_background)
        check("批量导出 JPG 格式正确（含 KB=0）", test_batch_jpeg_format)
        check("中心渐变向量化（等价 + 提速）", test_gradient_equivalence_and_speed)
        check("实时预览单飞", lambda: test_live_preview_single_flight(win, app))
        check("导出跟随当前选中项", lambda: test_export_follows_selection(win, app))
        check("高清照不小于标准照", test_hd_not_smaller_than_std)
        check("窗口缩放不压缩控件（滚动代替重叠）",
              lambda: test_resize_does_not_compress(win, app))
        check("规格清单（删 4 项 / 加自用1、自用2）", test_preset_sizes)
        check("数量输入框可见（不用 SpinBox）", lambda: test_count_field_visible(win))
        check("选项默认值（出血50/横距20/纵距20）", lambda: test_layout_option_defaults(win))
        check("排版按规格自动选高清图", lambda: test_layout_picks_hd_by_target(win))
        check("成品列表 HD 标记", lambda: test_ready_list_hd_marker(win))
        check("批量导出排版图（多页）", lambda: test_export_all_layouts(win, app))
        check("排版后台生成（不卡 UI）", lambda: test_generate_layout_async(win, app))
        check("导出 .part 失败自动回退", lambda: test_export_part_fallback(win))
        check("排版规格快捷项（自用2×8 / 自用1×9）", lambda: test_quick_add_shortcuts(win))
        check("成品证件照文件夹导入", lambda: test_ready_folder_import(win, app))
        check("默认纸张为 6 寸相纸", lambda: test_default_paper(win))
        check("旋转校正默认开启 + 下拉一致",
              lambda: test_defaults_and_dropdowns(win))
    finally:
        win._invoker.stop()

    print("\n" + "=" * 56)
    print(f"通过 {len(RESULT_PASS)} / {len(RESULT_PASS) + len(RESULT_FAIL)}")
    for name, _e in RESULT_FAIL:
        print(f"  [FAIL] {name}")
    print("=" * 56)
    return 0 if not RESULT_FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
