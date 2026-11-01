# -*- coding: utf-8 -*-
"""处理结果落盘存储
============================================================
3000 张批量处理时，std(4MB) + hd(4MB) + matting(10MB) 全留在内存约 54GB，
必然耗尽内存。这里把三张图写到临时目录的 PNG，内存只保留缩略图；
预览 / 导出 / 排版按需读取（带少量 LRU 缓存）。

对上层保持「像字典一样」的用法：`res.get("std")` 自动读盘。
"""
import os
import tempfile
import shutil
import time
import uuid
import cv2
from collections import OrderedDict


class ResultStore:
    """一次运行一个临时目录；退出时整体删除。"""

    def __init__(self, max_cached=3, root=None):
        base = os.environ.get("IDPHOTO_RESULT_CACHE")
        self.root = root or os.path.join(
            base or tempfile.gettempdir(), "idphoto_results", uuid.uuid4().hex[:8])
        os.makedirs(self.root, exist_ok=True)
        self._cache = OrderedDict()
        self._max = int(max_cached)

    def put(self, key, std, hd=None, matting=None):
        """把三张图写盘，返回 {kind: path}。"""
        paths = {}
        for name, img in (("std", std), ("hd", hd), ("matting", matting)):
            if img is None:
                continue
            path = os.path.join(self.root, f"{key}_{name}.png")
            try:
                if not cv2.imwrite(path, img):
                    continue
            except Exception:
                continue
            paths[name] = path
        return paths

    def load(self, path):
        if not path:
            return None
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            self._cache[path] = img
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)
        return img

    def drop(self, paths):
        for path in (paths or {}).values():
            self._cache.pop(path, None)
            try:
                os.remove(path)
            except OSError:
                pass

    def clear(self):
        self._cache.clear()
        shutil.rmtree(self.root, ignore_errors=True)

    @staticmethod
    def cleanup_stale(max_age_hours=24):
        """清掉上次异常退出留下的旧目录（启动时调用）。"""
        base = os.path.join(tempfile.gettempdir(), "idphoto_results")
        try:
            now = time.time()
            for name in os.listdir(base):
                d = os.path.join(base, name)
                if not os.path.isdir(d):
                    continue
                if now - os.path.getmtime(d) > max_age_hours * 3600:
                    shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


class StoredResult:
    """行为像 {"std": ndarray, "hd": ndarray, "matting": ndarray} 的轻量对象。

    实际只在 `.get(kind)` 时读盘（LRU 缓存 3 张），另外常驻一个缩略图供列表使用。
    """

    def __init__(self, store, paths, thumb=None, face_info=None, shapes=None):
        self._store = store
        self._paths = dict(paths or {})
        self._thumb = thumb
        self.face_info = face_info
        self.shapes = dict(shapes or {})

    def get(self, kind, default=None):
        if kind == "thumb":
            return self._thumb if self._thumb is not None else default
        path = self._paths.get(kind)
        if not path:
            return default
        img = self._store.load(path)
        return img if img is not None else default

    def __getitem__(self, kind):
        value = self.get(kind)
        if value is None:
            raise KeyError(kind)
        return value

    def __contains__(self, kind):
        return kind == "thumb" or kind in self._paths

    def keys(self):
        return list(self._paths)

    def paths(self):
        return dict(self._paths)


class LazyImage:
    """排版用的延迟加载引用：保存路径与尺寸，需要时才读成 ndarray。"""

    def __init__(self, store, path, shape=None):
        self._store = store
        self.path = path
        # 尺寸直接给出，列表刷新时不必为了 .shape 去读整张图
        self.shape = tuple(shape) if shape else (0, 0, 3)

    def load(self):
        return self._store.load(self.path) if self.path else None

    def __repr__(self):
        return f"LazyImage({self.path})"
