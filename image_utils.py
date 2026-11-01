# -*- coding: utf-8 -*-
"""
图像处理工具函数模块
负责图像格式转换、缩放、裁剪、美颜、水印等底层操作，不涉及业务逻辑。
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

from color_utils import hex_to_rgb_tuple

# 渐变背景的终止色（白色），与起始色做线性混合
GRADIENT_END_COLOR = (255.0, 255.0, 255.0)

# 字体缓存：页面标注每页都要用，重复 truetype 解析很慢
_FONT_CACHE = {}


def _get_cjk_font(size):
    """取一个能显示中文的 TrueType 字体（失败时回退 PIL 默认字体）。"""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    win_dir = os.environ.get("WINDIR", r"C:\Windows")
    for name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc"):
        path = os.path.join(win_dir, "Fonts", name)
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                _FONT_CACHE[size] = font
                return font
            except Exception:
                continue
    font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def draw_bottom_right_label(image_bgr, text, margin=18, font_size=None,
                            text_color=(70, 70, 70), bg_color=(255, 255, 255, 210),
                            raise_ratio=0.5):
    """在图片右下角画一行来源标注（带半透明白底，避免压住照片看不清）。

    raise_ratio：垂直方向额外上移的字高倍数（默认 0.5 = 半个字符高，
    避免标注贴底太紧、裁切/冲印时被修边吃掉）。
    只处理右下角的小块区域（不整页做 RGBA 合成），100 多页也不会明显变慢。
    返回同一张图（原地修改）。
    """
    if image_bgr is None or not text:
        return image_bgr
    height, width = image_bgr.shape[:2]
    size = int(font_size or max(14, round(width / 75)))
    font = _get_cjk_font(size)

    # 量文字尺寸（textbbox 的 left/top 可能非 0，绘制时要减掉）
    pad_x = max(6, size // 3)
    pad_y = max(3, size // 6)
    probe = Image.new("RGB", (1, 1))
    measure = ImageDraw.Draw(probe)

    def _measure(t):
        bb = measure.textbbox((0, 0), t, font=font)
        return bb, bb[2] - bb[0], bb[3] - bb[1]

    bbox, text_w, text_h = _measure(text)
    # 两级目录 + 长文件名可能超出页宽：从头部截断，保留文件名部分
    max_text_w = max(40, width - 2 * margin - 2 * pad_x)
    if text_w > max_text_w:
        while len(text) > 3 and font.getlength("…" + text) > max_text_w:
            text = text[1:]
        text = "…" + text
        bbox, text_w, text_h = _measure(text)

    box_w = min(width, text_w + pad_x * 2)
    box_h = min(height, text_h + pad_y * 2)

    x0 = max(0, width - margin - box_w)
    # 垂直方向比水平多让开半个字符高：标注更靠上，不会被页边裁掉
    y0 = max(0, height - margin - box_h - int(round(size * max(0.0, raise_ratio))))
    x1 = min(width, x0 + box_w)
    y1 = min(height, y0 + box_h)

    roi = image_bgr[y0:y1, x0:x1]
    pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, pil.width - 1, pil.height - 1], fill=bg_color)
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, font=font,
              fill=text_color + (255,))
    image_bgr[y0:y1, x0:x1] = cv2.cvtColor(
        np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    return image_bgr


def resize_image_esp(input_image: np.ndarray, max_side: int = 2000) -> np.ndarray:
    """
    将图像最长边缩放到 max_side，保持宽高比不变。
    若原图最长边已小于 max_side，则原样返回。
    """
    if input_image is None or input_image.size == 0:
        raise ValueError("输入图像为空")
    height = input_image.shape[0]
    width = input_image.shape[1]
    max_dim = max(height, width)
    if max_dim <= max_side:
        return input_image
    if height == max_dim:
        new_width = int((max_side / height) * width)
        new_height = max_side
    else:
        new_height = int((max_side / width) * height)
        new_width = max_side
    return cv2.resize(input_image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def image_to_bgr(input_image: np.ndarray) -> np.ndarray:
    """将任意通道数的图像转为 3 通道 BGR。"""
    if input_image is None:
        raise ValueError("输入图像为空")
    if len(input_image.shape) == 2:
        return cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    if input_image.shape[2] == 4:
        return input_image[:, :, 0:3]
    return input_image




def add_background_to_image(
    matting_image: np.ndarray,
    bgr_color: tuple,
    render_mode: int = 0,
) -> np.ndarray:
    """
    为透明背景的 BGRA 图像添加背景色。
    :param matting_image: 4 通道 BGRA 图像
    :param bgr_color: (B, G, R) 元组
    :param render_mode: 0=纯色, 1=上下渐变, 2=中心渐变
    :return: 3 通道 BGR 图像
    """
    if matting_image is None or matting_image.size == 0:
        raise ValueError("输入图像为空")
    if len(matting_image.shape) != 3 or matting_image.shape[2] != 4:
        raise ValueError("输入图像必须是 4 通道 BGRA 图像")

    height, width = matting_image.shape[:2]
    b, g, r, a = cv2.split(matting_image)
    alpha = a.astype(np.float32) / 255.0

    if render_mode == 0:
        # 纯色背景
        bg_b = np.full((height, width), bgr_color[0], dtype=np.float32)
        bg_g = np.full((height, width), bgr_color[1], dtype=np.float32)
        bg_r = np.full((height, width), bgr_color[2], dtype=np.float32)
    elif render_mode == 1:
        # 上下渐变：从指定色渐变到白色
        bg_b, bg_g, bg_r = _generate_gradient(bgr_color, width, height, mode="updown")
    else:
        # 中心渐变
        bg_b, bg_g, bg_r = _generate_gradient(bgr_color, width, height, mode="center")

    # Alpha 混合: result = foreground * alpha + background * (1 - alpha)
    out_b = b.astype(np.float32) * alpha + bg_b * (1.0 - alpha)
    out_g = g.astype(np.float32) * alpha + bg_g * (1.0 - alpha)
    out_r = r.astype(np.float32) * alpha + bg_r * (1.0 - alpha)

    result = cv2.merge((
        np.clip(out_b, 0, 255).astype(np.uint8),
        np.clip(out_g, 0, 255).astype(np.uint8),
        np.clip(out_r, 0, 255).astype(np.uint8),
    ))
    return result


def _generate_gradient(start_bgr: tuple, width: int, height: int, mode: str = "updown"):
    """生成渐变背景的三个通道。返回 (B, G, R) 各通道的 float32 矩阵。

    两处都用 numpy 向量化实现。旧版「中心渐变」是从最大半径到 1 逐圈
    `cv2.ellipse` 填充（1400×1000 要画 1400 圈，实测 511ms；2000×1400
    要 2145ms），而每张照片要算 2 次（标准照 + 高清照），是处理耗时里
    仅次于人脸检测的一块。向量化后是 O(像素)，同样的图 <10ms。
    """
    end_color = GRADIENT_END_COLOR  # 白色

    if mode == "updown":
        # 上下渐变：y 从 0 → height-1 时，颜色从 start 线性过渡到 end
        ratio = (np.arange(height, dtype=np.float32) / max(height, 1))[:, None]
        channels = []
        for c in range(3):
            ch = np.empty((height, width), dtype=np.float32)
            ch[:] = start_bgr[c] * (1.0 - ratio) + end_color[c] * ratio
            channels.append(ch)
        return tuple(channels)

    # 中心渐变：半径 = 像素到画布中心的欧氏距离，按最长边归一化。
    # 旧实现从大圈往小圈覆盖，等价于「中心 = 终止色(白)，边缘 = 起始色」，
    # 这里保持同样的视觉方向（中心白、四周底色）。
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = width / 2.0, height / 2.0
    max_axis = float(max(height, width, 1))
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    t = np.clip(dist / max_axis, 0.0, 1.0)      # 0 = 中心，1 = 最远角
    channels = []
    for c in range(3):
        channels.append(end_color[c] * (1.0 - t) + start_bgr[c] * t)
    return tuple(channels)


# ============================================================
#  美颜处理（磨皮 + 美白 + 肤色矫正，只作用于皮肤）
# ============================================================

# 正常肤色参考比例（由样张标定；用比例而非绝对值，避免把人“调亮/调暗”）
_SKIN_REF_BG = 0.657   # B / G
_SKIN_REF_RG = 1.390   # R / G
_SKIN_GAIN_MIN = 0.80  # 单通道增益下限，防止过度校正
_SKIN_GAIN_MAX = 1.40


# 人像抠图后的「透明像素」没有真实颜色，肤色检测要排除这些区域，
# 避免把边缘的杂色当成皮肤一起处理。
def _skin_mask(image_bgr: np.ndarray) -> np.ndarray:
    """肤色遮罩（0~1 浮点）：YCrCb 阈值 + 形态学去噪 + 羽化。"""
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)     # Y, Cr, Cb 下限
    upper = np.array([255, 173, 127], dtype=np.uint8)  # Y, Cr, Cb 上限
    mask = cv2.inRange(ycrcb, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)    # 去掉噪点
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)   # 补小洞
    mask = cv2.GaussianBlur(mask, (0, 0), 3)                 # 边缘羽化，避免硬边
    return mask.astype(np.float32) / 255.0


def _apply_skin_tone(image_bgr: np.ndarray, mask: np.ndarray, strength: int) -> np.ndarray:
    """肤色矫正：按皮肤区域估计偏色，算 RGB 增益后校正整张人像。

    只在皮肤上「测光」，但把增益应用到整张人像——这样脸、脖子、头发、
    衣服的颜色关系保持一致；背景是后期合成的，不受影响。

    判定依据是 B/G、R/G 两个比例（与亮度无关）：
        偏黄 → B/G 变小；偏绿/偏蓝 → R/G、B/G 同时偏小
    """
    if strength <= 0:
        return image_bgr
    selected = mask > 0.35
    area = int(selected.sum())
    if area < max(50, int(0.002 * selected.size)):   # 皮肤太少，估计不可靠
        return image_bgr

    mean_b, mean_g, mean_r = image_bgr[selected].mean(axis=0)
    if mean_g < 1.0:
        return image_bgr

    gain_b = _SKIN_REF_BG * mean_g / max(float(mean_b), 1.0)
    gain_r = _SKIN_REF_RG * mean_g / max(float(mean_r), 1.0)
    gains = np.clip([gain_b, 1.0, gain_r], _SKIN_GAIN_MIN, _SKIN_GAIN_MAX)

    t = min(max(strength, 0), 10) / 10.0
    gains = 1.0 + (gains - 1.0) * t

    lut = [np.clip(np.arange(256, dtype=np.float32) * g, 0, 255).astype(np.uint8)
           for g in gains]
    return cv2.merge([cv2.LUT(image_bgr[:, :, i], lut[i]) for i in range(3)])


def _apply_skin_even(image_bgr: np.ndarray, mask: np.ndarray, strength: int) -> np.ndarray:
    """面部颜色均衡：压平皮肤色度的「低频起伏」（红一块、白一块）。

    做法：在 LAB 的 a/b 通道里，用大尺度高斯提取低频色斑，
    再把低频相对皮肤均值的偏离按强度压回去——高频纹理（毛孔、细纹）保持不动。
    低频用「遮罩加权模糊」（分子分母都模糊）计算，避免头发/背景把脸边缘带脏。
    """
    if strength <= 0:
        return image_bgr
    if float(mask.sum()) < 50:
        return image_bgr

    height, width = image_bgr.shape[:2]
    m = mask.astype(np.float32)
    m_sum = float(m.sum())
    if m_sum < 50:
        return image_bgr

    # 低频信息不需要全分辨率：缩小 4 倍再模糊，速度快一个数量级
    scale = 4
    sw, sh = max(8, width // scale), max(8, height // scale)
    small = cv2.resize(image_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    m_small = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_AREA)
    sigma = max(2.0, min(height, width) * 0.02 / scale)   # 约脸宽的 2%
    denom = cv2.GaussianBlur(m_small, (0, 0), sigma) + 1e-6

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_small = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    k = min(max(strength, 0), 10) / 10.0 * 0.8
    for ch in (1, 2):                              # a（红绿）、b（黄蓝）
        value = lab[:, :, ch]
        mean = float((value * m).sum()) / m_sum
        low_small = cv2.GaussianBlur(lab_small[:, :, ch] * m_small, (0, 0), sigma) / denom
        low = cv2.resize(low_small, (width, height), interpolation=cv2.INTER_LINEAR)
        lab[:, :, ch] = value - k * (low - mean)

    evened = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8),
                          cv2.COLOR_LAB2BGR).astype(np.float32)
    original = image_bgr.astype(np.float32)
    # 只在遮罩内替换；LAB 往返的 ±1 误差在遮罩外被丢掉
    out = np.where(mask[..., None] > 0.004,
                   np.clip(evened, 0, 255), original).astype(np.uint8)
    return out


def apply_beauty(image_bgr: np.ndarray, params: dict, alpha=None) -> np.ndarray:
    """美颜：磨皮 + 肤色矫正 + 美白，只处理皮肤（矫正作用于整张人像）。

    params: {"smooth": 0~10, "whiten": 0~10, "correct": 0~10, "even": 0~10}
    alpha:  可选，uint8 人像透明度（BGRA 的 A 通道）；传入后效果只落在人像内，
            透明背景不会被误处理。

    磨皮 = 半分辨率双边滤波（保边）+ 回加 25% 高频细节，避免"塑料脸"；
    颜色均衡 = 压平皮肤色度的低频起伏（红一块白一块），保留高频纹理；
    肤色矫正 = 皮肤测光 → RGB 增益 → 整张人像校色；
    美白 = 在 LAB 的 L 通道提亮、轻微去黄，比 RGB 直接加亮度更自然。
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    if len(image_bgr.shape) != 3 or image_bgr.shape[2] != 3:
        return image_bgr

    smooth = int(params.get("smooth", 0) or 0)
    whiten = int(params.get("whiten", 0) or 0)
    correct = int(params.get("correct", 0) or 0)
    even = int(params.get("even", 0) or 0)
    if smooth <= 0 and whiten <= 0 and correct <= 0 and even <= 0:
        return image_bgr

    mask = _skin_mask(image_bgr)
    if alpha is not None and getattr(alpha, "shape", None) == image_bgr.shape[:2]:
        mask = mask * (alpha.astype(np.float32) / 255.0)
    if float(mask.max()) <= 0.01:
        return image_bgr

    # 先均匀色斑，再校色相，最后磨皮/提亮
    if even > 0:
        image_bgr = _apply_skin_even(image_bgr, mask, even)
    if correct > 0:
        image_bgr = _apply_skin_tone(image_bgr, mask, correct)

    # 只在「皮肤包围盒」里做浮点运算：整张图算会多花 3 倍时间，
    # 而皮肤通常只占画面一小块。pad 给滤波留一点上下文，避免边缘接缝。
    height, width = image_bgr.shape[:2]
    ys, xs = np.where(mask > 0.01)
    if ys.size == 0:
        return image_bgr
    pad = 24
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(height, int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(width, int(xs.max()) + 1 + pad)
    roi = image_bgr[y0:y1, x0:x1]
    roi_mask = mask[y0:y1, x0:x1]

    result = roi.astype(np.float32)

    # ---- 磨皮：半分辨率双边滤波（保边平滑）+ 高频回加 ----
    if smooth > 0:
        # 强度曲线：默认 3 就能看出效果，10 封顶 95%，避免塑料脸
        strength = (min(max(smooth, 0), 10) / 10.0) ** 0.85 * 0.95
        rh, rw = roi.shape[:2]
        # 半分辨率做双边：同样的 d 覆盖范围大 2 倍，效果更明显且更快
        small = cv2.resize(roi, (max(1, rw // 2), max(1, rh // 2)),
                           interpolation=cv2.INTER_AREA)
        small = cv2.bilateralFilter(small, 15, 100, 100)
        smoothed = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_LINEAR)
        base = cv2.GaussianBlur(roi, (0, 0), 2.0).astype(np.float32)
        detail = result - base
        blended = np.clip(smoothed.astype(np.float32) + detail * 0.25, 0, 255)
        k = (roi_mask * strength)[..., None]
        result = result * (1.0 - k) + blended * k

    # ---- 美白：LAB 的 L 提亮 + 轻微去黄 ----
    if whiten > 0:
        w = min(max(whiten, 0), 10) / 10.0
        before = result
        lab = cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8),
                           cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] = np.clip(lab[:, :, 0] + 16.0 * w * roi_mask, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] - 3.0 * w * roi_mask, 0, 255)
        whitened = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8),
                                cv2.COLOR_LAB2BGR).astype(np.float32)
        # LAB 往返本身有 ±1 取整误差：按遮罩混回，保证遮罩外像素逐位不变
        result = before + (whitened - before) * roi_mask[..., None]

    processed = np.clip(result, 0, 255).astype(np.uint8)
    out = image_bgr.copy()
    # 只替换遮罩覆盖的像素，其余（含透明区）与输入逐位一致
    out[y0:y1, x0:x1] = np.where(roi_mask[..., None] > 0.004, processed, roi)
    return out


# ============================================================
#  水印处理
# ============================================================

def apply_watermark(image_bgr: np.ndarray, params: dict) -> np.ndarray:
    """
    给 BGR 图像添加文字水印。
    params: {
        enable: bool,
        text: str,
        color: hex str,
        size: int,
        alpha: float 0-1,
        angle: int,
        spacing: int
    }
    """
    if not params.get("enable", False):
        return image_bgr
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    text = params.get("text", "")
    if not text:
        return image_bgr

    # 复制一份再绘制：原实现就地修改入参数组，若调用方仍持有该引用会被意外污染。
    # 此处改为在副本上绘制，返回内容与旧版完全一致，仅不再副作用入参。
    image_bgr = image_bgr.copy()

    height, width = image_bgr.shape[:2]
    color_hex = params.get("color", "#FFFFFF")
    rgb = hex_to_rgb_tuple(color_hex)
    bgr = (rgb[2], rgb[1], rgb[0])

    font_size = int(params.get("size", 20))
    alpha = float(params.get("alpha", 0.15))
    alpha = max(0.0, min(1.0, alpha))
    angle = int(params.get("angle", 30))
    spacing = int(params.get("spacing", 25))

    # 创建透明图层
    overlay = np.zeros((height, width, 4), dtype=np.uint8)

    # OpenCV 不支持中文，用 PIL 绘制
    from PIL import Image, ImageDraw, ImageFont
    pil_overlay = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGRA2RGBA))
    draw = ImageDraw.Draw(pil_overlay)

    try:
        font = ImageFont.truetype("msyh.ttc", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("Microsoft YaHei.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    # 估算文字尺寸
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # 按角度步进铺设水印
    rad = np.radians(angle)
    cos_val = np.cos(rad)
    tan_val = np.tan(rad)
    # 退化角度保护：角度接近 90/270 度时 cos 趋近 0、tan 趋于无穷，
    # 会导致内循环步长退化为 1（极密）且 col*tan 溢出。统一降级为水平铺排。
    if not np.isfinite(tan_val) or abs(cos_val) < 1e-6:
        tan_val = 0.0
        step_x = max(1, int(text_w + spacing))
    else:
        step_x = int((text_w + spacing) * cos_val)
        if step_x == 0:
            step_x = 1 if cos_val >= 0 else -1
    step_y = int((text_w + spacing) * np.sin(rad))
    row_step_y = text_h + spacing

    # 对角线铺设多行
    for row in range(-height, height * 2, max(1, row_step_y)):
        for col in range(-width, width * 2, abs(step_x) + 1):
            x = col
            y = row + int(col * tan_val)
            draw.text((x, y), text, font=font, fill=(rgb[0], rgb[1], rgb[2], int(255 * alpha)))

    overlay = cv2.cvtColor(np.array(pil_overlay), cv2.COLOR_RGBA2BGRA)

    # 混合到原图
    alpha_mask = overlay[:, :, 3].astype(np.float32) / 255.0
    for c in range(3):
        image_bgr[:, :, c] = (
            image_bgr[:, :, c].astype(np.float32) * (1 - alpha_mask) +
            overlay[:, :, c].astype(np.float32) * alpha_mask
        ).astype(np.uint8)

    return image_bgr


# ============================================================
#  辅助工具
# ============================================================

def get_nontransparent_bounds(image_bgra: np.ndarray, thresh: int = 127):
    """
    获取 4 通道图像中最大非透明区域的边界距离信息。
    """
    if image_bgra is None or image_bgra.size == 0:
        raise ValueError("输入图像为空")
    if len(image_bgra.shape) != 3 or image_bgra.shape[2] != 4:
        raise ValueError("输入图像必须是 4 通道 BGRA 图像")

    _, _, _, alpha = cv2.split(image_bgra)
    _, alpha_bin = cv2.threshold(alpha, thresh=thresh, maxval=255, type=0)
    contours, _ = cv2.findContours(alpha_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0, 0, 0, 0

    contour_areas = [cv2.contourArea(c) for c in contours]
    max_idx = contour_areas.index(max(contour_areas))
    x, y, w, h = cv2.boundingRect(contours[max_idx])

    height, width = image_bgra.shape[:2]
    return y, height - (y + h), x, width - (x + w)


def save_image_with_dpi(image_array: np.ndarray, output_path: str, dpi: int = 300):
    """
    以指定 DPI 保存图像为 PNG。
    """
    if image_array is None or image_array.size == 0:
        raise ValueError("输入图像为空")

    if len(image_array.shape) == 2:
        pil_img = Image.fromarray(image_array, mode="L")
    elif image_array.shape[2] == 4:
        rgba = cv2.cvtColor(image_array, cv2.COLOR_BGRA2RGBA)
        pil_img = Image.fromarray(rgba, mode="RGBA")
    else:
        rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb, mode="RGB")

    pil_img.save(output_path, format="PNG", dpi=(dpi, dpi))


def save_image_to_jpeg_kb(
    image_bgr: np.ndarray, output_path: str, target_kb: int, dpi: int = 300
):
    """将 BGR 图像保存为 JPEG，压缩到指定 KB 大小。"""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("输入图像为空")
    if target_kb <= 0:
        # 不限体积时也必须写 JPEG。
        # 旧实现这里转去调 save_image_with_dpi（固定写 PNG），于是
        # 「批量导出 + 格式选 JPEG + KB 限制=0」会产出扩展名 .jpg、
        # 内部却是 PNG 字节的文件（实测文件头 89 50 4E 47）。
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(
            output_path, format="JPEG", quality=95,
            dpi=(dpi, dpi), optimize=True)
        return

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    quality = 95
    while quality >= 1:
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality, dpi=(dpi, dpi))
        size_kb = len(buf.getvalue()) / 1024
        if size_kb <= target_kb or quality == 1:
            with open(output_path, "wb") as f:
                f.write(buf.getvalue())
            return
        quality -= 5
