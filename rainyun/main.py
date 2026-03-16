import logging
import os
import random
import re
import shutil
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations, permutations
from threading import Lock
from typing import Protocol, Sequence

import cv2
import ddddocr
import numpy as np
from .api.client import RainyunAPI
from selenium.common.exceptions import TimeoutException
from selenium.webdriver import ActionChains
from selenium.webdriver.support import expected_conditions as EC

from .config import Config, get_default_config
from .data.store import DataStore
from .browser.cookies import load_cookies
from .browser.locators import XPATH_CONFIG
from .browser.pages import LoginPage, RewardPage
from .browser.session import BrowserSession, RuntimeContext
from .utils.http import download_bytes, download_to_file
from .utils.image import decode_image_bytes, encode_image_bytes, normalize_gray, split_sprite_image

# 用户日志前缀（用于多账号区分）
_LOG_USER_PREFIX = ""


def _set_log_user(user: str | None) -> None:
    global _LOG_USER_PREFIX
    if user:
        _LOG_USER_PREFIX = f"用户 {user} "
    else:
        _LOG_USER_PREFIX = ""


def _set_log_prefix(prefix: str) -> None:
    global _LOG_USER_PREFIX
    _LOG_USER_PREFIX = prefix


def _get_log_prefix() -> str:
    return _LOG_USER_PREFIX


# 自定义异常：验证码处理过程中可重试的错误
class CaptchaRetryableError(Exception):
    """可重试的验证码处理错误（如下载失败、网络问题等）"""
    pass


class LazyDdddOcr:
    """延迟初始化的 ddddocr，只有首次调用才创建实例。"""

    def __init__(self, *, det: bool = False) -> None:
        self._det = det
        self._instance: ddddocr.DdddOcr | None = None

    def _ensure(self) -> ddddocr.DdddOcr:
        if self._instance is None:
            if self._det:
                logging.getLogger(__name__).info(f"{_get_log_prefix()}初始化 ddddocr(det)")
                self._instance = ddddocr.DdddOcr(det=True, show_ad=False)
            else:
                logging.getLogger(__name__).info(f"{_get_log_prefix()}初始化 ddddocr(ocr)")
                self._instance = ddddocr.DdddOcr(ocr=True, show_ad=False)
        return self._instance

    def classification(self, image_bytes: bytes):
        if self._det:
            raise AttributeError("当前实例为 det 模式，无法调用 classification")
        return self._ensure().classification(image_bytes)

    def detection(self, image_bytes: bytes):
        if not self._det:
            raise AttributeError("当前实例为 ocr 模式，无法调用 detection")
        return self._ensure().detection(image_bytes)

try:
    from .notify import configure, send

    print("✅ 通知模块加载成功")
except Exception as e:
    print(f"⚠️ 通知模块加载失败：{e}")

    def configure(_config: Config) -> None:
        pass

    def send(title, content):
        pass

# 服务器管理模块（可选功能，需要配置 API_KEY）
ServerManager = None
_server_manager_error = None
try:
    from .server.manager import ServerManager

    print("✅ 服务器管理模块加载成功")
except Exception as e:
    print(f"⚠️ 服务器管理模块加载失败：{e}")
    _server_manager_error = str(e)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 配置 logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

class _RingLogBuffer:
    """有上限的日志缓冲，避免内存无限增长。"""

    def __init__(self, max_lines: int = 600) -> None:
        self._buffer = deque(maxlen=max_lines)
        self._lock = Lock()

    def append(self, message: str) -> None:
        with self._lock:
            self._buffer.append(message)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def getvalue(self) -> str:
        with self._lock:
            return "\n".join(self._buffer)


class _RingLogHandler(logging.Handler):
    def __init__(self, buffer: _RingLogBuffer) -> None:
        super().__init__()
        self._buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        self._buffer.append(message)


log_capture_buffer = _RingLogBuffer(max_lines=600)
ring_handler = _RingLogHandler(log_capture_buffer)
ring_handler.setFormatter(formatter)
logger.addHandler(ring_handler)


@dataclass(frozen=True)
class MatchResult:
    positions: list[tuple[int, int]]
    similarities: list[float]
    method: str


class CaptchaMatcher(Protocol):
    name: str

    def match(
        self,
        background: np.ndarray,
        sprites: list[np.ndarray],
        bboxes: list[tuple[int, int, int, int]],
    ) -> MatchResult | None:
        ...


class CaptchaSolver(Protocol):
    def solve(
        self,
        background: np.ndarray,
        sprites: list[np.ndarray],
        bboxes: list[tuple[int, int, int, int]],
    ) -> MatchResult | None:
        ...


class StrategyCaptchaSolver:
    def __init__(self, matchers: Sequence[CaptchaMatcher]) -> None:
        self.matchers = list(matchers)

    def solve(
        self,
        background: np.ndarray,
        sprites: list[np.ndarray],
        bboxes: list[tuple[int, int, int, int]],
    ) -> MatchResult | None:
        prefix = _get_log_prefix()
        for matcher in self.matchers:
            result = matcher.match(background, sprites, bboxes)
            if result:
                logger.info(f"{prefix}验证码匹配策略命中: {matcher.name}")
                return result
            logger.warning(f"{prefix}验证码匹配策略失败: {matcher.name}")
        return None


class SiftMatcher:
    name = "sift"

    def __init__(self) -> None:
        self._sift = cv2.SIFT_create() if hasattr(cv2, "SIFT_create") else None
        if not self._sift:
            prefix = _get_log_prefix()
            logger.warning(f"{prefix}SIFT 不可用，将跳过 SiftMatcher")

    def match(
        self,
        background: np.ndarray,
        sprites: list[np.ndarray],
        bboxes: list[tuple[int, int, int, int]],
    ) -> MatchResult | None:
        if not self._sift:
            return None
        return build_match_result(
            background,
            sprites,
            bboxes,
            lambda sprite, spec: compute_sift_similarity(sprite, spec, self._sift),
            self.name,
        )


class TemplateMatcher:
    name = "template"

    def match(
        self,
        background: np.ndarray,
        sprites: list[np.ndarray],
        bboxes: list[tuple[int, int, int, int]],
    ) -> MatchResult | None:
        return build_match_result(
            background,
            sprites,
            bboxes,
            compute_template_similarity,
            self.name,
        )


def temp_path(ctx: RuntimeContext, filename: str) -> str:
    return os.path.join(ctx.temp_dir, filename)


def clear_temp_dir(temp_dir: str) -> None:
    if not os.path.exists(temp_dir):
        return
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)


def download_image(url: str, output_path: str, config: Config) -> bool:
    return download_to_file(url, output_path, config, log=logger)


def download_image_bytes(url: str, config: Config, fallback_path: str | None = None) -> bytes:
    prefix = _get_log_prefix()
    try:
        return download_bytes(
            url,
            timeout=config.download_timeout,
            max_retries=config.download_max_retries,
            retry_delay=config.download_retry_delay,
            log=logger,
        )
    except RuntimeError as e:
        if fallback_path:
            logger.warning(f"{prefix}内存下载失败，尝试降级为文件下载")
            if download_image(url, fallback_path, config):
                with open(fallback_path, "rb") as f:
                    return f.read()
        raise CaptchaRetryableError(f"验证码图片下载失败: {e}")


def get_url_from_style(style):
    # 修复：添加空值保护
    if not style:
        raise ValueError("style 属性为空，无法解析 URL")
    match = re.search(r"url\(([^)]+)\)", style, re.IGNORECASE)
    if not match:
        raise ValueError(f"无法从 style 中解析 URL: {style}")
    url = match.group(1).strip().strip('"').strip("'")
    return url


def get_width_from_style(style):
    # 修复：添加空值保护
    if not style:
        raise ValueError("style 属性为空，无法解析宽度")
    match = re.search(r"width\s*:\s*([\d.]+)px", style, re.IGNORECASE)
    if not match:
        raise ValueError(f"无法从 style 中解析宽度: {style}")
    return float(match.group(1))


def get_height_from_style(style):
    # 修复：添加空值保护
    if not style:
        raise ValueError("style 属性为空，无法解析高度")
    match = re.search(r"height\s*:\s*([\d.]+)px", style, re.IGNORECASE)
    if not match:
        raise ValueError(f"无法从 style 中解析高度: {style}")
    return float(match.group(1))


def get_element_size(element) -> tuple[float, float]:
    size = element.size or {}
    width = size.get("width", 0)
    height = size.get("height", 0)
    if not width or not height:
        raise ValueError("无法从元素尺寸解析宽高")
    return float(width), float(height)


def detect_captcha_bboxes(
    ctx: RuntimeContext,
    captcha_bytes: bytes,
    captcha_image: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    prefix = _get_log_prefix()
    payloads = [
        ("raw", captcha_bytes),
        ("reencode", encode_image_bytes(captcha_image, "验证码背景图")),
    ]
    for label, payload in payloads:
        try:
            bboxes = ctx.det.detection(payload)
            if bboxes:
                logger.info(f"{prefix}验证码检测成功({label}): {len(bboxes)} 个候选框")
                return bboxes
            logger.warning(f"{prefix}验证码检测结果为空({label})")
        except Exception as e:
            logger.warning(f"{prefix}验证码检测失败({label}): {e}")
    return []


def compute_sift_similarity(sprite: np.ndarray, spec: np.ndarray, sift) -> float:
    sprite_gray = normalize_gray(sprite)
    spec_gray = normalize_gray(spec)
    kp1, des1 = sift.detectAndCompute(sprite_gray, None)
    kp2, des2 = sift.detectAndCompute(spec_gray, None)
    if des1 is None or des2 is None:
        return 0.0
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m_n in matches if len(m_n) == 2 for m, n in [m_n] if m.distance < 0.8 * n.distance]
    if not matches or len(good) == 0:
        return 0.0
    return len(good) / len(matches)


def compute_template_similarity(sprite: np.ndarray, spec: np.ndarray) -> float:
    sprite_gray = normalize_gray(sprite)
    spec_gray = normalize_gray(spec)
    if sprite_gray is None or spec_gray is None or sprite_gray.size == 0 or spec_gray.size == 0:
        return 0.0
    if sprite_gray.shape != spec_gray.shape:
        sprite_gray = cv2.resize(sprite_gray, (spec_gray.shape[1], spec_gray.shape[0]))
    result = cv2.matchTemplate(spec_gray, sprite_gray, cv2.TM_CCOEFF_NORMED)
    return float(np.max(result))


def build_match_result(
    background: np.ndarray,
    sprites: list[np.ndarray],
    bboxes: list[tuple[int, int, int, int]],
    similarity_fn,
    method: str,
) -> MatchResult | None:
    prefix = _get_log_prefix()
    if not bboxes:
        logger.warning(f"{prefix}验证码检测结果为空，无法匹配")
        return None
    if len(sprites) != 3:
        logger.warning(f"{prefix}验证码小图数量异常: {len(sprites)}")
        return None
    best_positions: list[tuple[int, int] | None] = [None, None, None]
    best_scores: list[float | None] = [None, None, None]
    valid_specs: list[tuple[tuple[int, int], np.ndarray]] = []
    for bbox in bboxes:
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        if x2 <= x1 or y2 <= y1:
            continue
        spec = background[y1:y2, x1:x2]
        if spec.size == 0:
            continue
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        valid_specs.append((center, spec))
    if not valid_specs:
        return None
    if len(valid_specs) < len(sprites):
        for center, spec in valid_specs:
            for index, sprite in enumerate(sprites):
                if sprite is None or sprite.size == 0:
                    continue
                similarity = similarity_fn(sprite, spec)
                if best_scores[index] is None or similarity > best_scores[index]:
                    best_scores[index] = similarity
                    best_positions[index] = center
    else:
        sim_matrix: list[list[float]] = []
        for sprite in sprites:
            row: list[float] = []
            for _, spec in valid_specs:
                if sprite is None or sprite.size == 0:
                    row.append(0.0)
                    continue
                row.append(similarity_fn(sprite, spec))
            sim_matrix.append(row)

        best_perm: tuple[int, ...] | None = None
        best_scores_local: list[float] | None = None
        best_key: tuple[float, float, float] | None = None
        bbox_indices = range(len(valid_specs))
        for chosen in combinations(bbox_indices, len(sprites)):
            for perm in permutations(chosen):
                scores = [sim_matrix[i][perm[i]] for i in range(len(sprites))]
                min_score = min(scores)
                avg_score = sum(scores) / len(scores)
                sum_score = sum(scores)
                key = (min_score, avg_score, sum_score)
                if best_key is None or key > best_key:
                    best_key = key
                    best_perm = perm
                    best_scores_local = scores
        if best_perm is not None and best_scores_local is not None:
            for sprite_index, bbox_index in enumerate(best_perm):
                center, _ = valid_specs[bbox_index]
                best_positions[sprite_index] = center
                best_scores[sprite_index] = best_scores_local[sprite_index]
    if any(pos is None for pos in best_positions):
        return None
    return MatchResult(
        positions=[pos for pos in best_positions if pos is not None],
        similarities=[float(score) if score is not None else 0.0 for score in best_scores],
        method=method,
    )


def log_match_result(result: MatchResult) -> None:
    prefix = _get_log_prefix()
    for index, (position, similarity) in enumerate(zip(result.positions, result.similarities), start=1):
        x, y = position
        logger.info(
            f"{prefix}图案 {index} 位于 ({x},{y})，匹配率：{similarity:.4f}，策略：{result.method}"
        )


def process_captcha(ctx: RuntimeContext, retry_count: int = 0):
    """
    处理验证码逻辑（循环实现，避免递归栈溢出）
    - 整体重试上限由配置项 captcha_retry_limit 控制
    - 启用 captcha_retry_unlimited 后无限重试直到成功
    - 内部图片下载重试由配置项 download_max_retries 控制
    """
    prev_prefix = _get_log_prefix()
    user_label = ctx.config.display_name or ctx.config.rainyun_user
    _set_log_user(user_label)
    prefix = _get_log_prefix()

    def refresh_captcha() -> bool:
        try:
            reload_btn = ctx.driver.find_element(*XPATH_CONFIG["CAPTCHA_RELOAD"])
            time.sleep(2)
            reload_btn.click()
            time.sleep(2)
            return True
        except Exception as refresh_error:
            logger.error(f"{prefix}无法刷新验证码，放弃重试: {refresh_error}")
            return False

    solver = StrategyCaptchaSolver([SiftMatcher(), TemplateMatcher()])
    current_retry = retry_count
    try:
        while True:
            if not ctx.config.captcha_retry_unlimited and current_retry >= ctx.config.captcha_retry_limit:
                logger.error(f"{prefix}验证码重试次数过多，任务失败")
                return False
            if ctx.config.captcha_retry_unlimited and current_retry > 0:
                logger.info(f"{prefix}无限重试模式，当前第 {current_retry + 1} 次尝试")

            try:
                captcha_bytes, captcha_image, sprites = download_captcha_assets(ctx)
                if check_captcha(ctx, captcha_image, sprites):
                    logger.info(f"{prefix}开始识别验证码 (第 {current_retry + 1} 次尝试)")
                    bboxes = detect_captcha_bboxes(ctx, captcha_bytes, captcha_image)
                    if not bboxes:
                        logger.error(f"{prefix}验证码检测失败，正在重试")
                        save_captcha_samples(captcha_image, sprites, config=ctx.config, reason="no_bboxes")
                    else:
                        result = solver.solve(captcha_image, sprites, bboxes)
                        if result:
                            log_match_result(result)
                            if check_answer(result):
                                for position in result.positions:
                                    slide_bg = ctx.wait.until(
                                        EC.visibility_of_element_located(XPATH_CONFIG["CAPTCHA_BG"])
                                    )
                                    style = slide_bg.get_attribute("style")
                                    x, y = position
                                    width_raw, height_raw = captcha_image.shape[1], captcha_image.shape[0]
                                    try:
                                        width = get_width_from_style(style)
                                        height = get_height_from_style(style)
                                    except ValueError:
                                        width, height = get_element_size(slide_bg)
                                    x_offset, y_offset = float(-width / 2), float(-height / 2)
                                    final_x = int(x_offset + x / width_raw * width)
                                    final_y = int(y_offset + y / height_raw * height)
                                    ActionChains(ctx.driver).move_to_element_with_offset(
                                        slide_bg, final_x, final_y
                                    ).click().perform()
                                confirm = ctx.wait.until(EC.element_to_be_clickable(XPATH_CONFIG["CAPTCHA_SUBMIT"]))
                                logger.info(f"{prefix}提交验证码")
                                confirm.click()
                                time.sleep(5)
                                result_el = ctx.wait.until(
                                    EC.visibility_of_element_located(XPATH_CONFIG["CAPTCHA_OP"])
                                )
                                if 'show-success' in result_el.get_attribute("class"):
                                    logger.info(f"{prefix}验证码通过")
                                    return True
                                logger.error(f"{prefix}验证码未通过，正在重试")
                                save_captcha_samples(
                                    captcha_image, sprites, config=ctx.config, reason="submit_failed"
                                )
                            else:
                                logger.error(f"{prefix}验证码识别结果无效，正在重试")
                                save_captcha_samples(
                                    captcha_image, sprites, config=ctx.config, reason="answer_invalid"
                                )
                        else:
                            logger.error(f"{prefix}验证码匹配失败，正在重试")
                            save_captcha_samples(
                                captcha_image, sprites, config=ctx.config, reason="match_failed"
                            )
                else:
                    logger.error(f"{prefix}当前验证码识别率低，尝试刷新")

                if not refresh_captcha():
                    return False
                current_retry += 1
            except (TimeoutException, ValueError, CaptchaRetryableError) as e:
                logger.error(f"{prefix}验证码处理异常: {type(e).__name__} - {e}")
                if not refresh_captcha():
                    return False
                current_retry += 1
    finally:
        _set_log_prefix(prev_prefix)


def download_captcha_assets(ctx: RuntimeContext) -> tuple[bytes, np.ndarray, list[np.ndarray]]:
    prefix = _get_log_prefix()
    clear_temp_dir(ctx.temp_dir)
    slide_bg = ctx.wait.until(EC.visibility_of_element_located(XPATH_CONFIG["CAPTCHA_BG"]))
    img1_style = slide_bg.get_attribute("style")
    img1_url = get_url_from_style(img1_style)
    logger.info(f"{prefix}开始下载验证码图片(1): {img1_url}")
    captcha_bytes = download_image_bytes(img1_url, ctx.config, temp_path(ctx, "captcha.jpg"))
    sprite = ctx.wait.until(EC.visibility_of_element_located(XPATH_CONFIG["CAPTCHA_IMG_INSTRUCTION"]))
    img2_url = sprite.get_attribute("src")
    logger.info(f"{prefix}开始下载验证码图片(2): {img2_url}")
    sprite_bytes = download_image_bytes(img2_url, ctx.config, temp_path(ctx, "sprite.jpg"))
    captcha_image = decode_image_bytes(captcha_bytes, "验证码背景图")
    sprite_image = decode_image_bytes(sprite_bytes, "验证码小图")
    sprites = split_sprite_image(sprite_image)
    return captcha_bytes, captcha_image, sprites


def save_captcha_samples(
    captcha_image: np.ndarray | None,
    sprites: list[np.ndarray],
    *,
    config: Config,
    reason: str,
) -> None:
    """保存验证码样本用于排查。"""
    if not config.captcha_save_samples:
        return
    prefix = _get_log_prefix()
    try:
        base_dir = os.path.join("temp", "captcha_samples")
        os.makedirs(base_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        sample_dir = os.path.join(base_dir, f"{stamp}-{reason}-{random.randint(1000, 9999)}")
        os.makedirs(sample_dir, exist_ok=True)
        if captcha_image is not None and captcha_image.size > 0:
            cv2.imwrite(os.path.join(sample_dir, "background.jpg"), captcha_image)
        for index, sprite in enumerate(sprites, start=1):
            if sprite is None or sprite.size == 0:
                continue
            cv2.imwrite(os.path.join(sample_dir, f"sprite_{index}.jpg"), sprite)
        with open(os.path.join(sample_dir, "reason.txt"), "w", encoding="utf-8") as f:
            f.write(f"reason:{reason}\n")
    except Exception as e:
        logger.warning(f"{prefix}保存验证码样本失败: {e}")


def check_captcha(ctx: RuntimeContext, captcha_image: np.ndarray, sprites: list[np.ndarray]) -> bool:
    prefix = _get_log_prefix()
    if len(sprites) != 3:
        logger.error(f"{prefix}验证码小图数量异常，期望 3，实际 {len(sprites)}")
        save_captcha_samples(captcha_image, sprites, config=ctx.config, reason="sprite_count")
        return False
    low_confidence = 0
    for index, sprite in enumerate(sprites, start=1):
        sprite_bytes = encode_image_bytes(sprite, f"验证码小图{index}")
        if ctx.ocr.classification(sprite_bytes) in ["0", "1"]:
            low_confidence += 1
            logger.warning(f"{prefix}验证码小图 {index} 识别为低置信度标记")
    if low_confidence >= 2:
        logger.error(f"{prefix}低置信度小图过多，跳过本次识别")
        save_captcha_samples(captcha_image, sprites, config=ctx.config, reason="low_confidence")
        return False
    return True


# 检查是否存在重复坐标,快速判断识别错误
def check_answer(result: MatchResult, min_similarity: float = 0.25) -> bool:
    prefix = _get_log_prefix()
    if not result.positions or len(result.positions) < 3:
        logger.warning(
            f"{prefix}验证码识别坐标不足，当前仅有 {len(result.positions) if result.positions else 0} 个"
        )
        return False
    if len(result.similarities) < 3:
        logger.warning(f"{prefix}验证码匹配率不足，当前仅有 {len(result.similarities)} 个")
        return False
    if len(result.positions) != len(set(result.positions)):
        logger.warning(f"{prefix}验证码识别坐标重复: {result.positions}")
        return False
    min_match = min(result.similarities) if result.similarities else 0.0
    if min_match < min_similarity:
        logger.warning(
            f"{prefix}验证码最低匹配率 {min_match:.4f} 低于阈值 {min_similarity:.2f}，放弃提交"
        )
        return False
    return True


def run_with_config(config: Config) -> bool:
    ctx = None
    driver = None
    temp_dir = None
    debug = False
    session = None
    prefix = ""
    log_capture_buffer.clear()
    try:
        configure(config)
        timeout = config.timeout
        max_delay = config.max_delay
        user = config.rainyun_user
        pwd = config.rainyun_pwd
        debug = config.debug
        # 容器环境默认启用 Linux 模式
        linux = config.linux_mode
        display_name = config.display_name or user
        _set_log_user(display_name)
        prefix = _get_log_prefix()

        # 检查必要配置
        if not user or not pwd:
            logger.error(f"{prefix}请配置账号用户名和密码")
            return False

        api_key = config.rainyun_api_key
        api_client = RainyunAPI(api_key, config=config)

        logger.info(f"{prefix}━━━━━━ 雨云签到 v{config.app_version} ━━━━━━")
        if config.captcha_retry_unlimited:
            logger.warning(f"{prefix}已启用无限重试模式，验证码将持续重试直到成功或手动停止")

        # 初始积分记录
        start_points = 0
        if api_key:
            try:
                start_points = api_client.get_user_points()
                logger.info(f"{prefix}签到前初始积分: {start_points}")
            except Exception as e:
                logger.warning(f"{prefix}获取初始积分失败: {e}")

        delay = random.randint(0, max_delay)
        delay_sec = random.randint(0, 60)
        if not debug:
            logger.info(f"{prefix}随机延时等待 {delay} 分钟 {delay_sec} 秒")
            time.sleep(delay * 60 + delay_sec)
        logger.info(f"{prefix}准备 OCR/DET（延迟初始化）")
        ocr = LazyDdddOcr(det=False)
        det = LazyDdddOcr(det=True)
        logger.info(f"{prefix}初始化 Selenium")
        session = BrowserSession(config=config, debug=debug, linux=linux)
        driver, wait, temp_dir = session.start()
        ctx = RuntimeContext(
            driver=driver,
            wait=wait,
            ocr=ocr,
            det=det,
            temp_dir=temp_dir,
            api=api_client,
            config=config
        )

        login_page = LoginPage(ctx, captcha_handler=process_captcha)
        reward_page = RewardPage(ctx, captcha_handler=process_captcha)

        # 每次都重新登录，不使用 token/cookie
        logger.info(f"{prefix}正在发起首次或重新登录请求...")
        logged_in = login_page.login(user, pwd)

        if not logged_in:
            logger.error(f"{prefix}登录失败，任务终止")
            return False

        reward_page.handle_daily_reward(start_points)
        
        logger.info(f"{prefix}任务执行成功！")
        return True
    except Exception as e:
        logger.error(f"{prefix}脚本执行异常终止: {e}")
        return False

    finally:
        # === 核心逻辑：无论成功失败，这里都会执行 ===

        # 1. 关闭浏览器
        if session:
            session.close()

        # 2. 服务器到期检查和自动续费（需要配置 API_KEY）
        server_report = ""
        final_config = config or get_default_config()
        api_key = final_config.rainyun_api_key
        if api_key and ServerManager:
            logger.info(f"{prefix}━━━━━━ 开始检查服务器状态 ━━━━━━")
            try:
                manager = ServerManager(api_key, config=final_config)
                result = manager.check_and_renew()
                server_report = "\n\n" + manager.generate_report(result)
                logger.info(f"{prefix}服务器检查完成")
            except Exception as e:
                logger.error(f"{prefix}服务器检查失败: {e}")
                server_report = f"\n\n⚠️ 服务器检查失败: {e}"
        elif api_key and not ServerManager:
            # 修复：配置了 API_KEY 但模块加载失败时明确告警
            logger.error(f"{prefix}已配置 RAINYUN_API_KEY 但服务器管理模块加载失败: {_server_manager_error}")
            server_report = f"\n\n⚠️ 服务器管理模块加载失败: {_server_manager_error}"
        elif not api_key:
            logger.info(f"{prefix}未配置 RAINYUN_API_KEY，跳过服务器管理功能")

        # 3. 获取所有日志内容
        log_content = log_capture_buffer.getvalue()

        # 4. 发送通知（签到日志 + 服务器状态，一次性推送）
        logger.info("正在发送通知...")
        send("雨云签到", log_content + server_report)

        # 5. 释放内存
        if temp_dir and not debug:
            shutil.rmtree(temp_dir, ignore_errors=True)
        _set_log_user("")


def run() -> None:
    store = DataStore()
    data = store.load()
    if not data.accounts:
        logger.error("未配置任何账户，请先在 Web 面板中添加账户")
        return

    for account in data.accounts:
        if not account.enabled:
            continue
        config = Config.from_account(account, data.settings)
        success = run_with_config(config)
        account.last_checkin = datetime.now().isoformat()
        account.last_status = "success" if success else "failed"
        account_id = str(getattr(account, "id", "") or "").strip()
        account_name = str(getattr(account, "name", "") or "").strip()
        account_username = str(getattr(account, "username", "") or "").strip()
        user_label = account_name or account_username or account_id or "unknown"
        try:
            store.update_account(account)
        except Exception as exc:
            logger.error("用户 %s 回写账户状态失败: %s", user_label, exc)


if __name__ == "__main__":
    run()
