"""页面对象封装。"""

import logging
import re
import time
from typing import Callable

from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

from rainyun.browser.cookies import save_cookies
from rainyun.browser.locators import XPATH_CONFIG
from rainyun.browser.session import RuntimeContext
from rainyun.browser.urls import build_app_url

logger = logging.getLogger(__name__)
CaptchaHandler = Callable[[RuntimeContext], bool]


class LoginPage:
    _LOGIN_MAX_ATTEMPTS = 2
    _LOGIN_REDIRECT_WAIT_SECONDS = 20
    _LOGIN_CAPTCHA_WAIT_SECONDS = 8

    def __init__(self, ctx: RuntimeContext, captcha_handler: CaptchaHandler) -> None:
        self.ctx = ctx
        self.captcha_handler = captcha_handler

    def check_login_status(self) -> bool:
        """检查是否已登录。"""
        user_label = self.ctx.config.display_name or self.ctx.config.rainyun_user
        self.ctx.driver.get(build_app_url(self.ctx.config, "/dashboard"))
        time.sleep(3)
        # 如果跳转到登录页面，说明 cookie 失效
        if "login" in self.ctx.driver.current_url:
            logger.info(f"用户 {user_label} Cookie 已失效，需要重新登录")
            return False
        # 检查是否成功加载 dashboard
        if self.ctx.driver.current_url == build_app_url(self.ctx.config, "/dashboard"):
            logger.info(f"用户 {user_label} Cookie 有效，已登录")
            return True
        return False

    def _submit_login_form(self, user: str, pwd: str, user_label: str) -> bool:
        try:
            username = self.ctx.wait.until(EC.visibility_of_element_located((By.NAME, "login-field")))
            password = self.ctx.wait.until(EC.visibility_of_element_located((By.NAME, "login-password")))
            login_button = self.ctx.wait.until(
                EC.visibility_of_element_located((By.XPATH, XPATH_CONFIG["LOGIN_BTN"]))
            )
            username.clear()
            password.clear()
            username.send_keys(user)
            password.send_keys(pwd)
            login_button.click()
            return True
        except TimeoutException:
            logger.error(f"用户 {user_label} 页面加载超时，请尝试延长超时时间或切换到国内网络环境！")
            return False

    def _handle_login_captcha(self, user_label: str, wait_seconds: int) -> bool:
        try:
            captcha_wait = WebDriverWait(self.ctx.driver, wait_seconds, poll_frequency=0.5)
            captcha_wait.until(EC.visibility_of_element_located((By.ID, "tcaptcha_iframe_dy")))
            logger.warning(f"用户 {user_label} 触发验证码！")
            self.ctx.driver.switch_to.frame("tcaptcha_iframe_dy")
            if not self.captcha_handler(self.ctx):
                logger.error(f"用户 {user_label} 登录验证码识别失败")
                return False
            return True
        except TimeoutException:
            logger.info(f"用户 {user_label} 未触发验证码")
            return True
        finally:
            self.ctx.driver.switch_to.default_content()

    def _wait_login_redirect(self) -> bool:
        wait_seconds = max(self.ctx.config.timeout, self._LOGIN_REDIRECT_WAIT_SECONDS)
        redirect_wait = WebDriverWait(self.ctx.driver, wait_seconds, poll_frequency=0.5)
        try:
            redirect_wait.until(EC.url_contains("dashboard"))
            return True
        except TimeoutException:
            return False

    def login(self, user: str, pwd: str) -> bool:
        """执行登录流程。"""
        user_label = self.ctx.config.display_name or user
        logger.info(f"用户 {user_label} 发起登录请求")
        self.ctx.driver.get(build_app_url(self.ctx.config, "/auth/login"))
        for attempt in range(1, self._LOGIN_MAX_ATTEMPTS + 1):
            if not self._submit_login_form(user, pwd, user_label):
                return False

            captcha_wait_seconds = max(self.ctx.config.timeout, self._LOGIN_CAPTCHA_WAIT_SECONDS)
            if attempt > 1:
                captcha_wait_seconds = self._LOGIN_CAPTCHA_WAIT_SECONDS

            if not self._handle_login_captcha(user_label, wait_seconds=captcha_wait_seconds):
                return False

            time.sleep(2)  # 给页面一点点缓冲时间
            if self._wait_login_redirect():
                logger.info(f"用户 {user_label} 登录成功！")
                return True

            current_url = self.ctx.driver.current_url
            if attempt < self._LOGIN_MAX_ATTEMPTS and "/auth/login" in current_url:
                logger.warning(
                    "用户 %s 第 %s 次登录后仍停留登录页，自动重试一次。当前 URL: %s",
                    user_label,
                    attempt,
                    current_url,
                )
                continue
            break

        logger.error(f"用户 {user_label} 登录超时或失败！当前 URL: {self.ctx.driver.current_url}")
        return False


class RewardPage:
    _REWARD_PAGE_PATH = "/account/reward/earn"
    _REWARD_PAGE_URL_WAIT_SECONDS = 8
    _REWARD_PAGE_MENU_XPATH = "//a[contains(@href, '/account/reward/earn')]"
    _DAILY_SIGN_CLAIM_TEXTS = ("领取奖励", "去完成", "去签到")
    _DAILY_SIGN_CLAIM_XPATH = "//*[self::a or self::button][contains(normalize-space(.), '领取奖励') or contains(normalize-space(.), '去完成') or contains(normalize-space(.), '去签到')]"
    # 只在“每日签到”模块内匹配这些文案，避免“已完成”在其他任务卡片出现导致误判
    _DAILY_SIGN_DONE_PATTERNS = ("已完成", "已领取", "已签到", "明日再来")
    _DAILY_SIGN_SECTION_WAIT_SECONDS = 25
    _DAILY_SIGN_DONE_WAIT_SECONDS = 12

    def __init__(self, ctx: RuntimeContext, captcha_handler: CaptchaHandler) -> None:
        self.ctx = ctx
        self.captcha_handler = captcha_handler

    def _wait_reward_page_url(self, timeout: int | None = None) -> bool:
        if timeout is None:
            timeout = self._REWARD_PAGE_URL_WAIT_SECONDS
        wait = WebDriverWait(self.ctx.driver, timeout, poll_frequency=0.5)
        try:
            return bool(wait.until(EC.url_contains(self._REWARD_PAGE_PATH)))
        except TimeoutException:
            return False

    def _click_reward_menu_link(self) -> bool:
        """优先通过站内菜单跳转奖励页，避免直接重载被守卫重定向。"""

        try:
            links = self.ctx.driver.find_elements(By.XPATH, self._REWARD_PAGE_MENU_XPATH)
        except Exception:
            return False
        if not links:
            return False

        visible_links: list = []
        hidden_links: list = []
        for link in links:
            try:
                if link.is_displayed() and link.is_enabled():
                    visible_links.append(link)
                else:
                    hidden_links.append(link)
            except Exception:
                hidden_links.append(link)

        for link in [*visible_links, *hidden_links]:
            try:
                link.click()
                return True
            except Exception:
                try:
                    self.ctx.driver.execute_script("arguments[0].click();", link)
                    return True
                except Exception:
                    continue
        return False

    def open(self) -> bool:
        """打开奖励页；按“菜单点击→直接 URL→JS 跳转”顺序尝试。"""

        target_url = build_app_url(self.ctx.config, self._REWARD_PAGE_PATH)
        user_label = self.ctx.config.display_name or self.ctx.config.rainyun_user
        before_url = self.ctx.driver.current_url

        clicked = self._click_reward_menu_link()
        if clicked and self._wait_reward_page_url():
            logger.info("用户 %s 通过站内菜单进入奖励页", user_label)
            return True

        self.ctx.driver.get(target_url)
        if self._wait_reward_page_url():
            logger.info("用户 %s 通过直接 URL 进入奖励页", user_label)
            return True

        fallback_url = f"{target_url}?_ts={int(time.time() * 1000)}"
        try:
            self.ctx.driver.execute_script("window.location.assign(arguments[0]);", fallback_url)
        except Exception:
            self.ctx.driver.get(fallback_url)

        if self._wait_reward_page_url(timeout=max(self._REWARD_PAGE_URL_WAIT_SECONDS, 10)):
            logger.info("用户 %s 通过 JS 跳转进入奖励页", user_label)
            return True

        logger.warning(
            "用户 %s 奖励页导航未命中: from=%s, target=%s, current=%s, menu_clicked=%s",
            user_label,
            before_url,
            target_url,
            self.ctx.driver.current_url,
            clicked,
        )
        return False

    def _wait_daily_sign_section_ready(self, timeout: int | None = None) -> bool:
        """等待“每日签到”模块就绪。

        兼容 SPA 延迟渲染与局部结构调整：
        1) 优先等待标准 header/card 定位
        2) 兜底等待“每日签到”文本 + 可领取按钮同时出现
        """

        if timeout is None:
            timeout = max(self._DAILY_SIGN_SECTION_WAIT_SECONDS, self.ctx.config.timeout)

        wait = WebDriverWait(self.ctx.driver, timeout, poll_frequency=0.5)

        def _probe(driver) -> bool:
            if driver.find_elements(By.XPATH, XPATH_CONFIG["SIGN_IN_HEADER"]):
                return True
            if driver.find_elements(By.XPATH, XPATH_CONFIG["SIGN_IN_CARD"]):
                return True

            has_daily_sign_span = bool(
                driver.find_elements(By.XPATH, "//span[contains(normalize-space(.), '每日签到')]")
            )
            has_claim_button = bool(driver.find_elements(By.XPATH, self._DAILY_SIGN_CLAIM_XPATH))
            return has_daily_sign_span and has_claim_button

        try:
            return bool(wait.until(lambda driver: _probe(driver)))
        except TimeoutException:
            return False

    def _get_daily_sign_header_text(self) -> str:
        """读取“每日签到”卡片头部可见文本。

        注意：必须限定在每日签到模块范围内匹配，避免全页扫文案导致误判。
        """

        try:
            elements = self.ctx.driver.find_elements(By.XPATH, XPATH_CONFIG["SIGN_IN_HEADER"])
            if not elements:
                return ""
            header = elements[0]
            raw_text = (header.get_attribute("innerText") or header.text or "").strip()
            return re.sub(r"\s+", " ", raw_text).strip()
        except Exception:
            return ""

    def _get_daily_sign_card_text(self) -> str:
        """读取“每日签到”卡片可见文本（用于失败诊断）。"""

        try:
            elements = self.ctx.driver.find_elements(By.XPATH, XPATH_CONFIG["SIGN_IN_CARD"])
            if not elements:
                return ""
            card = elements[0]
            raw_text = (card.get_attribute("innerText") or card.text or "").strip()
            return re.sub(r"\s+", " ", raw_text).strip()
        except Exception:
            return ""

    def _get_daily_sign_snapshot(self) -> dict[str, str | int]:
        """采集每日签到模块的调试快照，用于定位偶发误判。"""

        header_text = self._get_daily_sign_header_text()
        card_text = self._get_daily_sign_card_text()
        try:
            header_count = len(self.ctx.driver.find_elements(By.XPATH, XPATH_CONFIG["SIGN_IN_HEADER"]))
        except Exception:
            header_count = -1
        try:
            button_count = len(self.ctx.driver.find_elements(By.XPATH, XPATH_CONFIG["SIGN_IN_BTN"]))
        except Exception:
            button_count = -1

        try:
            daily_sign_span_count = len(
                self.ctx.driver.find_elements(By.XPATH, "//span[contains(normalize-space(.), '每日签到')]")
            )
        except Exception:
            daily_sign_span_count = -1

        try:
            claim_button_count = len(self.ctx.driver.find_elements(By.XPATH, self._DAILY_SIGN_CLAIM_XPATH))
        except Exception:
            claim_button_count = -1

        try:
            page_source = self.ctx.driver.page_source or ""
            has_daily_sign_text = "yes" if "每日签到" in page_source else "no"
        except Exception:
            has_daily_sign_text = "unknown"

        current_url = (self.ctx.driver.current_url or "").strip()
        title = (self.ctx.driver.title or "").strip()
        return {
            "header_count": header_count,
            "button_count": button_count,
            "daily_sign_span_count": daily_sign_span_count,
            "claim_button_count": claim_button_count,
            "has_daily_sign_text": has_daily_sign_text,
            "current_url": current_url,
            "title": title[:80],
            "header_text": header_text,
            "card_excerpt": card_text[:180],
        }

    def _detect_daily_sign_done_pattern(self) -> str | None:
        header_text = self._get_daily_sign_header_text()
        for pattern in self._DAILY_SIGN_DONE_PATTERNS:
            if pattern in header_text:
                return pattern
        return None

    def _wait_daily_sign_done_pattern(self, timeout: int | None = None) -> str | None:
        if timeout is None:
            timeout = self._DAILY_SIGN_DONE_WAIT_SECONDS
        wait = WebDriverWait(self.ctx.driver, timeout, poll_frequency=0.5)
        try:
            return wait.until(lambda driver: self._detect_daily_sign_done_pattern() or False)
        except TimeoutException:
            return None

    def handle_daily_reward(self, start_points: int) -> dict:
        user_label = self.ctx.config.display_name or self.ctx.config.rainyun_user
        opened = self.open()
        if not opened:
            logger.warning("用户 %s 奖励页 URL 未及时命中，当前 URL: %s", user_label, self.ctx.driver.current_url)

        if not self._wait_daily_sign_section_ready():
            snapshot = self._get_daily_sign_snapshot()
            logger.warning(
                "用户 %s 奖励页首次加载未就绪，尝试刷新重试: url=%s, title=%s, header_count=%s, daily_sign_span_count=%s, claim_button_count=%s, has_daily_sign_text=%s",
                user_label,
                snapshot["current_url"],
                snapshot["title"],
                snapshot["header_count"],
                snapshot["daily_sign_span_count"],
                snapshot["claim_button_count"],
                snapshot["has_daily_sign_text"],
            )
            self.ctx.driver.refresh()
            if not self._wait_daily_sign_section_ready(timeout=max(self.ctx.config.timeout, 15)):
                snapshot = self._get_daily_sign_snapshot()
                logger.error(
                    "用户 %s 奖励页加载诊断: url=%s, title=%s, header_count=%s, button_count=%s, daily_sign_span_count=%s, claim_button_count=%s, has_daily_sign_text=%s, header_text=%s, card_excerpt=%s",
                    user_label,
                    snapshot["current_url"],
                    snapshot["title"],
                    snapshot["header_count"],
                    snapshot["button_count"],
                    snapshot["daily_sign_span_count"],
                    snapshot["claim_button_count"],
                    snapshot["has_daily_sign_text"],
                    snapshot["header_text"],
                    snapshot["card_excerpt"],
                )
                raise Exception("奖励页加载超时：未找到每日签到模块，可能页面结构已变更")

        done_pattern = self._detect_daily_sign_done_pattern()
        if done_pattern:
            logger.info(
                f":down_arrow: 用户 {user_label} 今日已签到（每日签到模块检测到：{done_pattern}），跳过签到流程"
            )
            current_points, earned = self._log_points(start_points)
            return {
                "status": "already_signed",
                "current_points": current_points,
                "earned": earned,
            }

        try:
            # 使用显式等待寻找可点击按钮（只针对“每日签到”模块内的领取按钮）
            earn = self.ctx.wait.until(EC.element_to_be_clickable((By.XPATH, XPATH_CONFIG["SIGN_IN_BTN"])))
            logger.info(f"用户 {user_label} 点击领取奖励")
            try:
                earn.click()
            except Exception:
                # 兜底：遇到遮罩/重渲染导致 click 失败时尝试 JS 点击
                self.ctx.driver.execute_script("arguments[0].click();", earn)
        except TimeoutException:
            done_pattern = self._detect_daily_sign_done_pattern()
            if done_pattern:
                logger.info(
                    f":down_arrow: 用户 {user_label} 今日已签到（每日签到模块检测到：{done_pattern}），跳过签到流程"
                )
                current_points, earned = self._log_points(start_points)
                return {
                    "status": "already_signed",
                    "current_points": current_points,
                    "earned": earned,
                }

            snapshot = self._get_daily_sign_snapshot()
            logger.error(
                "用户 %s 签到按钮诊断: header_count=%s, button_count=%s, header_text=%s, card_excerpt=%s",
                user_label,
                snapshot["header_count"],
                snapshot["button_count"],
                snapshot["header_text"],
                snapshot["card_excerpt"],
            )
            header_text = self._get_daily_sign_header_text()
            if any(claim_text in header_text for claim_text in self._DAILY_SIGN_CLAIM_TEXTS):
                raise Exception("未找到每日签到按钮（模块仍显示可领取），可能是页面渲染延迟或结构变更")
            raise Exception("未找到每日签到按钮，且未检测到已签到状态，可能页面结构已变更")

        logger.info(f"用户 {user_label} 处理验证码")
        try:
            self.ctx.wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, "tcaptcha_iframe_dy")))
            if not self.captcha_handler(self.ctx):
                logger.error(
                    f"用户 {user_label} 验证码重试次数过多，签到失败。当前页面状态: {self.ctx.driver.current_url}"
                )
                raise Exception("验证码识别重试次数过多，签到失败")
        except TimeoutException:
            # 极少数情况下可能不触发验证码：直接走状态判定，避免无意义失败
            logger.info(f"用户 {user_label} 未触发验证码")
        finally:
            self.ctx.driver.switch_to.default_content()

        done_pattern = self._wait_daily_sign_done_pattern()
        if not done_pattern:
            header_text = self._get_daily_sign_header_text()
            snapshot = self._get_daily_sign_snapshot()
            logger.error(
                "用户 %s 签到后状态诊断: header_count=%s, button_count=%s, header_text=%s, card_excerpt=%s",
                user_label,
                snapshot["header_count"],
                snapshot["button_count"],
                snapshot["header_text"],
                snapshot["card_excerpt"],
            )
            if any(claim_text in header_text for claim_text in self._DAILY_SIGN_CLAIM_TEXTS):
                raise Exception("验证码处理后每日签到仍显示可领取，未检测到完成状态，签到可能失败")
            raise Exception("验证码处理结束后未检测到每日签到完成状态，可能页面结构已变更")

        current_points, earned = self._log_points(start_points)
        logger.info(f"用户 {user_label} 签到成功（每日签到模块检测到：{done_pattern}）")
        return {
            "status": "signed",
            "current_points": current_points,
            "earned": earned,
        }

    def _log_points(self, start_points: int) -> tuple[int | None, int | None]:
        user_label = self.ctx.config.display_name or self.ctx.config.rainyun_user
        try:
            current_points = self.ctx.api.get_user_points()
            earned = current_points - start_points
            logger.info(
                f"用户 {user_label} 当前剩余积分: {current_points} (本次获得 {earned} 分) | 约为 {current_points / self.ctx.config.points_to_cny_rate:.2f} 元"
            )
            return current_points, earned
        except Exception:
            logger.info(f"用户 {user_label} 无法通过 API 获取当前积分信息")
            return None, None
