"""
Authentication management for Google Finance Beta.

Handles Google account login and session persistence.
First-time setup requires manual login, subsequent runs use saved state.
"""

import os
import logging
from pathlib import Path
from typing import Optional

from playwright.async_api import async_playwright, BrowserContext, Playwright

logger = logging.getLogger(__name__)

# Authentication state storage paths
AUTH_STATE_DIR = Path(__file__).parent.parent / "auth_state"
AUTH_STATE_FILE = AUTH_STATE_DIR / "google_finance_state.json"
USER_DATA_DIR = AUTH_STATE_DIR / "google_user_data"


class GoogleAuthManager:
    """Manages Google account authentication for Finance Beta."""

    @staticmethod
    def is_authenticated() -> bool:
        """Check if valid authentication state exists."""
        return AUTH_STATE_FILE.exists()

    @staticmethod
    def get_auth_state_path() -> Path:
        """Get the path to authentication state file."""
        return AUTH_STATE_FILE

    @staticmethod
    def get_user_data_dir() -> Path:
        """Get the path to user data directory."""
        return USER_DATA_DIR

    @staticmethod
    async def setup_authentication() -> bool:
        """
        Interactive setup: launches visible browser for user to log in.

        Returns:
            True if authentication was successful, False otherwise.
        """
        AUTH_STATE_DIR.mkdir(parents=True, exist_ok=True)

        async with async_playwright() as p:
            try:
                # Launch visible browser for manual login
                context = await p.chromium.launch_persistent_context(
                    user_data_dir=str(USER_DATA_DIR),
                    headless=False,
                    args=[
                        '--start-maximized',
                        '--disable-blink-features=AutomationControlled',
                    ],
                    viewport={'width': 1280, 'height': 800}
                )

                page = await context.new_page()
                await page.goto("https://accounts.google.com")

                print("\n" + "=" * 50)
                print("Google Finance 认证设置")
                print("=" * 50)
                print("\n请在浏览器中登录您的 Google 账号")
                print("登录完成后，返回此终端按 Enter 键继续...")
                print()

                input()  # Wait for user to complete login

                # Verify login by checking for profile elements
                try:
                    await page.goto("https://www.google.com")
                    # Try to find logged-in indicator
                    await page.wait_for_selector(
                        'a[aria-label*="Google Account"], img[data-noaft]',
                        timeout=5000
                    )
                    logger.info("Login verification successful")
                except Exception:
                    logger.warning("Could not verify login status, proceeding anyway")

                # Save authentication state
                await context.storage_state(path=str(AUTH_STATE_FILE))
                await context.close()

                print(f"\n✅ 认证状态已保存到: {AUTH_STATE_FILE}")
                return True

            except Exception as e:
                logger.error(f"Authentication setup failed: {e}")
                return False

    @staticmethod
    async def get_authenticated_context(playwright: Playwright) -> BrowserContext:
        """
        Get a browser context with authentication loaded.

        Args:
            playwright: Playwright instance

        Returns:
            Authenticated BrowserContext

        Raises:
            RuntimeError: If not authenticated
        """
        if not GoogleAuthManager.is_authenticated():
            raise RuntimeError(
                "未认证。请先运行 setup_google_auth.py 进行登录设置。\n"
                "Run: python setup_google_auth.py"
            )

        try:
            context = await playwright.chromium.launch_persistent_context(
                user_data_dir=str(USER_DATA_DIR),
                headless=True,
                storage_state=str(AUTH_STATE_FILE),
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                ],
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                viewport={'width': 1280, 'height': 800}
            )
            logger.info("Authenticated browser context created successfully")
            return context

        except Exception as e:
            logger.error(f"Failed to create authenticated context: {e}")
            raise RuntimeError(f"Failed to load authentication state: {e}")

    @staticmethod
    async def refresh_authentication() -> bool:
        """
        Refresh authentication by re-running setup.
        Use when authentication has expired.

        Returns:
            True if refresh was successful
        """
        # Remove old state
        if AUTH_STATE_FILE.exists():
            AUTH_STATE_FILE.unlink()

        # Run setup again
        return await GoogleAuthManager.setup_authentication()

    @staticmethod
    def clear_authentication():
        """Clear all authentication state."""
        import shutil

        if AUTH_STATE_FILE.exists():
            AUTH_STATE_FILE.unlink()
            logger.info(f"Removed {AUTH_STATE_FILE}")

        if USER_DATA_DIR.exists():
            shutil.rmtree(USER_DATA_DIR)
            logger.info(f"Removed {USER_DATA_DIR}")

        print("认证状态已清除。下次使用需要重新登录。")
