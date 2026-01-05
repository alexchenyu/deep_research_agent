#!/usr/bin/env python3
"""
Google Finance Beta 认证设置脚本

首次使用 Google Finance 工具前，需要运行此脚本进行 Google 账号登录。
登录状态会保存在 auth_state/ 目录中，后续使用无需再次登录。

使用方法:
    python setup_google_auth.py

如需清除认证状态:
    python setup_google_auth.py --clear
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from google_finance.auth import GoogleAuthManager


async def setup():
    """Run authentication setup."""
    print("\n" + "=" * 60)
    print("Google Finance Beta 认证设置")
    print("=" * 60)

    if GoogleAuthManager.is_authenticated():
        print("\n已检测到现有认证状态。")
        response = input("是否要重新认证？(y/N): ").strip().lower()
        if response != 'y':
            print("保持现有认证状态。")
            return True

    print("\n即将打开浏览器，请在浏览器中登录您的 Google 账号。")
    print("注意：建议使用个人账号而非工作账号。\n")

    success = await GoogleAuthManager.setup_authentication()

    if success:
        print("\n" + "=" * 60)
        print("认证设置完成！")
        print("现在可以使用 Google Finance 相关工具了。")
        print("=" * 60)
    else:
        print("\n认证设置失败，请重试。")

    return success


def clear():
    """Clear authentication state."""
    print("\n清除认证状态...")
    GoogleAuthManager.clear_authentication()
    print("完成。下次使用需要重新登录。")


def check_status():
    """Check current authentication status."""
    print("\n" + "=" * 60)
    print("Google Finance 认证状态")
    print("=" * 60)

    if GoogleAuthManager.is_authenticated():
        auth_path = GoogleAuthManager.get_auth_state_path()
        print(f"\n状态: 已认证 ✓")
        print(f"认证文件: {auth_path}")
    else:
        print(f"\n状态: 未认证 ✗")
        print("请运行 'python setup_google_auth.py' 进行认证设置。")


def main():
    parser = argparse.ArgumentParser(
        description="Google Finance Beta 认证管理工具"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="清除现有认证状态"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="检查当前认证状态"
    )

    args = parser.parse_args()

    if args.clear:
        clear()
    elif args.status:
        check_status()
    else:
        asyncio.run(setup())


if __name__ == "__main__":
    main()
