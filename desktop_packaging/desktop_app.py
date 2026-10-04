"""
desktop_packaging/desktop_app.py  -  轻量本机版桌面应用入口。

目录名特意不叫 "packaging"——那正好是 pip/setuptools/briefcase 自己都依赖的
真实 PyPI 包名（`import packaging` 解析版本号用），仓库根目录下建一个同名目录
会在 sys.path 解析时和真正的 packaging 库冲突（实测触发过 ModuleNotFoundError），
所以用 desktop_packaging 这个不会撞名的名字。

沿用现有 FastAPI + web/index.html 整套代码，不新建任何前端/后端逻辑——
本机版只是把它们打包运行在一个原生窗口里（pywebview），指向本地
127.0.0.1 上跑的同一个 uvicorn 实例。本地这个 FastAPI 实例该联网的地方
（登录/配额校验/上报）继续联网访问远程账号服务器（api/accounts/remote_client.py），
其它（本地 Ollama/vLLM 调用、core/pipeline.py 训练本身）完全保持本地不变。

不能用 Briefcase 默认的项目脚手架布局（Briefcase 期望 src/<app_name>/ 结构），
这里手写一个最小入口，靠 pyproject.toml 里的 [tool.briefcase] 配置指向它，
避免为了适配打包工具重排现有仓库结构。
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path

# 必须在 api.settings 第一次被 import 之前设置——pydantic-settings 的 Settings()
# 单例在模块导入时按当前环境变量构造好就不再变化，晚设置不生效。这里在任何
# uvicorn/api 相关 import 发生之前设置，保证 api/settings.py::settings.is_desktop_build
# 读到的是 True（Phase 6 离线宽限期只在本机版桌面构建下生效，网络版部署不设这个变量）。
os.environ.setdefault("IS_DESKTOP_BUILD", "1")

# uvicorn.run("api.main:app", ...) 的字符串形式是相对当前 sys.path 解析的——
# 直接 `python3 desktop_packaging/desktop_app.py` 运行时 sys.path[0] 是脚本所在
# 目录 desktop_packaging/，不是仓库根目录，导致 `import api` 失败（实测踩过）。
# 显式把仓库根目录（这个文件的上一级）插到 sys.path 最前面，不管从哪里/怎么调用
# 这个入口都能正确 import 到 api 包——Briefcase 打包后的实际目录布局会不一样，
# 这个修复对两种场景都需要，不是针对某一种调用方式的权宜之计。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_server(port: int) -> None:
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=port, log_level="warning")


def _wait_for_server(port: int, timeout: float = 15.0) -> bool:
    import httpx
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
            return True
        except Exception:
            time.sleep(0.2)
    return False


def main() -> None:
    import webview

    port = _find_free_port()
    server_thread = threading.Thread(target=_run_server, args=(port,), daemon=True)
    server_thread.start()
    _wait_for_server(port)

    webview.create_window("Vibe ML Studio", f"http://127.0.0.1:{port}/", width=1280, height=860)
    webview.start()


if __name__ == "__main__":
    main()
