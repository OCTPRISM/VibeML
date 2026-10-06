"""
api/main.py  -  VibeML API

启动（后端 + 前端一起，同一个地址）：
    cd automl_agent
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

    然后打开浏览器访问 http://localhost:8000 即可使用完整产品界面。
    API 交互文档：http://localhost:8000/docs
"""
from __future__ import annotations
import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.version import __version__
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from api.models import HealthResponse, QueueStatsResponse
from api.routes.tasks import router as tasks_router
from api.routes.datasets import router as datasets_router
from api.routes.conversations import router as conversations_router
from api.routes.auth import router as auth_router
from api.routes.accounts import router as accounts_router
from api.routes.providers import router as providers_router
from api.routes.api_tokens import router as api_tokens_router
from api.routes.attachments import router as attachments_router, ATTACHMENT_DIR
from api.routes.compute_profiles import router as compute_profiles_router
from api.routes.exports import router as exports_router
from api.accounts.oauth_google import router as oauth_google_router
from api.queue import job_queue
from api.settings import settings
from api.store import task_store
from api.conversation_store import conversation_store

WEB_DIR = Path(__file__).parent.parent / "web"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 训练进度回调是从 asyncio.to_thread() 的子线程里直接调用 push_event() 的
    # （on_event 在同步流水线代码内部触发）——子线程里没有"当前事件循环"，绑定
    # 真正在跑的这一个 loop，push_event()/push_live() 之后统一走
    # call_soon_threadsafe，不再依赖会静默退化成不安全跨线程操作的 try/except
    # 兜底（详见 api/store.py::TaskStore.bind_loop 的注释）
    loop = asyncio.get_running_loop()
    task_store.bind_loop(loop)
    conversation_store.bind_loop(loop)
    await job_queue.start()
    print(f"🚀 VibeML API v{__version__}  "
          f"API_KEY={'已设置' if os.environ.get('ANTHROPIC_API_KEY') else '未设置'}")
    if WEB_DIR.exists():
        print(f"   前端界面：http://localhost:8000/  （静态文件目录：{WEB_DIR}）")
    else:
        print(f"   ⚠  未找到前端目录 {WEB_DIR}，'/' 将只返回 API 信息 JSON")
    yield
    await job_queue.stop()
    print("⏹  API 关闭")

app = FastAPI(
    title="Vibe ML Studio API",
    description=(
        "零门槛对话式 AutoML — Phase 4\n\n"
        "**端点：**\n"
        "- `POST /api/tasks` 提交训练任务（进入限流队列）\n"
        "- `WS /api/tasks/{id}/stream` 实时接收训练事件\n"
        "- `POST /api/tasks/{id}/predict` 用训练模型预测\n"
        "- `POST /api/tasks/{id}/feedback` 部署后反馈 → 重训决策\n"
        "- `GET /api/tasks/queue/stats` 队列状态\n\n"
        "**前端界面**：访问根路径 `/` 打开完整的产品界面（web/index.html）。"
    ),
    version=__version__,
    lifespan=lifespan,
)

# allow_origins=["*"] + allow_credentials=True 本身就是浏览器规范不允许的组合
# （通配符 origin 不能配合携带凭据的请求）——refresh token 走 HttpOnly cookie 之后
# 这个组合是真的会被浏览器拒绝，不是理论问题，必须用显式 origin 列表。
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_allowed_origins,
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(tasks_router)
app.include_router(datasets_router)
app.include_router(conversations_router)
app.include_router(auth_router)
app.include_router(accounts_router)
app.include_router(providers_router)
app.include_router(api_tokens_router)
app.include_router(oauth_google_router)
app.include_router(attachments_router)
app.include_router(compute_profiles_router)
app.include_router(exports_router)

# 头像文件：单独一个静态目录挂载（跟 api/routes/auth.py::AVATAR_DIR 是同一个路径），
# 不影响上面 "/" 和 "/app.js" 两条手写路由的既有推理方式——这是一批新增的、数量会
# 随用户增长的文件，用 StaticFiles 比再手写一条路由更合适。
from api.routes.auth import AVATAR_DIR
app.mount("/avatars", StaticFiles(directory=AVATAR_DIR), name="avatars")
# 聊天附件里的图片同理需要能被浏览器直接预览（文档类附件不需要——提取出来的文本
# 已经随消息本身返回了，没有必要再把原始文件也回传浏览器）
app.mount("/attachments", StaticFiles(directory=ATTACHMENT_DIR), name="attachments")

# 前端静态文件：用显式路由而不是 StaticFiles 整体挂载，避免路由匹配顺序
# 意外遮盖 /api/* 或 FastAPI 自带的 /docs —— 目前只有两个静态文件
# (index.html / app.js)，显式路由比挂载整个目录更容易推理。
# 若后续静态资源变多，再切换为 StaticFiles(directory=WEB_DIR) 并确保
# 在 include_router(tasks_router) 之后挂载。

_NO_CACHE_HEADERS = {"Cache-Control": "no-cache"}
# FileResponse 默认不带 Cache-Control，浏览器会按启发式规则缓存（尤其是本地开发时
# 频繁改 app.js/index.html 却看不到最新代码，troubleshoot 时曾亲身踩过这个坑）——
# 用 no-cache（配合自带的 ETag/Last-Modified 做条件请求）而不是完全禁用缓存，
# 兼顾"每次都能拿到最新代码"和"没改动时仍可用 304 省流量"。

@app.get("/", include_in_schema=False)
async def root():
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path, headers=_NO_CACHE_HEADERS)
    return JSONResponse({"name": "VibeML API", "version": __version__, "docs": "/docs",
                          "note": f"前端文件未找到（期望路径：{index_path}）"})

@app.get("/app.js", include_in_schema=False)
async def app_js():
    app_js_path = WEB_DIR / "app.js"
    if app_js_path.exists():
        return FileResponse(app_js_path, media_type="text/javascript", headers=_NO_CACHE_HEADERS)
    return JSONResponse({"error": f"app.js 未找到（期望路径：{app_js_path}）"}, status_code=404)

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse()

@app.exception_handler(Exception)
async def global_exc(request, exc):
    # JSONResponse(content, status_code=...) 顺序——之前是 JSONResponse(500, {...})，
    # 把 500 当 content、字典当 status_code 传反了，遇到真的未捕获异常时这里会二次报错
    return JSONResponse({"error": type(exc).__name__, "message": str(exc)}, status_code=500)
