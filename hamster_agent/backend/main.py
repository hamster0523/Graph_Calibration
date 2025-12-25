#!/usr/bin/env python3
"""
OpenManus Backend API Server

主要的FastAPI应用程序入口点
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

import sys
from pathlib import Path

# 添加必要的路径到sys.path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

try:
    from api.routes import chat, workspace, config, websocket, flow_config
    from services.connection_manager import manager
    from services.agent_service import agent_service
except ImportError as e:
    # 如果导入失败，提供更好的错误信息
    print(f"Import error: {e}")
    print("Possible solutions:")
    print("   1. Run from project root: python -m backend.main")
    print("   2. Run from backend dir: python run_server.py")
    print("   3. Use launcher: python start_ui.py")
    sys.exit(1)

# 创建FastAPI应用
app = FastAPI(
    title="OpenManus Backend API",
    version="1.0.0",
    description="OpenManus Agent Web Interface Backend"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件服务
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=frontend_dir / "static"), name="static")

# 注册API路由
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(workspace.router, prefix="/api", tags=["workspace"])
app.include_router(config.router, prefix="/api", tags=["config"])
app.include_router(flow_config.router, prefix="/api", tags=["flow-config"])
app.include_router(websocket.router, tags=["websocket"])

# 主页路由 - 服务前端应用
@app.get("/")
async def serve_frontend():
    """服务前端应用"""
    from fastapi.responses import FileResponse
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    else:
        return {"message": "Frontend not found. Please build the frontend first."}

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "service": "OpenManus Backend",
        "version": "1.0.0",
        "agent_mode": "real" if not agent_service.is_demo_mode else "demo"
    }

@app.on_event("startup")
async def startup_event():
    """应用启动时的事件"""
    print("🔄 Initializing OpenManus Agent...")
    await agent_service.initialize_agent()

    if agent_service.is_demo_mode:
        print("🌟 Agent initialized in demo mode")
    else:
        print("🤖 Real OpenManus Agent initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时的事件"""
    print("🧹 Cleaning up Agent resources...")
    await agent_service.cleanup()
    print("✅ Cleanup completed")

def main():
    """主函数"""
    print("🚀 Starting OpenManus Backend Server...")
    print("=" * 50)
    print("📡 Backend API: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🔌 WebSocket: ws://localhost:8000/ws")
    print("🎨 Frontend: http://localhost:8000")
    print("\n🛑 Press Ctrl+C to stop the server")
    print("-" * 50)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info")

if __name__ == "__main__":
    main()
