"""
聊天相关的API路由
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime

try:
    from ...models.schemas import ChatMessage, ChatResponse, AgentStatus
    from ...services.agent_service import agent_service
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    try:
        from models.schemas import ChatMessage, ChatResponse, AgentStatus
        from services.agent_service import agent_service
    except ImportError:
        from backend.models.schemas import ChatMessage, ChatResponse, AgentStatus
        from backend.services.agent_service import agent_service

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(message: ChatMessage):
    """
    与Agent聊天
    
    发送消息给Agent并获取响应
    """
    try:
        if not message.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # 处理消息
        response = await agent_service.process_message(message)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")


@router.get("/status", response_model=AgentStatus)
async def get_agent_status():
    """
    获取Agent当前状态
    
    返回Agent的当前状态、进度等信息
    """
    try:
        return agent_service.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")


@router.post("/agent/reset")
async def reset_agent():
    """
    重置Agent状态
    
    重置Agent到初始状态
    """
    try:
        # 重置状态
        agent_service.status.status = "idle"
        agent_service.status.current_step = 0
        agent_service.status.last_action = "Reset to initial state"
        
        return {"message": "Agent reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting agent: {str(e)}")


@router.post("/agent/initialize")
async def initialize_agent():
    """
    初始化Agent
    
    重新初始化Agent连接
    """
    try:
        await agent_service.initialize_agent()
        
        return {
            "message": "Agent initialization attempted",
            "demo_mode": agent_service.is_demo_mode
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing agent: {str(e)}")


@router.get("/agent/info")
async def get_agent_info():
    """
    获取Agent信息
    
    返回Agent的基本信息和能力描述
    """
    try:
        llm_config_info = agent_service.get_llm_config_info()
        
        return {
            "name": "OpenManus Agent",
            "version": "1.0.0",
            "demo_mode": agent_service.is_demo_mode,
            "status": agent_service.get_status(),
            "llm_config": llm_config_info,
            "capabilities": [
                "Natural language conversation",
                "Data analysis and visualization",
                "Code generation and review", 
                "File processing and management",
                "Web browsing and automation",
                "Search and research tasks"
            ],
            "supported_formats": [
                "Text files (.txt, .md)",
                "Code files (.py, .js, .ts, etc.)",
                "Data files (.csv, .json, .xlsx)",
                "Images (.png, .jpg, .gif)",
                "Documents (.pdf, .docx)"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting agent info: {str(e)}")

@router.get("/agent/config")
async def get_agent_config():
    """
    获取Agent配置信息
    
    返回当前加载的LLM配置信息（隐藏敏感字段）
    """
    try:
        return agent_service.get_llm_config_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting agent config: {str(e)}")