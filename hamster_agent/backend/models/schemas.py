"""
数据模型和Pydantic Schemas
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """聊天消息模型"""
    message: str = Field(..., max_length=4000, description="用户消息内容")
    timestamp: Optional[str] = Field(None, description="消息时间戳")
    
    class Config:
        example = {
            "message": "Hello, how can you help me?",
            "timestamp": "2024-01-01T00:00:00Z"
        }


class ChatResponse(BaseModel):
    """聊天响应模型"""
    response: str = Field(..., description="Agent响应内容")
    status: str = Field(..., description="响应状态")
    timestamp: str = Field(..., description="响应时间戳")
    
    class Config:
        example = {
            "response": "Hello! I can help you with various tasks...",
            "status": "success",
            "timestamp": "2024-01-01T00:00:00Z"
        }


class AgentStatus(BaseModel):
    """Agent状态模型"""
    status: str = Field(..., description="当前状态: idle, processing, completed, error")
    current_step: int = Field(0, ge=0, description="当前执行步骤")
    max_steps: int = Field(20, ge=1, description="最大执行步骤数")
    last_action: Optional[str] = Field(None, description="最后执行的动作")
    
    class Config:
        example = {
            "status": "processing",
            "current_step": 5,
            "max_steps": 20,
            "last_action": "Analyzing user request"
        }


class WorkspaceFile(BaseModel):
    """工作空间文件模型"""
    name: str = Field(..., description="文件名")
    path: str = Field(..., description="相对路径")
    size: int = Field(..., ge=0, description="文件大小（字节）")
    modified: str = Field(..., description="修改时间")
    
    class Config:
        example = {
            "name": "example.txt",
            "path": "documents/example.txt",
            "size": 1024,
            "modified": "2024-01-01T00:00:00Z"
        }


class WorkspaceResponse(BaseModel):
    """工作空间响应模型"""
    files: List[WorkspaceFile] = Field([], description="文件列表")
    workspace_path: str = Field(..., description="工作空间路径")
    
    class Config:
        example = {
            "files": [
                {
                    "name": "example.txt",
                    "path": "example.txt",
                    "size": 1024,
                    "modified": "2024-01-01T00:00:00Z"
                }
            ],
            "workspace_path": "/workspace"
        }


class FileUploadResponse(BaseModel):
    """文件上传响应模型"""
    message: str = Field(..., description="上传结果消息")
    filename: str = Field(..., description="文件名")
    size: int = Field(..., ge=0, description="文件大小")
    path: str = Field(..., description="文件路径")
    
    class Config:
        example = {
            "message": "File uploaded successfully",
            "filename": "document.pdf",
            "size": 2048,
            "path": "document.pdf"
        }


class ConfigUpdate(BaseModel):
    """配置更新模型"""
    llm: Optional[Dict[str, Any]] = Field(None, description="LLM配置")
    browser: Optional[Dict[str, Any]] = Field(None, description="浏览器配置")
    search: Optional[Dict[str, Any]] = Field(None, description="搜索配置")
    sandbox: Optional[Dict[str, Any]] = Field(None, description="沙箱配置")
    
    class Config:
        example = {
            "llm": {
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 4096
            },
            "browser": {
                "headless": True,
                "disable_security": True
            }
        }


class WebSocketMessage(BaseModel):
    """WebSocket消息模型"""
    type: str = Field(..., description="消息类型")
    data: Dict[str, Any] = Field(..., description="消息数据")
    timestamp: Optional[str] = Field(None, description="时间戳")
    
    class Config:
        example = {
            "type": "user_message",
            "data": {
                "message": "Hello",
                "user_id": "user123"
            },
            "timestamp": "2024-01-01T00:00:00Z"
        }


class ErrorResponse(BaseModel):
    """错误响应模型"""
    detail: str = Field(..., description="错误详情")
    error_code: Optional[str] = Field(None, description="错误码")
    timestamp: str = Field(..., description="错误时间戳")
    
    class Config:
        example = {
            "detail": "Invalid request parameters",
            "error_code": "INVALID_PARAMS",
            "timestamp": "2024-01-01T00:00:00Z"
        }


# 新增流式消息模型
class AgentThinkData(BaseModel):
    """Agent思考数据"""
    content: str = Field(..., description="思考内容")
    step: int = Field(..., ge=0, description="当前步骤")
    reasoning: Optional[str] = Field(None, description="推理过程")


class AgentActionData(BaseModel):
    """Agent行动数据"""
    tool_name: str = Field(..., description="工具名称")
    tool_args: Dict[str, Any] = Field(..., description="工具参数")
    step: int = Field(..., ge=0, description="当前步骤")


class AgentObservationData(BaseModel):
    """Agent观察数据"""
    tool_name: str = Field(..., description="执行的工具名称")
    result: str = Field(..., description="执行结果")
    step: int = Field(..., ge=0, description="当前步骤")
    success: bool = Field(..., description="是否成功执行")


class AgentStreamMessage(BaseModel):
    """Agent流式消息"""
    message_type: str = Field(..., description="消息类型: think, act, observe, status, complete")
    data: Dict[str, Any] = Field(..., description="消息数据")
    step: int = Field(..., ge=0, description="当前步骤")
    total_steps: Optional[int] = Field(None, description="总步骤数")
    timestamp: str = Field(..., description="时间戳")
    
    class Config:
        example = {
            "message_type": "think",
            "data": {
                "content": "I need to analyze the user's request...",
                "reasoning": "The user is asking about data analysis"
            },
            "step": 1,
            "total_steps": 5,
            "timestamp": "2024-01-01T00:00:00Z"
        }


class FlowConfiguration(BaseModel):
    """Flow配置模型"""
    mode: str = Field(..., description="工作模式: single_agent, planning, game_data_analysis, data_analysis_flow")
    primaryAgent: str = Field(..., description="主要Agent类型")
    selectedAgents: List[str] = Field(default=[], description="选中的Agent列表(多Agent模式)")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Flow特定参数")
    
    class Config:
        example = {
            "mode": "data_analysis_flow",
            "primaryAgent": "DataAnalysisExpert", 
            "selectedAgents": ["ExcelCleanAgent", "GameDataAnalysisAgent"],
            "parameters": {
                "data_file_path": "C:\\data\\analysis.xlsx",
                "new_version_like": "v2024"
            }
        }


class FlowConfigResponse(BaseModel):
    """Flow配置响应"""
    success: bool = Field(..., description="配置是否成功")
    message: str = Field(..., description="响应消息")
    applied_config: FlowConfiguration = Field(..., description="已应用的配置")
    available_agents: List[str] = Field(..., description="可用的Agent列表")
    
    class Config:
        example = {
            "success": True,
            "message": "Flow configuration applied successfully",
            "applied_config": {
                "mode": "game_data_analysis",
                "primaryAgent": "DataAnalysisExpert",
                "selectedAgents": ["ExcelCleanAgent", "GameDataAnalysisAgent"]
            },
            "available_agents": ["manus", "DataAnalysisExpert", "ExcelCleanAgent"]
        }