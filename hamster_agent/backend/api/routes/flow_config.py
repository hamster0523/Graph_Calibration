"""
Flow配置相关的API路由
"""

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

try:
    from ...models.schemas import FlowConfigResponse, FlowConfiguration
    from ...services.agent_service import agent_service
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    try:
        from models.schemas import FlowConfigResponse, FlowConfiguration
        from services.agent_service import agent_service
    except ImportError:
        from backend.models.schemas import FlowConfigResponse, FlowConfiguration
        from backend.services.agent_service import agent_service

router = APIRouter()


@router.post("/flow-config", response_model=FlowConfigResponse)
async def configure_flow(config: FlowConfiguration):
    """
    配置Flow和Agent

    接收前端的flow配置请求，配置相应的agent或flow
    """
    try:
        print(f"🔧 Configuring flow: {config.mode}")
        print(f"   Primary agent: {config.primaryAgent}")
        print(f"   Selected agents: {config.selectedAgents}")
        if config.parameters:
            print(f"   Parameters: {config.parameters}")

        # 保存配置
        agent_service.flow_config = config

        # 清理现有的agent和flow
        await agent_service.cleanup()

        # 根据配置创建新的agent或flow
        if config.mode == "single_agent":
            await agent_service._initialize_single_agent(config.primaryAgent)
        else:
            await agent_service._initialize_flow(config)

        # 返回成功响应
        return FlowConfigResponse(
            success=True,
            message=f"Successfully configured {config.mode} mode",
            applied_config=config,
            available_agents=agent_service.get_available_agents(),
        )

    except Exception as e:
        print(f"❌ Error configuring flow: {e}")
        return FlowConfigResponse(
            success=False,
            message=f"Failed to configure flow: {str(e)}",
            applied_config=config,
            available_agents=agent_service.get_available_agents(),
        )


@router.get("/flow-config")
async def get_current_flow_config():
    """
    获取当前的Flow配置

    返回当前active的flow配置信息
    """
    try:
        current_config = agent_service.get_current_flow_config()
        return {
            "success": True,
            "current_config": current_config,
            "available_agents": agent_service.get_available_agents(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting flow config: {str(e)}"
        )


@router.get("/available-agents")
async def get_available_agents():
    """
    获取可用的Agent列表

    返回系统中所有可用的agent及其描述信息
    """
    try:
        agents = agent_service.get_available_agents()

        # 返回更详细的agent信息
        agent_details = []
        for agent_id in agents:
            agent_info = {
                "id": agent_id,
                "name": _get_agent_display_name(agent_id),
                "description": _get_agent_description(agent_id),
                "category": _get_agent_category(agent_id),
            }
            agent_details.append(agent_info)

        return {"success": True, "agents": agent_details}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting available agents: {str(e)}"
        )


@router.get("/available-flows")
async def get_available_flows():
    """
    获取可用的Flow类型列表

    返回系统中所有可用的flow类型及其描述信息
    """
    try:
        flows = [
            {
                "id": "single_agent",
                "name": "Single Agent",
                "description": "Use a single agent to handle all tasks",
                "category": "basic",
                "parameters": [],
            },
            {
                "id": "planning",
                "name": "Planning Flow",
                "description": "Multi-agent collaborative planning workflow",
                "category": "collaborative",
                "parameters": [],
            },
            {
                "id": "game_data_analysis",
                "name": "Game Data Analysis Flow",
                "description": "Specialized workflow for game data analysis",
                "category": "specialized",
                "parameters": [
                    {
                        "name": "data_file_path",
                        "type": "file",
                        "label": "Data File Path",
                        "description": "Path to the data file (CSV, Excel, etc.) to be analyzed",
                        "required": True,
                        "placeholder": "e.g., C:\\data\\analysis.xlsx",
                        "accept": ".csv,.xlsx,.xls,.json,.txt",
                    },
                    {
                        "name": "new_version_like",
                        "type": "text",
                        "label": "New Version Identifier",
                        "description": "String pattern to identify which records should be considered as new version",
                        "required": False,
                        "placeholder": "e.g., v2024, beta, latest, new",
                    },
                ],
            },
            {
                "id": "data_analysis_flow",
                "name": "Data Analysis Flow",
                "description": "Advanced data analysis workflow with file processing capabilities",
                "category": "specialized",
                "parameters": [
                    {
                        "name": "data_file_path",
                        "type": "file",
                        "label": "Data File Path",
                        "description": "Path to the data file (CSV, Excel, etc.) to be analyzed",
                        "required": True,
                        "placeholder": "e.g., C:\\data\\analysis.xlsx",
                        "accept": ".csv,.xlsx,.xls,.json,.txt",
                    },
                    {
                        "name": "new_version_like",
                        "type": "text",
                        "label": "New Version Identifier",
                        "description": "String pattern to identify which records should be considered as new version",
                        "required": False,
                        "placeholder": "e.g., v2024, beta, latest, new",
                    },
                ],
            },
        ]

        return {"success": True, "flows": flows}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting available flows: {str(e)}"
        )


def _get_agent_display_name(agent_id: str) -> str:
    """获取Agent的显示名称"""
    name_map = {
        "Manus": "Manus (General)",
        "DataAnalysisExpert": "Data Analysis Expert",
        "ExcelCleanAgent": "Excel Data Cleaner",
        "GameDataAnalysisAgent": "Game Data Analysis",
        "SWEAgent": "Software Engineering",
        "BrowserAgent": "Browser Automation",
        "AnalysisResultQnAAgent": "Analysis Result Q&A",
        "DataAnalysis": "Data Analysis (General)",
        "MultiDataAnalysisCoordinator": "Multi Data Analysis Coordinator",
        "KeyMetricAnalysisAgent": "Key Metric Analysis",
    }
    return name_map.get(agent_id, agent_id)


def _get_agent_description(agent_id: str) -> str:
    """获取Agent的描述"""
    desc_map = {
        "Manus": "General-purpose AI agent for various tasks",
        "DataAnalysisExpert": "Specialized in data analysis and statistical insights",
        "ExcelCleanAgent": "Expert in cleaning and preprocessing Excel data",
        "GameDataAnalysisAgent": "Specialized in game analytics and player behavior",
        "SWEAgent": "Expert in software development and code analysis",
        "BrowserAgent": "Automated web interaction and data extraction",
        "AnalysisResultQnAAgent": "Specialized in answering questions based on analysis results",
        "DataAnalysis": "General data analysis capabilities",
        "MultiDataAnalysisCoordinator": "Coordinates multiple data analysis processes",
        "KeyMetricAnalysisAgent": "Analyzes key performance indicators and metrics",
    }
    return desc_map.get(agent_id, f"Agent for {agent_id} tasks")


def _get_agent_category(agent_id: str) -> str:
    """获取Agent的分类"""
    category_map = {
        "Manus": "general",
        "DataAnalysisExpert": "data",
        "ExcelCleanAgent": "data",
        "GameDataAnalysisAgent": "specialized",
        "SWEAgent": "development",
        "BrowserAgent": "automation",
        "AnalysisResultQnAAgent": "qa",
        "DataAnalysis": "data",
        "MultiDataAnalysisCoordinator": "data",
        "KeyMetricAnalysisAgent": "data",
    }
    return category_map.get(agent_id, "other")
