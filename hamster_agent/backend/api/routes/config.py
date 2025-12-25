"""
配置管理相关的API路由
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any

try:
    from ...models.schemas import ConfigUpdate
    from ...services.config_service import config_service
except ImportError:
    try:
        from models.schemas import ConfigUpdate
        from services.config_service import config_service
    except ImportError:
        from backend.models.schemas import ConfigUpdate
        from backend.services.config_service import config_service

router = APIRouter()


@router.get("/config")
async def get_config():
    """
    获取当前配置
    
    返回系统的完整配置信息
    """
    try:
        config = config_service.get_current_config()
        
        # 隐藏敏感信息（API keys）
        safe_config = _sanitize_config(config)
        
        return safe_config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting configuration: {str(e)}")


@router.get("/config/status")
async def get_config_status():
    """
    获取配置状态
    
    返回配置文件的状态信息
    """
    try:
        return config_service.get_config_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting config status: {str(e)}")


@router.post("/config")
async def update_config(config_update: ConfigUpdate):
    """
    更新配置
    
    更新系统配置设置
    """
    try:
        # 验证配置数据
        config_data = config_update.dict(exclude_none=True)
        errors = config_service.validate_config(config_data)
        
        if errors:
            raise HTTPException(status_code=400, detail={
                "message": "Configuration validation failed",
                "errors": errors
            })
        
        # 更新配置
        success = config_service.update_config(config_update)
        
        if success:
            return {"message": "Configuration updated successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to update configuration")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating configuration: {str(e)}")


@router.post("/config/test")
async def test_config(test_data: Dict[str, Any]):
    """
    测试配置
    
    验证配置设置是否有效（如测试LLM连接）
    """
    try:
        if "llm" in test_data:
            llm_config = test_data["llm"]
            is_valid = config_service.test_llm_config(llm_config)
            
            return {
                "valid": is_valid,
                "message": "Configuration is valid" if is_valid else "Configuration test failed",
                "test_type": "llm"
            }
        
        return {"message": "No testable configuration provided"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Configuration test failed: {str(e)}")


@router.get("/config/models")
async def get_available_models():
    """
    获取可用模型
    
    返回所有支持的LLM模型列表
    """
    try:
        models = config_service.get_available_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting models: {str(e)}")


@router.get("/config/search-engines")
async def get_search_engines():
    """
    获取可用搜索引擎
    
    返回所有支持的搜索引擎列表
    """
    try:
        engines = config_service.get_search_engines()
        return {"search_engines": engines}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting search engines: {str(e)}")


@router.post("/config/backup")
async def backup_config():
    """
    备份配置
    
    创建当前配置的备份文件
    """
    try:
        result = config_service.backup_config()
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backup failed: {str(e)}")


@router.post("/config/restore-default")
async def restore_default_config():
    """
    恢复默认配置
    
    将配置重置为默认设置
    """
    try:
        success = config_service.restore_default_config()
        
        if success:
            return {"message": "Configuration restored to default"}
        else:
            raise HTTPException(status_code=500, detail="Failed to restore default configuration")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error restoring default config: {str(e)}")


@router.get("/config/validate")
async def validate_current_config():
    """
    验证当前配置
    
    检查当前配置的有效性
    """
    try:
        config = config_service.get_current_config()
        errors = config_service.validate_config(config) 
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "message": "Configuration is valid" if len(errors) == 0 else "Configuration has errors"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


def _sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """移除配置中的敏感信息"""
    safe_config = config.copy()
    
    # 隐藏API keys
    if "llm" in safe_config:
        llm_config = safe_config["llm"].copy()
        if "api_key" in llm_config and llm_config["api_key"]:
            # 只显示前4位和后4位
            api_key = llm_config["api_key"]
            if len(api_key) > 8:
                llm_config["api_key"] = f"{api_key[:4]}...{api_key[-4:]}"
            else:
                llm_config["api_key"] = "*" * len(api_key)
        safe_config["llm"] = llm_config
    
    return safe_config