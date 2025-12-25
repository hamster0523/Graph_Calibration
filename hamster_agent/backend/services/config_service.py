"""
配置管理服务
处理系统配置的读取、更新和验证
"""

import os
import json
import time
import tomllib
import toml
from pathlib import Path
from typing import Dict, Any, Optional, List

from ..models.schemas import ConfigUpdate


class ConfigService:
    """配置管理服务类"""

    def __init__(self):
        self.config_dir = Path(__file__).parent.parent.parent / "config"
        self.config_path = self.config_dir / "config.toml"
        self.example_config_path = self.config_dir / "config.example.toml"

        # 确保配置目录存在
        self.config_dir.mkdir(exist_ok=True)

    def get_current_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'rb') as f:
                    return tomllib.load(f)
            elif self.example_config_path.exists():
                with open(self.example_config_path, 'rb') as f:
                    return tomllib.load(f)
            else:
                return self._get_default_config()
        except Exception as e:
            print(f"Error reading config: {e}")
            return self._get_default_config()

    def update_config(self, config_update: ConfigUpdate) -> bool:
        """更新配置"""
        try:
            current_config = self.get_current_config()

            # 逐个更新配置节
            if config_update.llm:
                current_config["llm"] = {
                    **current_config.get("llm", {}),
                    **config_update.llm
                }

            if config_update.browser:
                current_config["browser"] = {
                    **current_config.get("browser", {}),
                    **config_update.browser
                }

            if config_update.search:
                current_config["search"] = {
                    **current_config.get("search", {}),
                    **config_update.search
                }

            if config_update.sandbox:
                current_config["sandbox"] = {
                    **current_config.get("sandbox", {}),
                    **config_update.sandbox
                }

            # 保存配置
            self._save_config(current_config)
            return True

        except Exception as e:
            print(f"Error updating config: {e}")
            return False

    def validate_config(self, config_data: Dict[str, Any]) -> Dict[str, str]:
        """验证配置数据"""
        errors = {}

        # 验证LLM配置
        if "llm" in config_data:
            llm_config = config_data["llm"]

            # 必需字段检查
            required_fields = ["api_key", "model", "base_url"]
            for field in required_fields:
                if not llm_config.get(field):
                    errors[f"llm.{field}"] = f"{field} is required"

            # 数值范围检查
            if "temperature" in llm_config:
                temp = llm_config["temperature"]
                if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                    errors["llm.temperature"] = "Temperature must be between 0 and 2"

            if "max_tokens" in llm_config:
                tokens = llm_config["max_tokens"]
                if not isinstance(tokens, int) or tokens < 1:
                    errors["llm.max_tokens"] = "Max tokens must be a positive integer"

        # 验证浏览器配置
        if "browser" in config_data:
            browser_config = config_data["browser"]

            if "max_content_length" in browser_config:
                max_len = browser_config["max_content_length"]
                if not isinstance(max_len, int) or max_len < 100:
                    errors["browser.max_content_length"] = "Max content length must be at least 100"

        # 验证沙箱配置
        if "sandbox" in config_data:
            sandbox_config = config_data["sandbox"]

            if "timeout" in sandbox_config:
                timeout = sandbox_config["timeout"]
                if not isinstance(timeout, int) or timeout < 10:
                    errors["sandbox.timeout"] = "Timeout must be at least 10 seconds"

        return errors

    def test_llm_config(self, llm_config: Dict[str, Any]) -> bool:
        """测试LLM配置"""
        try:
            required_fields = ["api_key", "base_url", "model"]
            if not all(llm_config.get(field) for field in required_fields):
                return False

            # 基本URL格式验证
            base_url = llm_config["base_url"]
            if not base_url.startswith(("http://", "https://")):
                return False

            # API key格式基本验证
            api_key = llm_config["api_key"]
            if len(api_key) < 10:  # 基本长度检查
                return False

            # 这里可以添加实际的API测试调用
            # 为了简化，我们只做基本验证
            return True

        except Exception as e:
            print(f"Error testing LLM config: {e}")
            return False

    def get_available_models(self) -> Dict[str, List[str]]:
        """获取可用的模型选项"""
        return {
            "openai": [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo"
            ],
            "anthropic": [
                "claude-3-5-sonnet-20241022",
                "claude-3-opus-20240229",
                "claude-3-haiku-20240307"
            ],
            "google": [
                "gemini-pro",
                "gemini-pro-vision"
            ],
            "ollama": [
                "llama3",
                "llama3:70b",
                "codellama",
                "mistral"
            ]
        }

    def get_search_engines(self) -> List[str]:
        """获取可用的搜索引擎"""
        return ["Google", "DuckDuckGo", "Baidu", "Bing"]

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "llm": {
                "model": "gpt-4o",
                "base_url": "https://api.openai.com/v1",
                "api_key": "",
                "max_tokens": 4096,
                "temperature": 0.0,
                "api_type": "openai",
                "api_version": "2023-05-15"
            },
            "browser": {
                "headless": False,
                "disable_security": True,
                "max_content_length": 2000
            },
            "search": {
                "engine": "Google",
                "fallback_engines": ["DuckDuckGo", "Baidu", "Bing"],
                "lang": "en",
                "country": "us"
            },
            "sandbox": {
                "use_sandbox": False,
                "image": "python:3.12-slim",
                "memory_limit": "512m",
                "timeout": 300
            }
        }

    def _save_config(self, config_data: Dict[str, Any]):
        """保存配置到文件"""
        try:
            # 确保配置目录存在
            self.config_path.parent.mkdir(exist_ok=True)

            # 写入TOML文件
            with open(self.config_path, 'w', encoding='utf-8') as f:
                toml.dump(config_data, f)

            print(f"Configuration saved to {self.config_path}")

        except Exception as e:
            raise Exception(f"Failed to save configuration: {e}")

    def backup_config(self) -> str:
        """备份当前配置"""
        try:
            if not self.config_path.exists():
                return "No configuration file to backup"

            backup_path = self.config_path.with_suffix(f".backup.{int(time.time())}")
            import shutil
            shutil.copy2(self.config_path, backup_path)

            return f"Configuration backed up to {backup_path}"

        except Exception as e:
            return f"Backup failed: {e}"

    def restore_default_config(self) -> bool:
        """恢复默认配置"""
        try:
            default_config = self._get_default_config()
            self._save_config(default_config)
            return True
        except Exception as e:
            print(f"Error restoring default config: {e}")
            return False

    def get_config_status(self) -> Dict[str, Any]:
        """获取配置状态信息"""
        return {
            "config_exists": self.config_path.exists(),
            "config_path": str(self.config_path),
            "example_exists": self.example_config_path.exists(),
            "config_dir": str(self.config_dir),
            "last_modified": (
                os.path.getmtime(self.config_path)
                if self.config_path.exists() else None
            )
        }


# 全局配置服务实例
config_service = ConfigService()
