"""
Agent服务
处理与Manus Agent的交互逻辑
"""

import asyncio
import logging
import re
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional

try:
    from ..models.schemas import (
        AgentActionData,
        AgentObservationData,
        AgentStatus,
        AgentStreamMessage,
        AgentThinkData,
        ChatMessage,
        ChatResponse,
        FlowConfigResponse,
        FlowConfiguration,
    )
    from .connection_manager import manager
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    try:
        from models.schemas import (
            AgentActionData,
            AgentObservationData,
            AgentStatus,
            AgentStreamMessage,
            AgentThinkData,
            ChatMessage,
            ChatResponse,
            FlowConfigResponse,
            FlowConfiguration,
        )
        from services.connection_manager import manager
    except ImportError:
        from backend.models.schemas import (
            AgentActionData,
            AgentObservationData,
            AgentStatus,
            AgentStreamMessage,
            AgentThinkData,
            ChatMessage,
            ChatResponse,
            FlowConfigResponse,
            FlowConfiguration,
        )
        from backend.services.connection_manager import manager


class StreamingLogHandler(logging.Handler):
    """自定义日志处理器 - 将关键日志信息转换为流式消息"""

    def __init__(self, broadcast_callback):
        super().__init__()
        self.broadcast = broadcast_callback
        self.current_step = 0
        self.total_steps = 20

        # 定义日志消息模式匹配
        self.log_patterns = {
            # Flow相关模式
            "flow_start": [
                r"Creating initial plan.*",
                r"开始执行.*",
                r"Execute.*flow.*",
                r"Starting.*",
            ],
            "plan_creation": [
                r"Creating initial plan.*",
                r"Plan creation.*",
                r"计划创建.*",
            ],
            "step_start": [
                r"执行步骤.*",
                r"Step.*执行.*",
                r"Processing step.*",
                r"Current step.*",
            ],
            "step_complete": [
                r"Step.*completed.*",
                r"步骤.*完成.*",
                r"Completed step.*",
            ],
            "agent_action": [
                r"Using.*agent.*",
                r"Agent.*executing.*",
                r"Tool.*called.*",
                r"Executing.*",
            ],
            "result": [r"Result.*", r"结果.*", r"Output.*", r"Generated.*"],
            "error": [r"Error.*", r"Failed.*", r"Exception.*", r"错误.*"],
        }

    def emit(self, record):
        """处理日志记录并转换为流式消息"""
        try:
            log_message = record.getMessage()
            message_type, data = self._classify_log_message(log_message, record)

            if message_type and self.broadcast:
                # 异步广播消息
                asyncio.create_task(self._async_broadcast(message_type, data))

        except Exception as e:
            # 避免在日志处理中产生无限循环
            pass

    def _classify_log_message(self, message: str, record) -> tuple:
        """根据日志内容分类并生成对应的流式消息"""

        # 检查每种模式
        for msg_type, patterns in self.log_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    return self._create_stream_data(msg_type, message, record)

        # 默认处理
        if record.levelno >= logging.WARNING:
            return self._create_stream_data("warning", message, record)
        elif record.levelno >= logging.INFO:
            return self._create_stream_data("info", message, record)

        return None, None

    def _create_stream_data(self, msg_type: str, message: str, record) -> tuple:
        """创建流式消息数据"""

        # 根据消息类型生成相应的数据结构
        if msg_type == "flow_start":
            self.current_step = 0
            return "start", {
                "description": message,
                "timestamp": datetime.now().isoformat(),
            }

        elif msg_type == "plan_creation":
            self.current_step = 1
            return "think_start", {
                "content": message,
                "reasoning": "正在创建执行计划...",
            }

        elif msg_type == "step_start":
            self.current_step += 1
            return "step_start", {
                "step": self.current_step,
                "description": message,
                "timestamp": datetime.now().isoformat(),
            }

        elif msg_type == "step_complete":
            return "step_complete", {
                "step": self.current_step,
                "result": message[:200],
                "description": f"步骤 {self.current_step} 完成",
            }

        elif msg_type == "agent_action":
            return "act", {
                "tool_name": self._extract_tool_name(message),
                "description": message,
                "timestamp": datetime.now().isoformat(),
            }

        elif msg_type == "result":
            return "observe", {
                "tool_name": "system",
                "result": message[:300],
                "success": True,
            }

        elif msg_type == "error":
            return "error", {
                "error": message,
                "description": f"执行过程中发生错误",
                "timestamp": datetime.now().isoformat(),
            }

        else:
            return "info", {
                "content": message,
                "level": record.levelname,
                "timestamp": datetime.now().isoformat(),
            }

    def _extract_tool_name(self, message: str) -> str:
        """从日志消息中提取工具名称"""
        # 尝试匹配常见的工具名称模式
        tool_patterns = [
            r"tool[:\s]+(\w+)",
            r"using\s+(\w+)",
            r"executing\s+(\w+)",
            r"(\w+)\s+agent",
        ]

        for pattern in tool_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(1)

        return "unknown_tool"

    async def _async_broadcast(self, message_type: str, data: Dict[str, Any]):
        """异步广播流式消息"""
        try:
            if self.broadcast:
                await self.broadcast(
                    message_type, data, self.current_step, self.total_steps
                )
        except Exception as e:
            # 静默处理广播错误，避免影响主流程
            pass


class LoggingInterceptor:
    """日志拦截器 - 临时替换logger处理器"""

    def __init__(self, broadcast_callback):
        self.broadcast = broadcast_callback
        self.original_handlers = {}
        self.stream_handler = StreamingLogHandler(broadcast_callback)

    def __enter__(self):
        """进入上下文时安装日志拦截器"""
        # 获取app.logger相关的logger
        loggers_to_intercept = [
            "app.flow",
            "app.agent",
            "app.logger",
            "app.flow.planning",
            "app.flow.data_analysis_flow",
            "app.agent.manus",
            "app.agent.base",
        ]

        for logger_name in loggers_to_intercept:
            try:
                logger = logging.getLogger(logger_name)
                # 保存原始处理器
                self.original_handlers[logger_name] = logger.handlers.copy()
                # 添加流式处理器
                logger.addHandler(self.stream_handler)
                # 设置适当的日志级别
                if logger.level == logging.NOTSET:
                    logger.setLevel(logging.INFO)
            except Exception as e:
                # 忽略不存在的logger
                pass

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时恢复原始日志处理器"""
        for logger_name, original_handlers in self.original_handlers.items():
            try:
                logger = logging.getLogger(logger_name)
                # 移除流式处理器
                if self.stream_handler in logger.handlers:
                    logger.removeHandler(self.stream_handler)
            except Exception as e:
                pass


class AgentService:
    """Agent服务类"""

    def __init__(self):
        self.current_agent: Optional[Any] = None  # 实际的Manus agent实例
        self.current_flow: Optional[Any] = None  # 当前flow实例
        self.status = AgentStatus(
            status="idle", current_step=0, max_steps=20, last_action="Ready")
        self.is_demo_mode = False  # 强制禁用演示模式
        self.llm_config = None  # LLM配置

        # Flow配置
        self.flow_config = FlowConfiguration(
            mode="single_agent", primaryAgent="manus", selectedAgents=[])

        self._load_config()

    def _load_config(self):
        """加载配置"""
        try:
            # 尝试导入app的配置
            from app.config import config as app_config

            self.llm_config = app_config.llm.get("default")
            if self.llm_config:
                print(
                    f"✅ Loaded LLM config: model={self.llm_config.model}, api_type={self.llm_config.api_type}"
                )
            else:
                print("⚠️ No default LLM config found")
        except ImportError:
            print("⚠️ App config not available, will use backend config service")
        except Exception as e:
            print(f"⚠️ Error loading app config: {e}")

        # 如果app配置不可用，尝试使用backend配置服务
        if not self.llm_config:
            try:
                # 尝试多种导入方式
                try:
                    from backend.services.config_service import config_service
                except ImportError:
                    try:
                        from services.config_service import config_service
                    except ImportError:
                        from .config_service import config_service

                backend_config = config_service.get_current_config()
                llm_config = backend_config.get("llm", {})
                if llm_config:
                    # 创建一个简单的配置对象
                    self.llm_config = type("LLMConfig", (), llm_config)()
                    print(
                        f"✅ Loaded backend LLM config: model={getattr(self.llm_config, 'model', 'unknown')}"
                    )
            except Exception as e:
                print(f"⚠️ Error loading backend config: {e}")

    async def initialize_agent(self):
        """初始化Agent（如果可用）"""
        try:
            # 强制不使用Demo模式 - 跳过复杂的Manus初始化
            print("🔧 Attempting to exit demo mode...")

            # 检查配置是否可用
            if self.llm_config:
                print(
                    f"🔧 Using LLM configuration: {getattr(self.llm_config, 'model', 'unknown')}"
                )
                if (
                    hasattr(self.llm_config, "api_key")
                    and not getattr(self.llm_config, "api_key", "").strip()
                ):
                    print("⚠️ Warning: LLM API key is empty")
                else:
                    # 如果有有效的API配置，不使用demo模式
                    self.is_demo_mode = False
                    print("✅ LLM configuration found, exiting demo mode")
                    return

            # 尝试导入并初始化真实的Manus agent (with timeout protection)
            from app.agent.manus import Manus

            self.current_agent = await Manus.create()
            self.is_demo_mode = False
            print("✅ Manus agent initialized successfully")

            # 更新最大步数
            if hasattr(self.current_agent, "max_steps"):
                self.status.max_steps = self.current_agent.max_steps

        except ImportError as e:
            print(f"⚠️ Manus agent not available, checking LLM config: {e}")
            # 如果有LLM配置，仍然可以不使用demo模式
            if (
                self.llm_config
                and hasattr(self.llm_config, "api_key")
                and getattr(self.llm_config, "api_key", "").strip()
            ):
                self.is_demo_mode = False
                print(
                    "✅ LLM configuration available, exiting demo mode despite Manus import failure"
                )
            else:
                self.is_demo_mode = True
        except Exception as e:
            print(f"❌ Error initializing Manus agent: {e}")
            # 如果有LLM配置，仍然可以不使用demo模式
            if (
                self.llm_config
                and hasattr(self.llm_config, "api_key")
                and getattr(self.llm_config, "api_key", "").strip()
            ):
                self.is_demo_mode = False
                print(
                    "✅ LLM configuration available, exiting demo mode despite Manus initialization failure"
                )
            else:
                self.is_demo_mode = True

    async def process_message(self, message: ChatMessage) -> ChatResponse:
        """处理聊天消息"""
        try:
            # 更新状态为处理中
            await self._update_status("processing", 1, "Processing user message")

            # 广播用户消息
            await self._broadcast_user_message(message)

            if self.is_demo_mode:
                # 演示模式：生成模拟响应
                response_text = await self._generate_demo_response(message.message)
            else:
                # 真实模式：使用Manus agent
                response_text = await self._process_with_agent(message.message)

            # 广播Agent响应
            await self._broadcast_agent_response(response_text)

            # 更新状态为完成
            await self._update_status("completed", 0, "Message processed successfully")

            return ChatResponse(
                response=response_text,
                status="success",
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            await self._update_status("error", 0, f"Error: {str(e)}")
            raise e

    async def _generate_demo_response(self, user_message: str) -> str:
        """生成演示响应（带流式输出）"""
        # 演示流式输出
        await self._broadcast_stream_message(
            "start", {"description": "演示模式 Agent 开始处理请求..."}, 0, 3
        )

        # 模拟思考阶段
        await self._broadcast_stream_message(
            "think_start", {"content": "正在分析用户请求..."}, 1, 3
        )

        await asyncio.sleep(0.5)  # 模拟思考时间

        await self._broadcast_stream_message(
            "think",
            {
                "content": f"用户询问: '{user_message}'. 我需要生成一个合适的回应。",
                "reasoning": "分析完成，准备生成回应",
            },
            1,
            3,
        )

        # 模拟执行阶段
        await self._broadcast_stream_message(
            "act",
            {"tool_name": "response_generator", "description": "执行回应生成器"},
            2,
            3,
        )

        await asyncio.sleep(0.5)  # 模拟执行时间

        # 生成回应逻辑
        message_lower = user_message.lower()

        if "hello" in message_lower or "hi" in message_lower:
            response = "Hello! I'm the OpenManus agent demo. How can I help you today?"
        elif "help" in message_lower:
            response = "I can assist you with various tasks including data analysis, code generation, file management, and web automation. What would you like me to help you with?"
        elif "test" in message_lower:
            response = "Test successful! The Web UI is working correctly. You can try uploading files, changing configurations, or asking me questions."
        elif "what can you do" in message_lower:
            response = "I can help you with:\n• Data analysis and visualization\n• Code review and generation\n• File processing and management\n• Web browsing and automation\n• Search and research tasks"
        else:
            response = f"I received your message: '{user_message}'. This is a demo response. In the full version, I would process this request and provide a detailed response based on my capabilities."

        # 模拟观察结果
        await self._broadcast_stream_message(
            "observe",
            {
                "tool_name": "response_generator",
                "result": f"生成回应: {response[:100]}...",
                "success": True,
            },
            2,
            3,
        )

        await asyncio.sleep(0.3)  # 最后的处理时间

        return response

    async def _process_with_agent(self, user_message: str) -> str:
        """使用真实Agent或Flow处理消息（支持流式输出）"""
        try:
            print(f"🔍 Processing message with:")
            print(f"   Current flow: {self.current_flow is not None}")
            print(f"   Current agent: {self.current_agent is not None}")
            print(f"   Flow config mode: {self.flow_config.mode}")
            print(f"   Demo mode: {self.is_demo_mode}")

            # 优先使用Flow
            if self.current_flow:
                print("✅ Using Flow mode")
                # 创建混合式Flow包装器 - 提供多层实时反馈
                stream_flow = HybridStreamingFlowWrapper(
                    self.current_flow, self._broadcast_stream_message
                )
                response = await stream_flow.run(user_message)
                return response or "Task completed successfully"
            elif self.current_agent:
                print("✅ Using Agent mode")
                # 创建流式Agent包装器
                stream_agent = StreamingAgentWrapper(
                    self.current_agent, self._broadcast_stream_message
                )
                response = await stream_agent.run(user_message)
                return response or "Task completed successfully"
            elif self.current_agent is None and not self.current_flow:
                print("⚠️ No Flow or Agent available, checking configuration...")

                # 如果配置为Flow模式但Flow不存在，尝试重新初始化
                if self.flow_config.mode != "single_agent":
                    print(
                        f"🔧 Attempting to reinitialize Flow: {self.flow_config.mode}"
                    )
                    await self._initialize_flow(self.flow_config)

                    if self.current_flow:
                        print("✅ Flow reinitialized successfully")
                        # 创建混合式Flow包装器 - 重新初始化后使用
                        stream_flow = HybridStreamingFlowWrapper(
                            self.current_flow, self._broadcast_stream_message
                        )
                        response = await stream_flow.run(user_message)
                        return response or "Task completed successfully"
                    else:
                        return f"Failed to initialize {self.flow_config.mode} flow. Please check your configuration."
                else:
                    # Single agent模式，尝试初始化Agent
                    await self.initialize_agent()
                    if self.current_agent:
                        # 创建流式Agent包装器
                        stream_agent = StreamingAgentWrapper(
                            self.current_agent, self._broadcast_stream_message
                        )
                        response = await stream_agent.run(user_message)
                        return response or "Task completed successfully"

            # 只有在single_agent模式且Agent不可用时才回退到LLM API
            if self.flow_config.mode == "single_agent":
                print("⚠️ Falling back to direct LLM API for single_agent mode")
                if (
                    self.llm_config
                    and hasattr(self.llm_config, "api_key")
                    and getattr(self.llm_config, "api_key", "").strip()
                ):
                    return await self._process_with_llm_direct(user_message)
                else:
                    return "No agent, flow, or LLM configuration available."
            else:
                return f"Flow mode ({self.flow_config.mode}) is not available. Please check your flow configuration."

        except Exception as e:
            print(f"Error processing with agent/flow: {e}")
            import traceback

            traceback.print_exc()
            return f"Error processing request: {str(e)}"

    async def _process_with_llm_direct(self, user_message: str) -> str:
        """直接使用LLM API处理消息"""
        try:
            import json

            import requests

            # 安全获取配置属性
            api_key = getattr(self.llm_config, "api_key", "")
            model = getattr(self.llm_config, "model", "gpt-3.5-turbo")
            base_url = getattr(
                self.llm_config, "base_url", "https://openrouter.ai/api/v1"
            )
            max_tokens = getattr(self.llm_config, "max_tokens", 4000)
            temperature = getattr(self.llm_config, "temperature", 0.7)

            if not api_key:
                return "LLM API密钥未配置，无法处理请求。"

            # 广播思考过程
            await self._broadcast_stream_message(
                "think", {"content": f"正在使用 {model} 处理您的请求..."}, 1, 3
            )

            # 准备API请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            data = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant. Please provide clear and accurate responses.",
                    },
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            # 广播动作
            await self._broadcast_stream_message(
                "action",
                {
                    "tool_name": "llm_api",
                    "args": {"model": model, "message": user_message[:50] + "..."},
                },
                2,
                3,
            )

            # 发送API请求
            response = requests.post(
                f"{base_url}/chat/completions", headers=headers, json=data, timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]

                # 广播观察结果
                await self._broadcast_stream_message(
                    "observe",
                    {
                        "tool_name": "llm_api",
                        "result": f"收到回复: {content[:100]}...",
                        "success": True,
                    },
                    3,
                    3,
                )

                return content
            else:
                error_msg = f"LLM API Error: {response.status_code} - {response.text}"
                print(error_msg)
                return f"抱歉，处理您的请求时出现错误：{error_msg}"

        except Exception as e:
            print(f"Error in direct LLM processing: {e}")
            return f"抱歉，处理您的请求时出现错误：{str(e)}"

    async def _update_status(self, status: str, step: int, action: str):
        """更新Agent状态"""
        self.status.status = status
        self.status.current_step = step
        self.status.last_action = action

        # 广播状态更新
        await manager.broadcast_json(
            {
                "type": "status_update",
                "data": self.status.dict(),
                "timestamp": datetime.now().isoformat(),
            }
        )

    async def _broadcast_user_message(self, message: ChatMessage):
        """广播用户消息"""
        await manager.broadcast_json(
            {
                "type": "user_message",
                "data": {
                    "message": message.message,
                    "timestamp": message.timestamp or datetime.now().isoformat(),
                },
            }
        )

    async def _broadcast_agent_response(self, response: str):
        """广播Agent响应"""
        await manager.broadcast_json(
            {
                "type": "agent_response",
                "data": {"response": response, "timestamp": datetime.now().isoformat()},
            }
        )

    def get_status(self) -> AgentStatus:
        """获取当前状态"""
        return self.status

    def get_llm_config_info(self) -> Dict[str, Any]:
        """获取LLM配置信息（隐藏敏感信息）"""
        if not self.llm_config:
            return {
                "available": False,
                "source": "none",
                "message": "No LLM configuration loaded",
            }

        try:
            config_info = {
                "available": True,
                "model": getattr(self.llm_config, "model", "unknown"),
                "api_type": getattr(self.llm_config, "api_type", "unknown"),
                "base_url": getattr(self.llm_config, "base_url", "unknown"),
                "max_tokens": getattr(self.llm_config, "max_tokens", "unknown"),
                "temperature": getattr(self.llm_config, "temperature", "unknown"),
                "api_key_configured": bool(
                    getattr(self.llm_config, "api_key", "").strip()
                ),
                "source": (
                    "app_config"
                    if hasattr(self.llm_config, "model")
                    else "backend_config"
                ),
            }

            # 隐藏API key，只显示是否配置了
            if hasattr(self.llm_config, "api_key"):
                api_key = getattr(self.llm_config, "api_key", "")
                if api_key.strip():
                    config_info["api_key_preview"] = (
                        api_key[:8] + "..." if len(api_key) > 8 else "***"
                    )
                else:
                    config_info["api_key_preview"] = "(not configured)"

            return config_info

        except Exception as e:
            return {"available": False, "source": "error", "error": str(e)}

    async def _broadcast_stream_message(
        self,
        message_type: str,
        data: Dict[str, Any],
        step: int,
        total_steps: Optional[int] = None,
    ):
        """广播流式消息"""
        stream_message = AgentStreamMessage(
            message_type=message_type,
            data=data,
            step=step,
            total_steps=total_steps or self.status.max_steps,
            timestamp=datetime.now().isoformat(),
        )

        # 调试输出
        print(f"🔄 Broadcasting stream message: {message_type} - Step {step}")
        print(f"   Data: {data}")
        print(f"   Active connections: {manager.get_connection_count()}")

        broadcast_data = {
            "type": "agent_stream",
            "data": stream_message.dict(),
            "timestamp": datetime.now().isoformat(),
        }

        print(f"   Broadcast data: {broadcast_data}")

        if manager.get_connection_count() == 0:
            print("⚠️ No active WebSocket connections to broadcast to!")
        else:
            await manager.broadcast_json(broadcast_data)
            print(
                f"✅ Message broadcasted to {manager.get_connection_count()} connections"
            )

    async def configure_flow(self, config: FlowConfiguration) -> FlowConfigResponse:
        """配置flow和agent"""
        try:
            print(f"🔧 Configuring flow: {config.mode}")
            print(f"   Primary agent: {config.primaryAgent}")
            print(f"   Selected agents: {config.selectedAgents}")
            print(f"   Parameters: {config.parameters}")

            # 保存配置
            self.flow_config = config

            # 清理现有的agent和flow
            await self.cleanup()

            # 根据配置创建新的agent或flow
            if config.mode == "single_agent":
                await self._initialize_single_agent(config.primaryAgent)
            else:
                await self._initialize_flow(config)

            # 检查初始化是否成功
            if config.mode == "single_agent" and not self.current_agent:
                raise Exception(
                    f"Failed to initialize single agent: {config.primaryAgent}"
                )
            elif config.mode != "single_agent" and not self.current_flow:
                raise Exception(f"Failed to initialize flow: {config.mode}")

            # 返回成功响应
            return FlowConfigResponse(
                success=True,
                message=f"Successfully configured {config.mode} mode",
                applied_config=config,
                available_agents=self.get_available_agents(),
            )

        except Exception as e:
            print(f"❌ Error configuring flow: {e}")
            import traceback

            traceback.print_exc()

            return FlowConfigResponse(
                success=False,
                message=f"Failed to configure flow: {str(e)}",
                applied_config=config,
                available_agents=self.get_available_agents(),
            )

    async def _initialize_single_agent(self, agent_type: str):
        """初始化单个Agent"""
        try:
            if agent_type == "manus":
                # 使用默认的Manus agent
                from app.agent.manus import Manus

                self.current_agent = await Manus.create()
            else:
                # 创建特定类型的agent
                agent_class = self._get_agent_class(agent_type)
                if agent_class:
                    self.current_agent = await agent_class.create()
                else:
                    raise ValueError(f"Unknown agent type: {agent_type}")

            self.current_flow = None
            self.is_demo_mode = False
            print(f"✅ Initialized single agent: {agent_type}")

        except Exception as e:
            print(
                f"⚠️ Failed to initialize agent {agent_type}, falling back to demo mode: {e}"
            )
            self.current_agent = None
            self.is_demo_mode = True

    async def _initialize_flow(self, config: FlowConfiguration):
        """初始化Flow"""
        try:
            print(f"🔧 Initializing Flow: {config.mode}")
            print(f"   Primary agent: {config.primaryAgent}")
            print(f"   Selected agents: {config.selectedAgents}")
            print(f"   Parameters: {config.parameters}")

            from app.flow.flow_factory import FlowFactory, FlowType

            # 创建agents字典
            agents = {}

            # 添加主要agent
            if config.primaryAgent:
                print(f"🔧 Creating primary agent: {config.primaryAgent}")
                primary_agent_class = self._get_agent_class(config.primaryAgent)
                if primary_agent_class:
                    try:
                        # 使用asyncio.wait_for添加60秒超时
                        agents[config.primaryAgent] = await asyncio.wait_for(
                            primary_agent_class.create(), timeout=60.0
                        )
                        print(f"✅ Primary agent created: {config.primaryAgent}")
                    except asyncio.TimeoutError:
                        print(
                            f"⏰ Primary agent creation timeout: {config.primaryAgent}"
                        )
                        print(
                            f"⚠️ Skipping primary agent {config.primaryAgent} due to timeout"
                        )
                    except Exception as e:
                        print(
                            f"❌ Failed to create primary agent {config.primaryAgent}: {e}"
                        )
                        print(
                            f"⚠️ Skipping primary agent {config.primaryAgent} due to error: {e}"
                        )
                else:
                    print(f"⚠️ Primary agent class not found: {config.primaryAgent}")
            else:
                print("⚠️ No primary agent specified")

            # 添加选中的agents
            for agent_type in config.selectedAgents:
                if agent_type not in agents:  # 避免重复
                    print(f"🔧 Creating selected agent: {agent_type}")
                    agent_class = self._get_agent_class(agent_type)
                    if agent_class:
                        try:
                            # 使用asyncio.wait_for添加60秒超时
                            agents[agent_type] = await asyncio.wait_for(
                                agent_class.create(), timeout=60.0
                            )
                            print(f"✅ Selected agent created: {agent_type}")
                        except asyncio.TimeoutError:
                            print(f"⏰ Selected agent creation timeout: {agent_type}")
                            # 不要因为一个Agent失败就终止整个流程
                            print(
                                f"⚠️ Skipping {agent_type} due to timeout, continuing with other agents..."
                            )
                            continue
                        except Exception as e:
                            print(
                                f"❌ Failed to create selected agent {agent_type}: {e}"
                            )
                            # 不要因为一个Agent失败就终止整个流程
                            print(
                                f"⚠️ Skipping {agent_type} due to error, continuing with other agents..."
                            )
                            continue
                    else:
                        print(f"⚠️ Agent class not found for {agent_type}, skipping...")

            print(f"✅ Total agents successfully created: {len(agents)}")

            # 如果没有成功创建任何Agent，使用默认的Manus agent
            if not agents:
                print(
                    "⚠️ No agents were created successfully, falling back to default Manus agent"
                )
                try:
                    from app.agent.manus import Manus

                    agents["manus"] = await Manus.create()
                    print("✅ Fallback Manus agent created successfully")
                except Exception as e:
                    print(f"❌ Even fallback Manus agent failed: {e}")
                    raise Exception("Failed to create any agents for the flow")

            # 映射flow类型
            flow_type_map = {
                "planning": FlowType.PLANNING,
                "game_data_analysis": FlowType.GAME_DATA_ANALYSIS,
                "data_analysis_flow": FlowType.DATA_ANALYSIS_FLOW,
            }

            flow_type = flow_type_map.get(config.mode)
            if not flow_type:
                raise ValueError(f"Unknown flow type: {config.mode}")

            print(f"🔧 Creating flow of type: {flow_type}")

            # 创建flow
            flow_kwargs = {}
            if config.parameters:
                flow_kwargs.update(config.parameters)
                print(f"🔧 Flow kwargs: {flow_kwargs}")

            self.current_flow = FlowFactory.create_flow(
                flow_type, agents, **flow_kwargs
            )
            self.current_agent = None  # Flow模式下不使用单个agent
            self.is_demo_mode = False
            print(f"✅ Initialized flow: {config.mode} with {len(agents)} agents")
            print(f"   Flow instance: {type(self.current_flow).__name__}")

        except Exception as e:
            print(f"❌ Failed to initialize flow {config.mode}: {e}")
            import traceback

            traceback.print_exc()

            self.current_flow = None
            self.current_agent = None
            self.is_demo_mode = True
            raise e  # 重新抛出异常，让调用者知道失败了

    def _get_agent_class(self, agent_type: str):
        """获取Agent类"""
        try:
            print(f"🔍 Getting agent class for: {agent_type}")

            agent_class_map = {
                "manus": "app.agent.manus.Manus",
                "Manus": "app.agent.manus.Manus",  # 添加大写版本
                "DataAnalysisExpert": "app.agent.DataAnalysisExpert.DataAnalysisExpert",
                "ExcelCleanAgent": "app.agent.excel_data_cleaner.ExcelCleanAgent",
                "GameDataAnalysisAgent": "app.agent.game_data_analysis.GameDataAnalysisAgent",
                "SWEAgent": "app.agent.swe.SWEAgent",
                "BrowserAgent": "app.agent.browser.BrowserAgent",
                "AnswerQuestionAgent": "app.agent.AnswerQuestionAgent.AnalysisResultQnAAgent",  # 修正为正确的类名
                "AnalysisResultQnAAgent": "app.agent.AnswerQuestionAgent.AnalysisResultQnAAgent",  # 修正为正确的类名
                "data_analysis": "app.agent.data_analysis.DataAnalysis",
                "DataAnalysis": "app.agent.data_analysis.DataAnalysis",  # 添加大写版本
                # 新增Game Data Analysis相关Agent - 更新为正确的路径和类名
                "MultiDataAnalysisCoordinator": "app.agent.lead_agent.MultiDataAnalysisCoordinator",  # 修正为正确的路径
                "KeyMetricAnalysisAgent": "app.agent.key_metric_analysis_agent.KeyMetricAnalysisAgent",
            }

            class_path = agent_class_map.get(agent_type)
            if not class_path:
                print(f"❌ Unknown agent type: {agent_type}")
                return None

            print(f"🔍 Importing class from: {class_path}")
            module_path, class_name = class_path.rsplit(".", 1)

            try:
                module = __import__(module_path, fromlist=[class_name])
                agent_class = getattr(module, class_name)

                # 检查类是否有create方法
                if hasattr(agent_class, "create"):
                    print(f"✅ Agent class {class_name} has create() method")
                    return agent_class
                elif hasattr(agent_class, "__init__"):
                    print(
                        f"⚠️ Agent class {class_name} has __init__ but no create() method"
                    )
                    # 创建一个包装器来处理没有create方法的类
                    return self._create_agent_wrapper(agent_class, class_name)
                else:
                    print(
                        f"❌ Agent class {class_name} has neither create() nor __init__ method"
                    )
                    return None

            except ImportError as e:
                print(f"❌ Failed to import {class_path}: {e}")
                return None
            except AttributeError as e:
                print(
                    f"❌ Failed to get class {class_name} from module {module_path}: {e}"
                )
                return None

        except Exception as e:
            print(f"❌ Error getting agent class {agent_type}: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _create_agent_wrapper(self, agent_class, class_name):
        """为没有create方法的Agent类创建包装器"""

        class AgentWrapper:
            def __init__(self, original_class):
                self.original_class = original_class

            async def create(self):
                """异步创建Agent实例"""
                try:
                    print(f"🔧 Creating {class_name} instance using __init__")

                    # 检查是否是ToolCallAgent或其子类
                    if hasattr(self.original_class, "__mro__"):
                        class_hierarchy = [
                            cls.__name__ for cls in self.original_class.__mro__
                        ]
                        print(f"  📋 Class hierarchy: {class_hierarchy}")

                        # 如果是ToolCallAgent的子类，需要特殊处理
                        if (
                            "ToolCallAgent" in class_hierarchy
                            or "ReActAgent" in class_hierarchy
                        ):
                            print(
                                f"  🔧 {class_name} is a ToolCallAgent, initializing with default config"
                            )
                            # ToolCallAgent类通常需要额外的初始化
                            instance = self.original_class()

                            # 尝试初始化MCP服务器（如果需要）
                            if hasattr(instance, "initialize_mcp_servers"):
                                try:
                                    await instance.initialize_mcp_servers()
                                    print(
                                        f"  ✅ MCP servers initialized for {class_name}"
                                    )
                                except Exception as e:
                                    print(
                                        f"  ⚠️ Failed to initialize MCP servers for {class_name}: {e}"
                                    )

                            # 设置初始化标志
                            if hasattr(instance, "_initialized"):
                                instance._initialized = True

                            print(f"  ✅ Successfully created {class_name} instance")
                            return instance
                        else:
                            # 普通类，直接实例化
                            instance = self.original_class()
                            print(f"  ✅ Successfully created {class_name} instance")
                            return instance
                    else:
                        # 直接实例化
                        instance = self.original_class()
                        print(f"  ✅ Successfully created {class_name} instance")
                        return instance

                except Exception as e:
                    print(f"  ❌ Failed to create {class_name} instance: {e}")
                    import traceback

                    traceback.print_exc()
                    raise e

        return AgentWrapper(agent_class)

    def get_available_agents(self) -> List[str]:
        """获取可用的Agent列表"""
        return [
            "Manus",
            "DataAnalysisExpert",
            "ExcelCleanAgent",
            "GameDataAnalysisAgent",
            "SWEAgent",
            "BrowserAgent",
            "AnalysisResultQnAAgent",
            "DataAnalysis",
            # Game Data Analysis 专用Agent
            "MultiDataAnalysisCoordinator",
            "KeyMetricAnalysisAgent",
            "AnalysisResultQnAAgent",
        ]

    def get_current_flow_config(self) -> FlowConfiguration:
        """获取当前flow配置"""
        return self.flow_config

    async def cleanup(self):
        """清理资源"""
        # 清理agent
        if self.current_agent and hasattr(self.current_agent, "cleanup"):
            try:
                await self.current_agent.cleanup()
            except Exception as e:
                print(f"Error cleaning up agent: {e}")

        # 清理flow
        if self.current_flow and hasattr(self.current_flow, "cleanup"):
            try:
                await self.current_flow.cleanup()
            except Exception as e:
                print(f"Error cleaning up flow: {e}")

        self.current_agent = None
        self.current_flow = None


class StreamingFlowWrapper:
    """Flow流式输出包装器 - 通过拦截日志实现流式输出"""

    def __init__(self, flow, broadcast_callback):
        self.flow = flow
        self.broadcast = broadcast_callback

    async def run(self, request: Optional[str] = None) -> str:
        """执行Flow并通过日志拦截实现流式输出"""
        try:
            # 广播开始消息
            await self.broadcast(
                "start",
                {
                    "description": "Data Analysis Flow开始执行任务...",
                    "request": request or "继续对话",
                },
                0,
                5,  # Flow的估计步骤数
            )

            # 使用日志拦截器捕获Flow内部的所有日志
            with LoggingInterceptor(self.broadcast):
                print(f"🔧 Executing flow with request: {request}")
                print(f"   Flow type: {type(self.flow).__name__}")
                print(
                    f"   Flow agents: {list(self.flow.agents.keys()) if hasattr(self.flow, 'agents') else 'No agents'}"
                )

                # 执行Flow - 现在所有内部日志都会被拦截并转换为流式消息
                result = await self.flow.execute(request)
                print(f"✅ Flow execution completed: {result}")

            # 广播完成消息
            await self.broadcast(
                "complete",
                {
                    "result": result or "Flow execution completed",
                    "description": "Data Analysis Flow执行完成",
                },
                5,
                5,
            )

            return result or "Flow execution completed successfully"

        except Exception as e:
            print(f"❌ Flow execution error: {e}")
            import traceback

            traceback.print_exc()

            await self.broadcast(
                "error",
                {"error": str(e), "description": "Flow执行过程中发生错误"},
                0,
                5,
            )
            return f"Flow execution failed: {str(e)}"


class StreamingAgentWrapper:
    """Agent流式输出包装器 - 通过日志拦截和原有逻辑结合实现流式输出"""

    def __init__(self, agent, broadcast_callback):
        self.agent = agent
        self.broadcast = broadcast_callback

    async def run(self, request: Optional[str] = None) -> str:
        """重写run方法，结合日志拦截实现详细的流式输出"""
        from app.sandbox.client import SANDBOX_CLIENT
        from app.schema import AgentState

        # 检查状态
        if self.agent.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.agent.state}")

        # 添加用户请求到内存
        if request:
            self.agent.update_memory("user", request)

        # 广播开始消息
        await self.broadcast(
            "start",
            {
                "description": "AI Agent开始执行任务...",
                "request": request or "继续对话",
            },
            0,
            self.agent.max_steps,
        )

        results = []

        # 使用日志拦截器 + 原有逻辑相结合
        with LoggingInterceptor(self.broadcast):
            # 使用状态上下文
            async with self.agent.state_context(AgentState.RUNNING):
                while (
                    self.agent.current_step < self.agent.max_steps
                    and self.agent.state != AgentState.FINISHED
                ):
                    self.agent.current_step += 1

                    # 广播步骤开始
                    await self.broadcast(
                        "step_start",
                        {
                            "step": self.agent.current_step,
                            "description": f"执行步骤 {self.agent.current_step}/{self.agent.max_steps}",
                        },
                        self.agent.current_step,
                        self.agent.max_steps,
                    )

                    try:
                        # 执行思考阶段 - 日志会被自动拦截
                        await self.broadcast(
                            "think_start",
                            {"content": "Agent正在分析当前情况并制定下一步计划..."},
                            self.agent.current_step,
                            self.agent.max_steps,
                        )

                        should_act = await self.agent.think()

                        # 获取思考内容
                        think_content = ""
                        if self.agent.messages and len(self.agent.messages) > 0:
                            last_message = self.agent.messages[-1]
                            if last_message.content:
                                think_content = last_message.content[:500]  # 限制长度

                        await self.broadcast(
                            "think",
                            {
                                "content": think_content,
                                "reasoning": "Agent已完成分析，决定下一步行动",
                                "will_act": should_act,
                            },
                            self.agent.current_step,
                            self.agent.max_steps,
                        )

                        if not should_act:
                            step_result = "思考完成 - 无需进一步行动"
                        else:
                            # 执行行动阶段 - 工具调用日志会被自动拦截
                            # 广播即将执行的工具
                            if (
                                hasattr(self.agent, "tool_calls")
                                and self.agent.tool_calls
                            ):
                                for tool_call in self.agent.tool_calls:
                                    await self.broadcast(
                                        "act",
                                        {
                                            "tool_name": tool_call.function.name,
                                            "tool_args": (
                                                tool_call.function.arguments[:200]
                                                if tool_call.function.arguments
                                                else ""
                                            ),
                                            "description": f"执行工具: {tool_call.function.name}",
                                        },
                                        self.agent.current_step,
                                        self.agent.max_steps,
                                    )

                            # 执行动作 - Agent内部的所有日志都会被拦截
                            step_result = await self.agent.act()

                            # 广播执行结果
                            if (
                                hasattr(self.agent, "tool_calls")
                                and self.agent.tool_calls
                            ):
                                for tool_call in self.agent.tool_calls:
                                    await self.broadcast(
                                        "observe",
                                        {
                                            "tool_name": tool_call.function.name,
                                            "result": (
                                                step_result[:300]
                                                if step_result
                                                else "执行完成"
                                            ),
                                            "success": True,
                                        },
                                        self.agent.current_step,
                                        self.agent.max_steps,
                                    )

                        # 检查是否卡住
                        if self.agent.is_stuck():
                            self.agent.handle_stuck_state()
                            await self.broadcast(
                                "observe",
                                {
                                    "tool_name": "system",
                                    "result": "检测到重复响应，正在调整策略...",
                                    "success": True,
                                },
                                self.agent.current_step,
                                self.agent.max_steps,
                            )

                        results.append(f"步骤 {self.agent.current_step}: {step_result}")

                        # 广播步骤完成
                        await self.broadcast(
                            "step_complete",
                            {
                                "step": self.agent.current_step,
                                "result": (
                                    step_result[:200] if step_result else "步骤完成"
                                ),
                                "description": f"步骤 {self.agent.current_step} 执行完成",
                            },
                            self.agent.current_step,
                            self.agent.max_steps,
                        )

                        # 如果任务完成，跳出循环
                        if self.agent.state == AgentState.FINISHED:
                            break

                    except Exception as e:
                        error_msg = f"步骤 {self.agent.current_step} 执行失败: {str(e)}"
                        results.append(error_msg)

                        # 广播错误
                        await self.broadcast(
                            "error",
                            {
                                "error": str(e),
                                "step": self.agent.current_step,
                                "description": error_msg,
                            },
                            self.agent.current_step,
                            self.agent.max_steps,
                        )
                        break

                # 检查是否达到最大步数
                if self.agent.current_step >= self.agent.max_steps:
                    self.agent.current_step = 0
                    self.agent.state = AgentState.IDLE
                    results.append(f"任务终止：达到最大步数 ({self.agent.max_steps})")

        # 清理沙箱
        await SANDBOX_CLIENT.cleanup()

        # 生成最终结果
        final_result = "\n".join(results) if results else "未执行任何步骤"

        # 广播完成消息
        await self.broadcast(
            "complete",
            {
                "result": final_result,
                "total_steps": self.agent.current_step,
                "description": "任务执行完成",
            },
            self.agent.current_step,
            self.agent.max_steps,
        )

        return final_result


# =============================================================================
# 方案2: 简化版日志拦截 + 装饰器模式
# =============================================================================


def create_streaming_wrapper(original_method, broadcast_callback, method_name):
    """创建流式输出装饰器"""

    @wraps(original_method)
    async def wrapper(*args, **kwargs):
        try:
            # 广播方法开始
            await broadcast_callback(
                f"{method_name}_start",
                {
                    "method": method_name,
                    "description": f"开始执行 {method_name}...",
                    "args_info": str(args[1:])[:100] if len(args) > 1 else "",
                },
                0,
                5,
            )

            # 执行原始方法
            result = await original_method(*args, **kwargs)

            # 广播方法完成
            await broadcast_callback(
                f"{method_name}_complete",
                {
                    "method": method_name,
                    "description": f"{method_name} 执行完成",
                    "result": str(result)[:200] if result else "Method completed",
                },
                1,
                5,
            )

            return result

        except Exception as e:
            # 广播错误
            await broadcast_callback(
                "error",
                {
                    "method": method_name,
                    "error": str(e),
                    "description": f"{method_name} 执行失败",
                },
                0,
                5,
            )
            raise

    return wrapper


class SimpleMethodInterceptor:
    """简化版方法拦截器 - 使用装饰器模式"""

    def __init__(self, broadcast_callback):
        self.broadcast = broadcast_callback
        self.patched_objects = []

    def __enter__(self):
        """进入上下文时安装装饰器"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时清理"""
        self.patched_objects.clear()

    def patch_flow_instance(self, flow_instance):
        """为Flow实例添加流式输出装饰器"""
        try:
            # 临时禁用method patching以避免Pydantic冲突
            # TODO: 实现更安全的方法包装机制
            print(f"⚠️ Method patching disabled for flow: {type(flow_instance).__name__}")
            ## pass

            # 装饰execute方法 - 使用setattr绕过Pydantic字段验证
            if hasattr(flow_instance, "execute"):
                original_execute = getattr(flow_instance, "execute")
                wrapped_execute = create_streaming_wrapper(
                    original_execute, self.broadcast, "flow_execute"
                )
                # 使用setattr直接设置到对象的__dict__来绕过Pydantic验证
                object.__setattr__(flow_instance, "execute", wrapped_execute)
                self.patched_objects.append(
                    (flow_instance, "execute", original_execute)
                )

            # 装饰_execute_step方法（如果存在）
            if hasattr(flow_instance, "_execute_step"):
                original_step = getattr(flow_instance, "_execute_step")
                wrapped_step = create_streaming_wrapper(
                    original_step, self.broadcast, "execute_step"
                )
                object.__setattr__(flow_instance, "_execute_step", wrapped_step)
                self.patched_objects.append(
                    (flow_instance, "_execute_step", original_step)
                )

        except Exception as e:
            print(f"⚠️ Failed to patch flow instance: {e}")
            import traceback
            traceback.print_exc()

    def patch_agent_instance(self, agent_instance):
        """为Agent实例添加流式输出装饰器"""
        try:
            # 装饰run方法
            if hasattr(agent_instance, "run"):
                original_run = agent_instance.run
                agent_instance.run = create_streaming_wrapper(
                    original_run, self.broadcast, "agent_run"
                )
                self.patched_objects.append((agent_instance, "run", original_run))

        except Exception as e:
            print(f"⚠️ Failed to patch agent instance: {e}")


class HybridStreamingFlowWrapper:
    """混合式Flow包装器 - 结合日志拦截和方法装饰"""

    def __init__(self, flow, broadcast_callback):
        self.flow = flow
        self.broadcast = broadcast_callback

    async def run(self, request: Optional[str] = None) -> str:
        """使用混合式方法执行Flow"""
        try:
            # 广播开始消息
            await self.broadcast(
                "start",
                {
                    "description": "Hybrid Data Analysis Flow开始执行...",
                    "request": request or "继续对话",
                    "flow_type": type(self.flow).__name__,
                },
                0,
                8,
            )

            # 方法1: 使用日志拦截器
            log_interceptor = LoggingInterceptor(self.broadcast)

            # 方法2: 使用方法装饰器
            method_interceptor = SimpleMethodInterceptor(self.broadcast)

            with log_interceptor:
                with method_interceptor:
                    # 为当前Flow实例添加装饰器
                    method_interceptor.patch_flow_instance(self.flow)

                    # 如果Flow有agents，也为它们添加装饰器
                    if hasattr(self.flow, "agents") and self.flow.agents:
                        for agent_key, agent in self.flow.agents.items():
                            method_interceptor.patch_agent_instance(agent)

                    # 执行Flow - 现在会有多层拦截
                    print(f"🔧 Executing hybrid flow with request: {request}")
                    result = await self.flow.execute(request)
                    print(f"✅ Hybrid flow execution completed: {result}")

            # 广播完成消息
            await self.broadcast(
                "complete",
                {
                    "result": result or "Hybrid Flow execution completed",
                    "description": "Hybrid Data Analysis Flow执行完成",
                    "flow_type": type(self.flow).__name__,
                },
                8,
                8,
            )

            return result or "Hybrid Flow execution completed successfully"

        except Exception as e:
            print(f"❌ Hybrid flow execution error: {e}")
            import traceback

            traceback.print_exc()

            await self.broadcast(
                "error",
                {"error": str(e), "description": "Hybrid Flow执行过程中发生错误"},
                0,
                8,
            )
            return f"Hybrid Flow execution failed: {str(e)}"


# 全局Agent服务实例
agent_service = AgentService()
