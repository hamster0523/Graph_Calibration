import json
import os
from datetime import datetime
from enum import Enum
from typing import Any, List, Literal, Optional, Union, Dict

from pydantic import BaseModel, Field


class Role(str, Enum):
    """Message role options"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


ROLE_VALUES = tuple(role.value for role in Role)
ROLE_TYPE = Literal[ROLE_VALUES]  # type: ignore


class ToolChoice(str, Enum):
    """Tool choice options"""

    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"


TOOL_CHOICE_VALUES = tuple(choice.value for choice in ToolChoice)
TOOL_CHOICE_TYPE = Literal[TOOL_CHOICE_VALUES]  # type: ignore


class AgentState(str, Enum):
    """Agent execution states"""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"


class Function(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    """Represents a tool/function call in a message"""

    id: str
    type: str = "function"
    function: Function


class Message(BaseModel):
    """Represents a chat message in the conversation"""

    role: ROLE_TYPE = Field(...)  # type: ignore
    content: Optional[str] = Field(default=None)
    tool_calls: Optional[List[ToolCall]] = Field(default=None)
    name: Optional[str] = Field(default=None)
    tool_call_id: Optional[str] = Field(default=None)
    base64_image: Optional[str] = Field(default=None)

    def __add__(self, other) -> List["Message"]:
        """支持 Message + list 或 Message + Message 的操作"""
        if isinstance(other, list):
            return [self] + other
        elif isinstance(other, Message):
            return [self, other]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'"
            )

    def __radd__(self, other) -> List["Message"]:
        """支持 list + Message 的操作"""
        if isinstance(other, list):
            return other + [self]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(other).__name__}' and '{type(self).__name__}'"
            )

    def to_dict(self) -> dict:
        """Convert message to dictionary format"""
        message = {"role": self.role}
        if self.content is not None:
            message["content"] = self.content
        if self.tool_calls is not None:
            message["tool_calls"] = [tool_call.dict() for tool_call in self.tool_calls]
        if self.name is not None:
            message["name"] = self.name
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.base64_image is not None:
            message["base64_image"] = self.base64_image
        return message

    @classmethod
    def user_message(
        cls, content: str, base64_image: Optional[str] = None
    ) -> "Message":
        """Create a user message"""
        return cls(role=Role.USER, content=content, base64_image=base64_image)

    @classmethod
    def system_message(cls, content: str) -> "Message":
        """Create a system message"""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def assistant_message(
        cls, content: Optional[str] = None, base64_image: Optional[str] = None
    ) -> "Message":
        """Create an assistant message"""
        return cls(role=Role.ASSISTANT, content=content, base64_image=base64_image)

    @classmethod
    def tool_message(
        cls, content: str, name, tool_call_id: str, base64_image: Optional[str] = None
    ) -> "Message":
        """Create a tool message"""
        return cls(
            role=Role.TOOL,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
            base64_image=base64_image,
        )

    @classmethod
    def from_tool_calls(
        cls,
        tool_calls: List[Any],
        content: Union[str, List[str]] = "",
        base64_image: Optional[str] = None,
        **kwargs,
    ):
        """Create ToolCallsMessage from raw tool calls.

        Args:
            tool_calls: Raw tool calls from LLM
            content: Optional message content
            base64_image: Optional base64 encoded image
        """
        formatted_calls = [
            {"id": call.id, "function": call.function.model_dump(), "type": "function"}
            for call in tool_calls
        ]
        return cls(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=formatted_calls,
            base64_image=base64_image,
            **kwargs,
        )


class Memory(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    max_messages: int = Field(default=100)

    def add_message(self, message: Message) -> None:
        """Add a message to memory"""
        self.messages.append(message)
        # Optional: Implement message limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def add_messages(self, messages: List[Message]) -> None:
        """Add multiple messages to memory"""
        self.messages.extend(messages)
        # Optional: Implement message limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def clear(self) -> None:
        """Clear all messages"""
        self.messages.clear()

    def get_recent_messages(self, n: int) -> List[Message]:
        """Get n most recent messages"""
        return self.messages[-n:]

    def to_dict_list(self) -> List[dict]:
        """Convert messages to list of dicts"""
        return [msg.to_dict() for msg in self.messages]


class AgentMemoryManager:

    def __init__(self, base_dir: str = "agent_memories"):
        from .config import config

        self.base_dir = os.path.join(config.root_path, base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

    async def save_memory(self, agent, task_id: Optional[str] = None):
        """保存agent记忆"""
        if not task_id:
            task_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        agent_name = agent.name.strip().replace(" ", "_")
        file_path = os.path.join(self.base_dir, f"{agent_name}_{task_id}.json")
        memory_dict = agent.memory.model_dump()

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(memory_dict, f, ensure_ascii=False, indent=2)

        return file_path

    async def load_memory(self, agent, file_path: str):
        """从文件加载记忆到agent"""
        with open(file_path, "r", encoding="utf-8") as f:
            memory_dict = json.load(f)

        memory = Memory(**memory_dict)
        agent.memory = memory
        return agent

    def list_memories(self, agent_name: Optional[str] = None):
        """列出保存的记忆文件"""
        files = os.listdir(self.base_dir)
        if agent_name:
            files = [f for f in files if f.startswith(f"{agent_name}_")]
        return files
    
    def find_last_create_chat_completion(self, file_path : str):
        agent_memory = json.load(open(file_path, "r"))
        agent_memory_message : List[Dict[str, Any]] = agent_memory['messages']
        for message in reversed(agent_memory_message):
            if message.get("role") == "assistant" and message.get("tool_calls"):
                tool_calls = message.get("tool_calls", [])
                for tool_call in tool_calls:
                    if (tool_call.get("type") == "function" and 
                        tool_call.get("function", {}).get("name") == "create_chat_completion"):
                        tool_call_id = tool_call.get("id")
                        for response_msg in agent_memory_message:
                            if (response_msg.get("role") == "tool" and 
                                response_msg.get("tool_call_id") == tool_call_id):
                                content = response_msg.get("content", "")
                                if content.startswith("Observed output of cmd `create_chat_completion` executed:\n"):
                                    content = content.replace("Observed output of cmd `create_chat_completion` executed:\n", "")
                                return content
            
        return ""

    def format_memory(
        self, memory_file_path: str, include_metadata: bool = True
    ) -> str:
        """
        格式化记忆文件中的消息为可读的字符串格式

        Args:
            memory_file_path: JSON记忆文件的路径
            include_metadata: 是否包含元数据信息（如tool_calls, tool_call_id等）

        Returns:
            格式化后的字符串
        """
        try:
            with open(memory_file_path, "r", encoding="utf-8") as f:
                memory_data = json.load(f)

            if "messages" not in memory_data:
                return "错误：记忆文件格式不正确，缺少messages字段"

            messages = memory_data["messages"]
            formatted_lines = []

            # 添加文件信息头部
            if include_metadata:
                file_name = os.path.basename(memory_file_path)
                formatted_lines.append(f"=== 记忆文件: {file_name} ===")
                formatted_lines.append(f"消息总数: {len(messages)}")
                formatted_lines.append("=" * 50)
                formatted_lines.append("")

            # 格式化每条消息
            for i, msg in enumerate(messages, 1):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                # 角色标题
                role_display = {
                    "user": "👤 用户",
                    "assistant": "🤖 助手",
                    "system": "⚙️ 系统",
                    "tool": "🔧 工具",
                }.get(role, f"❓ {role}")

                formatted_lines.append(f"[{i}] {role_display}:")

                # 消息内容
                if content:
                    # 处理多行内容的缩进
                    content_lines = content.split("\n")
                    for line in content_lines:
                        formatted_lines.append(f"    {line}")
                else:
                    formatted_lines.append("    <无内容>")

                # 包含元数据信息
                if include_metadata:
                    tool_calls = msg.get("tool_calls")
                    tool_call_id = msg.get("tool_call_id")
                    name = msg.get("name")
                    base64_image = msg.get("base64_image")

                    metadata_items = []
                    if tool_calls:
                        formatted_lines.append("    📞 工具调用:")
                        for tc in tool_calls:
                            func_name = tc.get("function", {}).get("name", "未知")
                            func_args = tc.get("function", {}).get("arguments", "{}")
                            formatted_lines.append(
                                f"        - {func_name}: {func_args}"
                            )

                    if tool_call_id:
                        formatted_lines.append(f"    🔗 工具调用ID: {tool_call_id}")

                    if name:
                        formatted_lines.append(f"    📝 名称: {name}")

                    if base64_image:
                        formatted_lines.append(f"    🖼️ 包含图片: 是")

                formatted_lines.append("")  # 消息间空行

            return "\n".join(formatted_lines)

        except FileNotFoundError:
            return f"错误：找不到记忆文件 {memory_file_path}"
        except json.JSONDecodeError as e:
            return f"错误：记忆文件JSON格式错误 - {str(e)}"
        except Exception as e:
            return f"错误：处理记忆文件时发生异常 - {str(e)}"

    def format_memory_simple(self, memory_file_path: str) -> str:
        """
        简单格式化记忆文件，只显示角色和内容

        Args:
            memory_file_path: JSON记忆文件的路径

        Returns:
            简化格式的字符串
        """
        try:
            with open(memory_file_path, "r", encoding="utf-8") as f:
                memory_data = json.load(f)

            if "messages" not in memory_data:
                return "错误：记忆文件格式不正确，缺少messages字段"

            messages = memory_data["messages"]
            formatted_lines = []

            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                if content:  # 只显示有内容的消息
                    role_prefix = {
                        "user": "User",
                        "assistant": "Assistant",
                        "system": "System",
                        "tool": "Tool",
                    }.get(role, role.title())

                    formatted_lines.append(f"{role_prefix}: {content}")
                    formatted_lines.append("")  # 消息间空行

            return "\n".join(formatted_lines)

        except Exception as e:
            return f"错误：{str(e)}"
