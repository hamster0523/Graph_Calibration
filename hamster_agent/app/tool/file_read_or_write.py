"""简化的本地文件读写工具。"""

import os
from pathlib import Path
from typing import Any, List, Literal, Optional

from app.exceptions import ToolError
from app.tool import BaseTool
from app.tool.base import ToolResult

Command = Literal[
    "view",
    "create",
    "write",
    "append",
]

# 常量
MAX_RESPONSE_LEN: int = 16000
TRUNCATED_MESSAGE: str = (
    "<response clipped><NOTE>为了节省上下文，只显示了文件的部分内容。"
    "如果需要查看更多内容，请使用view命令指定行范围。</NOTE>"
)

# 工具描述
_FILE_TOOL_DESCRIPTION = """简化的文件读写工具，支持查看、创建、写入和追加操作
* 仅支持本地文件操作
* `view` 命令显示文件内容或目录列表
* `create` 命令创建新文件（如果文件已存在会报错）
* `write` 命令覆盖写入文件内容
* `append` 命令追加内容到文件末尾
* 如果输出过长，会被截断并显示提示信息
"""


def maybe_truncate(
    content: str, truncate_after: Optional[int] = MAX_RESPONSE_LEN
) -> str:
    """如果内容超过指定长度，则截断内容并添加提示。"""
    if not truncate_after or len(content) <= truncate_after:
        return content
    return content[:truncate_after] + TRUNCATED_MESSAGE


class FileReadWriteTool(BaseTool):
    """简化的本地文件读写工具。"""

    name: str = "file_read_write_tool"
    description: str = _FILE_TOOL_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "description": "要执行的命令。可选项：`view`, `create`, `write`, `append`",
                "enum": ["view", "create", "write", "append"],
                "type": "string",
            },
            "path": {
                "description": "文件或目录的绝对路径",
                "type": "string",
            },
            "content": {
                "description": "create、write、append命令所需的文件内容",
                "type": "string",
            },
            "encoding": {
                "description": "文件编码（默认：utf-8）",
                "type": "string",
                "default": "utf-8",
            },
            "view_range": {
                "description": "view命令的可选参数，指定查看的行范围。例如[11, 12]显示第11-12行，[1, -1]显示全部",
                "items": {"type": "integer"},
                "type": "array",
            },
        },
        "required": ["command", "path"],
    }

    async def execute(
        self,
        command: Command,
        path: str,
        content: Optional[str] = None,
        encoding: str = "utf-8",
        view_range: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """执行文件操作命令。"""
        try:
            # 验证路径
            path_obj = Path(path)
            if not path_obj.is_absolute():
                raise ToolError(f"路径 {path} 不是绝对路径")

            # 根据命令执行相应操作
            if command == "view":
                result = await self._view(path_obj, view_range, encoding)
            elif command == "create":
                if content is None:
                    raise ToolError("create命令需要content参数")
                result = await self._create(path_obj, content, encoding)
            elif command == "write":
                if content is None:
                    raise ToolError("write命令需要content参数")
                result = await self._write(path_obj, content, encoding)
            elif command == "append":
                if content is None:
                    raise ToolError("append命令需要content参数")
                result = await self._append(path_obj, content, encoding)
            else:
                raise ToolError(f"不支持的命令: {command}")

            return result

        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"执行文件操作时发生错误: {str(e)}")

    async def _view(
        self,
        path: Path,
        view_range: Optional[List[int]] = None,
        encoding: str = "utf-8",
    ) -> ToolResult:
        """查看文件或目录内容。"""
        if not path.exists():
            raise ToolError(f"路径 {path} 不存在")

        if path.is_dir():
            # 显示目录内容
            if view_range:
                raise ToolError("查看目录时不支持view_range参数")

            try:
                items = []
                for item in sorted(path.iterdir()):
                    if item.name.startswith("."):
                        continue  # 跳过隐藏文件
                    if item.is_dir():
                        items.append(f"{item.name}/")
                    else:
                        items.append(item.name)

                content = f"目录 {path} 的内容:\n" + "\n".join(items)
                return ToolResult(output=content)
            except Exception as e:
                raise ToolError(f"读取目录失败: {str(e)}")
        else:
            # 显示文件内容
            try:
                with open(path, "r", encoding=encoding) as f:
                    file_content = f.read()

                # 应用查看范围
                if view_range:
                    if len(view_range) != 2:
                        raise ToolError("view_range应该包含两个整数")

                    lines = file_content.split("\n")
                    start_line, end_line = view_range

                    if start_line < 1:
                        raise ToolError("起始行号应该大于等于1")
                    if end_line != -1 and end_line < start_line:
                        raise ToolError("结束行号应该大于等于起始行号")

                    if end_line == -1:
                        selected_lines = lines[start_line - 1 :]
                    else:
                        selected_lines = lines[start_line - 1 : end_line]

                    file_content = "\n".join(selected_lines)
                    output = f"文件 {path} 的内容 (行 {start_line} 到 {end_line if end_line != -1 else '结尾'}):\n"
                else:
                    output = f"文件 {path} 的内容:\n"

                # 添加行号并截断如果太长
                numbered_content = self._add_line_numbers(
                    file_content, start_line=view_range[0] if view_range else 1
                )
                output += maybe_truncate(numbered_content)

                return ToolResult(output=output)
            except Exception as e:
                raise ToolError(f"读取文件失败: {str(e)}")

    async def _create(self, path: Path, content: str, encoding: str) -> ToolResult:
        """创建新文件。"""
        if path.exists():
            raise ToolError(f"文件 {path} 已存在，无法使用create命令覆盖")

        try:
            # 确保父目录存在
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding=encoding) as f:
                f.write(content)

            return ToolResult(output=f"文件创建成功: {path}")
        except Exception as e:
            raise ToolError(f"创建文件失败: {str(e)}")

    async def _write(self, path: Path, content: str, encoding: str) -> ToolResult:
        """覆盖写入文件内容。"""
        try:
            # 确保父目录存在
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding=encoding) as f:
                f.write(content)

            action = "覆盖写入" if path.exists() else "创建并写入"
            return ToolResult(output=f"文件{action}成功: {path}")
        except Exception as e:
            raise ToolError(f"写入文件失败: {str(e)}")

    async def _append(self, path: Path, content: str, encoding: str) -> ToolResult:
        """追加内容到文件末尾。"""
        try:
            # 确保父目录存在
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "a", encoding=encoding) as f:
                f.write(content)

            return ToolResult(output=f"内容追加成功: {path}")
        except Exception as e:
            raise ToolError(f"追加文件失败: {str(e)}")

    def _add_line_numbers(self, content: str, start_line: int = 1) -> str:
        """为内容添加行号。"""
        lines = content.split("\n")
        numbered_lines = []
        for i, line in enumerate(lines):
            line_num = start_line + i
            numbered_lines.append(f"{line_num:6}\t{line}")
        return "\n".join(numbered_lines)
