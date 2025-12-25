import os
from app.tool.base import BaseTool, ToolResult

class MarkdownSummaryWriterTool(BaseTool):
    """
    工具：将文本内容写入指定的Markdown文件（.md），用于保存分析总结等。
    """
    name: str = "markdown_summary_writer_tool"
    description: str = "Write a summary or any text content to a specified Markdown (.md) file."
    parameters: dict = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the Markdown (.md) file to write to."
            },
            "content": {
                "type": "string",
                "description": "The summary or text content to write into the file."
            },
            "mode": {
                "type": "string",
                "description": "Write mode: 'w' for overwrite, 'a' for append. Default is 'w'.",
                "default": "w"
            }
        },
        "required": ["file_path", "content"],
    }

    async def execute(self, file_path: str, content: str, mode: str = "w") -> ToolResult:
        """
        将内容写入指定的Markdown文件。
        Args:
            file_path: 目标md文件路径
            content: 要写入的内容
            mode: 写入模式，'w'覆盖，'a'追加
        Returns:
            ToolResult，output为写入成功提示
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, mode, encoding="utf-8") as f:
                f.write(content)
                if not content.endswith('\n'):
                    f.write('\n')
            return ToolResult(output=f"Summary written to {file_path} (mode={mode})")
        except Exception as e:
            return ToolResult(error=f"Failed to write summary to {file_path}: {str(e)}")
