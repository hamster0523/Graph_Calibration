import os
import pandas as pd
import asyncio

from app.tool.base import BaseTool, ToolResult
from app.logger import logger

class ExcelColumnNamesReadTool(BaseTool):
    """
    工具：读取CSV文件并返回所有列名。
    """
    name: str = "excel_column_names_read_tool"
    description: str = "Read and return all column names from a CSV file."
    parameters: dict = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the CSV file to read column names from."
            }
        },
        "required": ["file_path"],
    }

    async def execute(self, file_path: str) -> ToolResult:
        """
        读取CSV文件并返回所有列名。
        Args:
            file_path: CSV文件路径
        Returns:
            ToolResult，output为列名列表
        """
        if not os.path.exists(file_path):
            return ToolResult(error=f"CSV file not found: {file_path}")
        try:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, lambda: pd.read_csv(file_path, nrows=0))
            columns = list(df.columns)
            final_str = "[" + ", ".join(columns) + "]"
            logger.info(f"Read {len(columns)} columns from {file_path}: {final_str}")
            return ToolResult(output=final_str)
        except Exception as e:
            logger.error(f"Error reading columns from CSV: {e}")
            return ToolResult(error=f"Failed to read columns from CSV: {str(e)}")

if __name__ == "__main__":
    import argparse
    import asyncio
    parser = argparse.ArgumentParser(description="Read and print all column names from a CSV file.")
    parser.add_argument("file_path", help="Path to the CSV file")
    args = parser.parse_args()
    async def run():
        tool = ExcelColumnNamesReadTool()
        result = await tool.execute(args.file_path)
        if result.error:
            print(f"Error: {result.error}")
        else:
            print("Column names:")
            for col in result.output:
                print(col)
    asyncio.run(run())
