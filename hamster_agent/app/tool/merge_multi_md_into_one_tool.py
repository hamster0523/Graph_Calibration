import os
import glob
from typing import List

from app.tool.base import BaseTool, ToolResult
from app.exceptions import ToolError
from app.logger import logger

class MergeMultiMdIntoOneTool(BaseTool):
    """
    工具：递归搜索指定目录下的所有 Markdown 文件，并将它们合并为一个文件。
    """
    name: str = "merge_multi_md_into_one_tool"
    description: str = (
        "递归搜索目录中的所有 Markdown (.md) 文件，并将它们合并为一个单一的 Markdown 文件"
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "directory_path": {
                "type": "string",
                "description": "包含 Markdown 文件的目录路径",
            },
            "output_file_path": {
                "type": "string",
                "description": "合并后的 Markdown 文件的保存路径",
            },
            "sort_by": {
                "type": "string",
                "description": "文件排序方式: 'name'(按文件名), 'date'(按修改日期), 'path'(按文件路径)",
                "default": "name",
                "enum": ["name", "date", "path"]
            },
            "add_headers": {
                "type": "boolean",
                "description": "是否在每个文件内容前添加原始文件名作为标题",
                "default": True
            },
            "add_file_paths": {
                "type": "boolean",
                "description": "是否包含原始文件路径信息",
                "default": False
            }
        },
        "required": ["directory_path", "output_file_path"]
    }

    async def execute(
        self,
        directory_path: str,
        output_file_path: str,
        sort_by: str = "name",
        add_headers: bool = True,
        add_file_paths: bool = False,
    ) -> ToolResult:
        """
        执行 Markdown 文件合并过程
        
        Args:
            directory_path: 要搜索 Markdown 文件的目录路径
            output_file_path: 合并后文件的保存路径
            sort_by: 文件排序方式 ('name', 'date', 'path')
            add_headers: 是否添加包含原始文件名的标题
            add_file_paths: 是否包含原始文件路径
            
        Returns:
            ToolResult: 操作结果
        """
        try:
            # 检查目录是否存在
            if not os.path.isdir(directory_path):
                raise ToolError(f"目录不存在: {directory_path}")
            
            # 递归查找所有 Markdown 文件
            md_files = self._find_markdown_files(directory_path)
            if not md_files:
                return ToolResult(output=f"在 {directory_path} 中未找到 Markdown 文件")
            
            # 排除分析总结文件
            md_files = self._exclude_analysis_summary_md_files(md_files)
            
            # 按照指定方法排序文件
            sorted_files = self._sort_files(md_files, sort_by)
            
            # 合并文件
            merged_content = self._merge_files(sorted_files, add_headers, add_file_paths)
            
            # 创建输出目录（如果不存在）
            os.makedirs(os.path.dirname(os.path.abspath(output_file_path)), exist_ok=True)
            
            # 写入合并后的内容到输出文件
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(merged_content)
            
            return ToolResult(
                output=f"成功将 {len(sorted_files)} 个 Markdown 文件合并到 {output_file_path}"
            )
            
        except ToolError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            error_msg = f"合并 Markdown 文件时出错: {str(e)}"
            logger.error(error_msg)
            return ToolResult(error=error_msg)
        
    def _exclude_analysis_summary_md_files(self, md_files : List[str]) -> List[str]:
        """排除分析总结文件"""
        return [file for file in md_files if not file.endswith("analysis_summary.md")]

    def _find_markdown_files(self, directory_path: str) -> List[str]:
        """递归查找指定目录中的所有 Markdown 文件"""
        # 使用 glob 递归查找所有 .md 文件
        md_pattern = os.path.join(directory_path, '**', '*.md')
        return glob.glob(md_pattern, recursive=True)

    def _sort_files(self, files: List[str], sort_by: str) -> List[str]:
        """按指定方法排序文件列表"""
        if sort_by == "name":
            return sorted(files, key=lambda x: os.path.basename(x))
        elif sort_by == "date":
            return sorted(files, key=lambda x: os.path.getmtime(x))
        elif sort_by == "path":
            return sorted(files)
        else:
            # 默认按文件名排序
            return sorted(files, key=lambda x: os.path.basename(x))

    def _merge_files(self, files: List[str], add_headers: bool, add_file_paths: bool) -> str:
        """合并多个 Markdown 文件的内容"""
        merged_content = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read().strip()
                
                # 如果需要，添加标题
                if add_headers:
                    file_name = os.path.basename(file_path)
                    header = f"# {os.path.splitext(file_name)[0]}\n\n"
                    merged_content.append(header)
                
                # 如果需要，添加文件路径
                if add_file_paths:
                    path_info = f"*Source: `{file_path}`*\n\n"
                    merged_content.append(path_info)
                
                # 添加文件内容
                merged_content.append(file_content)
                
                # 在文件之间添加分隔符
                merged_content.append("\n\n---\n\n")
                
            except Exception as e:
                logger.warning(f"无法读取文件 {file_path}: {str(e)}")
                merged_content.append(f"*读取文件 {file_path} 时出错*\n\n")
        
        # 如果有内容，删除最后一个分隔符
        if merged_content and merged_content[-1] == "\n\n---\n\n":
            merged_content.pop()
            
        return "".join(merged_content)