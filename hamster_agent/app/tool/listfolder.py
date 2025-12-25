"""本地目录查看工具。"""

import os
import stat
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from app.exceptions import ToolError
from app.tool import BaseTool
from app.tool.base import ToolResult

# 常量
MAX_FILES_TO_SHOW = 500  # 最大显示文件数量
TRUNCATED_MESSAGE = "\n<注意>目录包含更多文件，但为了节省上下文，只显示了前{}个文件。\n使用filters参数可以过滤特定类型的文件。"

class ListFolderTool(BaseTool):
    """本地目录查看工具，显示目录下文件的详细信息。"""

    name: str = "list_folder_tool"
    description: str = (
        "本地目录查看工具。显示指定目录下所有文件和子目录的详细信息，"
        "包括文件大小、修改时间、权限等。支持递归查看和文件类型过滤。"
    )
    
    parameters: dict = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要查看的目录绝对路径",
            },
            "recursive": {
                "type": "boolean",
                "description": "是否递归显示子目录内容",
                "default": False
            },
            "show_hidden": {
                "type": "boolean",
                "description": "是否显示隐藏文件（以.开头的文件）",
                "default": False
            },
            "filters": {
                "type": "array",
                "items": {"type": "string"},
                "description": "文件扩展名过滤器，例如：['.csv', '.txt', '.json']。不指定则显示所有文件",
            },
            "sort_by": {
                "type": "string",
                "enum": ["name", "size", "modified", "type"],
                "description": "排序方式：name(名称)、size(大小)、modified(修改时间)、type(类型)",
                "default": "name"
            },
            "reverse_sort": {
                "type": "boolean",
                "description": "是否倒序排列",
                "default": False
            },
            "show_details": {
                "type": "boolean",  
                "description": "是否显示详细信息（文件大小、修改时间、权限等）",
                "default": True
            }
        },
        "required": ["path"]
    }

    async def execute(
        self,
        path: str,
        recursive: bool = False,
        show_hidden: bool = False,
        filters: Optional[List[str]] = None,
        sort_by: str = "name",
        reverse_sort: bool = False,
        show_details: bool = True,
        **kwargs: Any,
    ) -> ToolResult:
        """执行目录查看操作。"""
        try:
            # 验证路径
            path_obj = Path(path)
            if not path_obj.is_absolute():
                raise ToolError(f"路径 {path} 不是绝对路径")
            
            if not path_obj.exists():
                raise ToolError(f"路径 {path} 不存在")
            
            if not path_obj.is_dir():
                raise ToolError(f"路径 {path} 不是目录")

            # 收集文件信息
            file_info_list = await self._collect_files(
                path_obj, recursive, show_hidden, filters
            )
            
            # 排序
            file_info_list = self._sort_files(file_info_list, sort_by, reverse_sort)
            
            # 限制显示数量
            truncated = len(file_info_list) > MAX_FILES_TO_SHOW
            if truncated:
                file_info_list = file_info_list[:MAX_FILES_TO_SHOW]
            
            # 生成输出
            output = self._format_output(
                path_obj, file_info_list, show_details, truncated, filters
            )
            
            return ToolResult(output=output)
            
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"查看目录时发生错误: {str(e)}")

    async def _collect_files(
        self,
        path_obj: Path,
        recursive: bool,
        show_hidden: bool,
        filters: Optional[List[str]]
    ) -> List[dict]:
        """收集文件信息。"""
        file_info_list = []
        
        try:
            if recursive:
                # 递归遍历
                for root, dirs, files in os.walk(path_obj):
                    root_path = Path(root)
                    
                    # 处理目录
                    for dir_name in dirs:
                        if not show_hidden and dir_name.startswith('.'):
                            continue
                        
                        dir_path = root_path / dir_name
                        file_info = self._get_file_info(dir_path, root_path)
                        if file_info:
                            file_info_list.append(file_info)
                    
                    # 处理文件
                    for file_name in files:
                        if not show_hidden and file_name.startswith('.'):
                            continue
                        
                        file_path = root_path / file_name
                        
                        # 应用过滤器
                        if filters and not any(file_name.lower().endswith(ext.lower()) for ext in filters):
                            continue
                        
                        file_info = self._get_file_info(file_path, root_path)
                        if file_info:
                            file_info_list.append(file_info)
            else:
                # 非递归遍历
                for item in path_obj.iterdir():
                    if not show_hidden and item.name.startswith('.'):
                        continue
                    
                    # 对文件应用过滤器
                    if item.is_file() and filters:
                        if not any(item.name.lower().endswith(ext.lower()) for ext in filters):
                            continue
                    
                    file_info = self._get_file_info(item, path_obj)
                    if file_info:
                        file_info_list.append(file_info)
                        
        except PermissionError as e:
            raise ToolError(f"权限不足，无法访问目录: {e}")
        
        return file_info_list

    def _get_file_info(self, file_path: Path, base_path: Path) -> Optional[dict]:
        """获取单个文件或目录的信息。"""
        try:
            stat_info = file_path.stat()
            
            # 计算相对路径
            try:
                relative_path = file_path.relative_to(base_path)
            except ValueError:
                relative_path = file_path
            
            file_info = {
                'name': file_path.name,
                'path': str(file_path),
                'relative_path': str(relative_path),
                'is_dir': file_path.is_dir(),
                'is_file': file_path.is_file(),
                'size': stat_info.st_size,
                'size_human': self._format_size(stat_info.st_size),
                'modified': datetime.fromtimestamp(stat_info.st_mtime),
                'modified_str': datetime.fromtimestamp(stat_info.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                'permissions': stat.filemode(stat_info.st_mode),
                'extension': file_path.suffix.lower() if file_path.suffix else '',
                'type': 'directory' if file_path.is_dir() else ('file' if file_path.is_file() else 'other')
            }
            
            return file_info
            
        except (OSError, PermissionError):
            # 如果无法访问文件信息，返回基本信息
            return {
                'name': file_path.name,
                'path': str(file_path),
                'relative_path': str(file_path.relative_to(base_path)) if base_path in file_path.parents else str(file_path),
                'is_dir': False,
                'is_file': False,
                'size': 0,
                'size_human': '未知',
                'modified': None,
                'modified_str': '未知',
                'permissions': '未知',
                'extension': '',
                'type': '无法访问'
            }

    def _sort_files(self, file_info_list: List[dict], sort_by: str, reverse_sort: bool) -> List[dict]:
        """对文件列表进行排序。"""
        sort_keys = {
            'name': lambda x: (not x['is_dir'], x['name'].lower()),  # 目录优先，然后按名称
            'size': lambda x: (not x['is_dir'], x['size']),  # 目录优先，然后按大小
            'modified': lambda x: (not x['is_dir'], x['modified'] or datetime.min),  # 目录优先，然后按修改时间
            'type': lambda x: (x['type'], x['name'].lower())  # 按类型，然后按名称
        }
        
        sort_key = sort_keys.get(sort_by, sort_keys['name'])
        return sorted(file_info_list, key=sort_key, reverse=reverse_sort)

    def _format_size(self, size: int) -> str:
        """格式化文件大小为人类可读格式。"""
        size_float = float(size)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_float < 1024.0:
                if unit == 'B':
                    return f"{int(size_float)}"
                else:
                    return f"{size_float:.1f}{unit}"
            size_float /= 1024.0
        return f"{size_float:.1f}PB"

    def _format_output(
        self,
        path_obj: Path,
        file_info_list: List[dict],
        show_details: bool,
        truncated: bool,
        filters: Optional[List[str]]
    ) -> str:
        """格式化输出结果。"""
        lines = []
        
        # 标题
        lines.append(f"📁 目录: {path_obj}")
        lines.append("=" * 80)
        
        # 统计信息
        total_files = sum(1 for item in file_info_list if item['is_file'])
        total_dirs = sum(1 for item in file_info_list if item['is_dir'])
        total_size = sum(item['size'] for item in file_info_list if item['is_file'])
        
        lines.append(f"📊 统计: {total_dirs} 个目录, {total_files} 个文件, 总大小: {self._format_size(total_size)}")
        
        if filters:
            lines.append(f"🔍 过滤器: {', '.join(filters)}")
        
        lines.append("")
        
        if not file_info_list:
            lines.append("目录为空或没有符合条件的文件。")
        else:
            # 文件列表
            if show_details:
                # 详细格式
                lines.append(f"{'类型':<4} {'权限':<10} {'大小':<10} {'修改时间':<19} {'名称'}")
                lines.append("-" * 80)
                
                for item in file_info_list:
                    type_icon = "📁" if item['is_dir'] else "📄"
                    type_str = f"{type_icon}"
                    
                    name_display = item['relative_path'] 
                    if item['is_dir']:
                        name_display += "/"
                    
                    lines.append(f"{type_str:<4} {item['permissions']:<10} {item['size_human']:<10} {item['modified_str']:<19} {name_display}")
            else:
                # 简单格式
                dirs = [item for item in file_info_list if item['is_dir']]
                files = [item for item in file_info_list if item['is_file']]
                
                if dirs:
                    lines.append("📁 目录:")
                    for item in dirs:
                        lines.append(f"  {item['relative_path']}/")
                    lines.append("")
                
                if files:
                    lines.append("📄 文件:")
                    for item in files:
                        lines.append(f"  {item['relative_path']} ({item['size_human']})")
        
        # 截断提示
        if truncated:
            lines.append("")
            lines.append(TRUNCATED_MESSAGE.format(MAX_FILES_TO_SHOW))
        
        return "\n".join(lines)
