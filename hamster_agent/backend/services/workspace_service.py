"""
工作空间服务
处理文件上传、下载、删除等操作
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ..models.schemas import FileUploadResponse, WorkspaceFile, WorkspaceResponse


class WorkspaceService:
    """工作空间服务类"""

    def __init__(self, workspace_path: Optional[str] = None):
        # 设置工作空间路径
        if workspace_path:
            self.workspace_root = Path(workspace_path)
        else:
            # 默认路径：项目根目录下的workspace
            self.workspace_root = Path(__file__).parent.parent.parent / "workspace"

        # 确保工作空间目录存在
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        print(f"📁 Workspace initialized: {self.workspace_root}")

    async def list_files(self) -> WorkspaceResponse:
        """列出工作空间中的所有文件"""
        try:
            files: List[WorkspaceFile] = []

            # 递归遍历工作空间
            for item in self.workspace_root.rglob("*"):
                if item.is_file():
                    relative_path = item.relative_to(self.workspace_root)
                    file_info = WorkspaceFile(
                        name=item.name,
                        path=str(relative_path),
                        size=item.stat().st_size,
                        modified=datetime.fromtimestamp(
                            item.stat().st_mtime
                        ).isoformat(),
                    )
                    files.append(file_info)

            # 按修改时间排序（最新的在前）
            files.sort(key=lambda x: x.modified, reverse=True)

            return WorkspaceResponse(
                files=files, workspace_path=str(self.workspace_root)
            )

        except Exception as e:
            raise Exception(f"Error listing workspace files: {str(e)}")

    async def upload_file(self, filename: str, file_content) -> FileUploadResponse:
        """上传文件到工作空间"""
        try:
            print(f"📝 Processing upload for: {filename}")

            # 安全检查文件名
            safe_filename = self._sanitize_filename(filename)
            file_path = self.workspace_root / safe_filename

            # 防止路径穿越攻击
            if not file_path.resolve().is_relative_to(self.workspace_root.resolve()):
                raise Exception("Invalid file path")

            # 确保目标目录存在
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # 如果文件已存在，生成唯一名称
            if file_path.exists():
                original_path = file_path
                file_path = self._generate_unique_filename(file_path)
                print(f"📝 File exists, using unique name: {file_path.name}")

            # 保存文件
            print(f"💾 Saving file to: {file_path}")

            with open(file_path, "wb") as buffer:
                if hasattr(file_content, "read"):
                    # 文件对象
                    shutil.copyfileobj(
                        file_content, buffer, length=1024 * 1024
                    )  # 1MB缓冲区
                elif isinstance(file_content, bytes):
                    # 字节数据
                    buffer.write(file_content)
                else:
                    # 其他类型，尝试转换为字节
                    if hasattr(file_content, "encode"):
                        buffer.write(file_content.encode("utf-8"))
                    else:
                        raise Exception(
                            f"Unsupported file content type: {type(file_content)}"
                        )

            # 验证文件是否成功保存
            if not file_path.exists():
                raise Exception("File was not saved successfully")

            # 获取文件信息
            file_size = file_path.stat().st_size
            relative_path = file_path.relative_to(self.workspace_root)

            print(f"✅ File saved successfully: {file_path.name} ({file_size} bytes)")

            return FileUploadResponse(
                message="File uploaded successfully",
                filename=file_path.name,
                size=file_size,
                path=str(relative_path),
            )

        except Exception as e:
            print(f"❌ Upload error: {str(e)}")
            # 更具体的错误信息
            raise Exception(f"Error uploading file: {str(e)}")

    async def download_file(self, file_path: str) -> Path:
        """获取下载文件的路径"""
        try:
            # 安全检查路径
            safe_path = self._validate_file_path(file_path)
            full_path = self.workspace_root / safe_path

            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            if not full_path.is_file():
                raise ValueError(f"Path is not a file: {file_path}")

            return full_path

        except Exception as e:
            raise Exception(f"Error accessing file: {str(e)}")

    async def delete_file(self, file_path: str) -> str:
        """删除文件或目录"""
        try:
            # 安全检查路径
            safe_path = self._validate_file_path(file_path)
            full_path = self.workspace_root / safe_path

            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            if full_path.is_file():
                full_path.unlink()
                return f"File deleted: {file_path}"
            elif full_path.is_dir():
                shutil.rmtree(full_path)
                return f"Directory deleted: {file_path}"
            else:
                raise ValueError(f"Unknown file type: {file_path}")

        except Exception as e:
            raise Exception(f"Error deleting file: {str(e)}")

    def _sanitize_filename(self, filename: str) -> str:
        """清理文件名，移除危险字符"""
        # 移除路径分隔符和其他危险字符
        dangerous_chars = ["/", "\\", "..", ":", "*", "?", "<", ">", "|"]
        safe_filename = filename

        for char in dangerous_chars:
            safe_filename = safe_filename.replace(char, "_")

        # 限制文件名长度
        if len(safe_filename) > 255:
            name, ext = os.path.splitext(safe_filename)
            safe_filename = name[: 255 - len(ext)] + ext

        return safe_filename

    def _validate_file_path(self, file_path: str) -> Path:
        """验证文件路径安全性"""
        # 移除路径遍历攻击
        safe_path = Path(file_path)

        # 检查是否包含危险的路径组件
        if ".." in safe_path.parts:
            raise ValueError("Path traversal not allowed")

        # 确保路径是相对路径
        if safe_path.is_absolute():
            raise ValueError("Absolute paths not allowed")

        return safe_path

    def _generate_unique_filename(self, file_path: Path) -> Path:
        """生成唯一的文件名"""
        base_name = file_path.stem
        extension = file_path.suffix
        directory = file_path.parent

        counter = 1
        while True:
            new_name = f"{base_name}_{counter}{extension}"
            new_path = directory / new_name
            if not new_path.exists():
                return new_path
            counter += 1

    def get_workspace_stats(self) -> dict:
        """获取工作空间统计信息"""
        try:
            total_files = 0
            total_size = 0

            for item in self.workspace_root.rglob("*"):
                if item.is_file():
                    total_files += 1
                    total_size += item.stat().st_size

            return {
                "total_files": total_files,
                "total_size": total_size,
                "workspace_path": str(self.workspace_root),
                "free_space": shutil.disk_usage(self.workspace_root).free,
            }
        except Exception as e:
            return {"error": str(e)}


# 全局工作空间服务实例
workspace_service = WorkspaceService()
