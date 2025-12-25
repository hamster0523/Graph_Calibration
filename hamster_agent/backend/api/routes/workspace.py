"""
工作空间相关的API路由
"""

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

try:
    from ...models.schemas import FileUploadResponse, WorkspaceResponse
    from ...services.workspace_service import workspace_service
except ImportError:
    try:
        from models.schemas import FileUploadResponse, WorkspaceResponse
        from services.workspace_service import workspace_service
    except ImportError:
        from backend.models.schemas import FileUploadResponse, WorkspaceResponse
        from backend.services.workspace_service import workspace_service

router = APIRouter()


@router.get("/workspace/files", response_model=WorkspaceResponse)
async def list_workspace_files():
    """
    列出工作空间文件

    获取工作空间中所有文件的列表和信息
    """
    try:
        return await workspace_service.list_files()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workspace/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """上传文件到工作空间，支持各种文件类型的上传"""
    try:
        # 详细的调试信息
        print(f"📤 Upload request received:")
        print(f"  - Filename: {file.filename}")
        print(f"  - Content-Type: {file.content_type}")
        print(f"  - File size: {file.size if hasattr(file, 'size') else 'Unknown'}")

        if not file.filename:
            raise HTTPException(status_code=400, detail="No file selected")

        # 检查文件大小（限制为50MB）
        content = await file.read()
        file_size = len(content)
        print(f"  - Actual content size: {file_size} bytes")

        if file_size > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (max 50MB)")

        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file not allowed")

        # 上传文件
        result = await workspace_service.upload_file(file.filename, content)
        print(f"✅ Upload successful: {result.filename}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        # 记录详细错误日志
        import traceback

        error_details = traceback.format_exc()
        print(f"❌ Upload failed: {str(e)}")
        print(f"Full error: {error_details}")

        # 根据错误类型返回更合适的状态码
        if "file type" in str(e).lower() or "invalid content" in str(e).lower():
            raise HTTPException(status_code=422, detail=f"无法处理文件: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"服务器处理文件时发生错误: {str(e)}"
        )


@router.get("/workspace/download/{file_path:path}")
async def download_file(file_path: str):
    """
    下载工作空间中的文件

    根据文件路径下载指定文件
    """
    try:
        full_path = await workspace_service.download_file(file_path)

        return FileResponse(
            path=str(full_path),
            filename=full_path.name,
            media_type="application/octet-stream",
        )

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/workspace/delete/{file_path:path}")
async def delete_file(file_path: str):
    """
    删除工作空间中的文件

    根据文件路径删除指定文件或目录
    """
    try:
        message = await workspace_service.delete_file(file_path)
        return {"message": message}

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workspace/stats")
async def get_workspace_stats():
    """
    获取工作空间统计信息

    返回文件数量、总大小等统计信息
    """
    try:
        stats = workspace_service.get_workspace_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@router.post("/workspace/create-folder")
async def create_folder(folder_name: str):
    """
    创建文件夹

    在工作空间中创建新的文件夹
    """
    try:
        if not folder_name.strip():
            raise HTTPException(status_code=400, detail="Folder name cannot be empty")

        # 创建文件夹
        folder_path = workspace_service.workspace_root / folder_name.strip()
        folder_path.mkdir(parents=True, exist_ok=True)

        return {"message": f"Folder '{folder_name}' created successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating folder: {str(e)}")


@router.get("/workspace/search")
async def search_files(query: str):
    """
    搜索工作空间文件

    根据文件名或内容搜索文件
    """
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Search query cannot be empty")

        files = []
        query_lower = query.lower()

        # 搜索文件名
        for item in workspace_service.workspace_root.rglob("*"):
            if item.is_file() and query_lower in item.name.lower():
                relative_path = item.relative_to(workspace_service.workspace_root)
                files.append(
                    {
                        "name": item.name,
                        "path": str(relative_path),
                        "size": item.stat().st_size,
                        "type": "filename_match",
                    }
                )

        return {"files": files, "query": query}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
