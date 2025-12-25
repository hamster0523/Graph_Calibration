import base64
import os
from typing import List, Union
from pathlib import Path

from app.tool import BaseTool
from app.tool.base import ToolResult
from app.llm import LLM
from app.exceptions import ToolError
from app.schema import Message
from app.config import config

class ImageUnderstandTool(BaseTool):
    """A tool to read an image, convert it to base64, and use LLM to understand the image content."""

    name: str = "image_understand_tool"
    description: str = "Use this tool to read an image file, convert it to base64, and request LLM to understand and describe the image content."
    parameters: dict = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "The absolute path to the image file to analyze.",
            },
            "question": {
                "type": "string", 
                "description": "Optional specific question about the image. If not provided, will ask for general description.",
                "default": "请详细描述这张图片的内容，包括主要元素、颜色、布局等信息。"
            },
            "max_tokens": {
                "type": "integer",
                "description": "Maximum number of tokens for the LLM response.",
                "default": 1000
            }
        },
        "required": ["image_path"],
    }

    async def execute(
        self, 
        image_path: str, 
        question: str = "请详细描述这张图片的内容，包括主要元素、颜色、布局等信息。",
        max_tokens: int = 1000
    ) -> ToolResult:
        """
        Execute the image understanding task.
        
        Args:
            image_path: Path to the image file
            question: Question to ask about the image
            max_tokens: Maximum tokens for response
            
        Returns:
            ToolResult containing the LLM's understanding of the image
        """
        try:
            # Validate image path
            if not os.path.exists(image_path):
                return ToolResult(error=f"Image file not found: {image_path}")
            
            # Check if file is an image
            supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
            file_extension = Path(image_path).suffix.lower()
            if file_extension not in supported_formats:
                return ToolResult(error=f"Unsupported image format: {file_extension}. Supported formats: {', '.join(supported_formats)}")
            
            # Read and encode image to base64
            base64_image = await self._encode_image_to_base64(image_path)
            
            # Prepare prompt for LLM
            prompt = self._prepare_image_analysis_prompt(question, image_path)
            
            # Request LLM to analyze the image
            response = await self._request_llm_analysis(prompt, base64_image, max_tokens)
            
            # Format result
            result_content = f"📸 **图片分析结果**\n\n"
            result_content += f"📁 **图片路径**: {image_path}\n"
            result_content += f"❓ **分析问题**: {question}\n\n"
            result_content += f"🤖 **AI分析结果**:\n{response}\n"
            
            return ToolResult(output=result_content)
            
        except Exception as e:
            return ToolResult(error=f"Error analyzing image: {str(e)}")

    async def _encode_image_to_base64(self, image_path: str) -> str:
        """
        Read image file and encode it to base64 string.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded string of the image
        """
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                base64_encoded = base64.b64encode(image_data).decode('utf-8')
                
            # Get MIME type based on file extension
            file_extension = Path(image_path).suffix.lower()
            mime_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg', 
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.bmp': 'image/bmp',
                '.webp': 'image/webp'
            }
            mime_type = mime_type_map.get(file_extension, 'image/jpeg')
            
            return f"data:{mime_type};base64,{base64_encoded}"
            
        except Exception as e:
            raise ToolError(f"Failed to encode image to base64: {str(e)}")

    def _prepare_image_analysis_prompt(self, question: str, image_path: str) -> str:
        """
        Prepare the prompt for LLM image analysis.
        
        Args:
            question: The question to ask about the image
            image_path: Path to the image file
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""你是一个专业的图像分析专家。请仔细观察提供的图片并回答以下问题：

问题: {question}

图片文件: {os.path.basename(image_path)}

请提供详细、准确的分析，包括：
1. 图片的主要内容和元素
2. 颜色搭配和视觉效果
3. 布局和构图特点
4. 任何文字或标识信息
5. 整体风格和特征

请用中文回答，语言要专业且易懂。"""

        return prompt

    async def _request_llm_analysis(self, prompt: str, base64_image: str, max_tokens: int) -> str:
        """
        Request LLM to analyze the image.
        
        Args:
            prompt: The analysis prompt
            base64_image: Base64 encoded image
            max_tokens: Maximum tokens for response
            
        Returns:
            LLM response string
        """
        try:
            # Prepare messages for LLM using Message class
            messages: List[Union[dict, Message]] = [
                Message(role="user", content=prompt)
            ]
            
            # Prepare images list
            images: List[Union[str, dict]] = [base64_image]
            
            llm = LLM()

            # Call LLM with vision capability
            response = await llm.ask_with_images(
                messages=messages,
                images=images,
                temperature=0.7
            )
            
            return response.strip() if response else "无法获取图片分析结果"
                
        except Exception as e:
            raise ToolError(f"Failed to get LLM analysis: {str(e)}")

    def get_file_size_info(self, image_path: str) -> str:
        """
        Get human-readable file size information.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Formatted file size string
        """
        try:
            size_bytes = os.path.getsize(image_path)
            
            # Convert to human readable format
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size_bytes < 1024.0:
                    return f"{size_bytes:.1f} {unit}"
                size_bytes /= 1024.0
            return f"{size_bytes:.1f} TB"
            
        except Exception:
            return "Unknown size"
