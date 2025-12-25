import multiprocessing
import os
import subprocess
import sys
import tempfile
from io import StringIO
from typing import Dict

from app.config import config
from app.tool.base import BaseTool


class PythonExecute(BaseTool):
    """A tool for executing Python code with timeout and safety restrictions."""

    name: str = "python_execute"
    description: str = (
        "Executes Python code string. Note: Only print outputs are visible, function return values are not captured. Use print statements to see results."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute.",
            },
        },
        "required": ["code"],
    }

    python_path: str = config.run_flow_config.python_execute_path

    def _run_code(self, code: str, result_dict: dict, safe_globals: dict) -> None:
        original_stdout = sys.stdout
        try:
            output_buffer = StringIO()
            sys.stdout = output_buffer
            exec(code, safe_globals, safe_globals)
            result_dict["observation"] = output_buffer.getvalue()
            result_dict["success"] = True
        except Exception as e:
            result_dict["observation"] = str(e)
            result_dict["success"] = False
        finally:
            sys.stdout = original_stdout

    def _ensure_encoding_header(self, code: str) -> str:
        """
        确保代码有正确的编码声明，兼容shebang和已存在的声明
        """
        lines = code.splitlines()

        def has_encoding_decl(line: str) -> bool:
            """检测是否已有编码声明"""
            line_clean = line.strip().lower().replace(" ", "").replace("\t", "")
            return (
                line_clean.startswith("#-*-coding:utf-8-*-")
                or line_clean.startswith("#coding:utf-8")
                or line_clean.startswith("#coding=utf-8")
                or "coding:utf-8" in line_clean
                or "coding=utf-8" in line_clean
            )

        # 检查前两行是否已有编码声明
        if len(lines) >= 1 and has_encoding_decl(lines[0]):
            return code
        if len(lines) >= 2 and has_encoding_decl(lines[1]):
            return code

        # 选择编码声明
        encoding_decl = "# -*- coding: utf-8 -*-"

        # 处理shebang情况
        if lines and lines[0].strip().startswith("#!"):
            # shebang在第一行，编码声明插入第二行
            return "\n".join([lines[0], encoding_decl] + lines[1:])
        else:
            # 编码声明插入第一行
            return "\n".join([encoding_decl] + lines)

    def _run_code_2(self, code: str, result_dict: dict, safe_globals: dict) -> None:
        try:
            # 统一使用UTF-8编码，简化平台差异
            file_encoding = "utf-8"  # 不使用BOM，更通用
            io_encoding = "utf-8"

            # 确保代码有正确的编码声明，兼容shebang
            code = self._ensure_encoding_header(code)

            # 创建临时文件
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                encoding=file_encoding,
                newline="\n",  # 确保跨平台换行符一致性
            ) as f:
                f.write(code)
                temp_file = f.name

            # 使用指定的 Python 解释器
            interpreter = self.python_path if self.python_path else sys.executable

            # 根据平台设置环境变量
            env = os.environ.copy()
            if os.name == "nt":  # Windows平台
                env["PYTHONIOENCODING"] = io_encoding
                env["PYTHONLEGACYWINDOWSSTDIO"] = "0"
                # Windows下确保控制台使用UTF-8
                env["PYTHONUTF8"] = "1"  # Python 3.7+ 强制UTF-8模式
            else:  # Linux/Unix平台
                env["PYTHONIOENCODING"] = io_encoding
                # 确保Locale设置支持UTF-8
                if "LC_ALL" not in env:
                    env["LC_ALL"] = "C.UTF-8"
                if "LANG" not in env:
                    env["LANG"] = "C.UTF-8"

            result = subprocess.run(
                [interpreter, temp_file],
                capture_output=True,
                text=True,
                timeout=1000,
                encoding=io_encoding,
                errors="replace",  # 遇到编码错误时替换为占位符
                env=env,
            )

            # 处理结果
            if result.returncode == 0:
                result_dict["observation"] = result.stdout
                result_dict["success"] = True
            else:
                error_msg = result.stderr
                # 在Windows上可能需要额外处理编码问题
                if os.name == "nt" and "encoding" in error_msg.lower():
                    error_msg += "\n[提示] 如果遇到编码问题，请确保代码中的中文字符正确保存为UTF-8格式"
                result_dict["observation"] = f"Error: {error_msg}"
                result_dict["success"] = False

        except UnicodeEncodeError as e:
            result_dict["observation"] = (
                f"编码错误: {str(e)}。请检查代码中的特殊字符是否为有效的UTF-8编码。"
            )
            result_dict["success"] = False
        except UnicodeDecodeError as e:
            result_dict["observation"] = f"解码错误: {str(e)}。输出包含无法解码的字符。"
            result_dict["success"] = False
        except Exception as e:
            result_dict["observation"] = str(e)
            result_dict["success"] = False
        finally:
            if "temp_file" in locals():
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass  # 文件可能已被删除

    async def execute(
        self,
        code: str,
        timeout: int = 1000,
    ) -> Dict:
        """
        Executes the provided Python code with a timeout.

        Args:
            code (str): The Python code to execute.
            timeout (int): Execution timeout in seconds.

        Returns:
            Dict: Contains 'output' with execution output or error message and 'success' status.
        """

        with multiprocessing.Manager() as manager:
            result = manager.dict({"observation": "", "success": False})
            if isinstance(__builtins__, dict):
                safe_globals = {"__builtins__": __builtins__}
            else:
                safe_globals = {"__builtins__": __builtins__.__dict__.copy()}
            proc = multiprocessing.Process(
                target=self._run_code_2, args=(code, result, safe_globals)
            )
            proc.start()
            proc.join(timeout)

            # timeout process
            if proc.is_alive():
                proc.terminate()
                proc.join(1)
                return {
                    "observation": f"Execution timeout after {timeout} seconds",
                    "success": False,
                }
            return dict(result)
