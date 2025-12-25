from typing import final, Optional
from pydantic import Field
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path

from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.tool import Terminate, ToolCollection
from app.tool.chart_visualization.python_execute import NormalPythonExecute
from app.tool.ask_human import AskHuman
from app.tool.image_understand_tool import ImageUnderstandTool
from app.tool.file_read_or_write import FileReadWriteTool
from app.tool.listfolder import ListFolderTool

class QuestionType(Enum):
    DATA_SUMMARY = "数据摘要"
    VISUALIZATION = "可视化"
    COMPARISON = "对比分析" 
    TREND_ANALYSIS = "趋势分析"
    FILE_OPERATION = "文件操作"
    CALCULATION = "计算分析"
    GENERAL = "通用问题"

@dataclass
class ConversationContext:
    """对话上下文管理"""
    current_files: Optional[List[str]] = None  # 当前分析的文件
    last_analysis: Optional[Dict[str, Any]] = None  # 上次分析结果
    user_preferences: Optional[Dict[str, str]] = None  # 用户偏好
    question_history: Optional[List[str]] = None  # 问题历史
    
    def __post_init__(self):
        if self.current_files is None:
            self.current_files = []
        if self.last_analysis is None:
            self.last_analysis = {}
        if self.user_preferences is None:
            self.user_preferences = {}
        if self.question_history is None:
            self.question_history = []

class AnalysisResultQnAAgent(ToolCallAgent):
    """
    基于已有分析结果文件（如csv/json/txt等）对用户问题进行智能问答的Agent。
    用户输入exit时自动触发Terminate，其余输入自动调用分析工具回答。
    """

    name: str = "AnalysisResultQnAAgent"
    description: str = (
        "一个基于分析结果文件的智能问答代理，能够理解用户问题并结合已有分析结果自动作答。"
    )

    system_prompt: str = (
        "你是一个专业的数据分析结果问答专家，具备以下能力：\n"
        "1. 📊 数据理解：能够读取和理解各种格式的分析结果文件（CSV、JSON、图片等）\n"
        "2. 🔍 智能分析：根据用户问题自动选择最适合的分析方法和工具\n"
        "3. 📈 可视化：能够创建图表和可视化来回答用户问题\n"
        "4. 💡 洞察发现：从数据中提取关键洞察和业务建议\n"
        "5. 🎯 上下文记忆：记住对话历史，提供连贯的分析体验\n\n"
        
        "## 工作流程：\n"
        "1. 理解用户问题的核心意图\n"
        "2. 自动发现或询问相关的数据文件\n"
        "3. 选择合适的工具进行分析\n"
        "4. 提供清晰、专业的答案和可视化\n"
        "5. 主动询问是否需要进一步分析\n\n"
        
        "## 特殊指令：\n"
        "- 用户输入'exit'时立即终止会话\n"
        "- 用户输入'help'时显示帮助信息\n"
        "- 用户输入'files'时列出可用的分析文件\n"
        "- 始终保持专业、友好的对话风格\n"
        "- 主动提供数据洞察和业务建议"
    )
    next_step_prompt: str = "请继续提问，或输入exit结束。"

    max_observe: int = 20000
    max_steps: int = 100

    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            # 向用户提问
            AskHuman(),
            
            # 图片理解
            ImageUnderstandTool(),

            # python 工具
            NormalPythonExecute(),
            
            # 文件打开工具
            FileReadWriteTool(),
            
            # 查看一个目录
            ListFolderTool(),
            
            # 终止
            Terminate(),
        )
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.context = ConversationContext()

    async def run(self, user_input: Optional[str] = None, **kwargs):
        """主入口：持续对话模式或单次处理模式"""
        
        # 如果提供了具体的用户输入，则是单次处理模式
        if user_input is not None:
            return await self._process_user_input(user_input)
        
        # 否则进入持续对话模式
        return await self._start_conversation_loop()
    
    async def _start_conversation_loop(self):
        """启动持续对话循环"""
        print("🤖 分析结果问答助手已启动！")
        print("输入 'help' 获取帮助，输入 'exit' 结束对话\n")
        
        while True:
            try:
                # 从用户获取输入
                user_input = input("您: ").strip()
                
                if not user_input:
                    continue
                
                # 处理用户输入
                result = await self._process_user_input(user_input)
                
                # 显示结果
                if result:
                    print(f"\n🤖 助手: {result}\n")
                
                # 检查是否需要退出
                if user_input.lower() == "exit":
                    break
                    
            except KeyboardInterrupt:
                print("\n\n👋 检测到中断信号，正在退出...")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {str(e)}")
                print("请重试或输入 'help' 获取帮助\n")
        
        # 调用terminate工具
        terminate_tool = self._get_tool_by_type(Terminate)
        if terminate_tool:
            await terminate_tool.execute(status="success")
        
        print("👋 再见！感谢使用分析结果问答助手")
        return "对话已结束"

    async def _process_user_input(self, user_input: str) -> str:
        """处理单个用户输入"""
        try:
            return await self._safe_run(user_input)
        except Exception as e:
            error_msg = f"抱歉，处理您的问题时遇到了错误：{str(e)}\n\n"
            error_msg += "您可以：\n"
            error_msg += "- 重新描述您的问题\n"
            error_msg += "- 输入'help'获取帮助\n"
            error_msg += "- 输入'files'查看可用文件"
            return error_msg

    def _get_tool_by_type(self, tool_type):
        """获取指定类型的工具"""
        for tool in getattr(self.available_tools, 'tools', []):
            if isinstance(tool, tool_type):
                return tool
        return None

    async def _safe_run(self, user_input: str, **kwargs):
        """安全执行，包含错误处理的主逻辑"""
        
        user_input = user_input.strip()
        
        # 特殊命令处理（除了exit，因为在主循环中处理）
        if user_input.lower() == "help":
            return await self._show_help()
        
        elif user_input.lower() == "files":
            return await self._list_available_files()
        
        elif user_input.lower().startswith("context"):
            return await self._show_context()
        
        elif user_input.lower() == "exit":
            return "👋 正在退出..."
        
        # 更新上下文
        if self.context.question_history is None:
            self.context.question_history = []
        self.context.question_history.append(user_input)
        
        # 智能问题分类
        question_type = self._classify_question(user_input)
        
        # 推荐相关文件
        suggested_files = await self._suggest_relevant_files(user_input)
        
        # 构建增强的系统消息
        enhanced_input = self._enhance_user_input(user_input, question_type, suggested_files)
        
        try:
            result = await super().run(enhanced_input)
            self._update_context(user_input, result)
            return result
        finally:
            await self.cleanup()

    def _classify_question(self, user_input: str) -> QuestionType:
        """智能分类用户问题类型"""
        input_lower = user_input.lower()
        
        if any(word in input_lower for word in ['图表', '画图', '可视化', '绘制', 'plot', 'chart']):
            return QuestionType.VISUALIZATION
        elif any(word in input_lower for word in ['对比', '比较', '差异', 'compare', 'difference']):
            return QuestionType.COMPARISON
        elif any(word in input_lower for word in ['趋势', '变化', '增长', 'trend', 'change']):
            return QuestionType.TREND_ANALYSIS
        elif any(word in input_lower for word in ['计算', '统计', '平均', '求和', 'calculate', 'sum', 'mean']):
            return QuestionType.CALCULATION
        elif any(word in input_lower for word in ['摘要', '总结', '概述', 'summary', 'overview']):
            return QuestionType.DATA_SUMMARY
        elif any(word in input_lower for word in ['文件', '读取', '保存', 'file', 'read', 'save']):
            return QuestionType.FILE_OPERATION
        else:
            return QuestionType.GENERAL

    def _update_context(self, user_input: str, result: Any):
        """更新对话上下文"""
        # 确保 question_history 已初始化
        if self.context.question_history is None:
            self.context.question_history = []
        
        # 注意：question_history 已经在 run 方法中添加过了，这里不重复添加
        if len(self.context.question_history) > 10:  # 保持最近10个问题
            self.context.question_history.pop(0)
        
        # 确保 current_files 已初始化
        if self.context.current_files is None:
            self.context.current_files = []
        
        # 从结果中提取文件信息
        if hasattr(result, 'output') and result.output:
            output_str = str(result.output).lower()
            if 'csv' in output_str:
                # 提取文件路径等信息
                import re
                csv_matches = re.findall(r'[^\s]+\.csv', output_str)
                if csv_matches:
                    for match in csv_matches:
                        if match not in self.context.current_files:
                            self.context.current_files.append(match)
            
            # 保存最后的分析结果
            self.context.last_analysis = {
                'question': user_input,
                'result': str(result.output)[:500] if result.output else '',
                'timestamp': str(Path().cwd())  # 简单的时间戳替代
            }

    async def _discover_analysis_files(self, workspace_path: Optional[str] = None) -> Dict[str, List[Path]]:
        """自动发现工作区中的分析文件"""
        if workspace_path is None:
            workspace_path = str(config.workspace_root)
        
        file_types = {
            'csv_files': [],
            'json_files': [],
            'image_files': [],
            'report_files': []
        }
        
        workspace = Path(workspace_path)
        if workspace.exists():
            file_types['csv_files'] = list(workspace.glob('**/*.csv'))
            file_types['json_files'] = list(workspace.glob('**/*.json'))
            
            # 分别搜索不同的图片格式
            for ext in ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']:
                file_types['image_files'].extend(workspace.glob(f'**/*.{ext}'))
            
            # 分别搜索不同的报告格式
            for ext in ['txt', 'md', 'TXT', 'MD']:
                file_types['report_files'].extend(workspace.glob(f'**/*.{ext}'))
        
        return file_types

    async def _suggest_relevant_files(self, user_input: str) -> List[Path]:
        """根据用户问题推荐相关文件"""
        files = await self._discover_analysis_files()
        suggestions = []
        
        input_lower = user_input.lower()
        
        # 关键词匹配
        if 'unity' in input_lower:
            suggestions.extend([f for f in files['csv_files'] if 'unity' in str(f).lower()])
        if '版本' in input_lower or 'version' in input_lower:
            suggestions.extend([f for f in files['csv_files'] if 'version' in str(f).lower()])
        if '设备' in input_lower or 'device' in input_lower:
            suggestions.extend([f for f in files['csv_files'] if 'device' in str(f).lower()])
        
        return suggestions[:5]  # 返回前5个建议

    def _enhance_user_input(self, user_input: str, question_type: QuestionType, suggested_files: List[Path]) -> str:
        """增强用户输入，添加上下文信息"""
        enhanced = f"用户问题：{user_input}\n"
        enhanced += f"问题类型：{question_type.value}\n"
        
        if suggested_files:
            enhanced += f"推荐文件：{', '.join(str(f) for f in suggested_files[:3])}\n"
            
        if self.context.question_history:
            enhanced += f"对话历史：{' -> '.join(self.context.question_history[-3:])}\n"
        
        enhanced += "\n请基于以上信息智能回答用户问题。"
        return enhanced

    async def _show_help(self):
        """显示帮助信息"""
        help_text = """
🤖 **分析结果问答助手使用指南**

## 基本功能：
- 📊 数据摘要：询问"数据概况"、"总结分析结果"
- 📈 可视化：要求"画图"、"制作图表"、"可视化数据"
- 🔍 对比分析：询问"对比"、"差异分析"
- 📉 趋势分析：询问"趋势"、"变化情况"
- 🧮 计算分析：要求"计算"、"统计"特定指标

## 特殊命令：
- `help` - 显示此帮助信息
- `files` - 列出可用的分析文件
- `context` - 显示当前对话上下文
- `exit` - 结束对话

## 使用建议：
- 描述清楚你想了解的内容
- 可以指定具体的数据文件
- 询问时可以包含具体的指标名称
- 需要图表时明确说明图表类型

有任何问题都可以直接提问！
        """
        return help_text

    async def _list_available_files(self):
        """列出可用的分析文件"""
        files = await self._discover_analysis_files()
        
        file_list = "📁 **可用的分析文件：**\n\n"
        
        if files['csv_files']:
            file_list += "📊 **CSV数据文件：**\n"
            for f in files['csv_files'][:5]:
                file_list += f"  - {f.name}\n"
        
        if files['json_files']:
            file_list += "\n📋 **JSON结果文件：**\n"
            for f in files['json_files'][:5]:
                file_list += f"  - {f.name}\n"
        
        if files['image_files']:
            file_list += "\n🖼️ **图片文件：**\n"
            for f in files['image_files'][:5]:
                file_list += f"  - {f.name}\n"
        
        return file_list

    async def _show_context(self):
        """显示当前对话上下文"""
        context_info = "🧠 **当前对话上下文：**\n\n"
        
        if self.context.question_history:
            context_info += f"📝 最近问题：{' -> '.join(self.context.question_history[-3:])}\n"
        
        if self.context.current_files:
            context_info += f"📂 当前文件：{', '.join(self.context.current_files)}\n"
        
        return context_info

    async def cleanup(self):
        """清理资源"""
        # 这里可以添加任何需要清理的资源
        # 比如关闭文件句柄、数据库连接等
        await self.cleanup()

