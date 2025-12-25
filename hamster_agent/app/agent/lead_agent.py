from pydantic import Field
from typing import Dict, Optional

from app.agent.data_analysis import DataAnalysis
from app.config import config
from app.prompt.game_data_analysis import (
    COORDINATOR_NEXT_STEP_PROMPT,
    COORDINATOR_SYSTEM_PROMPT,
)
from app.tool import Terminate, ToolCollection
from app.tool.chart_visualization.python_execute import NormalPythonExecute
from app.tool.excel_analysis import *
from app.tool.merge_multi_md_into_one_tool import MergeMultiMdIntoOneTool
from app.tool.spawn_multi_data_analysis_tool import SpawnMultiDataAnalysisTool


class MultiDataAnalysisCoordinator(DataAnalysis):
    """
    一个协调代理，负责编排目录中多个CSV文件的数据分析任务。

    该代理的工作流程：
    1. 使用spawn_multi_data_analysis_tool启动多个代理并行分析CSV文件
    2. 使用merge_multi_md_into_one_tool将所有生成的markdown报告整合为单个文件

    该代理负责协调而非直接分析数据，管理整个批量分析过程。
    """

    name: str = "Multi_Data_Analysis_Coordinator"
    description: str = (
        "一个高级协调代理，负责编排和管理跨多个CSV文件的大规模批量数据分析操作。"
        "该代理充当主控制器，将任务委派给专业分析代理并整合结果。"
        "协调能力："
        "• 批量处理管理：自动发现并排队指定目录中的所有CSV/TSV文件进行并行分析"
        "• 多代理编排：生成和管理多个游戏数据分析代理来并发处理文件，最大化效率"
        "• 资源分配：通过可配置的并发限制智能分配工作负载到可用代理"
        "• 任务队列管理：维护和监控分析任务队列，确保所有文件处理完成"
        "• 错误处理和重试逻辑：实现健壮的重试机制和后验证以确保任务完成质量"
        "工作流程自动化："
        "• 阶段1 - 并行分析：使用SpawnMultiDataAnalysisTool启动多个专业代理同时处理CSV文件"
        "• 阶段2 - 结果整合：使用MergeMultiMdIntoOneTool递归收集并合并所有生成的markdown报告为统一文档"
        "• 元数据集成：自动定位并整合元数据文件（*.metadata.json，*column_analysis.json）以提高分析质量"
        "• 进度监控：跟踪所有生成代理的完成状态并提供全面的执行摘要"
        "专业特性："
        "• 双重验证系统：采用基于LLM和基于规则的验证来确保分析完成和质量"
        "• 内存管理：保存代理执行历史和中间结果以供调试和审计"
        "• 可扩展架构：通过高效的并行处理处理包含数十或数百个CSV文件的目录"
        "• 智能文件过滤：可以排除特定文件模式并处理复杂的目录结构"
        "• 统一报告：将分散的分析结果整合为具有适当格式的连贯、全面的报告"
        "该协调代理将手动的、顺序的数据分析转换为自动化的、并行的、可扩展的工作流程，"
        "能够处理整个数据集，同时保持分析质量并在所有文件中提供统一见解。"
    )

    system_prompt: str = COORDINATOR_SYSTEM_PROMPT
    next_step_prompt: str = COORDINATOR_NEXT_STEP_PROMPT

    max_observe: int = 15000
    max_steps: int = 25

    # Add coordination tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            SpawnMultiDataAnalysisTool(),
            # MergeMultiMdIntoOneTool(),
            NormalPythonExecute(),
            Terminate(),
        )
    )

    additional_request : Dict[str, Optional[str]] = Field(
        default={
            "fps" : None,
            "lag" : None,
            "memory" : None,
            "power" : None,
            "temperature" : None
    },
    description="附加请求，用于提供额外的分析上下文")

    async def update_system_prompt(self, appended_system_message: str):
        """
        更新系统提示，包含元数据文件路径。
        """
        self.system_prompt = self.system_prompt + appended_system_message

    async def update_additional_request(self, new_request: Dict[str, Optional[str]]):
        """
        更新附加请求字典。
        """
        self.additional_request = new_request
