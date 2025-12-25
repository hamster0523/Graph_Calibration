from pydantic import Field

from app.agent.manus import Manus
from app.tool import Terminate, ToolCollection
from app.tool.ask_human import AskHuman
from app.tool.excel_analysis.excel_clean_tool import ExcelCleanTool
from app.tool.excel_analysis.excel_split_multiple_substable_tool import ExcelSplitMultipleSubtableTool
from app.tool.markdown_summary_writer_tool import MarkdownSummaryWriterTool
from app.tool.excel_analysis.excel_split_to_key_metric_tool import ExcelSplitToKeyMetricTool

# 定义Excel分析代理的系统提示
EXCEL_CLEAN_SYSTEM_PROMPT = """
您是Excel分析代理，专门分析包含游戏性能数据的Excel/CSV文件。

分析Excel/CSV文件的工作流程包括以下步骤：
1. **清理**: 使用 `excel_clean_tool` 移除空列和常数值列以及Control组和Test组其中为零的数据
2. **拆分**: 使用 `excel_split_multiple_subtable_tool` 将数据按类别拆分为子表
3. **关键指标提取**: 使用 `excel_split_to_key_metric_tool` 提取关键性能指标
4. **以上步骤完成之后生成一个处理报告之后Terminate即可**
"""

# 定义引导工作流程的下一步提示
EXCEL_CLEAN_NEXT_PROMPT = """
基于Excel/CSV分析的当前状态，确定下一个合适的步骤：

如果文件还未清理：
- 使用 `excel_clean_tool` 通过移除空列/常数列来清理文件

如果文件已清理：
- 使用 `excel_split_multiple_subtable_tool` 将文件拆分为类别特定的子表

- 使用 `excel_split_to_key_metric_tool` 提取关键性能指标

如果都已经完成了
- 生成处理报告
- 使用 `Terminate` 结束流程
"""


class ExcelCleanAgent(Manus):
    """专门用于游戏性能数据Excel/CSV文件处理和分析的代理。"""

    name: str = "ExcelCleanAgent"
    description: str = (
        "一个功能全面的Excel/CSV数据处理代理，配备先进工具用于完整的数据管道管理。"
        "该代理可以处理从原始数据清理到结构化分析和报告的整个工作流程。"
        "核心功能："
        "• 数据清理：使用ExcelCleanTool自动移除空列、常量值列以及Control组和Test组其中为零的数据 "
        "• 智能数据拆分：使用ExcelSplitMultipleSubtableTool将大型数据集拆分为基于类别和设备标识的子表，同时保留公共标识符列 "
        "• 关键指标提取：使用ExcelSplitToKeyMetricTool从数据集中提取关键性能指标，便于后续分析 "
        "• 报告生成：使用MarkdownSummaryWriterTool创建全面的markdown摘要和建议 "
        "游戏性能数据专业化："
        "• 识别和保留关键列（设备ID(devicemodel)、测试参数(param_value)、 设备质量(realpicturequality)） "
        "• 为下游分析工作流程提供可操作的见解 "
        "该代理可以完全处理原始Excel/CSV文件，将其转换为干净、有组织且可用于分析的数据集，并提供详细文档。"
    )

    system_prompt: str = EXCEL_CLEAN_SYSTEM_PROMPT
    next_step_prompt: str = EXCEL_CLEAN_NEXT_PROMPT

    max_observe: int = 100000
    max_steps: int = 20

    # Set up the Excel analysis tools
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            ExcelCleanTool(),
            # ExcelColumnAnalysisTool(),
            ExcelSplitMultipleSubtableTool(),
            ExcelSplitToKeyMetricTool(),
            #ExcelJsonMetadataReadTool(),
            #ExcelColumnNamesReadTool(),
            MarkdownSummaryWriterTool(),
            AskHuman(),
            Terminate(),
        )
    )

    @classmethod
    async def create(cls, **kwargs) -> "ExcelCleanAgent":
        """Factory method to create and properly initialize a Manus instance."""
        instance = cls(**kwargs)
        await instance.initialize_mcp_servers()
        instance._initialized = True
        return instance

    async def think(self) -> bool:
        """Override think method to add Excel-specific logic if needed."""
        # Add any Excel-specific thinking logic here
        return await super().think()
