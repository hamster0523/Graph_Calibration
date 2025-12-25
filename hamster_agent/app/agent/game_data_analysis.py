from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.prompt.game_data_analysis import GAME_NEXT_STEP_PROMPT, GAME_SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection
from app.tool.chart_visualization.python_execute import NormalPythonExecute
from app.tool.detect_and_handle_outliers_iqr_tool import DetectAndHandleOutliersIqrTool
from app.tool.detect_and_handle_outliers_zscore_tool import (
    DetectAndHandleOutliersZscoreTool,
)
from app.tool.excel_analysis import *
from app.tool.excel_analysis import ControlTestAnalysisTool
from app.tool.format_datatime_tool import FormatDatetimeTool
from app.tool.markdown_summary_writer_tool import MarkdownSummaryWriterTool
from app.tool.ml_tools import *


class GameDataAnalysisAgent(ToolCallAgent):
    """
    一个使用规划来解决各种数据分析任务的数据分析代理。

    该代理扩展了ToolCallAgent，具备全面的工具集和功能，
    包括数据分析、图表可视化、数据报告。
    """

    name: str = "Game_Data_Analysis"
    description: str = (
        "一个游戏相关的数据分析代理，利用Python和数据可视化工具来解决多样化的数据分析任务"
    )

    system_prompt: str = GAME_SYSTEM_PROMPT
    next_step_prompt: str = GAME_NEXT_STEP_PROMPT

    max_observe: int = 20000
    max_steps: int = 25

    # Add general-purpose tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            NormalPythonExecute(),
            # VisualizationPrepare(),
            # DataVisualization(),
            # excel analysis tools
            ControlTestAnalysisTool(),
            ExcelJsonMetadataReadTool(),
            ExcelColumnNamesReadTool(),
            # markdown related tools
            MarkdownSummaryWriterTool(),
            # 异常值
            # DetectAndHandleOutliersIqrTool(),
            # DetectAndHandleOutliersZscoreTool(),
            # date tools
            # FormatDatetimeTool(),
            # data analysis tools
            # # ml tools
            # LabelEncodeTool(),
            # OneHotEncodeTool(),
            # FrequencyEncodeTool(),
            # TargetEncodeTool(),
            # CorrelationFeatureSelectionTool(),
            # VarianceFeatureSelectionTool(),
            # ScaleFeaturesTool(),
            # PCAAnalysisTool(),
            # RFETool(),
            # PolynomialFeaturesTool(),
            # FeatureCombinationsTool(),
            # ModelTrainingAndPredictionTool(),
            # terminate
            Terminate(),
        )
    )
