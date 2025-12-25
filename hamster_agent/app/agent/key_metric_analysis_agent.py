from pydantic import Field
import os

from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.prompt.key_metric_analysis import KEY_METRIC_SYSTEM_PROMPT, KEY_METRIC_NEXT_STEP_PROMPT
from app.tool import Terminate, ToolCollection
from app.tool.chart_visualization.python_execute import NormalPythonExecute

from app.tool.excel_analysis import KeyMetricAnalysisTool
from app.tool.excel_analysis import VersionAnalysisTool
from app.tool.file_read_or_write import FileReadWriteTool
from app.tool.excel_analysis import ExcelColumnNamesReadTool
from app.tool.listfolder import ListFolderTool

class KeyMetricAnalysisAgent(ToolCallAgent):

    name: str = "Key_Metric_Analysis_Agent"
    description: str = (
        "一个游戏性能指标关键指标Key_metric分析代理，利用Python和数据可视化工具来解决多样化的数据分析任务"
    )

    system_prompt: str = KEY_METRIC_SYSTEM_PROMPT.format(key_metric_file_path = os.path.join(config.workspace_root, "key_metric.csv"))
    next_step_prompt: str = KEY_METRIC_NEXT_STEP_PROMPT

    max_observe: int = 20000
    max_steps: int = 25

    # Add general-purpose tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            KeyMetricAnalysisTool(),
            VersionAnalysisTool(),
            
            NormalPythonExecute(),
            FileReadWriteTool(),
            ExcelColumnNamesReadTool(),
            ListFolderTool(),
            
            # terminate
            Terminate(),
        )
    )
