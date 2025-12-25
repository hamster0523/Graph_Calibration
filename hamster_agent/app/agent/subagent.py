from pydantic import Field, model_validator

from app.agent.manus import Manus
from app.tool import ToolCollection, Terminate
from app.tool.ask_human import AskHuman
from app.tool.excel_analysis.excel_clean_tool import ExcelCleanTool
from app.tool.excel_analysis.excel_column_analysis_tool import ExcelColumnAnalysisTool
from OpenManus_hamster.app.tool.excel_analysis.excel_split_subtable_by_type_tool import ExcelSplitSubtableTool
from app.tool.excel_analysis.excel_json_metadata_read_tool import ExcelJsonMetadataReadTool
from app.tool.excel_analysis.excel_column_names_read_tool import ExcelColumnNamesReadTool
from app.tool.markdown_summary_writer_tool import MarkdownSummaryWriterTool
from app.agent.data_analysis import DataAnalysis

class ManboAgent(DataAnalysis):
    """
    A data analysis agent that uses planning to solve various data analysis tasks.

    This agent extends ToolCallAgent with a comprehensive set of tools and capabilities,
    including Data Analysis, Chart Visualization, Data Report.
    """

    name: str = "Data_Analysis"
    description: str = "A game related data analytical agent that utilizes python and data visualization tools to solve diverse data analysis tasks"

    system_prompt: str = ""
    next_step_prompt: str = ""

    max_observe: int = 20000
    max_steps: int = 25
    
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            
        )
    )
    
