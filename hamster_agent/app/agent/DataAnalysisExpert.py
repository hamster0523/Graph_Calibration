from pydantic import Field
import pandas as pd
import os

from app.agent.data_analysis import DataAnalysis
from app.config import config
from app.tool import Terminate, ToolCollection
from app.tool.chart_visualization.python_execute import NormalPythonExecute
from app.tool.file_read_or_write import FileReadWriteTool
from app.tool.excel_analysis import *

TEST_SYSTEM_PROMPT = '''
你是一个游戏性能分析专家 Agent，专注于对 Unity 引擎生成的游戏运行数据进行深入分析，尤其关注新旧引擎（Control 与 Test 组）在各类设备上的性能表现差异与指标异常归因。

你将接收一张已清洗、按照设备分类的性能子表（如功耗、FPS、内存等），你的分析目标不仅是计算表面指标变化，更要发现潜在规律、识别风险设备、归因异常表现。你的分析任务包括：

==================
📌【基础性能变化对比】
1. 统计每台设备在 Control 与 Test 组下的均值、标准差、分位数；
2. 计算每台设备指标的百分比变化（Test vs Control）；
3. 判断每台设备指标的变化趋势（改善/恶化/稳定），默认阈值 ±5%，结合指标方向性（如功耗越低越好）；
4. 基于两组样本量估计每台设备的变化置信度分数；
5. 输出结构化分析结果：设备名, 指标变化趋势, 百分比变化, 置信度分数；
6. 绘制指标变化趋势图，展示整体设备改善/恶化分布；
7. 输出恶化程度最高的 Top-N 设备清单（按置信度排序）。

==================
📌【深入归因与高阶洞察任务】
你需要进一步探究指标异常背后的可能原因，重点包括：

- 自动识别控制组control与实验组test之间的关键性能差异；
- 基于设备、内存、架构、时间、配置等级等维度进行分组分析；
- 综合判断指标改善或退化的趋势与显著性；
- 支持从功耗效率、波动性、推荐准确性、GC负担等角度归因；
- 可输出关键设备表现列表、分层对比图、混淆矩阵、波动热力图等结构化内容；

你需要保持对数据结构与场景背景的理解，输出面向性能优化、引擎升级提供支持的分析结果。
==================

==================
这里有一些额外的关于csv文件列有关的元信息可以使用:
{meta_data_info}
==================
你应以数据为基础，具备一定领域知识推理能力，帮助用户发现非显性规律，输出可用于策略决策的表格、图表与结论性总结。
'''


TEST_NEXT_PROMPT = '''
请基于已有数据与分析目标，规划当前任务的下一步操作。你可以选择继续进行指标比较、趋势可视化、异常设备识别或深入归因分析等步骤。请明确输出当前的分析目标与拟采取的方法。
'''

class DataAnalysisExpert(DataAnalysis):
    name: str = "DataAnalysisExpert"
    description: str = (
        "一个游戏性能指标分析智能体，支持对功耗、FPS、GC等子表进行深入分析。"
        "除常规趋势判断外，该 Agent 具备设备分层对比、推荐配置准确性验证、性能波动识别、"
        "GC行为归因、功耗性价比建模、架构/内存差异比较、适配滞后分析等能力，"
        "可最终输出结构化指标趋势结果、归因图表、关键设备排名表及综合性能评分，"
        "辅助性能回归定位与引擎优化策略制定。"
    )


    system_prompt: str = TEST_SYSTEM_PROMPT
    next_step_prompt: str = TEST_NEXT_PROMPT

    max_observe: int = 20000
    max_steps: int = 25

    # Add coordination tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            ControlTestAnalysisTool(),
            NormalPythonExecute(),
            FileReadWriteTool(),
            ExcelJsonMetadataReadTool(),
            ExcelColumnNamesReadTool(),
            Terminate(),
        )
    )
    
    async def update_system_prompt(self, csv_file_path: str):
        """
        更新系统提示，包含元数据文件路径。
        """
        csv_columns = pd.read_csv(csv_file_path, nrows=0).columns.tolist()
        if config.run_flow_config.meta_data_name:
            metadata_file_path = os.path.join(
                config.workspace_root, config.run_flow_config.meta_data_name
            )
        read_tool = ExcelJsonMetadataReadTool()
        columns_meta_info = await read_tool.execute(
            json_metadata_path = metadata_file_path,
            column_names = csv_columns  
        )
        columns_meta_info = str(columns_meta_info.output)
        self.system_prompt = self.system_prompt.format(meta_data_info=columns_meta_info) 
        
        