import json
import os
from typing import Dict, List

import pandas as pd

from app.config import config
from app.exceptions import ToolError
from app.logger import logger
from app.tool.base import BaseTool, ToolResult

# 预定义的关键性能指标
KEY_PERFORMANCE_METRICS = [
        "avgfps_unity",        # 平均帧率
        "totalgcallocsize",    # 总GC分配大小
        "gt_50",               # 大于50ms的次数
        "current_avg",         # 当前平均值
        "battleendtotalpss",   # 战斗结束总PSS
        "bigjankper10min"      # 每10分钟大卡顿次数
    ]

class ExcelSplitToKeyMetricTool(BaseTool):

    name: str = "excel_split_to_key_metric_tool"
    description: str = (
        "从CSV文件中提取关键性能指标的专用工具。该工具会提取预定义的6个关键指标"
        "(avgfps_unity, totalgcallocsize, gt_50, current_avg, battleendtotalpss, bigjankper10min)、"
        "从JSON元数据文件中标记为'is_key_metric': '2'的公共列，以及param_value列用于区分control/test组，"
        "生成一个包含所有关键指标的key_metric.csv子表。"
    )
    
    parameters: dict = {
        "type": "object",
        "properties": {
            "csv_file_path": {
                "type": "string",
                "description": "需要提取关键指标的CSV文件路径",
            },
        },
        "required": ["csv_file_path"],
    }

    async def execute(
        self,
        csv_file_path: str,
        output_filename: str = "key_metric.csv",
        output_dir = None,
    ) -> ToolResult:
        """
        Extract key performance metrics from CSV file to create a key_metric.csv subtable.

        Args:
            csv_file_path: Path to the source CSV file
            output_filename: Name of the output key metrics file
            output_dir: Optional output directory for the key metrics file

        Returns:
            ToolResult with summary of the key metrics extraction
        """
        try:
            # Validate input file
            if not os.path.exists(csv_file_path):
                raise ToolError(f"CSV file not found: {csv_file_path}")

            # Check for JSON metadata file
            if config.run_flow_config.meta_data_name is None:
                raise ToolError(
                    f"JSON metadata file name is not configured in workspace root directory"
                )

            json_metadata_path = os.path.join(
                config.workspace_root, config.run_flow_config.meta_data_name
            )

            if not os.path.exists(json_metadata_path):
                raise ToolError(f"JSON metadata file not found: {json_metadata_path}")

            # Read JSON metadata
            try:
                with open(json_metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                logger.info(
                    f"Successfully loaded JSON metadata from {json_metadata_path}"
                )
            except json.JSONDecodeError as e:
                raise ToolError(f"Invalid JSON format in metadata file: {str(e)}")
            except Exception as e:
                raise ToolError(f"Error reading JSON metadata file: {str(e)}")

            # Read CSV file
            try:
                df = pd.read_csv(csv_file_path)
                logger.info(f"Successfully loaded CSV file with shape: {df.shape}")
            except Exception as e:
                raise ToolError(f"Error reading CSV file: {str(e)}")

            # Extract common columns from metadata (is_key_metric = "2")
            common_columns = self._extract_common_columns_from_metadata(metadata)
            
            # 确保包含 param_value 列用于区分 control/test 组
            if 'param_value' not in common_columns and 'param_value' in df.columns:
                common_columns.append('param_value')
                logger.info("Added param_value column for control/test group identification")
            
            logger.info(f"Found {len(common_columns)} common columns from metadata: {common_columns}")

            # Validate common columns exist in CSV
            missing_common_cols = [
                col for col in common_columns if col not in df.columns
            ]
            if missing_common_cols:
                logger.warning(
                    f"Common columns not found in CSV: {missing_common_cols}"
                )
                # Filter to only existing common columns
                common_columns = [col for col in common_columns if col in df.columns]

            # Check for key performance metrics in CSV
            available_key_metrics = []
            missing_key_metrics = []
            
            for metric in KEY_PERFORMANCE_METRICS:
                if metric in df.columns:
                    available_key_metrics.append(metric)
                else:
                    missing_key_metrics.append(metric)

            if not available_key_metrics:
                raise ToolError(
                    f"None of the key performance metrics found in CSV. "
                    f"Expected metrics: {', '.join(KEY_PERFORMANCE_METRICS)}"
                )

            if missing_key_metrics:
                logger.warning(
                    f"Missing key performance metrics in CSV: {missing_key_metrics}"
                )

            logger.info(f"Available key metrics: {available_key_metrics}")

            # Prepare output directory
            if output_dir is None:
                csv_dir = os.path.dirname(csv_file_path)
                output_dir = csv_dir

            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Output directory prepared: {output_dir}")

            # Combine common columns and key metrics
            all_columns = common_columns + available_key_metrics
            
            # Remove duplicates while preserving order
            final_columns = []
            for col in all_columns:
                if col not in final_columns:
                    final_columns.append(col)

            logger.info(f"Final columns for key metrics table: {final_columns}")

            # Create key metrics subtable
            key_metrics_df = df[final_columns].copy()

            # Save key metrics file
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                key_metrics_df.to_csv(output_path, index=False, encoding="utf-8")
                logger.info(f"Successfully created key metrics file: {output_path}")
            except Exception as e:
                raise ToolError(f"Error saving key metrics file: {str(e)}")

            # Generate analysis summary
            analysis_summary = self._analyze_key_metrics(key_metrics_df, available_key_metrics, common_columns)

            # Generate summary report
            summary = self._generate_summary_report(
                csv_file_path,
                json_metadata_path,
                output_path,
                key_metrics_df.shape,
                common_columns,
                available_key_metrics,
                missing_key_metrics,
                analysis_summary
            )

            return ToolResult(output=summary)

        except ToolError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            error_msg = f"Unexpected error in key metrics extraction tool: {str(e)}"
            logger.error(error_msg)
            return ToolResult(error=error_msg)

    def _extract_common_columns_from_metadata(self, metadata) -> List[str]:
        """Extract common columns from metadata where is_key_metric = "2"."""
        
        common_columns = []
        
        # Handle both list and single object JSON formats
        if isinstance(metadata, list):
            metadata_items = metadata
        else:
            metadata_items = [metadata]

        for item in metadata_items:
            if isinstance(item, dict) and "Field" in item and "is_key_metric" in item:
                field_name = item["Field"]
                is_key_metric = str(item["is_key_metric"])  # Convert to string for comparison
                
                if is_key_metric == "2":
                    common_columns.append(field_name)
                    logger.info(f"Found common column from metadata: {field_name}")

        logger.info(f"Extracted {len(common_columns)} common columns from metadata")
        return common_columns

    def _analyze_key_metrics(
        self, 
        df: pd.DataFrame, 
        key_metrics: List[str], 
        common_columns: List[str]
    ) -> Dict:
        """Analyze the key metrics data for basic statistics."""
        
        analysis = {
            "data_overview": {
                "total_records": len(df),
                "total_columns": len(df.columns),
                "common_columns_count": len(common_columns),
                "key_metrics_count": len(key_metrics)
            },
            "key_metrics_stats": {},
            "data_quality": {
                "missing_values_per_column": {},
                "complete_records": 0
            }
        }

        # Analyze each key metric
        for metric in key_metrics:
            if metric in df.columns:
                metric_data = pd.to_numeric(df[metric], errors='coerce')
                
                analysis["key_metrics_stats"][metric] = {
                    "count": metric_data.count(),
                    "missing": metric_data.isna().sum(),
                    "mean": metric_data.mean() if metric_data.count() > 0 else None,
                    "std": metric_data.std() if metric_data.count() > 0 else None,
                    "min": metric_data.min() if metric_data.count() > 0 else None,
                    "max": metric_data.max() if metric_data.count() > 0 else None,
                    "median": metric_data.median() if metric_data.count() > 0 else None
                }

        # Data quality analysis
        for col in df.columns:
            missing_count = df[col].isna().sum()
            analysis["data_quality"]["missing_values_per_column"][col] = missing_count

        # Count complete records (no missing values in key metrics)
        key_metrics_df = df[key_metrics]
        complete_records = len(key_metrics_df.dropna())
        analysis["data_quality"]["complete_records"] = complete_records

        return analysis

    def _generate_summary_report(
        self,
        csv_file_path: str,
        json_metadata_path: str,
        output_path: str,
        output_shape: tuple,
        common_columns: List[str],
        available_key_metrics: List[str],
        missing_key_metrics: List[str],
        analysis_summary: Dict
    ) -> str:
        """Generate a comprehensive summary report of the key metrics extraction."""

        summary = "关键指标提取工具执行完成\n"
        summary += "=" * 60 + "\n\n"

        summary += f"**输入文件信息:**\n"
        summary += f"- 源CSV文件: {csv_file_path}\n"
        summary += f"- JSON元数据文件: {json_metadata_path}\n"
        summary += f"- 输出文件: {output_path}\n\n"

        summary += f"**关键指标提取结果:**\n"
        summary += f"- 输出文件形状: {output_shape[0]} 行 × {output_shape[1]} 列\n"
        summary += f"- 成功提取的关键指标数量: {len(available_key_metrics)}\n"
        summary += f"- 公共列数量: {len(common_columns)}\n\n"

        summary += f"**成功提取的关键指标:**\n"
        for i, metric in enumerate(available_key_metrics, 1):
            stats = analysis_summary["key_metrics_stats"].get(metric, {})
            summary += f"{i}. {metric}\n"
            if stats:
                summary += f"   - 有效数据: {stats['count']} 条\n"
                summary += f"   - 缺失数据: {stats['missing']} 条\n"
                if stats['mean'] is not None:
                    summary += f"   - 平均值: {stats['mean']:.2f}\n"
                if stats['median'] is not None:
                    summary += f"   - 中位数: {stats['median']:.2f}\n"
                if stats['std'] is not None:
                    summary += f"   - 标准差: {stats['std']:.2f}\n"
            summary += "\n"

        if missing_key_metrics:
            summary += f"**缺失的关键指标:**\n"
            for metric in missing_key_metrics:
                summary += f"- {metric} (未在源CSV中找到)\n"
            summary += "\n"

        summary += f"**公共列 (is_key_metric=2 + param_value):**\n"
        if common_columns:
            for i, col in enumerate(common_columns, 1):
                missing_count = analysis_summary["data_quality"]["missing_values_per_column"].get(col, 0)
                col_type = "(param_value)" if col == "param_value" else "(metadata)"
                summary += f"{i}. {col} {col_type} (缺失值: {missing_count})\n"
        else:
            summary += "- 未找到标记为公共列的字段\n"
        summary += "\n"

        summary += f"**数据质量概览:**\n"
        summary += f"- 总记录数: {analysis_summary['data_overview']['total_records']}\n"
        summary += f"- 关键指标完整记录数: {analysis_summary['data_quality']['complete_records']}\n"
        summary += f"- 数据完整度: {analysis_summary['data_quality']['complete_records'] / analysis_summary['data_overview']['total_records'] * 100:.1f}%\n\n"

        summary += f"**缺失值统计:**\n"
        for col, missing_count in analysis_summary["data_quality"]["missing_values_per_column"].items():
            if missing_count > 0:
                missing_pct = missing_count / analysis_summary['data_overview']['total_records'] * 100
                summary += f"- {col}: {missing_count} 条 ({missing_pct:.1f}%)\n"
        summary += "\n"

        summary += f"**操作结果:**\n"
        summary += f"✅ 成功提取 {len(available_key_metrics)} 个关键性能指标\n"
        summary += f"✅ 包含 {len(common_columns)} 个公共列用于数据关联\n"
        summary += f"✅ 生成关键指标子表: {output_path}\n"
        summary += f"✅ 数据形状: {output_shape[0]} 行 × {output_shape[1]} 列\n"

        if missing_key_metrics:
            summary += f"⚠️ 注意: {len(missing_key_metrics)} 个预定义关键指标未在源数据中找到\n"

        summary += f"\n**使用建议:**\n"
        summary += f"- 该关键指标表可用于核心性能分析\n"
        summary += f"- param_value列用于区分control/test组，支持对比分析\n"
        summary += f"- 建议结合control_test_analysis_tool进行深度对比分析\n"
        summary += f"- 可使用公共列进行设备型号等维度的分组分析\n"
        summary += f"- 关注数据完整度，考虑处理缺失值对分析的影响\n"
        summary += f"- 6个关键指标涵盖帧率、内存、卡顿等核心性能维度\n"

        return summary