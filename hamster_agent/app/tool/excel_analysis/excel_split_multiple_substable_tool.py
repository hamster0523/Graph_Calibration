import json
import os
from typing import Dict, List

import pandas as pd

from app.config import config
from app.exceptions import ToolError
from app.logger import logger
from app.tool.base import BaseTool, ToolResult


class ExcelSplitMultipleSubtableTool(BaseTool):

    name: str = "excel_split_multiple_subtable_tool"
    description: str = (
        "基于JSON元数据文件中的列分类信息对CSV文件进行两级拆分的工具。首先按照性能指标的大分类（内存、功耗、其他、平均帧率、卡顿、温度）将数据拆分为6个主要类别的子表，然后在每个大分类内部再按照设备质量等级（如realpicturequality字段的不同值）进行进一步细分，最终生成分层次的多个子表文件，便于进行分类别和分质量等级的深入分析。"
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "csv_file_path": {
                "type": "string",
                "description": "需要拆分的已清理CSV文件的路径",
            },
            "common_columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "需要包含在每个子表中的公共列名列表。这些通常是包含元数据或上下文信息的列，如param_value、devicemodel、platform、hardware_model、OS_version或其他标识符，这些列为所有性能指标提供上下文，应该存在于每个子表中以便进行适当的分析。",
                "default": ["param_value", "devicemodel"],
            },
            "quality_column": {
                "type": "string",
                "description": "用于设备质量分类的列名（例如realpicturequality），该列的不同值将作为第二级拆分的依据",
                "default": "realpicturequality",
            },
        },
        "required": ["csv_file_path"],
    }

    async def execute(
        self,
        csv_file_path: str,
        common_columns: List[str],
        quality_column: str = "realpicturequality",
    ) -> ToolResult:
        """
        Split a CSV file into category-based subtables and further split by device quality levels.

        The 6 categories are:
        - 内存 (Memory)
        - 功耗 (Power)
        - 其他 (Others)
        - 平均帧率 (FPS)
        - 卡顿 (Lag)
        - 温度 (Temperature)

        Each category will be further split by device quality levels (e.g., realpicturequality).

        Args:
            csv_file_path: Path to the CSV file to be split
            json_metadata_path: Path to JSON file containing column metadata with Classification field
            common_columns: List of columns to include in all subtables
            quality_column: Column name for device quality classification
            output_dir: Optional output directory for subtables

        Returns:
            ToolResult with summary of created subtables
        """
        try:
            # Validate input files
            if not os.path.exists(csv_file_path):
                raise ToolError(f"CSV file not found: {csv_file_path}")

            if config.run_flow_config.meta_data_name is None:
                raise ToolError(
                    f"JSON metadata file name is not located in workspace root directory"
                )

            json_metadata_path = os.path.join(
                config.workspace_root, config.run_flow_config.meta_data_name
            )

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

            # Validate quality column exists
            if quality_column not in df.columns:
                raise ToolError(
                    f"Quality column '{quality_column}' not found in CSV file"
                )

            # Define the 6 classification categories
            categories = {
                "内存": "memory",
                "功耗": "power",
                "其他": "others",
                "平均帧率": "fps",
                "卡顿": "lag",
                "温度": "temperature",
            }

            # Create field to classification mapping
            field_classification_map = {}

            # Handle both list and single object JSON formats
            if isinstance(metadata, list):
                metadata_items = metadata
            else:
                metadata_items = [metadata]

            for item in metadata_items:
                if (
                    isinstance(item, dict)
                    and "Field" in item
                    and "Classification" in item
                ):
                    field_name = item["Field"]
                    classification = item["Classification"]
                    field_classification_map[field_name] = classification

            logger.info(
                f"Created field classification mapping for {len(field_classification_map)} fields"
            )

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

            if not common_columns:
                raise ToolError("No valid common columns found in the CSV file")

            # Get unique quality levels
            quality_levels = sorted(df[quality_column].unique())
            logger.info(f"Found {len(quality_levels)} quality levels: {quality_levels}")

            # Prepare output directory
            csv_dir = os.path.dirname(csv_file_path)
            output_dir = os.path.join(csv_dir, "splitsubtable_multiple")

            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Output directory prepared: {output_dir}")

            # Group columns by classification
            columns_by_category = {category: [] for category in categories.keys()}
            unclassified_columns = []

            for col in df.columns:
                if col in common_columns or col == quality_column:
                    continue  # Skip common columns and quality column as they'll be handled separately

                classification = field_classification_map.get(col)
                if classification in categories:
                    columns_by_category[classification].append(col)
                else:
                    unclassified_columns.append(col)
                    # Add unclassified columns to "其他" category
                    columns_by_category["其他"].append(col)

            if unclassified_columns:
                logger.info(
                    f"Unclassified columns added to '其他' category: {unclassified_columns}"
                )

            # Create subtables
            created_subtables = {}
            total_files_created = 0

            for category_cn, category_en in categories.items():
                category_columns = columns_by_category[category_cn]

                if not category_columns:
                    logger.info(
                        f"No columns found for category '{category_cn}', skipping subtable creation"
                    )
                    continue

                # Create category directory
                category_dir = os.path.join(output_dir, f"{category_en}_subtables")
                os.makedirs(category_dir, exist_ok=True)
                logger.info(f"Created category directory: {category_dir}")

                # Prepare columns for this category (common columns + category-specific columns, without quality column for final output)
                subtable_columns_with_quality = (
                    common_columns + [quality_column] + category_columns
                )
                subtable_columns_final = common_columns + category_columns

                # Create base subtable with quality column for splitting
                base_subtable = df[subtable_columns_with_quality].copy()

                # Split by quality levels
                quality_subtables = {}
                for quality_level in quality_levels:
                    # Filter data for this quality level
                    quality_data = base_subtable[
                        base_subtable[quality_column] == quality_level
                    ].copy()

                    if quality_data.empty:
                        logger.warning(
                            f"No data found for quality level '{quality_level}' in category '{category_cn}'"
                        )
                        continue

                    # Remove quality column from final output
                    quality_data = quality_data[subtable_columns_final].copy()

                    # Save quality-specific subtable
                    quality_filename = f"{category_en}_{quality_level}_subtable.csv"
                    quality_path = os.path.join(category_dir, quality_filename)

                    try:
                        quality_data.to_csv(quality_path, index=False, encoding="utf-8")
                        logger.info(
                            f"Created quality subtable: {quality_path} with shape {quality_data.shape}"
                        )

                        quality_subtables[quality_level] = {
                            "filename": quality_filename,
                            "path": quality_path,
                            "shape": quality_data.shape,
                            "columns": subtable_columns_final,
                            "data_count": len(quality_data),
                        }
                        total_files_created += 1

                    except Exception as e:
                        logger.error(
                            f"Error saving quality subtable for category {category_cn}, quality {quality_level}: {str(e)}"
                        )
                        raise ToolError(
                            f"Failed to save quality subtable for category {category_cn}, quality {quality_level}: {str(e)}"
                        )

                # Store category information
                if quality_subtables:
                    created_subtables[category_cn] = {
                        "category_english": category_en,
                        "category_dir": category_dir,
                        "category_columns": category_columns,
                        "quality_subtables": quality_subtables,
                        "total_quality_levels": len(quality_subtables),
                        "total_columns": len(subtable_columns_final),
                        "feature_columns_count": len(category_columns),
                        "common_columns_count": len(common_columns),
                    }

            # Generate summary report
            summary = self._generate_summary_report(
                csv_file_path,
                json_metadata_path,
                output_dir,
                created_subtables,
                common_columns,
                field_classification_map,
                quality_column,
                quality_levels,
                total_files_created,
            )

            return ToolResult(output=summary)

        except ToolError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            error_msg = (
                f"Unexpected error in excel split multiple subtable tool: {str(e)}"
            )
            logger.error(error_msg)
            return ToolResult(error=error_msg)

    def _generate_summary_report(
        self,
        csv_file_path: str,
        json_metadata_path: str,
        output_dir: str,
        created_subtables: Dict,
        common_columns: List[str],
        field_classification_map: Dict,
        quality_column: str,
        quality_levels: List,
        total_files_created: int,
    ) -> str:
        """Generate a comprehensive summary report of the splitting operation."""

        summary = "Excel多级分表工具执行完成\n"
        summary += "=" * 60 + "\n\n"

        summary += f"**输入文件信息:**\n"
        summary += f"- CSV文件: {csv_file_path}\n"
        summary += f"- JSON元数据文件: {json_metadata_path}\n"
        summary += f"- 输出目录: {output_dir}\n"
        summary += f"- 设备质量分类列: {quality_column}\n\n"

        summary += f"**设备质量分级:**\n"
        summary += f"- 发现 {len(quality_levels)} 个质量等级: {', '.join(map(str, quality_levels))}\n\n"

        summary += f"**公共列配置:**\n"
        summary += f"- 公共列数量: {len(common_columns)}\n"
        summary += f"- 公共列列表: {', '.join(common_columns)}\n\n"

        summary += f"**分类统计:**\n"
        summary += f"- 总字段数量: {len(field_classification_map)}\n"
        summary += f"- 成功创建分类目录数量: {len(created_subtables)}\n"
        summary += f"- 总共生成子表文件数量: {total_files_created}\n\n"

        summary += f"**生成的分类子表详情:**\n"
        for category_cn, info in created_subtables.items():
            summary += f"\n📊 **{category_cn}分类 ({info['category_english']})**\n"
            summary += f"   - 分类目录: {info['category_dir']}\n"
            summary += f"   - 特征列数量: {info['feature_columns_count']}\n"
            summary += f"   - 公共列数量: {info['common_columns_count']}\n"
            summary += f"   - 总列数: {info['total_columns']}\n"
            summary += f"   - 质量等级子表数量: {info['total_quality_levels']}\n"

            if info["category_columns"]:
                summary += f"   - 特征列列表: {', '.join(info['category_columns'])}\n"

            summary += f"   - 质量等级子表详情:\n"
            for quality_level, quality_info in info["quality_subtables"].items():
                summary += f"     * {quality_level}: {quality_info['filename']} ({quality_info['shape'][0]} 行 × {quality_info['shape'][1]} 列)\n"

        summary += f"\n**字段分类映射:**\n"
        categories = ["内存", "功耗", "其他", "平均帧率", "卡顿", "温度"]
        for category in categories:
            category_fields = [
                field
                for field, classification in field_classification_map.items()
                if classification == category
            ]
            summary += f"- {category}: {len(category_fields)} 个字段\n"
            if category_fields:
                # Show first few fields, truncate if too many
                displayed_fields = category_fields[:5]
                if len(category_fields) > 5:
                    displayed_fields.append(f"... 等{len(category_fields)-5}个更多字段")
                summary += f"  └─ {', '.join(displayed_fields)}\n"

        summary += f"\n**文件结构:**\n"
        summary += f"输出目录结构:\n"
        summary += f"{output_dir}/\n"
        for category_cn, info in created_subtables.items():
            category_en = info["category_english"]
            summary += f"├── {category_en}_subtables/\n"
            for quality_level in info["quality_subtables"].keys():
                summary += f"│   ├── {category_en}_{quality_level}_subtable.csv\n"

        summary += f"\n**操作结果:**\n"
        summary += f"✅ 成功将原始CSV文件按 {len(created_subtables)} 个分类进行分割\n"
        summary += f"✅ 每个分类按 {len(quality_levels)} 个质量等级进一步细分\n"
        summary += f"✅ 总共生成 {total_files_created} 个子表文件\n"
        summary += f"✅ 所有子表均包含 {len(common_columns)} 个公共列用于数据关联\n"
        summary += f"✅ 所有文件已保存至: {output_dir}\n"

        if len(created_subtables) < 6:
            missing_categories = set(
                ["内存", "功耗", "其他", "平均帧率", "卡顿", "温度"]
            ) - set(created_subtables.keys())
            summary += f"\n⚠️ 注意: 以下分类未找到对应字段，未生成子表: {', '.join(missing_categories)}\n"

        summary += f"\n**使用建议:**\n"
        summary += f"- 各质量等级的子表可独立进行性能分析\n"
        summary += f"- 利用公共列进行跨子表数据关联分析\n"
        summary += f"- 可按质量等级进行同类设备的横向对比分析\n"
        summary += f"- 可按性能指标类别进行不同质量等级的纵向分析\n"
        summary += f"- 针对不同性能指标类别和质量等级采用专门的分析方法\n"
        summary += f"- 建议对各类别和质量等级分别进行统计分析和可视化\n"

        return summary
