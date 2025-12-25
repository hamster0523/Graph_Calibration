import os
import json
from typing import List, Dict, Any, Optional
import asyncio
import re

from app.exceptions import ToolError
from app.tool.base import BaseTool, ToolResult
from app.logger import logger
from app.config import config

# 关键性能指标关键词
KEY_PERFORMANCE_INDICATORS = [
        "avgfps_unity",      # 平均帧率
        "totalgcallocsize",  # GC分配内存大小
        "gt_50",             # GC次数 
        "current_avg",       # 当前电流均值(mA)
        "battleendtotalpss", # 战斗结束PSS内存
        "bigjankper10min"    # 每10分钟大卡顿次数
    ]
    
# 辅助分类字段关键词
AUXILIARY_FIELDS = [
        "devicemodel",    # 设备型号
        "unityversion",   # 引擎版本
        "param_value",    # 实验分组
        "accountid",      # 账户ID
        "uid",            # 用户ID
        "zoneid",         # 区域ID
        "channelid",      # 渠道ID
        "region",         # 地区
        "date",           # 日期
        "time",           # 时间
        "timestamp",      # 时间戳
        "logymd",         # 日志日期
        "platform",       # 平台
        "version",        # 版本
        "build"           # 构建版本
    ]

class ExcelJsonMetadataReadTool(BaseTool):
    """
    A tool for reading and extracting column metadata from a JSON metadata file.
    
    This tool reads a JSON metadata file and extracts the metadata for specified columns,
    with enhanced classification based on key performance metrics and auxiliary fields.
    """
    
    name: str = "excel_json_metadata_read_tool"
    description: str = "Read and extract column metadata from a JSON metadata file for specified columns with performance metric classification."
    parameters: dict = {
        "type": "object",
        "properties": {
            "column_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of column names to extract metadata for. If empty, metadata for all columns will be returned.",
                "default": []
            },
            "include_reasoning": {
                "type": "boolean",
                "description": "Whether to include the AI-generated reasoning about the column's purpose in the results.",
                "default": True,
            },
            "classify_metrics": {
                "type": "boolean",
                "description": "Whether to classify columns as key metrics, auxiliary fields, or others based on predefined criteria.",
                "default": True
            }
        },
        "required": [],
    }

    async def execute(
        self,
        column_names: Optional[List[str]] = None,
        include_reasoning: bool = True,
        classify_metrics: bool = True,
        json_metadata_path: Optional[str] = None
    ) -> ToolResult:
        """
        Read and extract column metadata from a JSON metadata file.
        """
        try:
            # Use fixed JSON metadata path
            json_metadata_path = os.path.join(config.workspace_root, "classification_results.json")
            
            # Validate input
            if not os.path.exists(json_metadata_path):
                return ToolResult(error=f"JSON metadata file not found: {json_metadata_path}")
            
            # Read the JSON metadata file
            metadata = await self._read_json_metadata(json_metadata_path)
            
            # Extract column metadata
            column_metadata = await self._extract_column_metadata(
                metadata, column_names, include_reasoning, classify_metrics
            )
            
            # Generate enhanced summary with metric classification
            summary = self._generate_enhanced_summary(column_metadata, column_names)
            
            # Format output
            output_lines = []
            output_lines.append("=== 列元信息分析结果 ===\n")
            output_lines.append(f"📁 源文件: {json_metadata_path}")
            output_lines.append(f"📊 分析列数: {len(column_metadata)}")
            output_lines.append(f"{summary}\n")
            
            # Add metric classification summary if enabled
            if classify_metrics:
                classification_summary = self._generate_classification_summary(column_metadata)
                output_lines.append("🎯 **指标分类统计:**")
                output_lines.append(classification_summary)
            
            output_lines.append("\n📋 **详细列信息:**")
            
            # Format each column's metadata
            for col_name, col_data in column_metadata.items():
                output_lines.append(f"\n🔹 **{col_name}**")
                
                # Basic info
                if 'Type' in col_data:
                    output_lines.append(f"   📝 类型: {col_data['Type']}")
                if 'Description' in col_data:
                    output_lines.append(f"   📄 描述: {col_data['Description']}")
                if 'Classification' in col_data:
                    output_lines.append(f"   🏷️  分类: {col_data['Classification']}")
                
                # Enhanced classification
                if classify_metrics and 'enhanced_classification' in col_data:
                    classification = col_data['enhanced_classification']
                    if classification['is_key_metric'] == "1":
                        output_lines.append(f"   ⭐ **关键性能指标** (匹配: {classification['match_reason']})")
                    elif classification['is_key_metric'] == "2":
                        output_lines.append(f"   🔧 **辅助分析字段** (匹配: {classification['match_reason']})")
                    else:
                        output_lines.append(f"   📊 其他字段")
                
                # Sample values if available
                if 'sample_values' in col_data and col_data['sample_values']:
                    sample_str = ', '.join(str(v) for v in col_data['sample_values'][:3])
                    output_lines.append(f"   🔍 示例值: {sample_str}")
                
                # Reasoning if requested
                if include_reasoning and 'Reason' in col_data:
                    reason = col_data['Reason'][:200] + "..." if len(col_data['Reason']) > 200 else col_data['Reason']
                    output_lines.append(f"   💭 推理: {reason}")
            
            return ToolResult(output="\n".join(output_lines))
            
        except Exception as e:
            logger.error(f"Error extracting column metadata: {e}")
            return ToolResult(error=f"Failed to extract column metadata: {str(e)}")
    
    async def _read_json_metadata(self, json_metadata_path: str) -> Dict[str, Any]:
        """Read the JSON metadata file asynchronously."""
        try:
            loop = asyncio.get_event_loop()
            
            def read_json():
                with open(json_metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            metadata = await loop.run_in_executor(None, read_json)
            logger.info(f"Successfully read JSON metadata file: {json_metadata_path}")
            return metadata
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON metadata file: {e}")
            raise ToolError(f"Invalid JSON format in metadata file: {str(e)}")
        except Exception as e:
            logger.error(f"Error reading JSON metadata file: {e}")
            raise ToolError(f"Failed to read JSON metadata file: {str(e)}")
    
    async def _extract_column_metadata(
        self,
        metadata: Any,
        column_names: Optional[List[str]],
        include_reasoning: bool,
        classify_metrics: bool
    ) -> Dict[str, Dict[str, Any]]:
        """Extract metadata for specified columns with enhanced classification."""
        
        # Handle different JSON structures
        columns_data = None
        if isinstance(metadata, list):
            # Handle array format like your example
            columns_data = {}
            for i, item in enumerate(metadata):
                if isinstance(item, dict):
                    field_name = item.get('Field', f'column_{i}')
                    columns_data[field_name] = item
        elif isinstance(metadata, dict):
            # Handle object format
            if "columns_analysis" in metadata:
                columns_data = metadata["columns_analysis"]
            elif "columns" in metadata:
                columns_data = metadata["columns"]
            else:
                # Try to use the whole metadata as columns data
                columns_data = metadata
        
        if not columns_data:
            raise ToolError("Invalid metadata format: Unable to find column information.")
        
        # Filter columns if specified
        if column_names:
            filtered_data = {}
            for col_name in column_names:
                if col_name in columns_data:
                    filtered_data[col_name] = columns_data[col_name]
            columns_data = filtered_data
        
        # Process each column
        result = {}
        for col_name, col_data in columns_data.items():
            if not isinstance(col_data, dict):
                continue
                
            processed_data = dict(col_data)  # Copy original data
            
            # Remove reasoning if not requested
            if not include_reasoning and 'category_reasoning' in processed_data:
                del processed_data['category_reasoning']
            if not include_reasoning and 'Reason' in processed_data:
                del processed_data['Reason']
            
            # Add enhanced classification if requested
            if classify_metrics:
                processed_data['enhanced_classification'] = self._classify_column(col_name, col_data)
            
            result[col_name] = processed_data
        
        return result
    
    def _classify_column(self, col_name: str, col_data: Dict[str, Any]) -> Dict[str, str]:
        """Classify a column as key metric (1), auxiliary field (2), or other (3)."""
        col_name_lower = col_name.lower()
        description = col_data.get('Description', '').lower()
        
        # Check for key performance indicators (is_key_metric = "1")
        for kpi in KEY_PERFORMANCE_INDICATORS:
            if kpi.lower() in col_name_lower:
                return {
                    'is_key_metric': "1",
                    'match_reason': f'字段名包含关键指标 "{kpi}"'
                }
        
        # Check description for key metrics
        key_metric_patterns = [
            ('fps', '帧率'),
            ('gc', 'GC内存'),
            ('jank', '卡顿'),
            ('current', '电流'),
            ('pss', 'PSS内存'),
            ('memory', '内存'),
            ('performance', '性能'),
            ('latency', '延迟'),
            ('frametime', '帧时间')
        ]
        
        for pattern, chinese_name in key_metric_patterns:
            if pattern in col_name_lower or pattern in description:
                return {
                    'is_key_metric': "1",
                    'match_reason': f'包含关键性能模式 "{chinese_name}"'
                }
        
        # Check for auxiliary fields (is_key_metric = "2")
        for aux_field in AUXILIARY_FIELDS:
            if aux_field.lower() in col_name_lower:
                return {
                    'is_key_metric': "2",
                    'match_reason': f'字段名包含辅助分析字段 "{aux_field}"'
                }
        
        # Check for auxiliary patterns
        auxiliary_patterns = [
            ('device', '设备'),
            ('model', '型号'),
            ('version', '版本'),
            ('id', '标识符'),
            ('date', '日期'),
            ('time', '时间'),
            ('platform', '平台'),
            ('channel', '渠道'),
            ('region', '地区'),
            ('zone', '区域'),
            ('account', '账户'),
            ('user', '用户')
        ]
        
        for pattern, chinese_name in auxiliary_patterns:
            if pattern in col_name_lower or pattern in description:
                return {
                    'is_key_metric': "2",
                    'match_reason': f'包含辅助分析模式 "{chinese_name}"'
                }
        
        # Default to other (is_key_metric = "3")
        return {
            'is_key_metric': "3",
            'match_reason': '不匹配关键指标或辅助字段模式'
        }
    
    def _generate_enhanced_summary(self, column_metadata: Dict, requested_columns: Optional[List[str]]) -> str:
        """Generate an enhanced summary with metric information."""
        if requested_columns:
            found_columns = [col for col in requested_columns if col in column_metadata]
            missing_columns = [col for col in requested_columns if col not in column_metadata]
            
            summary_parts = [f"✅ 找到 {len(found_columns)}/{len(requested_columns)} 个请求的列"]
            if missing_columns:
                summary_parts.append(f"❌ 缺失列: {', '.join(missing_columns)}")
        else:
            summary_parts = [f"✅ 提取了所有 {len(column_metadata)} 个列的元数据"]
        
        return "\n".join(summary_parts)
    
    def _generate_classification_summary(self, column_metadata: Dict) -> str:
        """Generate a summary of metric classifications."""
        key_metrics = []
        auxiliary_fields = []
        others = []
        
        for col_name, col_data in column_metadata.items():
            if 'enhanced_classification' in col_data:
                classification = col_data['enhanced_classification']['is_key_metric']
                if classification == "1":
                    key_metrics.append(col_name)
                elif classification == "2":
                    auxiliary_fields.append(col_name)
                else:
                    others.append(col_name)
        
        summary_lines = []
        summary_lines.append(f"   ⭐ 关键性能指标 ({len(key_metrics)}个): {', '.join(key_metrics) if key_metrics else '无'}")
        summary_lines.append(f"   🔧 辅助分析字段 ({len(auxiliary_fields)}个): {', '.join(auxiliary_fields) if auxiliary_fields else '无'}")
        summary_lines.append(f"   📊 其他字段 ({len(others)}个): {', '.join(others) if others else '无'}")
        
        return "\n".join(summary_lines)


# Command-line interface for testing
if __name__ == "__main__":
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(description="Read and extract column metadata from a JSON metadata file")
    parser.add_argument("--columns", nargs="+", help="Column names to extract metadata for")
    parser.add_argument("--no-reasoning", action="store_true", help="Exclude AI-generated reasoning from the results")
    parser.add_argument("--no-classify", action="store_true", help="Disable metric classification")
    
    args = parser.parse_args()
    
    async def run_tool():
        tool = ExcelJsonMetadataReadTool()
        result = await tool.execute(
            args.columns, 
            not args.no_reasoning,
            not args.no_classify
        )
        
        if result.error:
            print(f"Error: {result.error}")
        else:
            print(result.output)
    
    asyncio.run(run_tool())
