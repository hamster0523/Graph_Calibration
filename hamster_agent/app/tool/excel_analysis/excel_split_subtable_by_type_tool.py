import os
import json
import pandas as pd
from typing import List, Dict, Optional
import asyncio

from app.exceptions import ToolError
from app.tool.base import BaseTool, ToolResult
from app.logger import logger

class ExcelSplitSubtableTool(BaseTool):
    """
    A tool for splitting CSV files into subtables based on column categories.
    
    This tool splits a cleaned CSV file into multiple subtables based on column categories
    defined in a JSON metadata file. Each subtable contains columns of a specific category
    plus user-specified common columns.
    
    Common columns typically include metadata and context information such as:
    - Device identifiers (device_id, device_name, hardware_model)
    - Software information (app_version, build_number, OS_version)
    - Test context (test_date, test_id, test_duration, test_environment)
    - User/session information (user_id, session_id)
    
    These common columns provide necessary context for all performance metrics and
    ensure that each subtable can be analyzed independently while maintaining
    the connection to the specific test conditions.
    """
    
    name: str = "excel_split_subtable_tool"
    description: str = "Split a CSV file into multiple subtables based on column categories from a JSON metadata file."
    parameters: dict = {
        "type": "object",
        "properties": {
            "csv_file_path": {
                "type": "string",
                "description": "Path to the cleaned CSV file to split into subtables",
            },
            "json_metadata_path": {
                "type": "string",
                "description": "Path to the JSON metadata file containing column categories",
            },
            "common_columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of column names to include in every subtable. These are typically columns containing metadata or context information such as device_id, version, build_number, test_date, platform, hardware_model, OS_version, or other identifiers that provide context for all performance metrics and should be present in every subtable for proper analysis.",
            },
            "output_dir": {
                "type": "string",
                "description": "Optional directory to save the subtables. If not provided, creates a 'splitsubtable' directory in the same location as the CSV file.",
            },
        },
        "required": ["csv_file_path", "json_metadata_path", "common_columns"],
    }
    
    async def execute(
        self,
        csv_file_path: str,
        json_metadata_path: str,
        common_columns: List[str],
        output_dir: Optional[str] = None,
    ) -> ToolResult | ToolError:
        """
        Split a CSV file into subtables based on column categories.
        
        Args:
            csv_file_path: Path to the cleaned CSV file.
            json_metadata_path: Path to the JSON metadata file containing column categories.
            common_columns: List of column names to include in every subtable. These typically
                           include metadata columns like device_id, version, build_number,
                           test_date, platform, hardware specs, and other contextual information
                           that's relevant for all performance metrics.
            output_dir: Directory to save the subtables.
            
        Returns:
            A string summarizing the results of the operation.
        """
        try:
            output_files = await self._split_csv_into_subtables(
                csv_file_path, json_metadata_path, common_columns, output_dir
            )
            
            # Prepare summary result
            num_subtables = len(output_files)
            if num_subtables == 0:
                return ToolResult(
                    output="No subtables were created. Check that the CSV and JSON metadata files contain valid data."
                )
            
            categories = list(output_files.keys())
            summary = f"Successfully split CSV into {num_subtables} subtables by category:\n"
            
            for category, filepath in output_files.items():
                summary += f"- {category}: {filepath}\n"
                
            return ToolResult(output=summary)
            
        except Exception as e:
            logger.error(f"Error splitting CSV into subtables: {e}")
            error = f"Failed to split CSV into subtables: {str(e)}"
            return ToolError(error)
        
    async def _split_csv_into_subtables(
        self,
        csv_file_path: str,
        json_metadata_path: str,
        common_columns: List[str],
        output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Split a cleaned CSV file into multiple subtables based on column categories.
        
        Args:
            csv_file_path: Path to the cleaned CSV file.
            json_metadata_path: Path to the JSON metadata file containing column categories.
            common_columns: List of column names to include in every subtable. These typically
                           include metadata columns like device_id, version, build_number,
                           test_date, platform, hardware specs, and other contextual information
                           that's relevant for all performance metrics.
            output_dir: Directory to save the subtables. If None, creates a 'splitsubtable' 
                        directory in the same location as the CSV file.
        
        Returns:
            Dictionary mapping category names to output file paths.
        """
        # Validate inputs
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
        
        if not os.path.exists(json_metadata_path):
            raise FileNotFoundError(f"JSON metadata file not found: {json_metadata_path}")
        
        # Read the CSV file - run in a thread to avoid blocking
        try:
            # Use run_in_executor to handle file I/O asynchronously
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, lambda: pd.read_csv(csv_file_path))
            logger.info(f"Successfully read CSV file with {len(df)} rows and {len(df.columns)} columns")
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            raise
        
        # Read the JSON metadata - run in a thread to avoid blocking
        try:
            def read_json():
                with open(json_metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            metadata = await loop.run_in_executor(None, read_json)
            logger.info(f"Successfully read JSON metadata file")
        except Exception as e:
            logger.error(f"Error reading JSON metadata file: {e}")
            raise
        
        # Verify that common columns exist in the CSV
        missing_columns = [col for col in common_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The following common columns are missing from the CSV: {missing_columns}")
            
        # Log information about the common columns found
        logger.info(f"Using {len(common_columns)} common columns: {', '.join(common_columns)}")
        logger.info("These common columns will be included in all subtables to maintain context")
        
        # Group columns by basic_category
        columns_by_category = {}
        
        # Extract column information from metadata
        column_info = metadata.get("columns_analysis", {})
        
        for column_name, column_data in column_info.items():
            if column_name in df.columns:
                category = column_data.get("basic_category")
                if category:
                    if category not in columns_by_category:
                        columns_by_category[category] = []
                    columns_by_category[category].append(column_name)
        
        logger.info(f"Found {len(columns_by_category)} categories: {', '.join(columns_by_category.keys())}")
        
        # Create output directory if it doesn't exist
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(csv_file_path), "splitsubtable")
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Create and save subtables
        output_files = {}
        
        for category, category_columns in columns_by_category.items():
            # Create subtable with category columns + common columns
            subtable_columns = list(set(common_columns + category_columns))
            
            # Ensure all columns exist in the DataFrame
            valid_columns = [col for col in subtable_columns if col in df.columns]
            
            if len(valid_columns) > 0:
                subtable = df[valid_columns]
                
                # Create output file path
                output_file = os.path.join(output_dir, f"{category}.csv")
                
                # Save subtable to CSV - run in a thread to avoid blocking
                await loop.run_in_executor(
                    None, 
                    lambda: subtable.to_csv(output_file, index=False)
                )
                # logger.info(f"Created subtable for category '{category}' with {len(valid_columns)} columns, saved to {output_file}")
                
                output_files[category] = output_file
            else:
                logger.warning(f"No valid columns found for category '{category}', skipping")
        
        return output_files
            
# Command-line interface for testing the tool directly
if __name__ == "__main__":
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(description="Split a CSV file into subtables based on column categories")
    parser.add_argument("csv_file", help="Path to the cleaned CSV file")
    parser.add_argument("json_metadata", help="Path to the JSON metadata file")
    parser.add_argument("--common-columns", nargs="+", required=True, 
                       help="Column names to include in every subtable (e.g., device_id, app_version, test_date)")
    parser.add_argument("--output-dir", help="Directory to save the subtables")
    
    args = parser.parse_args()
    
    async def run_tool():
        tool = ExcelSplitSubtableTool()
        result = await tool.execute(args.csv_file, args.json_metadata, args.common_columns, args.output_dir)
        print(result)
    
    asyncio.run(run_tool())