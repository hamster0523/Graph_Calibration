import pandas as pd
from typing import Union, List, Optional
import os

from app.tool.base import BaseTool, ToolResult
from app.exceptions import ToolError
from app.logger import logger

def detect_and_handle_outliers_zscore(data: pd.DataFrame, columns: Union[str, List[str]], threshold: float = 3.0, method: str = 'clip') -> pd.DataFrame:
    """
    Detect and handle outliers in specified columns using the Z-score method.
    Args:
        data (pd.DataFrame): The input DataFrame.
        columns (str or List[str]): The name(s) of the column(s) to check for outliers.
        threshold (float, optional): The Z-score threshold to identify outliers. Defaults to 3.0.
        method (str, optional): The method to handle outliers. Options: 'clip', 'remove'. Defaults to 'clip'.
    Returns:
        pd.DataFrame: The DataFrame with outliers handled.
    """
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"Column '{column}' must be numeric.")

        mean = data[column].mean()
        std = data[column].std()
        z_scores = (data[column] - mean) / std

        if method == 'clip':
            # Define the bounds
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            # Apply clipping only to values exceeding the threshold
            data.loc[z_scores > threshold, column] = upper_bound
            data.loc[z_scores < -threshold, column] = lower_bound
        elif method == 'remove':
            data = data[abs(z_scores) <= threshold]
        else:
            raise ValueError("Invalid method. Choose from 'clip' or 'remove'.")

    return data


class DetectAndHandleOutliersZscoreTool(BaseTool):
    """
    Tool for detecting and handling outliers in numerical columns using the Z-score method.
    """
    
    name: str = "detect_and_handle_outliers_zscore"
    description: str = (
        "Detect and handle outliers in specified columns using the Z-score method. "
        "This tool is useful for identifying and managing extreme values in numerical features "
        "based on their distance from the mean in terms of standard deviations."
    )
    
    parameters: dict = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the CSV/Excel file containing the data to process"
            },
            "columns": {
                "type": ["string", "array"],
                "description": "The name(s) of the column(s) to check for outliers. Can be a single column name or a list of column names.",
                "items": {"type": "string"}
            },
            "threshold": {
                "type": "number",
                "description": "The Z-score threshold to identify outliers. Values with absolute Z-scores above this threshold are considered outliers. Typically 3.0 or 2.5.",
                "default": 3.0,
                "minimum": 0.1,
                "maximum": 10.0
            },
            "method": {
                "type": "string",
                "description": "The method to handle outliers.",
                "enum": ["clip", "remove"],
                "default": "clip"
            },
            "output_file_path": {
                "type": "string",
                "description": "Path where the processed data should be saved. If not provided, will save to 'zscore_processed.csv' in the same directory as the input file."
            }
        },
        "required": ["file_path", "columns"]
    }

    async def execute(
        self,
        file_path: str,
        columns: Union[str, List[str]],
        threshold: float = 3.0,
        method: str = "clip",
        output_file_path: Optional[str] = None,
    ) -> ToolResult:
        """
        Execute outlier detection and handling using Z-score method.
        
        Args:
            file_path: Path to the input data file
            columns: Column name(s) to process for outliers
            threshold: Z-score threshold for outlier detection
            method: Method to handle outliers ('clip' or 'remove')
            output_file_path: Optional output file path (defaults to 'zscore_processed.csv' in input directory)
            
        Returns:
            ToolResult with processing summary
        """
        try:
            # Validate inputs
            if threshold <= 0:
                raise ToolError("Threshold must be a positive number")
            
            if method not in ["clip", "remove"]:
                raise ToolError("Method must be either 'clip' or 'remove'")
            
            # Load data
            try:
                if file_path.endswith('.csv'):
                    data = pd.read_csv(file_path)
                elif file_path.endswith(('.xlsx', '.xls')):
                    data = pd.read_excel(file_path)
                else:
                    raise ToolError("Unsupported file format. Please use CSV or Excel files.")
            except FileNotFoundError:
                raise ToolError(f"File not found: {file_path}")
            except Exception as e:
                raise ToolError(f"Error loading file: {str(e)}")
            
            # Convert single column to list
            if isinstance(columns, str):
                columns = [columns]
            
            # Validate columns exist and are numeric
            missing_columns = [col for col in columns if col not in data.columns]
            if missing_columns:
                raise ToolError(f"Columns not found in data: {missing_columns}")
            
            non_numeric_columns = [col for col in columns if not pd.api.types.is_numeric_dtype(data[col])]
            if non_numeric_columns:
                raise ToolError(f"Non-numeric columns cannot be processed: {non_numeric_columns}")
            
            # Store original data info
            original_shape = data.shape
            outliers_info = {}
            
            # Process each column and collect statistics
            for column in columns:
                mean = data[column].mean()
                std = data[column].std()
                z_scores = (data[column] - mean) / std
                
                # Count outliers before processing
                outliers_mask = abs(z_scores) > threshold
                outliers_count = outliers_mask.sum()
                outliers_info[column] = {
                    'outliers_detected': int(outliers_count),
                    'mean': float(mean),
                    'std': float(std),
                    'threshold': threshold
                }
                
                if outliers_count > 0:
                    logger.info(f"Found {outliers_count} outliers in column '{column}' (threshold: {threshold})")
            
            # Apply outlier handling
            processed_data = detect_and_handle_outliers_zscore(
                data=data.copy(),
                columns=columns,
                threshold=threshold,
                method=method
            )
            
            # Determine output path
            if output_file_path is None:
                output_file_path = os.path.join(file_path, "zscore_processed.csv")
            
            # Save processed data
            try:
                if output_file_path.endswith('.csv'):
                    processed_data.to_csv(output_file_path, index=False)
                elif output_file_path.endswith(('.xlsx', '.xls')):
                    processed_data.to_excel(output_file_path, index=False)
                else:
                    # Default to CSV
                    if not output_file_path.endswith('.csv'):
                        output_file_path += '.csv'
                    processed_data.to_csv(output_file_path, index=False)
            except Exception as e:
                raise ToolError(f"Error saving processed data: {str(e)}")
            
            # Generate summary
            final_shape = processed_data.shape
            total_outliers = sum(info['outliers_detected'] for info in outliers_info.values())
            
            summary = f"Outlier detection and handling completed using Z-score method:\n"
            summary += f"- Input file: {file_path}\n"
            summary += f"- Output file: {output_file_path}\n"
            summary += f"- Processed columns: {', '.join(columns)}\n"
            summary += f"- Method: {method}\n"
            summary += f"- Z-score threshold: {threshold}\n"
            summary += f"- Original data shape: {original_shape}\n"
            summary += f"- Final data shape: {final_shape}\n"
            summary += f"- Total outliers detected: {total_outliers}\n"
            
            if method == "remove":
                rows_removed = original_shape[0] - final_shape[0]
                summary += f"- Rows removed: {rows_removed}\n"
            
            summary += "\nColumn-wise outlier statistics:\n"
            for column, info in outliers_info.items():
                summary += f"  {column}: {info['outliers_detected']} outliers (mean: {info['mean']:.4f}, std: {info['std']:.4f})\n"
            
            return ToolResult(output=summary)
            
        except ToolError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            error_msg = f"Unexpected error in outlier detection: {str(e)}"
            logger.error(error_msg)
            return ToolResult(error=error_msg)


# Usage example and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_tool():
        tool = DetectAndHandleOutliersZscoreTool()
        
        # Test with sample data
        import numpy as np
        
        # Create sample data with outliers
        np.random.seed(42)
        normal_data = np.random.normal(50, 10, 1000)
        outliers = [150, -50, 200, -100]  # Extreme outliers
        data_with_outliers = np.concatenate([normal_data, outliers])
        
        sample_df = pd.DataFrame({
            'value1': data_with_outliers,
            'value2': np.random.normal(30, 5, len(data_with_outliers)),
            'category': ['A'] * len(data_with_outliers)
        })
        
        # Save sample data
        sample_df.to_csv('test_outliers.csv', index=False)
        
        # Test the tool
        result = await tool.execute(
            file_path='test_outliers.csv',
            columns=['value1', 'value2'],
            threshold=3.0,
            method='clip',
            output_file_path='test_outliers_processed.csv'
        )
        
        print(result.output if result.output else result.error)
    
    asyncio.run(test_tool())