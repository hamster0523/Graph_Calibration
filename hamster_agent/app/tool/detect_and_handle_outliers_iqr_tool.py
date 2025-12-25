import pandas as pd
from typing import Union, List, Optional
import os

from app.tool.base import BaseTool, ToolResult
from app.exceptions import ToolError
from app.logger import logger

def detect_and_handle_outliers_iqr(data: pd.DataFrame, columns: Union[str, List[str]], factor: float = 1.5, method: str = 'clip') -> pd.DataFrame:
    """
    Detect and handle outliers in specified columns using the Interquartile Range (IQR) method.

    Args:
        data (pd.DataFrame): The input DataFrame.
        columns (str or List[str]): The name(s) of the column(s) to check for outliers.
        factor (float, optional): The IQR factor to determine the outlier threshold. Defaults to 1.5.
        method (str, optional): The method to handle outliers. Options: 'clip', 'remove'. Defaults to 'clip'.

    Returns:
        pd.DataFrame: The DataFrame with outliers handled.
    """
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"Column '{column}' must be numeric.")

        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        if method == 'clip':
            data[column] = data[column].clip(lower_bound, upper_bound)
        elif method == 'remove':
            data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
        else:
            raise ValueError("Invalid method. Choose from 'clip' or 'remove'.")

    return data

class DetectAndHandleOutliersIqrTool(BaseTool):
    """
    Tool for detecting and handling outliers in numerical columns using the Interquartile Range (IQR) method.
    Useful for identifying and managing extreme values without assuming a specific distribution of the data.
    """
    
    name: str = "detect_and_handle_outliers_iqr"
    description: str = (
        "Detect and handle outliers in specified columns using the Interquartile Range (IQR) method. "
        "This tool is useful for identifying and managing extreme values in numerical features without "
        "assuming a specific distribution of the data. Particularly effective when the data distribution "
        "is unknown, non-normal, or when the dataset is small or contains extreme outliers. Less sensitive "
        "to extreme outliers compared to the Z-score method but may be less precise for normally distributed data."
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
            "factor": {
                "type": "number",
                "description": "The IQR factor to determine the outlier threshold. Typically 1.5 for outliers or 3.0 for extreme outliers. Higher values detect only more extreme outliers.",
                "default": 1.5,
                "minimum": 0.1,
                "maximum": 10.0
            },
            "method": {
                "type": "string",
                "description": "The method to handle outliers. 'clip': cap outliers to threshold values, 'remove': delete rows containing outliers (not recommended for test sets).",
                "enum": ["clip", "remove"],
                "default": "clip"
            },
            "return_mask": {
                "type": "boolean",
                "description": "If True, return a boolean mask indicating outliers instead of removing them. Useful for analysis and visualization.",
                "default": False
            },
            "output_file_path": {
                "type": "string",
                "description": "Path where the processed data should be saved. If not provided, will save to 'iqr_processed.csv' in the same directory as the input file."
            }
        },
        "required": ["file_path", "columns"]
    }

    async def execute(
        self,
        file_path: str,
        columns: Union[str, List[str]],
        factor: float = 1.5,
        method: str = "clip",
        return_mask: bool = False,
        output_file_path: Optional[str] = None,
    ) -> ToolResult:
        """
        Execute outlier detection and handling using IQR method.
        
        This method does not assume any specific data distribution and is less sensitive
        to extreme outliers compared to the Z-score method. The choice of factor affects
        the range of what is considered an outlier.
        
        Args:
            file_path: Path to the input data file (CSV or Excel)
            columns: Column name(s) to process for outliers (single string or list)
            factor: IQR factor for outlier threshold (1.5 for outliers, 3.0 for extreme outliers)
            method: Method to handle outliers ('clip' or 'remove')
            return_mask: If True, return boolean mask instead of processing data
            output_file_path: Optional output file path (defaults to 'iqr_processed.csv' in input directory)
            
        Returns:
            ToolResult with detailed outlier detection summary including IQR statistics,
            outlier counts, and processing results
        """
        try:
            # Validate inputs
            if factor <= 0:
                raise ToolError("Factor must be a positive number")
            
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
            outlier_masks = {}
            
            # Process each column and collect statistics
            for column in columns:
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                # Create outlier mask
                outlier_mask = (data[column] < lower_bound) | (data[column] > upper_bound)
                outliers_count = outlier_mask.sum()
                
                outliers_info[column] = {
                    'outliers_detected': int(outliers_count),
                    'Q1': float(Q1),
                    'Q3': float(Q3),
                    'IQR': float(IQR),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'factor': factor,
                    'total_values': len(data[column]),
                    'outlier_percentage': float(outliers_count / len(data[column]) * 100)
                }
                
                outlier_masks[column] = outlier_mask
                
                if outliers_count > 0:
                    logger.info(f"Found {outliers_count} outliers in column '{column}' (factor: {factor})")
            
            # If return_mask is True, save mask information and return
            if return_mask:
                mask_summary = f"Outlier detection mask generated using IQR method:\n"
                mask_summary += f"- Input file: {file_path}\n"
                mask_summary += f"- Processed columns: {', '.join(columns)}\n"
                mask_summary += f"- IQR factor: {factor}\n"
                mask_summary += f"- Data shape: {original_shape}\n"
                
                mask_summary += "\nOutlier detection results:\n"
                for column, info in outliers_info.items():
                    mask_summary += f"  {column}:\n"
                    mask_summary += f"    - Outliers detected: {info['outliers_detected']} ({info['outlier_percentage']:.2f}%)\n"
                    mask_summary += f"    - Q1: {info['Q1']:.4f}, Q3: {info['Q3']:.4f}, IQR: {info['IQR']:.4f}\n"
                    mask_summary += f"    - Bounds: [{info['lower_bound']:.4f}, {info['upper_bound']:.4f}]\n"
                
                # Note: In a real implementation, you might want to save the mask to a file
                mask_summary += "\nNote: Boolean masks generated but not saved to file (return_mask=True)\n"
                
                return ToolResult(output=mask_summary)
            
            # Apply outlier handling
            processed_data = detect_and_handle_outliers_iqr(
                data=data.copy(),
                columns=columns,
                factor=factor,
                method=method
            )
            
            # Determine output path
            if output_file_path is None:
                output_file_path = os.path.join(file_path, "iqr_processed.csv")
            
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
            
            summary = f"Outlier detection and handling completed using IQR method:\n"
            summary += f"- Input file: {file_path}\n"
            summary += f"- Output file: {output_file_path}\n"
            summary += f"- Processed columns: {', '.join(columns)}\n"
            summary += f"- Method: {method}\n"
            summary += f"- IQR factor: {factor} ({'outliers' if factor == 1.5 else 'extreme outliers' if factor == 3.0 else 'custom threshold'})\n"
            summary += f"- Original data shape: {original_shape}\n"
            summary += f"- Final data shape: {final_shape}\n"
            summary += f"- Total outliers detected: {total_outliers}\n"
            
            if method == "remove":
                rows_removed = original_shape[0] - final_shape[0]
                summary += f"- Rows removed: {rows_removed}\n"
                summary += f"- WARNING: Using 'remove' method deletes data entries - not recommended for test sets\n"
            
            summary += f"\nMETHOD CHARACTERISTICS:\n"
            summary += f"- Does not assume any specific data distribution\n"
            summary += f"- Less sensitive to extreme outliers compared to Z-score method\n"
            summary += f"- May be less precise for normally distributed data\n"
            
            summary += "\nColumn-wise IQR outlier statistics:\n"
            for column, info in outliers_info.items():
                summary += f"  {column}:\n"
                summary += f"    - Outliers: {info['outliers_detected']} ({info['outlier_percentage']:.2f}%)\n"
                summary += f"    - Q1: {info['Q1']:.4f}, Q3: {info['Q3']:.4f}, IQR: {info['IQR']:.4f}\n"
                summary += f"    - Outlier bounds: [{info['lower_bound']:.4f}, {info['upper_bound']:.4f}]\n"
            
            return ToolResult(output=summary)
            
        except ToolError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            error_msg = f"Unexpected error in IQR outlier detection: {str(e)}"
            logger.error(error_msg)
            return ToolResult(error=error_msg)


# Usage example and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_tool():
        tool = DetectAndHandleOutliersIqrTool()
        
        # Test with sample data
        import numpy as np
        
        # Create sample data with outliers
        np.random.seed(42)
        normal_data = np.random.normal(50, 10, 1000)
        # Add some outliers
        outliers = [100, 0, 120, -20, 150]  # Clear outliers
        data_with_outliers = np.concatenate([normal_data, outliers])
        
        sample_df = pd.DataFrame({
            'value1': data_with_outliers,
            'value2': np.random.exponential(20, len(data_with_outliers)),  # Non-normal distribution
            'category': ['A'] * len(data_with_outliers)
        })
        
        # Save sample data
        sample_df.to_csv('test_iqr_outliers.csv', index=False)
        
        # Test the tool
        result = await tool.execute(
            file_path='test_iqr_outliers.csv',
            columns=['value1', 'value2'],
            factor=1.5,
            method='clip',
            return_mask=False,
            output_file_path='test_iqr_outliers_processed.csv'
        )
        
        print(result.output if result.output else result.error)
        
        # Test mask generation
        print("\n" + "="*50)
        print("Testing mask generation:")
        
        mask_result = await tool.execute(
            file_path='test_iqr_outliers.csv',
            columns=['value1'],
            factor=1.5,
            return_mask=True
        )
        
        print(mask_result.output if mask_result.output else mask_result.error)
    
    asyncio.run(test_tool())