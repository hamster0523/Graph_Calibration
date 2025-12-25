import pandas as pd
from typing import Union, List, Optional
import os

from app.tool.base import BaseTool, ToolResult
from app.exceptions import ToolError
from app.logger import logger


def format_datetime(data: pd.DataFrame, columns: Union[str, List[str]], format: str = '%Y-%m-%d %H:%M:%S', errors: str = 'coerce') -> pd.DataFrame:
    """
    Format datetime columns in a DataFrame to a specified format.

    Args:
        data (pd.DataFrame): The input DataFrame.
        columns (str or List[str]): Column label or sequence of labels to format.
        format (str, optional): The desired output format for datetime. 
                                Defaults to '%Y-%m-%d %H:%M:%S'.
        errors (str, optional): How to handle parsing errors. 
                                Options: 'raise', 'coerce', 'ignore'. Defaults to 'coerce'.

    Returns:
        pd.DataFrame: The DataFrame with formatted datetime columns.
    """
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")

        # First, ensure the column is in datetime format
        data[column] = pd.to_datetime(data[column], errors=errors)

        # Then, format the datetime column
        data[column] = data[column].dt.strftime(format)

    return data


class FormatDatetimeTool(BaseTool):
    """
    Tool for formatting datetime columns in CSV/Excel files to a specified format.
    Useful for datetime standardization, data preprocessing, and ensuring consistent datetime formats.
    """
    
    name: str = "format_datetime"
    description: str = (
        "Format datetime columns in a DataFrame to a specified format. This tool is useful for "
        "standardizing date and time representations across your dataset, datetime standardization, "
        "data preprocessing, and ensuring consistent datetime formats for analysis and reporting. "
        "The method first converts specified columns to datetime using pd.to_datetime before formatting. "
        "Supports various error handling modes: 'coerce' sets invalid parsing to NaT, 'ignore' returns "
        "original input for invalid parsing, and 'raise' throws exceptions for parsing errors."
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
                "description": "The name(s) of the column(s) to format as datetime. Can be a single column name or a list of column names.",
                "items": {"type": "string"}
            },
            "format": {
                "type": "string",
                "description": "The desired output format for datetime using Python's strftime format codes. Examples: '%Y-%m-%d %H:%M:%S' (2023-01-15 14:30:00), '%Y-%m-%d' (2023-01-15), '%d/%m/%Y' (15/01/2023), '%B %d, %Y' (January 15, 2023). Ensure the format matches your expected datetime structure.",
                "default": "%Y-%m-%d %H:%M:%S"
            },
            "errors": {
                "type": "string",
                "description": "How to handle parsing errors when converting to datetime. 'coerce': invalid parsing will be set to NaT (Not a Time), 'ignore': invalid parsing will return the original input, 'raise': invalid parsing will raise an exception.",
                "enum": ["raise", "coerce", "ignore"],
                "default": "coerce"
            },
            "output_file_path": {
                "type": "string",
                "description": "Path where the processed data should be saved. If not provided, will save to 'after_formatting.csv' in the same directory as the input file."
            }
        },
        "required": ["file_path", "columns"]
    }

    async def execute(
        self,
        file_path: str,
        columns: Union[str, List[str]],
        format: str = "%Y-%m-%d %H:%M:%S",
        errors: str = "coerce",
        output_file_path: Optional[str] = None,
    ) -> ToolResult:
        """
        Execute datetime formatting for specified columns.
        
        This method first converts the specified columns to datetime using pd.to_datetime 
        before applying the desired format. Consider the impact of datetime formatting 
        on your data analysis and model performance.
        
        Args:
            file_path: Path to the input data file (CSV or Excel)
            columns: Column name(s) to format as datetime (single string or list)
            format: Desired datetime format using Python's strftime/strptime format codes
            errors: Error handling mode - 'raise', 'coerce', or 'ignore'
            output_file_path: Optional output file path (defaults to 'after_formatting.csv' in input directory)
            
        Returns:
            ToolResult with detailed formatting summary including before/after samples,
            null value changes, and conversion statistics
        """
        try:
            # Validate inputs
            if errors not in ["raise", "coerce", "ignore"]:
                raise ToolError("errors parameter must be one of: 'raise', 'coerce', 'ignore'")
            
            # Validate datetime format
            try:
                import datetime
                datetime.datetime.now().strftime(format)
            except ValueError as e:
                raise ToolError(f"Invalid datetime format '{format}': {str(e)}. Please use valid Python strftime format codes. Examples: '%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d/%m/%Y', '%B %d, %Y'")
            
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
            
            # Validate columns exist
            missing_columns = [col for col in columns if col not in data.columns]
            if missing_columns:
                raise ToolError(f"Columns not found in data: {missing_columns}")
            
            # Store original data info
            original_shape = data.shape
            formatting_info = {}
            
            # Process each column and collect statistics
            for column in columns:
                original_sample = data[column].head(3).tolist()
                null_count_before = data[column].isnull().sum()
                
                try:
                    # Apply datetime formatting
                    data = format_datetime(
                        data=data,
                        columns=[column],
                        format=format,
                        errors=errors
                    )
                    
                    null_count_after = data[column].isnull().sum()
                    formatted_sample = data[column].head(3).tolist()
                    
                    formatting_info[column] = {
                        'original_sample': original_sample,
                        'formatted_sample': formatted_sample,
                        'null_before': int(null_count_before),
                        'null_after': int(null_count_after),
                        'conversion_issues': int(null_count_after - null_count_before)
                    }
                    
                    logger.info(f"Formatted column '{column}' to format '{format}'")
                    
                except Exception as e:
                    if errors == "raise":
                        raise ToolError(f"Error formatting column '{column}': {str(e)}")
                    else:
                        logger.warning(f"Warning formatting column '{column}': {str(e)}")
                        formatting_info[column] = {
                            'error': str(e),
                            'status': 'failed'
                        }
            
            # Determine output path
            if output_file_path is None:
                output_file_path = os.path.join(file_path, "after_formatting.csv")
            
            # Save processed data
            try:
                if output_file_path.endswith('.csv'):
                    data.to_csv(output_file_path, index=False)
                elif output_file_path.endswith(('.xlsx', '.xls')):
                    data.to_excel(output_file_path, index=False)
                else:
                    # Default to CSV
                    if not output_file_path.endswith('.csv'):
                        output_file_path += '.csv'
                    data.to_csv(output_file_path, index=False)
            except Exception as e:
                raise ToolError(f"Error saving processed data: {str(e)}")
            
            # Generate summary
            final_shape = data.shape
            successful_columns = [col for col, info in formatting_info.items() if 'error' not in info]
            failed_columns = [col for col, info in formatting_info.items() if 'error' in info]
            
            summary = f"Datetime formatting completed successfully:\n"
            summary += f"- Input file: {file_path}\n"
            summary += f"- Output file: {output_file_path}\n"
            summary += f"- Target format: {format} (using Python strftime format codes)\n"
            summary += f"- Error handling: {errors}\n"
            summary += f"- Data shape: {original_shape} -> {final_shape}\n"
            summary += f"- Successfully formatted columns: {len(successful_columns)}\n"
            summary += f"- Failed columns: {len(failed_columns)}\n"
            summary += f"\nNOTE: The method first converted columns to datetime using pd.to_datetime before formatting.\n"
            summary += f"Consider the impact of this datetime formatting on your data analysis and model performance.\n"
            
            if successful_columns:
                summary += "\nSuccessfully formatted columns:\n"
                for column in successful_columns:
                    info = formatting_info[column]
                    summary += f"  {column}:\n"
                    summary += f"    - Sample before: {info['original_sample']}\n"
                    summary += f"    - Sample after: {info['formatted_sample']}\n"
                    summary += f"    - Null values before: {info['null_before']}\n"
                    summary += f"    - Null values after: {info['null_after']}\n"
                    if info['conversion_issues'] > 0:
                        summary += f"    - Conversion issues: {info['conversion_issues']} values became NaT (Not a Time) due to invalid parsing\n"
                    else:
                        summary += f"    - No conversion issues detected\n"
            
            if failed_columns:
                summary += "\nFailed columns:\n"
                for column in failed_columns:
                    error = formatting_info[column]['error']
                    summary += f"  {column}: {error}\n"
            
            return ToolResult(output=summary)
            
        except ToolError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            error_msg = f"Unexpected error in datetime formatting: {str(e)}"
            logger.error(error_msg)
            return ToolResult(error=error_msg)


# Usage example and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_tool():
        tool = FormatDatetimeTool()
        
        # Create sample data with various datetime formats
        sample_data = pd.DataFrame({
            'date1': ['2023-01-15 14:30:00', '2023-02-20 09:15:30', '2023-03-10 18:45:15'],
            'date2': ['15/01/2023', '20/02/2023', '10/03/2023'],
            'timestamp': ['1674645000', '1676887730', '1678468515'],
            'value': [100, 200, 300]
        })
        
        # Save sample data
        sample_data.to_csv('test_datetime.csv', index=False)
        
        # Test the tool
        result = await tool.execute(
            file_path='test_datetime.csv',
            columns=['date1', 'date2'],
            format='%Y-%m-%d',
            errors='coerce',
            output_file_path='test_datetime_formatted.csv'
        )
        
        print(result.output if result.output else result.error)
    
    asyncio.run(test_tool())