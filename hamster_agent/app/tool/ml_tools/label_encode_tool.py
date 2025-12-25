import pandas as pd
import os
import warnings
from typing import Union, List, Optional

from sklearn.preprocessing import LabelEncoder
from app.tool.base import BaseTool, ToolResult
from app.exceptions import ToolError
from app.logger import logger

def label_encode(data: pd.DataFrame, 
                 columns: Union[str, List[str]]) -> pd.DataFrame:
    """
    Perform label encoding on specified categorical columns. The resulting columns 
    will follow the format 'original_column_encoded'.

    Args:
        data (pd.DataFrame): The input DataFrame.
        columns (str or List[str]): Column label or list of column labels to encode.

    Returns:
        pd.DataFrame: DataFrame with label encoded columns.

    Example:
        >>> df = pd.DataFrame({'fruit': ['apple', 'banana', 'apple', 'cherry']})
        >>> label_encode(df, 'fruit')
           fruit  fruit_encoded
        0  apple              0
        1  banana             1
        2  apple              0
        3  cherry             2
    """
    if isinstance(columns, str):
        columns = [columns]

    result = data.copy()

    # Check if specified columns exist
    missing_columns = set(columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} not found in the DataFrame.")

    for col in columns:
        col_data = data[col]

        # Only encode columns of type 'category' or 'object'
        if pd.api.types.is_categorical_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
            encoder = LabelEncoder()
            encoded_col_name = f"{col}_encoded"
            result[encoded_col_name] = encoder.fit_transform(col_data.astype(str))
        else:
            warnings.warn(f"Column '{col}' is {col_data.dtype}, which is not categorical. Skipping encoding.")

    return result


class LabelEncodeTool(BaseTool):
    """
    Tool for performing label encoding on categorical columns in CSV/Excel files.
    Label encoding converts categorical text values to numeric labels (0, 1, 2, ...).
    Useful for ordinal categorical data where the order matters.
    """
    
    name: str = "label_encode"
    description: str = (
        "Perform label encoding on specified categorical columns. Label encoding converts categorical "
        "text values to numeric labels (0, 1, 2, ...) in alphabetical order. The resulting columns "
        "will follow the format 'original_column_encoded'. This encoding is most suitable for ordinal "
        "categorical data where there is a meaningful order among categories. For nominal categorical "
        "data with no inherent order, consider using one-hot encoding instead to avoid implying "
        "false ordinal relationships."
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
                "description": "Column label or list of column labels to encode. Should contain categorical (text) data that can be meaningfully ordered.",
                "items": {"type": "string"}
            },
            "drop_original": {
                "type": "boolean",
                "description": "If True, drop the original categorical columns after encoding. If False, keep both original and encoded columns.",
                "default": False
            },
            "output_file_path": {
                "type": "string",
                "description": "Path where the processed data should be saved. If not provided, will create 'label_encoded.csv' in the same directory as the input file."
            }
        },
        "required": ["file_path", "columns"]
    }

    async def execute(
        self,
        file_path: str,
        columns: Union[str, List[str]],
        drop_original: bool = False,
        output_file_path: Optional[str] = None,
    ) -> ToolResult:
        """
        Execute label encoding for specified categorical columns.
        
        Label encoding assigns numeric labels (0, 1, 2, ...) to categorical values in alphabetical order.
        Best suited for ordinal categorical data where the order of categories is meaningful.
        For nominal categorical data, consider one-hot encoding to avoid implying false relationships.
        
        Args:
            file_path: Path to the input data file (CSV or Excel)
            columns: Column name(s) to encode (single string or list)
            drop_original: Whether to drop original columns after encoding
            output_file_path: Optional output file path (defaults to create new file)
            
        Returns:
            ToolResult with detailed encoding summary including label mappings,
            encoding statistics, and warnings for non-categorical columns
        """
        try:
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
            encoding_info = {}
            warnings_list = []
            skipped_columns = []
            
            # Analyze each column before encoding
            result = data.copy()
            
            for column in columns:
                col_data = data[column]
                unique_values = col_data.dropna().unique()
                unique_count = len(unique_values)
                null_count = col_data.isnull().sum()
                
                # Check if column is categorical or object type
                if pd.api.types.is_categorical_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
                    # Perform label encoding
                    encoder = LabelEncoder()
                    encoded_col_name = f"{column}_encoded"
                    
                    # Handle null values by converting to string
                    col_data_str = col_data.astype(str)
                    encoded_values = encoder.fit_transform(col_data_str)
                    result[encoded_col_name] = encoded_values
                    
                    # Create label mapping for documentation
                    label_mapping = dict(zip(encoder.classes_, range(len(encoder.classes_))))
                    
                    encoding_info[column] = {
                        'unique_categories': int(unique_count),
                        'null_values': int(null_count),
                        'encoded_column': encoded_col_name,
                        'label_mapping': label_mapping,
                        'encoding_range': f"0 to {len(encoder.classes_) - 1}"
                    }
                    
                    logger.info(f"Label encoded column '{column}' -> '{encoded_col_name}' with {len(encoder.classes_)} unique labels")
                    
                else:
                    # Skip non-categorical columns with warning
                    warning_msg = f"Column '{column}' is {col_data.dtype}, which is not categorical. Skipping encoding."
                    warnings_list.append(warning_msg)
                    skipped_columns.append(column)
                    logger.warning(warning_msg)
            
            # Drop original columns if requested
            if drop_original:
                encoded_columns = [col for col in columns if col not in skipped_columns]
                if encoded_columns:
                    result = result.drop(encoded_columns, axis=1)
            
            # Determine output path
            if output_file_path is None:
                output_file_path = os.path.join(os.path.dirname(file_path), "label_encoded.csv")
            
            # Save processed data
            try:
                if output_file_path.endswith('.csv'):
                    result.to_csv(output_file_path, index=False)
                elif output_file_path.endswith(('.xlsx', '.xls')):
                    result.to_excel(output_file_path, index=False)
                else:
                    # Default to CSV
                    if not output_file_path.endswith('.csv'):
                        output_file_path += '.csv'
                    result.to_csv(output_file_path, index=False)
            except Exception as e:
                raise ToolError(f"Error saving processed data: {str(e)}")
            
            # Generate summary
            final_shape = result.shape
            successfully_encoded = len(encoding_info)
            columns_added = final_shape[1] - original_shape[1] + (successfully_encoded if drop_original else 0)
            
            summary = f"Label encoding completed successfully:\n"
            summary += f"- Input file: {file_path}\n"
            summary += f"- Output file: {output_file_path}\n"
            summary += f"- Requested columns: {', '.join(columns)}\n"
            summary += f"- Successfully encoded: {successfully_encoded} columns\n"
            summary += f"- Skipped columns: {len(skipped_columns)} columns\n"
            summary += f"- Drop original: {drop_original}\n"
            summary += f"- Original data shape: {original_shape}\n"
            summary += f"- Final data shape: {final_shape}\n"
            summary += f"- New encoded columns created: {columns_added}\n"
            
            if warnings_list:
                summary += f"\nWARNINGS:\n"
                for warning in warnings_list:
                    summary += f"- {warning}\n"
            
            if skipped_columns:
                summary += f"\nSkipped columns (non-categorical): {', '.join(skipped_columns)}\n"
            
            summary += f"\nENCODING CHARACTERISTICS:\n"
            summary += f"- Assigns numeric labels (0, 1, 2, ...) to categorical values\n"
            summary += f"- Labels are assigned in alphabetical order of category names\n"
            summary += f"- Best for ordinal categorical data with meaningful order\n"
            summary += f"- For nominal data, consider one-hot encoding to avoid false ordinal implications\n"
            summary += f"- Creates new columns with '_encoded' suffix\n"
            
            if encoding_info:
                summary += "\nColumn-wise encoding details:\n"
                for column, info in encoding_info.items():
                    summary += f"  {column} -> {info['encoded_column']}:\n"
                    summary += f"    - Categories: {info['unique_categories']}\n"
                    summary += f"    - Encoding range: {info['encoding_range']}\n"
                    summary += f"    - Null values: {info['null_values']}\n"
                    
                    # Show label mapping (limit to first 10 for readability)
                    mapping_items = list(info['label_mapping'].items())
                    sample_mapping = mapping_items[:10]
                    summary += f"    - Label mapping: {dict(sample_mapping)}\n"
                    if len(mapping_items) > 10:
                        summary += f"    - ... and {len(mapping_items) - 10} more mappings\n"
            
            return ToolResult(output=summary)
            
        except ToolError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            error_msg = f"Unexpected error in label encoding: {str(e)}"
            logger.error(error_msg)
            return ToolResult(error=error_msg)


# Usage example and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_tool():
        tool = LabelEncodeTool()
        
        # Create sample data with categorical columns
        sample_data = pd.DataFrame({
            'fruit': ['apple', 'banana', 'cherry', 'apple', 'banana'],
            'size': ['small', 'large', 'medium', 'large', 'small'],
            'quality': ['good', 'excellent', 'fair', 'good', 'excellent'],
            'price': [1.5, 2.5, 1.8, 1.6, 2.3],
            'quantity': [10, 5, 8, 12, 6]
        })
        
        # Save sample data
        sample_data.to_csv('test_categorical_label.csv', index=False)
        
        # Test the tool
        result = await tool.execute(
            file_path='test_categorical_label.csv',
            columns=['fruit', 'size', 'quality'],
            drop_original=False,
            output_file_path='test_categorical_label_encoded.csv'
        )
        
        print(result.output if result.output else result.error)
        
        # Test with non-categorical column (should show warning)
        print("\n" + "="*50)
        print("Testing with non-categorical column:")
        
        mixed_result = await tool.execute(
            file_path='test_categorical_label.csv',
            columns=['fruit', 'price'],  # price is numeric, should be skipped
            drop_original=False
        )
        
        print(mixed_result.output if mixed_result.output else mixed_result.error)
    
    asyncio.run(test_tool())