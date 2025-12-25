import pandas as pd
from typing import Union, List, Optional
import os
from sklearn.preprocessing import OneHotEncoder

from app.tool.base import BaseTool, ToolResult
from app.exceptions import ToolError
from app.logger import logger

def one_hot_encode(data: pd.DataFrame, 
                   columns: Union[str, List[str]], 
                   drop_original: bool = False, 
                   handle_unknown: str = 'error') -> pd.DataFrame:
    """
    Perform one-hot encoding on specified categorical columns. The resulting columns 
    will follow the format 'original_column_value'.

    Args:
        data (pd.DataFrame): The input DataFrame.
        columns (str or List[str]): Column label or list of column labels to encode.
        handle_unknown (str, optional): How to handle unknown categories. Options are 'error' or 'ignore'. Defaults to 'error'.
        drop_original (bool, optional): If True, drop original columns. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with one-hot encoded columns.

    Example:
        >>> df = pd.DataFrame({'color': ['red', 'blue', 'green']})
        >>> one_hot_encode(df, 'color')
           color_blue  color_green  color_red
        0           0            0          1
        1           1            0          0
        2           0            1          0
    """
    if isinstance(columns, str):
        columns = [columns]

    # Check if specified columns exist
    missing_columns = set(columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} not found in the DataFrame.")

    # Perform one-hot encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown=handle_unknown)
    encoded = encoder.fit_transform(data[columns])
    
    # Create new column names in the format 'original_column_value'
    new_columns = [f"{col}_{val}" for col, vals in zip(columns, encoder.categories_) for val in vals]
    
    # Create a new DataFrame with encoded values
    encoded_df = pd.DataFrame(encoded, columns=new_columns, index=data.index)
    
    # Combine with original DataFrame
    result = pd.concat([data, encoded_df], axis=1)
    
    # Drop original columns if specified
    if drop_original:
        result = result.drop(columns, axis=1)
    
    return result


class OneHotEncodeTool(BaseTool):
    """
    Tool for performing one-hot encoding on categorical columns in CSV/Excel files.
    Useful for encoding categorical variables with no ordinal relationship for machine learning models.
    """
    
    name: str = "one_hot_encode"
    description: str = (
        "Perform one-hot encoding on specified categorical columns. The resulting columns will follow "
        "the format 'original_column_value'. This tool is essential for encoding categorical variables "
        "with no ordinal relationship, especially useful for machine learning models that cannot handle "
        "categorical data directly (e.g., linear regression, neural networks). Best for categorical "
        "variables with relatively few unique categories. Creates binary columns for each category, "
        "but may lead to 'curse of dimensionality' with high-cardinality variables."
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
                "description": "Column label or list of column labels to encode. Should contain categorical data with no inherent order among categories.",
                "items": {"type": "string"}
            },
            "handle_unknown": {
                "type": "string",
                "description": "How to handle unknown categories during transform. 'error': raise error for unknown categories, 'ignore': create all-zero row for unknown categories.",
                "enum": ["error", "ignore"],
                "default": "error"
            },
            "drop_original": {
                "type": "boolean",
                "description": "If True, drop the original categorical columns after encoding. If False, keep both original and encoded columns.",
                "default": False
            },
            "max_categories": {
                "type": "integer",
                "description": "Maximum number of unique categories allowed per column to prevent dimensionality explosion. Set to 0 to disable limit.",
                "default": 50,
                "minimum": 0
            },
            "output_file_path": {
                "type": "string",
                "description": "Path where the processed data should be saved. If not provided, will create 'one_hot_encoded.csv' in the same directory as the input file."
            }
        },
        "required": ["file_path", "columns"]
    }

    async def execute(
        self,
        file_path: str,
        columns: Union[str, List[str]],
        handle_unknown: str = "error",
        drop_original: bool = False,
        max_categories: int = 50,
        output_file_path: Optional[str] = None,
    ) -> ToolResult:
        """
        Execute one-hot encoding for specified categorical columns.
        
        One-hot encoding creates a new binary column for each category in the original column.
        Suitable for nominal categorical data where there's no inherent order among categories.
        Consider other encoding methods for high-cardinality features to avoid dimensionality issues.
        
        Args:
            file_path: Path to the input data file (CSV or Excel)
            columns: Column name(s) to encode (single string or list)
            handle_unknown: How to handle unknown categories ('error' or 'ignore')
            drop_original: Whether to drop original columns after encoding
            max_categories: Maximum unique categories per column (0 = no limit)
            output_file_path: Optional output file path (defaults to overwrite input)
            
        Returns:
            ToolResult with detailed encoding summary including new column information,
            dimensionality changes, and categorical statistics
        """
        try:
            # Validate inputs
            if handle_unknown not in ["error", "ignore"]:
                raise ToolError("handle_unknown must be either 'error' or 'ignore'")
            
            if max_categories < 0:
                raise ToolError("max_categories must be non-negative")
            
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
            
            # Analyze each column before encoding
            for column in columns:
                unique_values = data[column].dropna().unique()
                unique_count = len(unique_values)
                null_count = data[column].isnull().sum()
                
                # Check if column appears to be categorical
                if pd.api.types.is_numeric_dtype(data[column]) and unique_count > 10:
                    warning_msg = f"Column '{column}' appears to be numeric with {unique_count} unique values. Consider if one-hot encoding is appropriate."
                    warnings_list.append(warning_msg)
                    logger.warning(warning_msg)
                
                # Check cardinality
                if max_categories > 0 and unique_count > max_categories:
                    raise ToolError(
                        f"Column '{column}' has {unique_count} unique categories, exceeding max_categories={max_categories}. "
                        f"High-cardinality encoding may lead to 'curse of dimensionality'. "
                        f"Consider using other encoding methods or increase max_categories limit."
                    )
                
                encoding_info[column] = {
                    'unique_categories': int(unique_count),
                    'null_values': int(null_count),
                    'categories': unique_values.tolist()[:10],  # Show first 10 categories
                    'will_create_columns': int(unique_count)
                }
                
                logger.info(f"Column '{column}': {unique_count} categories will create {unique_count} new binary columns")
            
            # Calculate total new columns that will be created
            total_new_columns = sum(info['will_create_columns'] for info in encoding_info.values())
            
            # Apply one-hot encoding
            try:
                processed_data = one_hot_encode(
                    data=data.copy(),
                    columns=columns,
                    drop_original=drop_original,
                    handle_unknown=handle_unknown
                )
            except Exception as e:
                raise ToolError(f"Error during one-hot encoding: {str(e)}")
            
            # Determine output path
            if output_file_path is None:
                output_file_path = os.path.join(file_path, "one_hot_encoded.csv")
            
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
            columns_added = final_shape[1] - original_shape[1] + (len(columns) if drop_original else 0)
            
            summary = f"One-hot encoding completed successfully:\n"
            summary += f"- Input file: {file_path}\n"
            summary += f"- Output file: {output_file_path}\n"
            summary += f"- Encoded columns: {', '.join(columns)}\n"
            summary += f"- Handle unknown: {handle_unknown}\n"
            summary += f"- Drop original: {drop_original}\n"
            summary += f"- Original data shape: {original_shape}\n"
            summary += f"- Final data shape: {final_shape}\n"
            summary += f"- New binary columns created: {columns_added}\n"
            summary += f"- Total categories encoded: {sum(info['unique_categories'] for info in encoding_info.values())}\n"
            
            if warnings_list:
                summary += f"\nWARNINGS:\n"
                for warning in warnings_list:
                    summary += f"- {warning}\n"
            
            summary += f"\nENCODING CHARACTERISTICS:\n"
            summary += f"- Creates binary columns for each category (format: 'original_column_value')\n"
            summary += f"- Suitable for nominal categorical data with no inherent order\n"
            summary += f"- May increase dimensionality significantly with high-cardinality variables\n"
            summary += f"- Best for categorical variables with relatively few unique categories\n"
            
            summary += "\nColumn-wise encoding details:\n"
            for column, info in encoding_info.items():
                summary += f"  {column}:\n"
                summary += f"    - Categories: {info['unique_categories']}\n"
                summary += f"    - New columns created: {info['will_create_columns']}\n"
                summary += f"    - Null values: {info['null_values']}\n"
                summary += f"    - Sample categories: {', '.join(map(str, info['categories']))}\n"
                if len(info['categories']) == 10 and info['unique_categories'] > 10:
                    summary += f"    - ... and {info['unique_categories'] - 10} more categories\n"
            
            # Get list of new encoded columns
            if drop_original:
                new_columns = [col for col in processed_data.columns if col not in data.drop(columns, axis=1).columns]
            else:
                new_columns = [col for col in processed_data.columns if col not in data.columns]
            
            if new_columns:
                summary += f"\nNew encoded columns created:\n"
                sample_new_cols = new_columns[:20]  # Show first 20 new columns
                summary += f"  {', '.join(sample_new_cols)}\n"
                if len(new_columns) > 20:
                    summary += f"  ... and {len(new_columns) - 20} more columns\n"
            
            return ToolResult(output=summary)
            
        except ToolError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            error_msg = f"Unexpected error in one-hot encoding: {str(e)}"
            logger.error(error_msg)
            return ToolResult(error=error_msg)


# Usage example and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_tool():
        tool = OneHotEncodeTool()
        
        # Create sample data with categorical columns
        sample_data = pd.DataFrame({
            'color': ['red', 'blue', 'green', 'red', 'blue'],
            'size': ['small', 'large', 'medium', 'large', 'small'],
            'material': ['wood', 'metal', 'plastic', 'wood', 'metal'],
            'price': [10.5, 25.0, 15.0, 12.0, 22.5],
            'quantity': [1, 2, 1, 3, 2]
        })
        
        # Save sample data
        sample_data.to_csv('test_categorical.csv', index=False)
        
        # Test the tool
        result = await tool.execute(
            file_path='test_categorical.csv',
            columns=['color', 'size'],
            handle_unknown='error',
            drop_original=False,
            max_categories=10,
            output_file_path='test_categorical_encoded.csv'
        )
        
        print(result.output if result.output else result.error)
        
        # Test with high cardinality warning
        print("\n" + "="*50)
        print("Testing high cardinality warning:")
        
        high_card_result = await tool.execute(
            file_path='test_categorical.csv',
            columns=['color'],
            max_categories=2  # This should trigger an error
        )
        
        print(high_card_result.error if high_card_result.error else high_card_result.output)
    
    asyncio.run(test_tool())