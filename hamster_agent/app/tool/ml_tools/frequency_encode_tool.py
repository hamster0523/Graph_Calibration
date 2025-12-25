import pandas as pd
import os
import warnings
from typing import Union, List, Optional

from app.tool.base import BaseTool, ToolResult
from app.exceptions import ToolError
from app.logger import logger

def frequency_encode(data: pd.DataFrame, 
                     columns: Union[str, List[str]], 
                     drop_original: bool = False) -> pd.DataFrame:
    """
    Perform frequency encoding on specified categorical columns. The resulting columns 
    will follow the format 'original_column_freq'.

    Args:
        data (pd.DataFrame): The input DataFrame.
        columns (str or List[str]): Column label or list of column labels to encode.
        drop_original (bool, optional): If True, drop original columns. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with frequency encoded columns.

    Example:
        >>> df = pd.DataFrame({'fruit': ['apple', 'banana', 'apple', 'cherry']})
        >>> frequency_encode(df, 'fruit')
           fruit  fruit_freq
        0  apple        0.50
        1  banana       0.25
        2  apple        0.50
        3  cherry       0.25
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
        frequency = col_data.value_counts(normalize=True)
        encoded_col_name = f"{col}_freq"
        result[encoded_col_name] = col_data.map(frequency)

    return result


class FrequencyEncodeTool(BaseTool):
    """
    Tool for performing frequency encoding on categorical columns in CSV/Excel files.
    Frequency encoding replaces each category with its relative frequency in the dataset.
    Particularly useful for high-cardinality categorical variables where one-hot encoding would create too many features.
    """
    
    name: str = "frequency_encode"
    description: str = (
        "Perform frequency encoding on specified categorical columns. The resulting columns will follow "
        "the format 'original_column_freq'. Frequency encoding replaces each category with its relative "
        "frequency in the dataset, which can capture information about the importance of each category. "
        "This encoding is particularly useful for high-cardinality categorical variables where one-hot "
        "encoding would create too many features, and when the frequency of categories is informative "
        "for the target variable. Works well with both tree-based and linear models."
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
                "description": "Column label or list of column labels to encode. Should contain categorical data, especially high-cardinality variables where frequency information is meaningful.",
                "items": {"type": "string"}
            },
            "drop_original": {
                "type": "boolean",
                "description": "If True, drop the original categorical columns after encoding. If False, keep both original and encoded columns.",
                "default": False
            },
            "handle_missing": {
                "type": "string",
                "description": "How to handle missing values in categorical columns. 'include': treat NaN as a category, 'exclude': ignore NaN values in frequency calculation.",
                "enum": ["include", "exclude"],
                "default": "exclude"
            },
            "output_file_path": {
                "type": "string",
                "description": "Path where the processed data should be saved. If not provided, will create 'frequency_encoded.csv' in the same directory as the input file."
            }
        },
        "required": ["file_path", "columns"]
    }

    async def execute(
        self,
        file_path: str,
        columns: Union[str, List[str]],
        drop_original: bool = False,
        handle_missing: str = "exclude",
        output_file_path: Optional[str] = None,
    ) -> ToolResult:
        """
        Execute frequency encoding for specified categorical columns.
        
        Frequency encoding replaces each category with its relative frequency (proportion) in the dataset.
        This preserves information about category distribution and is especially useful for high-cardinality
        categorical variables. The frequency values can be informative features for machine learning models.
        
        Args:
            file_path: Path to the input data file (CSV or Excel)
            columns: Column name(s) to encode (single string or list)
            drop_original: Whether to drop original columns after encoding
            handle_missing: How to handle NaN values ('include' or 'exclude')
            output_file_path: Optional output file path (defaults to create new file)
            
        Returns:
            ToolResult with detailed encoding summary including frequency statistics,
            distribution information, and warnings about potential ordinality implications
        """
        try:
            # Validate inputs
            if handle_missing not in ["include", "exclude"]:
                raise ToolError("handle_missing must be either 'include' or 'exclude'")
            
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
            
            # Apply frequency encoding
            result = data.copy()
            
            for column in columns:
                col_data = data[column]
                unique_values = col_data.dropna().unique() if handle_missing == "exclude" else col_data.unique()
                unique_count = len(unique_values)
                null_count = col_data.isnull().sum()
                
                # Check if this might introduce false ordinality
                if unique_count > 10:
                    warning_msg = f"Column '{column}' has {unique_count} unique categories. Frequency encoding may introduce false sense of ordinality among categories."
                    warnings_list.append(warning_msg)
                    logger.warning(warning_msg)
                
                # Calculate frequencies
                if handle_missing == "include":
                    # Include NaN as a category
                    frequency = col_data.value_counts(normalize=True, dropna=False)
                else:
                    # Exclude NaN from frequency calculation
                    frequency = col_data.value_counts(normalize=True, dropna=True)
                
                # Apply frequency encoding
                encoded_col_name = f"{column}_freq"
                result[encoded_col_name] = col_data.map(frequency)
                
                # If excluding NaN but NaN exists, those will become NaN in encoded column
                if handle_missing == "exclude" and null_count > 0:
                    encoded_null_count = result[encoded_col_name].isnull().sum()
                else:
                    encoded_null_count = 0
                
                # Calculate frequency statistics
                freq_min = frequency.min()
                freq_max = frequency.max()
                freq_mean = frequency.mean()
                
                encoding_info[column] = {
                    'unique_categories': int(unique_count),
                    'null_values': int(null_count),
                    'encoded_column': encoded_col_name,
                    'frequency_range': f"{freq_min:.4f} to {freq_max:.4f}",
                    'mean_frequency': float(freq_mean),
                    'encoded_null_count': int(encoded_null_count),
                    'most_frequent_category': frequency.index[0],
                    'most_frequent_value': float(frequency.iloc[0]),
                    'least_frequent_category': frequency.index[-1],
                    'least_frequent_value': float(frequency.iloc[-1])
                }
                
                logger.info(f"Frequency encoded column '{column}' -> '{encoded_col_name}' with frequencies ranging from {freq_min:.4f} to {freq_max:.4f}")
            
            # Drop original columns if requested
            if drop_original:
                result = result.drop(columns, axis=1)
            
            # Determine output path
            if output_file_path is None:
                output_file_path = os.path.join(os.path.dirname(file_path), "frequency_encoded.csv")
            
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
            columns_added = final_shape[1] - original_shape[1] + (len(columns) if drop_original else 0)
            
            summary = f"Frequency encoding completed successfully:\n"
            summary += f"- Input file: {file_path}\n"
            summary += f"- Output file: {output_file_path}\n"
            summary += f"- Encoded columns: {', '.join(columns)}\n"
            summary += f"- Handle missing: {handle_missing}\n"
            summary += f"- Drop original: {drop_original}\n"
            summary += f"- Original data shape: {original_shape}\n"
            summary += f"- Final data shape: {final_shape}\n"
            summary += f"- New frequency columns created: {columns_added}\n"
            
            if warnings_list:
                summary += f"\nWARNINGS:\n"
                for warning in warnings_list:
                    summary += f"- {warning}\n"
            
            summary += f"\nENCODING CHARACTERISTICS:\n"
            summary += f"- Replaces each category with its relative frequency (0.0 to 1.0)\n"
            summary += f"- Preserves information about category distribution in the dataset\n"
            summary += f"- Useful for high-cardinality categorical variables\n"
            summary += f"- May introduce false sense of ordinality among categories\n"
            summary += f"- Works well with both tree-based and linear models\n"
            summary += f"- Frequency information can be predictive when category popularity matters\n"
            
            summary += "\nColumn-wise encoding details:\n"
            for column, info in encoding_info.items():
                summary += f"  {column} -> {info['encoded_column']}:\n"
                summary += f"    - Categories: {info['unique_categories']}\n"
                summary += f"    - Frequency range: {info['frequency_range']}\n"
                summary += f"    - Mean frequency: {info['mean_frequency']:.4f}\n"
                summary += f"    - Null values: {info['null_values']}\n"
                summary += f"    - Most frequent: '{info['most_frequent_category']}' (freq: {info['most_frequent_value']:.4f})\n"
                summary += f"    - Least frequent: '{info['least_frequent_category']}' (freq: {info['least_frequent_value']:.4f})\n"
                if info['encoded_null_count'] > 0:
                    summary += f"    - Encoded null values: {info['encoded_null_count']}\n"
            
            # Get list of new encoded columns
            if drop_original:
                new_columns = [col for col in result.columns if col not in data.drop(columns, axis=1).columns]
            else:
                new_columns = [col for col in result.columns if col not in data.columns]
            
            if new_columns:
                summary += f"\nNew frequency-encoded columns created:\n"
                summary += f"  {', '.join(new_columns)}\n"
            
            summary += f"\nAPPLICABLE SITUATIONS:\n"
            summary += f"- High-cardinality categorical variables (many unique categories)\n"
            summary += f"- When frequency/popularity of categories is informative\n"
            summary += f"- Alternative to one-hot encoding for memory efficiency\n"
            summary += f"- When category distribution matters for prediction\n"
            summary += f"- Both tree-based models (Random Forest, XGBoost) and linear models\n"
            
            return ToolResult(output=summary)
            
        except ToolError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            error_msg = f"Unexpected error in frequency encoding: {str(e)}"
            logger.error(error_msg)
            return ToolResult(error=error_msg)


# Usage example and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_tool():
        tool = FrequencyEncodeTool()
        
        # Create sample data with categorical columns (including high-cardinality)
        sample_data = pd.DataFrame({
            'city': ['New York', 'London', 'Paris', 'New York', 'London', 'New York', 'Tokyo', 'Paris'],
            'product': ['laptop', 'phone', 'tablet', 'laptop', 'laptop', 'phone', 'tablet', 'laptop'],
            'brand': ['Apple', 'Samsung', 'Apple', 'Dell', 'Apple', 'Samsung', 'Apple', 'Dell'],
            'price': [1000, 800, 600, 900, 1200, 850, 650, 950],
            'sales': [50, 30, 20, 40, 60, 35, 25, 45]
        })
        
        # Save sample data
        sample_data.to_csv('test_frequency.csv', index=False)
        
        # Test the tool
        result = await tool.execute(
            file_path='test_frequency.csv',
            columns=['city', 'product', 'brand'],
            drop_original=False,
            handle_missing='exclude',
            output_file_path='test_frequency_encoded.csv'
        )
        
        print(result.output if result.output else result.error)
        
        # Test with missing values
        print("\n" + "="*50)
        print("Testing with missing values:")
        
        # Add some missing values
        sample_data_with_nan = sample_data.copy()
        sample_data_with_nan.loc[1, 'city'] = None
        sample_data_with_nan.loc[3, 'brand'] = None
        sample_data_with_nan.to_csv('test_frequency_nan.csv', index=False)
        
        nan_result = await tool.execute(
            file_path='test_frequency_nan.csv',
            columns=['city', 'brand'],
            handle_missing='include'  # Include NaN as a category
        )
        
        print(nan_result.output if nan_result.output else nan_result.error)
    
    asyncio.run(test_tool())
