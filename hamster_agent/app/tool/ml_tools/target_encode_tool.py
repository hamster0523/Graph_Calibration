import pandas as pd
import numpy as np
import os
from typing import Union, List, Optional
from scipy.special import expit

from app.tool.base import BaseTool, ToolResult
from app.exceptions import ToolError
from app.logger import logger

def target_encode(data: pd.DataFrame, 
                  columns: Union[str, List[str]], 
                  target: str, 
                  min_samples_leaf: int = 1, 
                  smoothing: float = 1.0) -> pd.DataFrame:
    """
    Perform target encoding on specified categorical columns. The resulting columns 
    will follow the format 'original_column_target_enc'.

    Args:
        data (pd.DataFrame): The input DataFrame.
        columns (str or List[str]): Column label or list of column labels to encode.
        target (str): The name of the target column.
        min_samples_leaf (int, optional): Minimum samples to take category average into account. Defaults to 1.
        smoothing (float, optional): Smoothing effect to balance categorical average vs prior. Defaults to 1.0.

    Returns:
        pd.DataFrame: DataFrame with target encoded columns.

    Example:
        >>> df = pd.DataFrame({
    'fruit': ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple', 'cherry', 'banana', 'apple', 'cherry', 'kiwi'],
    'region': ['north', 'north', 'south', 'south', 'north', 'south', 'north', 'south', 'north', 'north', 'south'],
    'price': [1, 0, 1, 0, 2, 3, 1, 0, 1, 2, 3]
})
        >>> target_encode(data, ['fruit', 'region'], 'price', min_samples_leaf=2, smoothing=2.0)
            fruit  region  price  fruit_price_enc  region_price_enc
        0    apple   north      1          1.437566           1.509699
        1   banana   north      0          0.912568           1.509699
        2    apple   south      1          1.437566           1.250000
        3   cherry   south      0          0.796902           1.250000
        4   banana   north      2          0.912568           1.509699
        5    apple   south      3          1.437566           1.250000
        6   cherry   north      1          0.796902           1.509699
        7   banana   south      0          0.912568           1.250000
        8    apple   north      1          1.437566           1.509699
        9   cherry   north      2          0.796902           1.509699
        10    kiwi   south      3          1.750000           1.250000
    """
    if isinstance(columns, str):
        columns = [columns]

    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in the DataFrame.")

    if min_samples_leaf < 0:
        raise ValueError(f"min_samples_leaf should be non-negative, but got {min_samples_leaf}.")
    
    if smoothing <= 0:
        raise ValueError(f"smoothing should be positive, but got {smoothing}.")

    result = data.copy()
    prior = data[target].mean()

    # Check if specified columns exist
    missing_columns = set(columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} not found in the DataFrame.")

    for col in columns:
        col_data = data[col]
        averages = data.groupby(col)[target].agg(["count", "mean"])
        
        # Calculate the smoothing factor using a sigmoid function
        smoothing_factor = expit((averages["count"] - min_samples_leaf) / smoothing)
        
        # Calculate the smoothed averages
        averages["smooth"] = prior * (1 - smoothing_factor) + averages["mean"] * smoothing_factor
        
        # Map the smooth values back to the original data
        encoded_col_name = f"{col}_target_enc"
        result[encoded_col_name] = col_data.map(averages["smooth"]).fillna(prior)  # Fill new categories with global prior

    return result


class TargetEncodeTool(BaseTool):
    """
    Tool for performing target encoding on categorical columns in CSV/Excel files.
    Target encoding replaces categorical values with the mean of the target variable for that category.
    Particularly effective for high-cardinality features in supervised learning tasks.
    """
    
    name: str = "target_encode"
    description: str = (
        "Perform target encoding on specified categorical columns. Target encoding replaces each "
        "categorical value with the mean of the target variable for that category, which can capture "
        "complex relationships between categorical variables and the target. The resulting columns "
        "follow the format 'original_column_target_enc'. This encoding is particularly useful for "
        "high-cardinality categorical variables in supervised learning tasks where there's a clear "
        "relationship between categories and the target variable. Includes smoothing to prevent "
        "overfitting, especially for categories with few samples."
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
                "description": "Column label or list of column labels to encode. Should contain categorical data that has a relationship with the target variable.",
                "items": {"type": "string"}
            },
            "target": {
                "type": "string",
                "description": "The name of the target column in the DataFrame. Must be a numeric column for calculating means."
            },
            "min_samples_leaf": {
                "type": "integer",
                "description": "Minimum samples to take category average into account. Categories with fewer samples will be more influenced by the global prior.",
                "default": 1,
                "minimum": 0
            },
            "smoothing": {
                "type": "number",
                "description": "Smoothing effect to balance categorical average vs prior. Higher values give more weight to the global mean for small categories.",
                "default": 1.0,
                "minimum": 0.01
            },
            "drop_original": {
                "type": "boolean",
                "description": "If True, drop the original categorical columns after encoding. If False, keep both original and encoded columns.",
                "default": False
            },
            "cross_validate": {
                "type": "boolean",
                "description": "If True, use out-of-fold encoding to prevent data leakage. Recommended for model training to avoid overfitting.",
                "default": False
            },
            "output_file_path": {
                "type": "string",
                "description": "Path where the processed data should be saved. If not provided, will create 'target_encoded.csv' in the same directory as the input file."
            }
        },
        "required": ["file_path", "columns", "target"]
    }

    async def execute(
        self,
        file_path: str,
        columns: Union[str, List[str]],
        target: str,
        min_samples_leaf: int = 1,
        smoothing: float = 1.0,
        drop_original: bool = False,
        cross_validate: bool = False,
        output_file_path: Optional[str] = None,
    ) -> ToolResult:
        """
        Execute target encoding for specified categorical columns.
        
        Target encoding replaces each category with the smoothed mean of the target variable
        for that category. This can capture complex relationships between categorical features
        and the target, making it particularly effective for high-cardinality variables.
        
        Args:
            file_path: Path to the input data file (CSV or Excel)
            columns: Column name(s) to encode (single string or list)
            target: Name of the target column (must be numeric)
            min_samples_leaf: Minimum samples for category averaging
            smoothing: Smoothing factor to balance category mean vs global prior
            drop_original: Whether to drop original columns after encoding
            cross_validate: Whether to use out-of-fold encoding to prevent leakage
            output_file_path: Optional output file path (defaults to create new file)
            
        Returns:
            ToolResult with detailed encoding summary including target statistics,
            encoding mappings, smoothing effects, and data leakage warnings
        """
        try:
            # Validate inputs
            if min_samples_leaf < 0:
                raise ToolError("min_samples_leaf must be non-negative")
            
            if smoothing <= 0:
                raise ToolError("smoothing must be positive")
            
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
            
            # Validate target column exists and is numeric
            if target not in data.columns:
                raise ToolError(f"Target column '{target}' not found in data")
            
            if not pd.api.types.is_numeric_dtype(data[target]):
                raise ToolError(f"Target column '{target}' must be numeric for target encoding")
            
            # Validate columns exist
            missing_columns = [col for col in columns if col not in data.columns]
            if missing_columns:
                raise ToolError(f"Columns not found in data: {missing_columns}")
            
            # Store original data info
            original_shape = data.shape
            encoding_info = {}
            warnings_list = []
            
            # Calculate global target statistics
            target_mean = data[target].mean()
            target_std = data[target].std()
            target_null_count = data[target].isnull().sum()
            
            if target_null_count > 0:
                warning_msg = f"Target column '{target}' has {target_null_count} null values. These will affect encoding quality."
                warnings_list.append(warning_msg)
                logger.warning(warning_msg)
            
            # Apply target encoding
            if cross_validate:
                # Implement out-of-fold encoding to prevent data leakage
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                result = data.copy()
                
                for column in columns:
                    encoded_values = np.full(len(data), target_mean)
                    
                    for train_idx, val_idx in kf.split(data):
                        train_data = data.iloc[train_idx]
                        
                        # Calculate target means for training fold
                        averages = train_data.groupby(column)[target].agg(["count", "mean"])
                        prior = train_data[target].mean()
                        
                        # Apply smoothing
                        smoothing_factor = expit((averages["count"] - min_samples_leaf) / smoothing)
                        averages["smooth"] = prior * (1 - smoothing_factor) + averages["mean"] * smoothing_factor
                        
                        # Apply to validation fold
                        val_encoded = data.iloc[val_idx][column].map(averages["smooth"]).fillna(prior)
                        encoded_values[val_idx] = val_encoded
                    
                    encoded_col_name = f"{column}_target_enc"
                    result[encoded_col_name] = encoded_values
                    
                    # Calculate encoding statistics
                    unique_categories = data[column].nunique()
                    category_stats = data.groupby(column)[target].agg(["count", "mean", "std"]).fillna(0)
                    
                    encoding_info[column] = {
                        'unique_categories': int(unique_categories),
                        'encoded_column': encoded_col_name,
                        'target_correlation': float(np.corrcoef(result[encoded_col_name], data[target])[0, 1]),
                        'encoding_range': f"{result[encoded_col_name].min():.4f} to {result[encoded_col_name].max():.4f}",
                        'cross_validated': True,
                        'smoothing_applied': True,
                        'min_category_samples': int(category_stats['count'].min()),
                        'max_category_samples': int(category_stats['count'].max()),
                        'mean_category_samples': float(category_stats['count'].mean())
                    }
            else:
                # Standard target encoding (potential for data leakage)
                result = target_encode(
                    data=data.copy(),
                    columns=columns,
                    target=target,
                    min_samples_leaf=min_samples_leaf,
                    smoothing=smoothing
                )
                
                # Add data leakage warning
                warning_msg = "Standard target encoding may cause data leakage. Consider using cross_validate=True for model training."
                warnings_list.append(warning_msg)
                logger.warning(warning_msg)
                
                # Calculate encoding statistics
                for column in columns:
                    encoded_col_name = f"{column}_target_enc"
                    unique_categories = data[column].nunique()
                    category_stats = data.groupby(column)[target].agg(["count", "mean", "std"]).fillna(0)
                    
                    encoding_info[column] = {
                        'unique_categories': int(unique_categories),
                        'encoded_column': encoded_col_name,
                        'target_correlation': float(np.corrcoef(result[encoded_col_name], data[target])[0, 1]),
                        'encoding_range': f"{result[encoded_col_name].min():.4f} to {result[encoded_col_name].max():.4f}",
                        'cross_validated': False,
                        'smoothing_applied': True,
                        'min_category_samples': int(category_stats['count'].min()),
                        'max_category_samples': int(category_stats['count'].max()),
                        'mean_category_samples': float(category_stats['count'].mean())
                    }
            
            # Drop original columns if requested
            if drop_original:
                result = result.drop(columns, axis=1)
            
            # Determine output path
            if output_file_path is None:
                output_file_path = os.path.join(os.path.dirname(file_path), "target_encoded.csv")
            
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
            
            summary = f"Target encoding completed successfully:\n"
            summary += f"- Input file: {file_path}\n"
            summary += f"- Output file: {output_file_path}\n"
            summary += f"- Encoded columns: {', '.join(columns)}\n"
            summary += f"- Target column: {target}\n"
            summary += f"- Min samples leaf: {min_samples_leaf}\n"
            summary += f"- Smoothing factor: {smoothing}\n"
            summary += f"- Cross validation: {cross_validate}\n"
            summary += f"- Drop original: {drop_original}\n"
            summary += f"- Original data shape: {original_shape}\n"
            summary += f"- Final data shape: {final_shape}\n"
            summary += f"- New encoded columns created: {columns_added}\n"
            
            summary += f"\nTARGET VARIABLE STATISTICS:\n"
            summary += f"- Target mean: {target_mean:.4f}\n"
            summary += f"- Target std: {target_std:.4f}\n"
            summary += f"- Target null values: {target_null_count}\n"
            
            if warnings_list:
                summary += f"\nWARNINGS:\n"
                for warning in warnings_list:
                    summary += f"- {warning}\n"
            
            summary += f"\nENCODING CHARACTERISTICS:\n"
            summary += f"- Replaces categories with smoothed target means\n"
            summary += f"- Captures complex category-target relationships\n"
            summary += f"- Effective for high-cardinality categorical variables\n"
            summary += f"- Smoothing prevents overfitting for small categories\n"
            summary += f"- Cross-validation recommended to prevent data leakage\n"
            summary += f"- Can be sensitive to target outliers\n"
            
            summary += "\nColumn-wise encoding details:\n"
            for column, info in encoding_info.items():
                summary += f"  {column} -> {info['encoded_column']}:\n"
                summary += f"    - Categories: {info['unique_categories']}\n"
                summary += f"    - Encoding range: {info['encoding_range']}\n"
                summary += f"    - Target correlation: {info['target_correlation']:.4f}\n"
                summary += f"    - Cross validated: {info['cross_validated']}\n"
                summary += f"    - Category samples: {info['min_category_samples']} to {info['max_category_samples']} (avg: {info['mean_category_samples']:.1f})\n"
            
            # Get list of new encoded columns
            if drop_original:
                new_columns = [col for col in result.columns if col not in data.drop(columns, axis=1).columns]
            else:
                new_columns = [col for col in result.columns if col not in data.columns]
            
            if new_columns:
                summary += f"\nNew target-encoded columns created:\n"
                summary += f"  {', '.join(new_columns)}\n"
            
            summary += f"\nAPPLICABLE SITUATIONS:\n"
            summary += f"- High-cardinality categorical variables in supervised learning\n"
            summary += f"- When categories have clear relationship with target variable\n"
            summary += f"- Tree-based models and linear models benefit from this encoding\n"
            summary += f"- When one-hot encoding creates too many dimensions\n"
            summary += f"- Regression and classification tasks with numeric targets\n"
            
            summary += f"\nDATA LEAKAGE PREVENTION:\n"
            summary += f"- Use cross_validate=True for model training datasets\n"
            summary += f"- Apply encoding parameters from training to test data\n"
            summary += f"- Consider target encoding as part of feature engineering pipeline\n"
            summary += f"- Monitor for overfitting, especially with small categories\n"
            
            return ToolResult(output=summary)
            
        except ToolError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            error_msg = f"Unexpected error in target encoding: {str(e)}"
            logger.error(error_msg)
            return ToolResult(error=error_msg)


# Usage example and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_tool():
        tool = TargetEncodeTool()
        
        # Create sample data with categorical columns and numeric target
        sample_data = pd.DataFrame({
            'fruit': ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple', 'cherry', 'banana', 'apple', 'cherry', 'kiwi'],
            'region': ['north', 'north', 'south', 'south', 'north', 'south', 'north', 'south', 'north', 'north', 'south'],
            'store_type': ['premium', 'regular', 'premium', 'regular', 'premium', 'regular', 'premium', 'regular', 'premium', 'regular', 'premium'],
            'price': [1.5, 0.8, 1.2, 0.6, 2.1, 3.2, 1.1, 0.9, 1.4, 2.0, 3.5]
        })
        
        # Save sample data
        sample_data.to_csv('test_target_encoding.csv', index=False)
        
        # Test the tool with standard encoding
        result = await tool.execute(
            file_path='test_target_encoding.csv',
            columns=['fruit', 'region'],
            target='price',
            min_samples_leaf=2,
            smoothing=1.0,
            drop_original=False,
            cross_validate=False,
            output_file_path='test_target_encoded.csv'
        )
        
        print(result.output if result.output else result.error)
        
        # Test with cross-validation
        print("\n" + "="*50)
        print("Testing with cross-validation:")
        
        cv_result = await tool.execute(
            file_path='test_target_encoding.csv',
            columns=['fruit', 'region', 'store_type'],
            target='price',
            min_samples_leaf=1,
            smoothing=2.0,
            cross_validate=True
        )
        
        print(cv_result.output if cv_result.output else cv_result.error)
    
    asyncio.run(test_tool())