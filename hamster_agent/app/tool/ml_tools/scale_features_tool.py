import pandas as pd
import numpy as np
import os
import warnings
from typing import Union, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from app.tool.base import BaseTool, ToolResult
from app.exceptions import ToolError
from app.logger import logger

def scale_features(data: pd.DataFrame, 
                   columns: Union[str, List[str]], 
                   method: str = 'standard', 
                   copy: bool = True) -> pd.DataFrame:
    """
    Scale numerical features in the specified columns of a DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame.
        columns (str or List[str]): Column label or sequence of labels of numerical features to scale.
        method (str, optional): The scaling method to use. 
            Options: 'standard' for StandardScaler, 
                     'minmax' for MinMaxScaler, 
                     'robust' for RobustScaler. 
            Defaults to 'standard'.
        copy (bool, optional): If False, try to avoid a copy and do inplace scaling instead. 
            This is not guaranteed to always work inplace; e.g. if the data is not a NumPy array or scipy.sparse CSR matrix, 
            a copy may still be returned. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame with scaled features

    Raises:
        ValueError: If any of the specified columns are not numerical or if duplicate columns are not identical.
    """
    if isinstance(columns, str):
        columns = [columns]

    # Check if specified columns exist
    missing_columns = set(columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} not found in the DataFrame.")

    # Handle duplicate columns
    unique_columns = []
    for col in columns:
        if col in unique_columns:
            continue
        col_data = data[col]
        if isinstance(col_data, pd.DataFrame):
            # Check if all duplicate columns are identical
            if col_data.nunique().eq(1).all():
                print(f"Warning: Duplicate identical columns found for '{col}'. Only one instance will be scaled.")
                unique_columns.append(col)
            else:
                raise ValueError(f"Duplicate non-identical columns found for '{col}'. Please resolve this before scaling.")
        else:
            unique_columns.append(col)

    # Check if all specified columns are numerical
    non_numeric_cols = [col for col in unique_columns if not pd.api.types.is_numeric_dtype(data[col])]
    if non_numeric_cols:
        raise ValueError(f"The following columns are not numerical: {non_numeric_cols}. "
                         "Please only specify numerical columns for scaling.")

    # Select the appropriate scaler
    if method == 'standard':
        scaler = StandardScaler(copy=copy)
    elif method == 'minmax':
        scaler = MinMaxScaler(copy=copy)
    elif method == 'robust':
        scaler = RobustScaler(copy=copy)
    else:
        raise ValueError("Invalid method. Choose 'standard', 'minmax', or 'robust'.")

    # Create a copy of the dataframe if required
    if copy:
        data = data.copy()

    # Fit and transform the selected columns
    scaled_data = scaler.fit_transform(data[unique_columns])

    # Replace the original columns with scaled data
    data[unique_columns] = scaled_data

    return data


class ScaleFeaturesTool(BaseTool):
    """
    Tool for scaling numerical features in CSV/Excel files using various scaling methods.
    Essential for preparing numerical data for machine learning models that are sensitive 
    to the scale of input features (e.g., neural networks, SVM, K-means clustering).
    """
    
    name: str = "scale_features"
    description: str = (
        "Scale numerical features in the specified columns of a DataFrame using various scaling methods. "
        "This tool is essential for data preprocessing and preparing numerical features for machine learning "
        "models that are sensitive to feature scales. Supports StandardScaler (standardization), "
        "MinMaxScaler (normalization), and RobustScaler (outlier-resistant scaling). Particularly "
        "important for algorithms like neural networks, SVM, and K-means clustering."
    )
    
    parameters: dict = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the CSV/Excel file containing the numerical features to scale"
            },
            "columns": {
                "type": ["string", "array"],
                "description": "Column label or sequence of labels of numerical features to scale. Only numerical columns are supported.",
                "items": {"type": "string"}
            },
            "method": {
                "type": "string",
                "description": "The scaling method to use. 'standard': standardization (mean=0, std=1); 'minmax': normalization (0-1 range); 'robust': outlier-resistant scaling using median and IQR.",
                "enum": ["standard", "minmax", "robust"],
                "default": "standard"
            },
            "feature_range": {
                "type": "array",
                "description": "For MinMaxScaler only: the target range for scaling. Default is [0, 1].",
                "items": {"type": "number"},
                "default": [0, 1],
                "minItems": 2,
                "maxItems": 2
            },
            "handle_outliers": {
                "type": "boolean",
                "description": "If True and outliers are detected, recommend using RobustScaler regardless of chosen method.",
                "default": True
            },
            "outlier_threshold": {
                "type": "number",
                "description": "Z-score threshold for outlier detection. Values above this threshold suggest using RobustScaler.",
                "default": 3.0,
                "minimum": 1.0
            },
            "output_file_path": {
                "type": "string",
                "description": "Path where the processed data should be saved. If not provided, will create 'scaled_features.csv' in the same directory as the input file."
            }
        },
        "required": ["file_path", "columns"]
    }

    async def execute(
        self,
        file_path: str,
        columns: Union[str, List[str]],
        method: str = "standard",
        feature_range: List[float] = [0, 1],
        handle_outliers: bool = True,
        outlier_threshold: float = 3.0,
        output_file_path: Optional[str] = None,
    ) -> ToolResult:
        """
        Execute feature scaling for specified numerical columns.
        
        Applies the chosen scaling method to numerical features, which is essential
        for machine learning algorithms sensitive to feature scales. Provides
        outlier detection and scaling method recommendations.
        
        Args:
            file_path: Path to the input data file (CSV or Excel)
            columns: Column name(s) to scale (single string or list)
            method: Scaling method ('standard', 'minmax', 'robust')
            feature_range: Target range for MinMaxScaler [min, max]
            handle_outliers: Whether to detect and warn about outliers
            outlier_threshold: Z-score threshold for outlier detection
            output_file_path: Optional output file path (defaults to create new file)
            
        Returns:
            ToolResult with detailed scaling summary including statistics,
            outlier analysis, method characteristics, and scaling recommendations
        """
        try:
            # Validate inputs
            if method not in ["standard", "minmax", "robust"]:
                raise ToolError("method must be one of: 'standard', 'minmax', 'robust'")
            
            if len(feature_range) != 2 or feature_range[0] >= feature_range[1]:
                raise ToolError("feature_range must be [min, max] where min < max")
            
            if outlier_threshold < 1.0:
                raise ToolError("outlier_threshold must be >= 1.0")
            
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
            warnings_list = []
            
            # Validate columns are numerical
            non_numeric_cols = [col for col in columns if not pd.api.types.is_numeric_dtype(data[col])]
            if non_numeric_cols:
                raise ToolError(f"Non-numerical columns found: {non_numeric_cols}. Only numerical columns can be scaled.")
            
            # Calculate original statistics
            original_stats = {}
            outlier_info = {}
            
            for col in columns:
                col_data = data[col].dropna()
                original_stats[col] = {
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'median': float(col_data.median()),
                    'q25': float(col_data.quantile(0.25)),
                    'q75': float(col_data.quantile(0.75)),
                    'null_count': int(data[col].isnull().sum()),
                    'unique_values': int(col_data.nunique())
                }
                
                # Outlier detection
                if handle_outliers and len(col_data) > 0:
                    z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                    outliers = col_data[z_scores > outlier_threshold]
                    outlier_info[col] = {
                        'outlier_count': len(outliers),
                        'outlier_percentage': float(len(outliers) / len(col_data) * 100),
                        'max_z_score': float(z_scores.max()) if len(z_scores) > 0 else 0.0,
                        'has_outliers': len(outliers) > 0
                    }
                    
                    if len(outliers) > 0 and method in ['standard', 'minmax']:
                        warning_msg = f"Column '{col}' has {len(outliers)} outliers ({outlier_info[col]['outlier_percentage']:.1f}%). Consider using 'robust' scaling method."
                        warnings_list.append(warning_msg)
                        logger.warning(warning_msg)
            
            # Check for missing values
            null_counts = data[columns].isnull().sum()
            if null_counts.sum() > 0:
                warning_msg = f"Missing values found in scaling columns: {dict(null_counts[null_counts > 0])}"
                warnings_list.append(warning_msg)
                logger.warning(warning_msg)
            
            # Check for constant features
            constant_features = [col for col in columns if data[col].nunique() <= 1]
            if constant_features:
                warning_msg = f"Constant features detected: {constant_features}. Scaling will have no effect."
                warnings_list.append(warning_msg)
                logger.warning(warning_msg)
            
            # Apply scaling based on method
            try:
                if method == 'minmax' and feature_range != [0, 1]:
                    # Use custom feature range for MinMaxScaler
                    scaler = MinMaxScaler(feature_range=tuple(feature_range))
                    result_data = data.copy()
                    scaled_values = scaler.fit_transform(data[columns])
                    result_data[columns] = scaled_values
                else:
                    # Use the original function for standard cases
                    result_data = scale_features(
                        data=data.copy(),
                        columns=columns,
                        method=method,
                        copy=True
                    )
            except Exception as e:
                raise ToolError(f"Error during scaling: {str(e)}")
            
            # Calculate scaled statistics
            scaled_stats = {}
            for col in columns:
                col_data = result_data[col].dropna()
                scaled_stats[col] = {
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'median': float(col_data.median()),
                    'range': float(col_data.max() - col_data.min())
                }
            
            # Determine output path
            if output_file_path is None:
                output_file_path = os.path.join(os.path.dirname(file_path), "scaled_features.csv")
            
            # Save processed data
            try:
                if output_file_path.endswith('.csv'):
                    result_data.to_csv(output_file_path, index=False)
                elif output_file_path.endswith(('.xlsx', '.xls')):
                    result_data.to_excel(output_file_path, index=False)
                else:
                    # Default to CSV
                    if not output_file_path.endswith('.csv'):
                        output_file_path += '.csv'
                    result_data.to_csv(output_file_path, index=False)
            except Exception as e:
                raise ToolError(f"Error saving processed data: {str(e)}")
            
            # Generate summary
            final_shape = result_data.shape
            
            summary = f"Feature scaling completed successfully:\n"
            summary += f"- Input file: {file_path}\n"
            summary += f"- Output file: {output_file_path}\n"
            summary += f"- Scaling method: {method}\n"
            summary += f"- Columns scaled: {', '.join(columns)}\n"
            summary += f"- Original data shape: {original_shape}\n"
            summary += f"- Final data shape: {final_shape}\n"
            
            if method == 'minmax' and feature_range != [0, 1]:
                summary += f"- Feature range: [{feature_range[0]}, {feature_range[1]}]\n"
            
            if warnings_list:
                summary += f"\nWARNINGS:\n"
                for warning in warnings_list:
                    summary += f"- {warning}\n"
            
            summary += f"\nSCALING METHOD CHARACTERISTICS:\n"
            if method == 'standard':
                summary += f"- StandardScaler: Standardizes features to mean=0, std=1\n"
                summary += f"- Assumes normal distribution, sensitive to outliers\n"
                summary += f"- Good for algorithms assuming normally distributed data\n"
                summary += f"- Preserves shape of original distribution\n"
            elif method == 'minmax':
                summary += f"- MinMaxScaler: Scales features to range {feature_range}\n"
                summary += f"- Preserves relationships between original values\n"
                summary += f"- Sensitive to outliers (can compress main data range)\n"
                summary += f"- Good when you need bounded values\n"
            else:  # robust
                summary += f"- RobustScaler: Uses median and IQR for scaling\n"
                summary += f"- Robust to outliers, doesn't remove them\n"
                summary += f"- Good for data with many outliers\n"
                summary += f"- Centers data around median, scales by IQR\n"
            
            summary += f"\nCOLUMN-WISE SCALING ANALYSIS:\n"
            for col in columns:
                orig = original_stats[col]
                scaled = scaled_stats[col]
                
                summary += f"  {col}:\n"
                summary += f"    Original: mean={orig['mean']:.4f}, std={orig['std']:.4f}, range=[{orig['min']:.4f}, {orig['max']:.4f}]\n"
                summary += f"    Scaled:   mean={scaled['mean']:.4f}, std={scaled['std']:.4f}, range=[{scaled['min']:.4f}, {scaled['max']:.4f}]\n"
                
                if handle_outliers and col in outlier_info:
                    outlier = outlier_info[col]
                    summary += f"    Outliers: {outlier['outlier_count']} ({outlier['outlier_percentage']:.1f}%), max Z-score: {outlier['max_z_score']:.2f}\n"
                
                if orig['null_count'] > 0:
                    summary += f"    Missing values: {orig['null_count']}\n"
                
                if orig['unique_values'] <= 1:
                    summary += f"    Warning: Constant feature (only {orig['unique_values']} unique value)\n"
            
            if handle_outliers:
                total_outliers = sum(info['outlier_count'] for info in outlier_info.values())
                if total_outliers > 0:
                    summary += f"\nOUTLIER ANALYSIS:\n"
                    summary += f"- Total outliers detected: {total_outliers}\n"
                    summary += f"- Outlier threshold: {outlier_threshold} standard deviations\n"
                    summary += f"- Recommendation: Consider RobustScaler for outlier-heavy data\n"
            
            summary += f"\nAPPLICABLE SITUATIONS:\n"
            summary += f"- Preprocessing for machine learning algorithms\n"
            summary += f"- Neural networks (require similar input scales)\n"
            summary += f"- SVM and kernel methods (distance-based algorithms)\n"
            summary += f"- K-means clustering and other distance-based methods\n"
            summary += f"- Principal Component Analysis (PCA)\n"
            summary += f"- Gradient descent optimization\n"
            
            summary += f"\nIMPORTANT CONSIDERATIONS:\n"
            summary += f"- Apply scaling after train/test split to avoid data leakage\n"
            summary += f"- Use same scaler parameters for test data as training data\n"
            summary += f"- Tree-based algorithms (Random Forest, XGBoost) don't require scaling\n"
            summary += f"- Consider outlier impact when choosing scaling method\n"
            summary += f"- Categorical features should be encoded, not scaled\n"
            summary += f"- Save scaler parameters for consistent preprocessing in production\n"
            
            if constant_features:
                summary += f"\nCONSTANT FEATURES DETECTED:\n"
                summary += f"  {', '.join(constant_features)}\n"
                summary += f"  Consider removing these features as they provide no information\n"
            
            return ToolResult(output=summary)
            
        except ToolError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            error_msg = f"Unexpected error in feature scaling: {str(e)}"
            logger.error(error_msg)
            return ToolResult(error=error_msg)


# Usage example and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_tool():
        tool = ScaleFeaturesTool()
        
        # Create sample data with different scales and outliers
        np.random.seed(42)
        n_samples = 100
        
        # Features with different scales
        feature1 = np.random.normal(0, 1, n_samples)  # Standard normal
        feature2 = np.random.normal(100, 15, n_samples)  # Different scale
        feature3 = np.random.uniform(0, 1000, n_samples)  # Large range
        feature4 = np.random.normal(0, 1, n_samples)
        
        # Add some outliers
        feature4[0] = 10  # Strong outlier
        feature4[1] = -8  # Strong outlier
        
        sample_data = pd.DataFrame({
            'feature1': feature1,
            'feature2': feature2,
            'feature3': feature3,
            'feature4_with_outliers': feature4,
            'constant_feature': np.ones(n_samples),
            'categorical': np.random.choice(['A', 'B', 'C'], n_samples),
            'target': np.random.normal(0, 1, n_samples)
        })
        
        # Add some missing values
        sample_data.loc[5:7, 'feature1'] = np.nan
        
        # Save sample data
        sample_data.to_csv('test_scaling.csv', index=False)
        
        # Test with StandardScaler
        result = await tool.execute(
            file_path='test_scaling.csv',
            columns=['feature1', 'feature2', 'feature3', 'feature4_with_outliers'],
            method='standard',
            handle_outliers=True,
            outlier_threshold=2.5,
            output_file_path='test_scaled_standard.csv'
        )
        
        print(result.output if result.output else result.error)
        
        # Test with RobustScaler for outlier-heavy data
        print("\n" + "="*50)
        print("Testing with RobustScaler:")
        
        robust_result = await tool.execute(
            file_path='test_scaling.csv',
            columns=['feature4_with_outliers'],
            method='robust',
            handle_outliers=True
        )
        
        print(robust_result.output if robust_result.output else robust_result.error)
        
        # Test with MinMaxScaler and custom range
        print("\n" + "="*50)
        print("Testing with MinMaxScaler and custom range:")
        
        minmax_result = await tool.execute(
            file_path='test_scaling.csv',
            columns=['feature2', 'feature3'],
            method='minmax',
            feature_range=[-1, 1],
            handle_outliers=False
        )
        
        print(minmax_result.output if minmax_result.output else minmax_result.error)
    
    asyncio.run(test_tool())
