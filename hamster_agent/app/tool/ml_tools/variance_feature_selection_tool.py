import pandas as pd
import numpy as np
import os
import warnings
from typing import Union, List, Optional
from sklearn.feature_selection import VarianceThreshold

from app.tool.base import BaseTool, ToolResult
from app.exceptions import ToolError
from app.logger import logger

def variance_feature_selection(data: pd.DataFrame, threshold: float = 0.0, columns: Union[str, List[str]] = None) -> pd.DataFrame:
    """
    Perform feature selection based on variance analysis.

    Args:
        data (pd.DataFrame): The input DataFrame containing features.
        threshold (float, optional): Features with a variance lower than this threshold will be removed. 
                                        Defaults to 0.0.
        columns (str or List[str], optional): Column label or sequence of labels to consider. 
                                                If None, use all columns. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with selected features and their variances.
    """
    if columns is None:
        columns = data.columns
    elif isinstance(columns, str):
        columns = [columns]

    # Select specified columns
    X = data[columns]

    # Initialize VarianceThreshold
    selector = VarianceThreshold(threshold=threshold)

    # Fit the selector
    selector.fit(X)

    # Get the mask of selected features
    feature_mask = selector.get_support()

    # Get the variances
    variances = selector.variances_

    # Create a DataFrame with selected features and their variances
    selected_features = pd.DataFrame({
        'feature': X.columns[feature_mask],
        'variance': variances[feature_mask]
    }).sort_values('variance', ascending=False)

    return selected_features


class VarianceFeatureSelectionTool(BaseTool):
    """
    Tool for performing feature selection based on variance analysis.
    Helps identify and remove features with low variance, which often contribute little to model performance.
    Particularly useful for removing constant or near-constant features in preprocessing pipelines.
    """
    
    name: str = "variance_feature_selection"
    description: str = (
        "Perform feature selection based on variance analysis. This tool helps identify and remove "
        "features with low variance, which often contribute little to model performance. It's particularly "
        "useful for dimensionality reduction and removing constant or near-constant features that provide "
        "little discriminative information. The tool analyzes the variance of each feature and removes "
        "those below a specified threshold."
    )
    
    parameters: dict = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the CSV/Excel file containing the dataset with features to analyze"
            },
            "threshold": {
                "type": "number",
                "description": "Features with variance lower than this threshold will be removed. 0.0 removes constant features; 0.16 removes binary features present in >80% of samples.",
                "default": 0.0,
                "minimum": 0.0
            },
            "columns": {
                "type": ["string", "array", "null"],
                "description": "Column label or sequence of labels to consider. If None, all numeric columns will be analyzed.",
                "items": {"type": "string"},
                "default": None
            },
            "scale_features": {
                "type": "boolean",
                "description": "Whether to scale features before variance analysis. Recommended when features are on different scales.",
                "default": False
            },
            "include_categorical": {
                "type": "boolean", 
                "description": "Whether to include categorical features in analysis. If True, categorical features will be encoded first.",
                "default": False
            },
            "output_file_path": {
                "type": "string",
                "description": "Path where the processed data should be saved. If not provided, will create 'variance_selected.csv' in the same directory as the input file."
            }
        },
        "required": ["file_path"]
    }

    async def execute(
        self,
        file_path: str,
        threshold: float = 0.0,
        columns: Optional[Union[str, List[str]]] = None,
        scale_features: bool = False,
        include_categorical: bool = False,
        output_file_path: Optional[str] = None,
    ) -> ToolResult:
        """
        Execute variance-based feature selection.
        
        Analyzes the variance of features and removes those with variance below the threshold.
        This helps eliminate constant or near-constant features that provide little information
        for machine learning models.
        
        Args:
            file_path: Path to the input data file (CSV or Excel)
            threshold: Minimum variance threshold for feature retention
            columns: Specific columns to analyze (None for all numeric columns)
            scale_features: Whether to scale features before variance analysis
            include_categorical: Whether to include categorical features (will be encoded)
            output_file_path: Optional output file path (defaults to create new file)
            
        Returns:
            ToolResult with detailed variance analysis including selected features,
            removed features, variance statistics, and scaling recommendations
        """
        try:
            # Validate inputs
            if threshold < 0:
                raise ToolError("threshold must be non-negative")
            
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
            
            # Store original data info
            original_shape = data.shape
            warnings_list = []
            
            # Determine columns to analyze
            if columns is None:
                # Select numeric columns by default
                numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                
                if include_categorical:
                    # Add categorical columns (will be encoded)
                    categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
                    analysis_columns = numeric_columns + categorical_columns
                else:
                    analysis_columns = numeric_columns
                    if len(data.select_dtypes(include=['object', 'category']).columns) > 0:
                        warning_msg = "Categorical columns excluded. Set include_categorical=True to include them."
                        warnings_list.append(warning_msg)
                        logger.warning(warning_msg)
            else:
                # Use specified columns
                if isinstance(columns, str):
                    analysis_columns = [columns]
                else:
                    analysis_columns = list(columns)
                
                # Validate columns exist
                missing_columns = [col for col in analysis_columns if col not in data.columns]
                if missing_columns:
                    raise ToolError(f"Columns not found in data: {missing_columns}")
            
            if len(analysis_columns) == 0:
                raise ToolError("No columns available for variance analysis")
            
            # Prepare data for analysis
            analysis_data = data[analysis_columns].copy()
            
            # Handle categorical columns if included
            categorical_cols = analysis_data.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols and include_categorical:
                warning_msg = f"Encoding categorical columns: {categorical_cols}"
                warnings_list.append(warning_msg)
                logger.warning(warning_msg)
                
                # Simple label encoding for categorical columns
                from sklearn.preprocessing import LabelEncoder
                for col in categorical_cols:
                    le = LabelEncoder()
                    analysis_data[col] = le.fit_transform(analysis_data[col].astype(str))
            
            # Handle missing values
            null_counts = analysis_data.isnull().sum()
            if null_counts.sum() > 0:
                warning_msg = f"Missing values found: {dict(null_counts[null_counts > 0])}"
                warnings_list.append(warning_msg)
                logger.warning(warning_msg)
                
                # Fill missing values with mean for numeric, mode for categorical
                for col in analysis_data.columns:
                    if analysis_data[col].dtype in ['object', 'category']:
                        analysis_data[col].fillna(analysis_data[col].mode().iloc[0] if len(analysis_data[col].mode()) > 0 else 'missing', inplace=True)
                    else:
                        analysis_data[col].fillna(analysis_data[col].mean(), inplace=True)
            
            # Scale features if requested
            original_variances = analysis_data.var()
            if scale_features:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                analysis_data_scaled = pd.DataFrame(
                    scaler.fit_transform(analysis_data),
                    columns=analysis_data.columns,
                    index=analysis_data.index
                )
                analysis_data_for_variance = analysis_data_scaled
                warning_msg = "Features scaled before variance analysis"
                warnings_list.append(warning_msg)
                logger.info(warning_msg)
            else:
                analysis_data_for_variance = analysis_data
                
                # Check if scaling might be needed
                numeric_cols = analysis_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    ranges = analysis_data[numeric_cols].max() - analysis_data[numeric_cols].min()
                    if ranges.max() / ranges.min() > 100:  # Large scale difference
                        warning_msg = "Features have very different scales. Consider setting scale_features=True."
                        warnings_list.append(warning_msg)
                        logger.warning(warning_msg)
            
            # Perform variance selection
            try:
                variance_results = variance_feature_selection(
                    data=analysis_data_for_variance,
                    threshold=threshold,
                    columns=None  # Use all columns in the prepared data
                )
            except Exception as e:
                raise ToolError(f"Error in variance analysis: {str(e)}")
            
            # Calculate additional statistics
            all_variances = analysis_data_for_variance.var()
            selected_features = variance_results['feature'].tolist() if len(variance_results) > 0 else []
            removed_features = [col for col in analysis_columns if col not in selected_features]
            
            # Variance statistics
            variance_stats = {
                'min_variance': float(all_variances.min()),
                'max_variance': float(all_variances.max()),
                'mean_variance': float(all_variances.mean()),
                'median_variance': float(all_variances.median()),
                'std_variance': float(all_variances.std())
            }
            
            # Identify constant and near-constant features
            constant_features = [col for col in analysis_columns if all_variances[col] == 0.0]
            near_constant_features = [col for col in analysis_columns if 0.0 < all_variances[col] <= 0.01]
            
            # Create output dataset
            if len(selected_features) > 0:
                # Include selected features plus any non-analyzed columns
                other_columns = [col for col in data.columns if col not in analysis_columns]
                output_columns = selected_features + other_columns
                result_data = data[output_columns].copy()
            else:
                # If no features selected, keep non-analyzed columns only
                other_columns = [col for col in data.columns if col not in analysis_columns]
                if other_columns:
                    result_data = data[other_columns].copy()
                    warning_msg = "No features passed variance threshold. Only non-analyzed columns retained."
                    warnings_list.append(warning_msg)
                else:
                    result_data = pd.DataFrame()
                    warning_msg = "No features passed variance threshold. Output dataset is empty."
                    warnings_list.append(warning_msg)
            
            # Determine output path
            if output_file_path is None:
                output_file_path = os.path.join(os.path.dirname(file_path), "variance_selected.csv")
            
            # Save processed data
            try:
                if len(result_data) > 0 and len(result_data.columns) > 0:
                    if output_file_path.endswith('.csv'):
                        result_data.to_csv(output_file_path, index=False)
                    elif output_file_path.endswith(('.xlsx', '.xls')):
                        result_data.to_excel(output_file_path, index=False)
                    else:
                        # Default to CSV
                        if not output_file_path.endswith('.csv'):
                            output_file_path += '.csv'
                        result_data.to_csv(output_file_path, index=False)
                else:
                    warning_msg = "No data to save - no features selected"
                    warnings_list.append(warning_msg)
            except Exception as e:
                raise ToolError(f"Error saving processed data: {str(e)}")
            
            # Generate summary
            final_shape = result_data.shape if len(result_data.columns) > 0 else (0, 0)
            
            summary = f"Variance-based feature selection completed:\n"
            summary += f"- Input file: {file_path}\n"
            summary += f"- Output file: {output_file_path}\n"
            summary += f"- Variance threshold: {threshold}\n"
            summary += f"- Features scaled: {scale_features}\n"
            summary += f"- Categorical included: {include_categorical}\n"
            summary += f"- Original data shape: {original_shape}\n"
            summary += f"- Final data shape: {final_shape}\n"
            summary += f"- Features analyzed: {len(analysis_columns)}\n"
            summary += f"- Features selected: {len(selected_features)}\n"
            summary += f"- Features removed: {len(removed_features)}\n"
            
            if warnings_list:
                summary += f"\nWARNINGS:\n"
                for warning in warnings_list:
                    summary += f"- {warning}\n"
            
            summary += f"\nVARIANCE STATISTICS:\n"
            summary += f"- Minimum variance: {variance_stats['min_variance']:.6f}\n"
            summary += f"- Maximum variance: {variance_stats['max_variance']:.6f}\n"
            summary += f"- Mean variance: {variance_stats['mean_variance']:.6f}\n"
            summary += f"- Median variance: {variance_stats['median_variance']:.6f}\n"
            summary += f"- Standard deviation of variances: {variance_stats['std_variance']:.6f}\n"
            
            if constant_features:
                summary += f"\nCONSTANT FEATURES (variance = 0.0):\n"
                for feat in constant_features:
                    summary += f"  - {feat}: {original_variances[feat]:.6f}\n"
            
            if near_constant_features:
                summary += f"\nNEAR-CONSTANT FEATURES (0.0 < variance ≤ 0.01):\n"
                for feat in near_constant_features:
                    summary += f"  - {feat}: {original_variances[feat]:.6f}\n"
            
            if len(variance_results) > 0:
                summary += f"\nSELECTED FEATURES (sorted by variance):\n"
                for _, row in variance_results.iterrows():
                    original_var = original_variances[row['feature']]
                    summary += f"  - {row['feature']}: {original_var:.6f}"
                    if scale_features:
                        summary += f" (scaled: {row['variance']:.6f})"
                    summary += "\n"
            
            if removed_features:
                summary += f"\nREMOVED FEATURES (below threshold):\n"
                for feat in removed_features:
                    summary += f"  - {feat}: {original_variances[feat]:.6f}\n"
            
            summary += f"\nVARIANCE ANALYSIS CHARACTERISTICS:\n"
            summary += f"- Identifies and removes low-variance features\n"
            summary += f"- Threshold 0.0 removes constant features\n"
            summary += f"- Higher thresholds remove near-constant features\n"
            summary += f"- For binary features: 0.16 threshold removes features >80% uniform\n"
            summary += f"- Does not consider feature-target relationships\n"
            summary += f"- Most effective for numerical features\n"
            
            summary += f"\nAPPLICABLE SITUATIONS:\n"
            summary += f"- Preprocessing step in machine learning pipelines\n"
            summary += f"- Dimensionality reduction for high-dimensional datasets\n"
            summary += f"- Removing constant or near-constant features\n"
            summary += f"- Feature engineering and data cleaning\n"
            summary += f"- Computational efficiency improvement\n"
            
            summary += f"\nRECOMMENDATIONS:\n"
            summary += f"- Combine with other feature selection methods\n"
            summary += f"- Consider feature scaling if features have different units\n"
            summary += f"- Be cautious with small datasets (unreliable variance estimates)\n"
            summary += f"- High variance doesn't guarantee feature importance\n"
            summary += f"- Consider domain knowledge when interpreting results\n"
            
            if categorical_cols and not include_categorical:
                summary += f"\nEXCLUDED CATEGORICAL FEATURES:\n"
                summary += f"  {', '.join(categorical_cols)}\n"
                summary += f"  Set include_categorical=True to include them in analysis\n"
            
            return ToolResult(output=summary)
            
        except ToolError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            error_msg = f"Unexpected error in variance feature selection: {str(e)}"
            logger.error(error_msg)
            return ToolResult(error=error_msg)


# Usage example and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_tool():
        tool = VarianceFeatureSelectionTool()
        
        # Create sample data with different variance characteristics
        np.random.seed(42)
        n_samples = 100
        
        # Create features with different variances
        constant_feature = np.ones(n_samples)  # Zero variance
        low_variance_feature = np.random.normal(0, 0.01, n_samples)  # Very low variance
        medium_variance_feature = np.random.normal(0, 1, n_samples)  # Normal variance
        high_variance_feature = np.random.normal(0, 10, n_samples)  # High variance
        binary_feature = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])  # Low variance binary
        categorical_feature = np.random.choice(['A', 'B', 'C'], n_samples)
        
        sample_data = pd.DataFrame({
            'constant': constant_feature,
            'low_variance': low_variance_feature,
            'medium_variance': medium_variance_feature,
            'high_variance': high_variance_feature,
            'binary_low_var': binary_feature,
            'categorical': categorical_feature,
            'target': np.random.normal(0, 1, n_samples)
        })
        
        # Save sample data
        sample_data.to_csv('test_variance_selection.csv', index=False)
        
        # Test the tool with default threshold (removes constant features)
        result = await tool.execute(
            file_path='test_variance_selection.csv',
            threshold=0.0,
            columns=None,
            scale_features=False,
            include_categorical=False,
            output_file_path='test_variance_selected.csv'
        )
        
        print(result.output if result.output else result.error)
        
        # Test with higher threshold and feature scaling
        print("\n" + "="*50)
        print("Testing with higher threshold and scaling:")
        
        high_threshold_result = await tool.execute(
            file_path='test_variance_selection.csv',
            threshold=0.1,
            scale_features=True,
            include_categorical=True
        )
        
        print(high_threshold_result.output if high_threshold_result.output else high_threshold_result.error)
    
    asyncio.run(test_tool())