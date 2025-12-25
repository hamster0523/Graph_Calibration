import pandas as pd
import numpy as np
import os
import warnings
from typing import Union, List, Optional
from scipy.stats import spearmanr, kendalltau

from app.tool.base import BaseTool, ToolResult
from app.exceptions import ToolError
from app.logger import logger

def correlation_feature_selection(data: pd.DataFrame, target: str, method: str = 'pearson', threshold: float = 0.5) -> pd.DataFrame:
    """
    Perform feature selection based on correlation analysis.

    Args:
        data (pd.DataFrame): The input DataFrame containing features and target.
        target (str): The name of the target column.
        method (str, optional): The correlation method to use. 
            Options: 'pearson', 'spearman', 'kendall'. Defaults to 'pearson'.
        threshold (float, optional): The correlation threshold for feature selection. 
            Features with absolute correlation greater than this value will be selected. 
            Defaults to 0.5.

    Returns:
        pd.DataFrame: DataFrame with selected features and their correlation with the target.
    """
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in the DataFrame.")

    # Separate features and target
    X = data.drop(columns=[target])
    y = data[target]

    # Calculate correlation
    if method == 'spearman':
        corr_matrix, _ = spearmanr(X, y)
        corr_with_target = pd.Series(corr_matrix[-1][:-1], index=X.columns)
    else:
        corr_with_target = X.apply(lambda x: x.corr(y, method=method))

    # Select features based on threshold
    selected_features = corr_with_target[abs(corr_with_target) > threshold]

    return pd.DataFrame({
        'feature': selected_features.index,
        'correlation': selected_features.values
    }).sort_values('correlation', key=abs, ascending=False)


class CorrelationFeatureSelectionTool(BaseTool):
    """
    Tool for performing feature selection based on correlation analysis with the target variable.
    Helps identify features that have strong linear or monotonic relationships with the target,
    useful for dimensionality reduction and identifying important features for predictive modeling.
    """
    
    name: str = "correlation_feature_selection"
    description: str = (
        "Perform feature selection based on correlation analysis. This tool helps identify features "
        "that have a strong correlation with the target variable, which is useful for feature selection, "
        "dimensionality reduction, and identifying important features for predictive modeling. "
        "Supports multiple correlation methods (Pearson, Spearman, Kendall) to capture different "
        "types of relationships between features and target variables."
    )
    
    parameters: dict = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the CSV/Excel file containing the dataset with features and target variable"
            },
            "target": {
                "type": "string",
                "description": "The name of the target column in the DataFrame. Must be a numeric column for correlation analysis."
            },
            "method": {
                "type": "string",
                "description": "The correlation method to use. 'pearson': linear relationships, sensitive to outliers; 'spearman': rank-based, captures monotonic relationships; 'kendall': rank-based, good for small samples.",
                "enum": ["pearson", "spearman", "kendall"],
                "default": "pearson"
            },
            "threshold": {
                "type": "number",
                "description": "The correlation threshold for feature selection. Features with absolute correlation greater than this value will be selected.",
                "default": 0.5,
                "minimum": 0.0,
                "maximum": 1.0
            },
            "include_target": {
                "type": "boolean",
                "description": "Whether to include the target column in the output dataset. If False, only selected features are included.",
                "default": True
            },
            "remove_multicollinear": {
                "type": "boolean",
                "description": "If True, remove highly correlated features among themselves to reduce multicollinearity.",
                "default": False
            },
            "multicollinearity_threshold": {
                "type": "number",
                "description": "Threshold for removing multicollinear features. Features with correlation above this value will be examined for removal.",
                "default": 0.9,
                "minimum": 0.0,
                "maximum": 1.0
            },
            "output_file_path": {
                "type": "string",
                "description": "Path where the processed data should be saved. If not provided, will create 'correlation_selected.csv' in the same directory as the input file."
            }
        },
        "required": ["file_path", "target"]
    }

    async def execute(
        self,
        file_path: str,
        target: str,
        method: str = "pearson",
        threshold: float = 0.5,
        include_target: bool = True,
        remove_multicollinear: bool = False,
        multicollinearity_threshold: float = 0.9,
        output_file_path: Optional[str] = None,
    ) -> ToolResult:
        """
        Execute correlation-based feature selection.
        
        Analyzes correlation between features and target variable to select the most
        relevant features. Supports different correlation methods and can handle
        multicollinearity removal for cleaner feature sets.
        
        Args:
            file_path: Path to the input data file (CSV or Excel)
            target: Name of the target column (must be numeric)
            method: Correlation method ('pearson', 'spearman', 'kendall')
            threshold: Minimum absolute correlation for feature selection
            include_target: Whether to include target in output dataset
            remove_multicollinear: Whether to remove highly correlated features
            multicollinearity_threshold: Threshold for multicollinearity removal
            output_file_path: Optional output file path (defaults to create new file)
            
        Returns:
            ToolResult with detailed correlation analysis including selected features,
            correlation statistics, multicollinearity analysis, and method-specific insights
        """
        try:
            # Validate inputs
            if method not in ["pearson", "spearman", "kendall"]:
                raise ToolError("method must be one of: 'pearson', 'spearman', 'kendall'")
            
            if not (0.0 <= threshold <= 1.0):
                raise ToolError("threshold must be between 0.0 and 1.0")
            
            if not (0.0 <= multicollinearity_threshold <= 1.0):
                raise ToolError("multicollinearity_threshold must be between 0.0 and 1.0")
            
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
            
            # Validate target column
            if target not in data.columns:
                raise ToolError(f"Target column '{target}' not found in data")
            
            if not pd.api.types.is_numeric_dtype(data[target]):
                raise ToolError(f"Target column '{target}' must be numeric for correlation analysis")
            
            # Store original data info
            original_shape = data.shape
            warnings_list = []
            
            # Identify numeric features
            numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
            if target in numeric_features:
                numeric_features.remove(target)
            
            if len(numeric_features) == 0:
                raise ToolError("No numeric features found for correlation analysis")
            
            # Check for non-numeric features
            non_numeric_features = [col for col in data.columns if col not in numeric_features and col != target]
            if non_numeric_features:
                warning_msg = f"Non-numeric features excluded from analysis: {non_numeric_features}"
                warnings_list.append(warning_msg)
                logger.warning(warning_msg)
            
            # Create dataset with only numeric features and target
            analysis_data = data[numeric_features + [target]].copy()
            
            # Handle missing values
            null_counts = analysis_data.isnull().sum()
            if null_counts.sum() > 0:
                warning_msg = f"Missing values found: {dict(null_counts[null_counts > 0])}"
                warnings_list.append(warning_msg)
                logger.warning(warning_msg)
                
                # Remove rows with missing values for correlation analysis
                analysis_data = analysis_data.dropna()
                if len(analysis_data) == 0:
                    raise ToolError("No data remaining after removing missing values")
            
            # Perform correlation analysis
            try:
                correlation_results = correlation_feature_selection(
                    data=analysis_data,
                    target=target,
                    method=method,
                    threshold=threshold
                )
            except Exception as e:
                raise ToolError(f"Error in correlation analysis: {str(e)}")
            
            # Calculate additional statistics
            X = analysis_data.drop(columns=[target])
            y = analysis_data[target]
            
            # Full correlation matrix for all features
            if method == 'pearson':
                full_corr_matrix = X.corr(method='pearson')
                target_correlations = X.corrwith(y, method='pearson')
            elif method == 'spearman':
                full_corr_matrix = X.corr(method='spearman')
                target_correlations = X.corrwith(y, method='spearman')
            else:  # kendall
                full_corr_matrix = X.corr(method='kendall')
                target_correlations = X.corrwith(y, method='kendall')
            
            # Selected features
            if len(correlation_results) > 0:
                selected_features = correlation_results['feature'].tolist()
            else:
                selected_features = []
                warning_msg = f"No features selected with correlation threshold {threshold}"
                warnings_list.append(warning_msg)
                logger.warning(warning_msg)
            
            # Multicollinearity analysis
            multicollinear_pairs = []
            removed_features = []
            
            if remove_multicollinear and len(selected_features) > 1:
                # Check for multicollinearity among selected features
                selected_corr_matrix = full_corr_matrix.loc[selected_features, selected_features]
                
                # Find highly correlated pairs
                for i in range(len(selected_features)):
                    for j in range(i+1, len(selected_features)):
                        feat1, feat2 = selected_features[i], selected_features[j]
                        corr_val = abs(selected_corr_matrix.loc[feat1, feat2])
                        
                        if corr_val > multicollinearity_threshold:
                            multicollinear_pairs.append((feat1, feat2, corr_val))
                            
                            # Remove feature with lower target correlation
                            target_corr1 = abs(target_correlations[feat1])
                            target_corr2 = abs(target_correlations[feat2])
                            
                            if target_corr1 < target_corr2 and feat1 not in removed_features:
                                removed_features.append(feat1)
                            elif feat2 not in removed_features:
                                removed_features.append(feat2)
                
                # Update selected features
                selected_features = [f for f in selected_features if f not in removed_features]
                
                if removed_features:
                    warning_msg = f"Removed {len(removed_features)} features due to multicollinearity: {removed_features}"
                    warnings_list.append(warning_msg)
                    logger.warning(warning_msg)
            
            # Create output dataset
            output_columns = selected_features.copy()
            if include_target:
                output_columns.append(target)
            
            # Use original data for output (not the analysis_data which may have dropped rows)
            if output_columns:
                result_data = data[output_columns].copy()
            else:
                # If no features selected, just include target
                result_data = data[[target]].copy() if include_target else pd.DataFrame()
            
            # Determine output path
            if output_file_path is None:
                output_file_path = os.path.join(os.path.dirname(file_path), "correlation_selected.csv")
            
            # Save processed data
            try:
                if len(result_data) > 0:
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
            final_shape = result_data.shape if len(result_data) > 0 else (0, 0)
            
            summary = f"Correlation-based feature selection completed:\n"
            summary += f"- Input file: {file_path}\n"
            summary += f"- Output file: {output_file_path}\n"
            summary += f"- Target column: {target}\n"
            summary += f"- Correlation method: {method}\n"
            summary += f"- Correlation threshold: {threshold}\n"
            summary += f"- Original data shape: {original_shape}\n"
            summary += f"- Final data shape: {final_shape}\n"
            summary += f"- Total numeric features analyzed: {len(numeric_features)}\n"
            summary += f"- Features selected: {len(selected_features)}\n"
            summary += f"- Features removed (multicollinearity): {len(removed_features)}\n"
            
            if warnings_list:
                summary += f"\nWARNINGS:\n"
                for warning in warnings_list:
                    summary += f"- {warning}\n"
            
            summary += f"\nCORRELATION METHOD CHARACTERISTICS:\n"
            if method == 'pearson':
                summary += f"- Pearson: Measures linear relationships, sensitive to outliers\n"
                summary += f"- Assumes normal distribution and linear relationships\n"
                summary += f"- Best for continuous variables with linear associations\n"
            elif method == 'spearman':
                summary += f"- Spearman: Rank-based, captures monotonic relationships\n"
                summary += f"- Robust to outliers and non-normal distributions\n"
                summary += f"- Good for ordinal data and non-linear monotonic relationships\n"
            else:  # kendall
                summary += f"- Kendall: Rank-based, good for small sample sizes\n"
                summary += f"- More robust than Spearman for small datasets\n"
                summary += f"- Measures concordance between variable pairs\n"
            
            if len(correlation_results) > 0:
                summary += f"\nSELECTED FEATURES (sorted by absolute correlation):\n"
                for _, row in correlation_results.iterrows():
                    summary += f"  {row['feature']}: {row['correlation']:.4f}\n"
            else:
                summary += f"\nNo features met the correlation threshold of {threshold}\n"
            
            if multicollinear_pairs:
                summary += f"\nMULTICOLLINEARITY ANALYSIS:\n"
                summary += f"- Highly correlated feature pairs (>{multicollinearity_threshold}):\n"
                for feat1, feat2, corr_val in multicollinear_pairs:
                    removal_status = "(removed)" if feat1 in removed_features or feat2 in removed_features else "(kept)"
                    summary += f"  {feat1} ↔ {feat2}: {corr_val:.4f} {removal_status}\n"
            
            # Target correlation statistics
            summary += f"\nTARGET CORRELATION STATISTICS:\n"
            summary += f"- Target mean: {y.mean():.4f}\n"
            summary += f"- Target std: {y.std():.4f}\n"
            summary += f"- Strongest positive correlation: {target_correlations.max():.4f} ({target_correlations.idxmax()})\n"
            summary += f"- Strongest negative correlation: {target_correlations.min():.4f} ({target_correlations.idxmin()})\n"
            summary += f"- Mean absolute correlation: {abs(target_correlations).mean():.4f}\n"
            
            summary += f"\nAPPLICABLE SITUATIONS:\n"
            summary += f"- Feature selection for machine learning models\n"
            summary += f"- Dimensionality reduction in high-dimensional datasets\n"
            summary += f"- Identifying important features for predictive modeling\n"
            summary += f"- Exploratory data analysis and feature importance ranking\n"
            summary += f"- Preprocessing step before applying complex algorithms\n"
            
            summary += f"\nLIMITATIONS AND CONSIDERATIONS:\n"
            summary += f"- Does not capture feature interactions or non-linear relationships\n"
            summary += f"- May miss important features with weak individual correlations\n"
            summary += f"- Consider domain knowledge when interpreting results\n"
            summary += f"- Correlation does not imply causation\n"
            summary += f"- High correlations between features (multicollinearity) can be problematic\n"
            
            if non_numeric_features:
                summary += f"\nEXCLUDED NON-NUMERIC FEATURES:\n"
                summary += f"  {', '.join(non_numeric_features)}\n"
                summary += f"  Consider encoding categorical features before correlation analysis\n"
            
            return ToolResult(output=summary)
            
        except ToolError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            error_msg = f"Unexpected error in correlation feature selection: {str(e)}"
            logger.error(error_msg)
            return ToolResult(error=error_msg)


# Usage example and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_tool():
        tool = CorrelationFeatureSelectionTool()
        
        # Create sample data with features and target
        np.random.seed(42)
        n_samples = 100
        
        # Create features with different correlation strengths
        target = np.random.normal(0, 1, n_samples)
        feature1 = target * 0.8 + np.random.normal(0, 0.3, n_samples)  # High correlation
        feature2 = target * 0.6 + np.random.normal(0, 0.5, n_samples)  # Medium correlation
        feature3 = target * 0.3 + np.random.normal(0, 0.8, n_samples)  # Low correlation
        feature4 = np.random.normal(0, 1, n_samples)  # No correlation
        feature5 = feature1 * 0.9 + np.random.normal(0, 0.1, n_samples)  # Multicollinear with feature1
        
        sample_data = pd.DataFrame({
            'feature1': feature1,
            'feature2': feature2,
            'feature3': feature3,
            'feature4': feature4,
            'feature5': feature5,
            'categorical': np.random.choice(['A', 'B', 'C'], n_samples),
            'target': target
        })
        
        # Save sample data
        sample_data.to_csv('test_correlation_selection.csv', index=False)
        
        # Test the tool with Pearson correlation
        result = await tool.execute(
            file_path='test_correlation_selection.csv',
            target='target',
            method='pearson',
            threshold=0.4,
            include_target=True,
            remove_multicollinear=True,
            multicollinearity_threshold=0.8,
            output_file_path='test_correlation_selected.csv'
        )
        
        print(result.output if result.output else result.error)
        
        # Test with Spearman correlation
        print("\n" + "="*50)
        print("Testing with Spearman correlation:")
        
        spearman_result = await tool.execute(
            file_path='test_correlation_selection.csv',
            target='target',
            method='spearman',
            threshold=0.3,
            remove_multicollinear=False
        )
        
        print(spearman_result.output if spearman_result.output else spearman_result.error)
    
    asyncio.run(test_tool())