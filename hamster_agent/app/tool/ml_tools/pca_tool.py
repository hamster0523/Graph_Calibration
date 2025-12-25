import pandas as pd
import numpy as np
import os
import warnings
from typing import Union, List, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from app.tool.base import BaseTool, ToolResult
from app.exceptions import ToolError
from app.logger import logger

def perform_pca(data: pd.DataFrame, n_components: Union[int, float, str] = 0.95, columns: Union[str, List[str]] = None, scale: bool = True) -> pd.DataFrame:
    """
    Perform Principal Component Analysis (PCA) on the specified columns of a DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame.
        n_components (int, float, or str, optional): Number of components to keep.
            If int, it represents the exact number of components.
            If float between 0 and 1, it represents the proportion of variance to be retained.
            If 'mle', Minka's MLE is used to guess the dimension.
            Defaults to 0.95 (95% of variance).
        columns (str or List[str], optional): Column label or sequence of labels to consider.
            If None, use all columns. Defaults to None.
        scale (bool, optional): Whether to scale the data before applying PCA.
            Recommended when features are not on the same scale. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame with PCA results

    Example:
        >>> df = pd.DataFrame({'feature1': [1, 2, 3, 4, 5], 'feature2': [2, 4, 5, 4, 5], 'feature3': [3, 6, 7, 8, 9]})
        >>> perform_pca(df, n_components=2)
                  PC1        PC2
        0  -2.121320  -0.707107
        1  -0.707107   0.707107
        2   0.000000   0.000000
        3   0.707107  -0.707107
        4   2.121320   0.707107
    """
    if columns is None:
        columns = data.columns
    elif isinstance(columns, str):
        columns = [columns]

    X = data[columns]

    # Check for non-numeric data types
    non_numeric_cols = X.select_dtypes(exclude=['number']).columns
    if not non_numeric_cols.empty:
        raise ValueError(f"Non-numeric data types detected in columns: {list(non_numeric_cols)}. "
                         "Please ensure all features are properly encoded and scaled before applying PCA.")

    # Warn if data doesn't seem to be scaled
    if (X.std() > 10).any():
        warnings.warn("Some features have high standard deviations. "
                      "Consider scaling your data before applying PCA for better results.")

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X)

    # Create a DataFrame with PCA results
    pca_df = pd.DataFrame(
        data=pca_result,
        columns=[f'PC{i+1}' for i in range(pca_result.shape[1])]
    )

    return pca_df


class PCAAnalysisTool(BaseTool):
    """
    Tool for performing Principal Component Analysis (PCA) on CSV/Excel files.
    Useful for dimensionality reduction, feature extraction, data visualization, 
    and handling multicollinearity in machine learning pipelines.
    """
    
    name: str = "perform_pca"
    description: str = (
        "Perform Principal Component Analysis (PCA) on specified columns of a DataFrame. This tool is "
        "useful for dimensionality reduction, feature extraction, and data visualization. PCA transforms "
        "the data into a set of orthogonal (uncorrelated) components that capture the maximum variance "
        "in the data. It's particularly effective for handling multicollinearity and reducing the "
        "dimensionality of high-dimensional datasets while preserving the most important information."
    )
    
    parameters: dict = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the CSV/Excel file containing the numerical features for PCA analysis"
            },
            "n_components": {
                "type": ["integer", "number", "string"],
                "description": "Number of components to keep. If int: exact number of components; If float (0-1): proportion of variance to retain; If 'mle': use Minka's MLE to estimate optimal dimensions.",
                "default": 0.95
            },
            "columns": {
                "type": ["string", "array", "null"],
                "description": "Column label or sequence of labels to consider for PCA. If None, all numerical columns will be used.",
                "items": {"type": "string"},
                "default": None
            },
            "scale": {
                "type": "boolean",
                "description": "Whether to scale the data before applying PCA. Highly recommended when features are not on the same scale.",
                "default": True
            },
            "include_original": {
                "type": "boolean",
                "description": "Whether to include original columns alongside PCA components in the output.",
                "default": False
            },
            "save_loadings": {
                "type": "boolean",
                "description": "Whether to save component loadings (feature contributions) to a separate file.",
                "default": True
            },
            "output_file_path": {
                "type": "string",
                "description": "Path where the PCA-transformed data should be saved. If not provided, will create 'pca_transformed.csv' in the same directory as the input file."
            }
        },
        "required": ["file_path"]
    }

    async def execute(
        self,
        file_path: str,
        n_components: Union[int, float, str] = 0.95,
        columns: Optional[Union[str, List[str]]] = None,
        scale: bool = True,
        include_original: bool = False,
        save_loadings: bool = True,
        output_file_path: Optional[str] = None,
    ) -> ToolResult:
        """
        Execute Principal Component Analysis for dimensionality reduction.
        
        Transforms high-dimensional data into a lower-dimensional representation
        while preserving the maximum amount of variance. Useful for visualization,
        noise reduction, and handling multicollinearity.
        
        Args:
            file_path: Path to the input data file (CSV or Excel)
            n_components: Number/proportion of components to keep or 'mle'
            columns: Specific columns for PCA (None for all numeric columns)
            scale: Whether to standardize features before PCA
            include_original: Whether to include original features in output
            save_loadings: Whether to save component loadings to separate file
            output_file_path: Optional output file path (defaults to create new file)
            
        Returns:
            ToolResult with detailed PCA analysis including explained variance,
            component interpretation, loadings analysis, and dimensionality insights
        """
        try:
            # Validate n_components
            if isinstance(n_components, (int, float)):
                if isinstance(n_components, float) and not (0 < n_components <= 1):
                    raise ToolError("When n_components is float, it must be between 0 and 1")
                if isinstance(n_components, int) and n_components < 1:
                    raise ToolError("When n_components is int, it must be >= 1")
            elif isinstance(n_components, str) and n_components != 'mle':
                raise ToolError("When n_components is string, it must be 'mle'")
            
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
            
            # Determine columns for PCA
            if columns is None:
                # Use all numeric columns
                numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_columns) == 0:
                    raise ToolError("No numeric columns found for PCA analysis")
                analysis_columns = numeric_columns
                
                # Warn about excluded columns
                non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
                if non_numeric_cols:
                    warning_msg = f"Non-numeric columns excluded from PCA: {non_numeric_cols}"
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
                
                # Validate columns are numeric
                non_numeric_cols = [col for col in analysis_columns if not pd.api.types.is_numeric_dtype(data[col])]
                if non_numeric_cols:
                    raise ToolError(f"Non-numeric columns found: {non_numeric_cols}. PCA requires numeric data.")
            
            if len(analysis_columns) < 2:
                raise ToolError("PCA requires at least 2 features for meaningful analysis")
            
            # Check for missing values
            analysis_data = data[analysis_columns]
            null_counts = analysis_data.isnull().sum()
            if null_counts.sum() > 0:
                warning_msg = f"Missing values found: {dict(null_counts[null_counts > 0])}"
                warnings_list.append(warning_msg)
                logger.warning(warning_msg)
                
                # Remove rows with missing values
                analysis_data = analysis_data.dropna()
                if len(analysis_data) == 0:
                    raise ToolError("No data remaining after removing missing values")
            
            # Validate n_components against data dimensions
            max_components = min(len(analysis_data), len(analysis_columns))
            if isinstance(n_components, int) and n_components > max_components:
                raise ToolError(f"n_components ({n_components}) cannot exceed min(n_samples, n_features) = {max_components}")
            
            # Check for constant features
            constant_features = analysis_data.columns[analysis_data.var() == 0].tolist()
            if constant_features:
                warning_msg = f"Constant features detected: {constant_features}. These will be excluded from PCA."
                warnings_list.append(warning_msg)
                logger.warning(warning_msg)
                analysis_data = analysis_data.drop(columns=constant_features)
                analysis_columns = [col for col in analysis_columns if col not in constant_features]
            
            # Calculate original statistics
            original_stats = {
                'mean': analysis_data.mean(),
                'std': analysis_data.std(),
                'var': analysis_data.var(),
                'min': analysis_data.min(),
                'max': analysis_data.max()
            }
            
            # Check if scaling is needed
            if not scale and (analysis_data.std() > 10).any():
                warning_msg = "Some features have high standard deviations. Consider setting scale=True for better PCA results."
                warnings_list.append(warning_msg)
                logger.warning(warning_msg)
            
            # Perform PCA
            try:
                # Prepare data
                X = analysis_data.values
                feature_names = analysis_data.columns.tolist()
                
                # Scale if requested
                if scale:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                else:
                    X_scaled = X
                
                # Fit PCA
                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(X_scaled)
                
                # Create PCA DataFrame
                n_components_actual = pca_result.shape[1]
                pca_columns = [f'PC{i+1}' for i in range(n_components_actual)]
                pca_df = pd.DataFrame(
                    data=pca_result,
                    columns=pca_columns,
                    index=analysis_data.index
                )
                
            except Exception as e:
                raise ToolError(f"Error during PCA computation: {str(e)}")
            
            # Calculate PCA statistics
            explained_variance_ratio = pca.explained_variance_ratio_
            explained_variance = pca.explained_variance_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            # Component loadings (how much each original feature contributes to each PC)
            loadings = pca.components_.T * np.sqrt(explained_variance)
            loadings_df = pd.DataFrame(
                loadings,
                columns=pca_columns,
                index=feature_names
            )
            
            # Find dominant features for each component
            component_interpretation = {}
            for i, pc in enumerate(pca_columns):
                abs_loadings = np.abs(loadings_df[pc])
                top_features = abs_loadings.nlargest(3)
                component_interpretation[pc] = {
                    'explained_variance': float(explained_variance_ratio[i]),
                    'cumulative_variance': float(cumulative_variance[i]),
                    'top_features': [(feat, float(loadings_df.loc[feat, pc])) for feat in top_features.index]
                }
            
            # Create output dataset
            if include_original:
                # Include non-analyzed columns
                other_columns = [col for col in data.columns if col not in analysis_columns]
                if other_columns:
                    result_data = pd.concat([
                        data[other_columns].loc[pca_df.index],
                        pca_df
                    ], axis=1)
                else:
                    result_data = pca_df
            else:
                result_data = pca_df
            
            # Determine output paths
            if output_file_path is None:
                output_file_path = os.path.join(os.path.dirname(file_path), "pca_transformed.csv")
            
            loadings_file_path = None
            if save_loadings:
                loadings_file_path = os.path.join(os.path.dirname(file_path), "pca_loadings.csv")
            
            # Save processed data
            try:
                if output_file_path.endswith('.csv'):
                    result_data.to_csv(output_file_path, index=False)
                elif output_file_path.endswith(('.xlsx', '.xls')):
                    result_data.to_excel(output_file_path, index=False)
                else:
                    if not output_file_path.endswith('.csv'):
                        output_file_path += '.csv'
                    result_data.to_csv(output_file_path, index=False)
                
                # Save loadings if requested
                if save_loadings:
                    loadings_df.to_csv(loadings_file_path)
                    
            except Exception as e:
                raise ToolError(f"Error saving processed data: {str(e)}")
            
            # Generate summary
            final_shape = result_data.shape
            variance_retained = float(cumulative_variance[-1])
            dimensionality_reduction = len(analysis_columns) - n_components_actual
            
            summary = f"Principal Component Analysis completed successfully:\n"
            summary += f"- Input file: {file_path}\n"
            summary += f"- Output file: {output_file_path}\n"
            if save_loadings:
                summary += f"- Loadings file: {loadings_file_path}\n"
            summary += f"- Features analyzed: {len(analysis_columns)}\n"
            summary += f"- Components extracted: {n_components_actual}\n"
            summary += f"- Dimensionality reduction: {dimensionality_reduction} features\n"
            summary += f"- Variance retained: {variance_retained:.1%}\n"
            summary += f"- Features scaled: {scale}\n"
            summary += f"- Original data shape: {original_shape}\n"
            summary += f"- Final data shape: {final_shape}\n"
            
            if warnings_list:
                summary += f"\nWARNINGS:\n"
                for warning in warnings_list:
                    summary += f"- {warning}\n"
            
            summary += f"\nPCA CHARACTERISTICS:\n"
            summary += f"- Linear transformation capturing maximum variance\n"
            summary += f"- Components are orthogonal (uncorrelated)\n"
            summary += f"- Sensitive to feature scaling - scaling {'applied' if scale else 'not applied'}\n"
            summary += f"- Assumes linear relationships between features\n"
            summary += f"- Results in loss of feature interpretability\n"
            
            summary += f"\nCOMPONENT ANALYSIS:\n"
            for pc, info in component_interpretation.items():
                summary += f"  {pc}:\n"
                summary += f"    - Explained variance: {info['explained_variance']:.1%}\n"
                summary += f"    - Cumulative variance: {info['cumulative_variance']:.1%}\n"
                summary += f"    - Top contributing features:\n"
                for feat, loading in info['top_features']:
                    summary += f"      * {feat}: {loading:.3f}\n"
            
            summary += f"\nVARIANCE BREAKDOWN:\n"
            summary += f"- Total variance explained: {variance_retained:.1%}\n"
            summary += f"- Variance per component: {', '.join([f'{v:.1%}' for v in explained_variance_ratio])}\n"
            
            if n_components_actual >= 2:
                summary += f"- First 2 components explain: {cumulative_variance[1]:.1%} of variance\n"
            
            # Recommendations based on variance retained
            if variance_retained < 0.8:
                summary += f"\nRECOMMENDATION: Only {variance_retained:.1%} variance retained. Consider:\n"
                summary += f"- Increasing n_components for better information preservation\n"
                summary += f"- Checking if PCA is appropriate for this dataset\n"
            elif variance_retained > 0.95:
                summary += f"\nRECOMMENDATION: High variance retained ({variance_retained:.1%}). Consider:\n"
                summary += f"- Reducing n_components for more dimensionality reduction\n"
                summary += f"- Current setting provides good information preservation\n"
            
            summary += f"\nAPPLICABLE SITUATIONS:\n"
            summary += f"- Dimensionality reduction for high-dimensional datasets\n"
            summary += f"- Feature extraction and noise reduction\n"
            summary += f"- Data visualization (using first 2-3 components)\n"
            summary += f"- Handling multicollinearity in regression problems\n"
            summary += f"- Preprocessing for machine learning algorithms\n"
            summary += f"- Exploratory data analysis and pattern discovery\n"
            
            summary += f"\nLIMITATIONS AND CONSIDERATIONS:\n"
            summary += f"- Assumes linear relationships between features\n"
            summary += f"- May not be suitable for categorical data\n"
            summary += f"- Loss of feature interpretability\n"
            summary += f"- Sensitive to outliers and feature scaling\n"
            summary += f"- May not preserve local data structure\n"
            summary += f"- Consider non-linear methods (t-SNE, UMAP) for complex data\n"
            
            if constant_features:
                summary += f"\nEXCLUDED CONSTANT FEATURES:\n"
                summary += f"  {', '.join(constant_features)}\n"
            
            return ToolResult(output=summary)
            
        except ToolError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            error_msg = f"Unexpected error in PCA analysis: {str(e)}"
            logger.error(error_msg)
            return ToolResult(error=error_msg)


# Usage example and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_tool():
        tool = PCAAnalysisTool()
        
        # Create sample data with correlated features
        np.random.seed(42)
        n_samples = 100
        
        # Create correlated features
        base_feature = np.random.normal(0, 1, n_samples)
        feature1 = base_feature + np.random.normal(0, 0.3, n_samples)
        feature2 = base_feature * 1.5 + np.random.normal(0, 0.5, n_samples)
        feature3 = -base_feature + np.random.normal(0, 0.4, n_samples)
        feature4 = np.random.normal(0, 1, n_samples)  # Independent
        feature5 = feature1 * 100 + np.random.normal(0, 10, n_samples)  # Different scale
        
        sample_data = pd.DataFrame({
            'feature1': feature1,
            'feature2': feature2,
            'feature3': feature3,
            'feature4': feature4,
            'feature5': feature5,
            'categorical': np.random.choice(['A', 'B', 'C'], n_samples),
            'target': base_feature + np.random.normal(0, 0.2, n_samples)
        })
        
        # Save sample data
        sample_data.to_csv('test_pca.csv', index=False)
        
        # Test PCA with variance-based component selection
        result = await tool.execute(
            file_path='test_pca.csv',
            n_components=0.9,  # Keep 90% of variance
            columns=None,  # Use all numeric columns
            scale=True,
            include_original=False,
            save_loadings=True,
            output_file_path='test_pca_transformed.csv'
        )
        
        print(result.output if result.output else result.error)
        
        # Test with fixed number of components
        print("\n" + "="*50)
        print("Testing with fixed number of components:")
        
        fixed_result = await tool.execute(
            file_path='test_pca.csv',
            n_components=3,  # Keep exactly 3 components
            columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'],
            scale=True,
            include_original=True
        )
        
        print(fixed_result.output if fixed_result.output else fixed_result.error)
        
        # Test with MLE component selection
        print("\n" + "="*50)
        print("Testing with MLE component selection:")
        
        mle_result = await tool.execute(
            file_path='test_pca.csv',
            n_components='mle',
            scale=True,
            save_loadings=False
        )
        
        print(mle_result.output if mle_result.output else mle_result.error)
    
    asyncio.run(test_tool())