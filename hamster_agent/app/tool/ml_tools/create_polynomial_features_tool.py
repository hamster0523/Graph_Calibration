import pandas as pd
import numpy as np
import os
import warnings
from typing import Union, List, Optional
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from itertools import combinations

from app.tool.base import BaseTool, ToolResult
from app.exceptions import ToolError
from app.logger import logger

def create_polynomial_features(data: pd.DataFrame, 
                               columns: Union[str, List[str]], 
                               degree: int = 2, 
                               interaction_only: bool = False, 
                               include_bias: bool = False) -> pd.DataFrame:
    """
    Create polynomial features from specified columns of a DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame.
        columns (str or List[str]): Column label or list of column labels to use for creating polynomial features.
        degree (int, optional): The degree of the polynomial features. Defaults to 2.
        interaction_only (bool, optional): If True, only interaction features are produced. Defaults to False.
        include_bias (bool, optional): If True, include a bias column (all 1s). Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with original and new polynomial features.

    Raises:
        ValueError: If specified columns are not numeric or if invalid parameters are provided.
    """
    if isinstance(columns, str):
        columns = [columns]

    if degree < 1:
        raise ValueError("Degree must be at least 1.")

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
                print(f"Warning: Duplicate identical columns found for '{col}'. Only one instance will be used for polynomial features.")
                unique_columns.append(col)
            else:
                raise ValueError(f"Duplicate non-identical columns found for '{col}'. Please resolve this before creating polynomial features.")
        else:
            unique_columns.append(col)

    # Check if all specified columns are numeric
    for col in unique_columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise ValueError(f"Column '{col}' is {data[col].dtype}, which is not numeric. Polynomial features require numeric data.")

    X = data[unique_columns]
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
    poly_features = poly.fit_transform(X)

    feature_names = poly.get_feature_names_out(unique_columns)
    poly_df = pd.DataFrame(poly_features, columns=feature_names, index=data.index)

    # Remove duplicate columns (original features)
    poly_df = poly_df.loc[:, ~poly_df.columns.duplicated()]

    result = pd.concat([data, poly_df], axis=1)

    if result.shape[1] > 1000:
        warnings.warn("The resulting DataFrame has over 1000 columns. "
                      "This may lead to computational issues and overfitting.")

    return result


class PolynomialFeaturesTool(BaseTool):
    """
    Tool for creating polynomial features from numerical columns in CSV/Excel files.
    Useful for capturing non-linear relationships and interactions between features.
    Particularly effective for enhancing linear models to learn non-linear patterns.
    """
    
    name: str = "create_polynomial_features"
    description: str = (
        "Create polynomial features from specified numerical columns of a DataFrame. This tool is "
        "essential for capturing non-linear relationships between features and target variables. "
        "It generates polynomial combinations and interactions that can help linear models learn "
        "complex patterns, enhance feature spaces for any model type, and reveal hidden interactions "
        "between variables. The tool supports various polynomial degrees and interaction options."
    )
    
    parameters: dict = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the CSV/Excel file containing the dataset with numerical columns for polynomial feature creation"
            },
            "columns": {
                "type": ["string", "array"],
                "description": "Column label or list of column labels to use for creating polynomial features. All specified columns must be numerical.",
                "items": {"type": "string"}
            },
            "degree": {
                "type": "integer",
                "description": "The degree of the polynomial features. Higher degrees capture more complex relationships but increase overfitting risk.",
                "default": 2,
                "minimum": 1,
                "maximum": 5
            },
            "interaction_only": {
                "type": "boolean",
                "description": "If True, only interaction features are produced (no pure polynomial terms like x^2). Useful for capturing feature interactions without individual polynomial terms.",
                "default": False
            },
            "include_bias": {
                "type": "boolean",
                "description": "If True, include a bias column (constant term of all 1s). Useful for some regression contexts.",
                "default": False
            },
            "analyze_correlations": {
                "type": "boolean",
                "description": "Whether to analyze correlations between original features and new polynomial features to identify the most impactful combinations.",
                "default": True
            },
            "correlation_threshold": {
                "type": "number",
                "description": "Minimum correlation threshold for highlighting significant polynomial features when analyze_correlations is True.",
                "default": 0.1,
                "minimum": 0.0,
                "maximum": 1.0
            },
            "max_features_warning": {
                "type": "integer",
                "description": "Issue warning if resulting DataFrame exceeds this number of features.",
                "default": 1000,
                "minimum": 100
            },
            "output_file_path": {
                "type": "string",
                "description": "Path where the data with polynomial features should be saved. If not provided, will create 'polynomial_features.csv' in the same directory as the input file."
            }
        },
        "required": ["file_path", "columns"]
    }

    async def execute(
        self,
        file_path: str,
        columns: Union[str, List[str]],
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = False,
        analyze_correlations: bool = True,
        correlation_threshold: float = 0.1,
        max_features_warning: int = 1000,
        output_file_path: Optional[str] = None,
    ) -> ToolResult:
        """
        Execute polynomial feature creation for capturing non-linear relationships.
        
        Creates polynomial combinations and interactions from numerical features
        to enhance model capacity for learning complex patterns. Particularly
        valuable for linear models that need to capture non-linear relationships.
        
        Args:
            file_path: Path to the input data file (CSV or Excel)
            columns: Column name(s) for polynomial feature creation (must be numerical)
            degree: Polynomial degree (1-5, higher = more complex but riskier)
            interaction_only: Only create interaction terms (no pure polynomials)
            include_bias: Include constant bias term
            analyze_correlations: Analyze correlations between features
            correlation_threshold: Minimum correlation for highlighting features
            max_features_warning: Warning threshold for total feature count
            output_file_path: Optional output file path (defaults to create new file)
            
        Returns:
            ToolResult with detailed polynomial feature analysis including
            feature statistics, correlation analysis, complexity warnings,
            and usage recommendations for different model types
        """
        try:
            # Validate inputs
            if degree < 1 or degree > 5:
                raise ToolError("degree must be between 1 and 5")
            
            if not (0.0 <= correlation_threshold <= 1.0):
                raise ToolError("correlation_threshold must be between 0.0 and 1.0")
            
            if max_features_warning < 100:
                raise ToolError("max_features_warning must be >= 100")
            
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
            
            # Standardize columns input
            if isinstance(columns, str):
                columns_list = [columns]
            else:
                columns_list = list(columns)
            
            if len(columns_list) == 0:
                raise ToolError("At least one column must be specified")
            
            # Validate columns exist
            missing_columns = set(columns_list) - set(data.columns)
            if missing_columns:
                raise ToolError(f"Columns not found in data: {list(missing_columns)}")
            
            # Remove duplicate columns
            unique_columns = []
            duplicates_found = []
            for col in columns_list:
                if col not in unique_columns:
                    unique_columns.append(col)
                else:
                    duplicates_found.append(col)
            
            warnings_list = []
            if duplicates_found:
                warning_msg = f"Duplicate columns removed: {duplicates_found}"
                warnings_list.append(warning_msg)
            
            # Validate numeric columns
            non_numeric_cols = []
            numeric_cols = []
            for col in unique_columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    numeric_cols.append(col)
                else:
                    non_numeric_cols.append(col)
            
            if non_numeric_cols:
                raise ToolError(f"Non-numeric columns found: {non_numeric_cols}. Polynomial features require numeric data.")
            
            if len(numeric_cols) == 0:
                raise ToolError("No numeric columns available for polynomial feature creation")
            
            # Store original data info
            original_shape = data.shape
            original_feature_count = len(numeric_cols)
            
            # Check for missing values
            missing_data = data[numeric_cols].isnull().sum()
            if missing_data.sum() > 0:
                warning_msg = f"Missing values found in features: {dict(missing_data[missing_data > 0])}"
                warnings_list.append(warning_msg)
                logger.warning(warning_msg)
            
            # Calculate expected feature count
            if interaction_only:
                # Only interaction terms
                expected_features = 0
                for i in range(2, degree + 1):
                    expected_features += len(list(combinations(range(len(numeric_cols)), i)))
            else:
                # All polynomial terms up to degree
                from math import comb
                expected_features = sum(comb(len(numeric_cols) + d - 1, d) for d in range(1, degree + 1))
            
            if include_bias:
                expected_features += 1
            
            total_expected_features = original_shape[1] + expected_features
            
            # Warn about potential computational issues
            if total_expected_features > max_features_warning:
                warning_msg = f"Expected {total_expected_features} total features (>{max_features_warning}). Consider reducing degree or feature count."
                warnings_list.append(warning_msg)
                logger.warning(warning_msg)
            
            if degree > 3 and len(numeric_cols) > 5:
                warning_msg = f"High degree ({degree}) with many features ({len(numeric_cols)}) may cause overfitting"
                warnings_list.append(warning_msg)
            
            # Analyze feature statistics before transformation
            feature_stats = {}
            for col in numeric_cols:
                col_data = data[col].dropna()
                feature_stats[col] = {
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'range': float(col_data.max() - col_data.min()),
                    'missing_count': int(data[col].isnull().sum())
                }
            
            # Create polynomial features
            try:
                X = data[numeric_cols]
                poly = PolynomialFeatures(
                    degree=degree, 
                    interaction_only=interaction_only, 
                    include_bias=include_bias
                )
                poly_features = poly.fit_transform(X)
                
                # Get feature names
                feature_names = poly.get_feature_names_out(numeric_cols)
                poly_df = pd.DataFrame(poly_features, columns=feature_names, index=data.index)
                
                # Remove original features to avoid duplication
                new_feature_names = [name for name in feature_names if name not in numeric_cols]
                new_poly_df = poly_df[new_feature_names]
                
                # Combine with original data
                result_data = pd.concat([data, new_poly_df], axis=1)
                
            except Exception as e:
                raise ToolError(f"Error creating polynomial features: {str(e)}")
            
            # Analyze new features
            new_features_created = len(new_feature_names)
            final_shape = result_data.shape
            
            # Identify feature types
            feature_types = {
                'bias': [],
                'linear': [],
                'polynomial': [],
                'interaction': []
            }
            
            for name in new_feature_names:
                if name == '1':  # bias term
                    feature_types['bias'].append(name)
                elif ' ' not in name:  # single feature
                    if '^' in name:
                        feature_types['polynomial'].append(name)
                    else:
                        feature_types['linear'].append(name)
                else:  # interaction term
                    feature_types['interaction'].append(name)
            
            # Correlation analysis
            correlation_results = {}
            if analyze_correlations and new_features_created > 0:
                try:
                    # Calculate correlations between original and new features
                    corr_matrix = result_data[numeric_cols + new_feature_names].corr()
                    
                    # Find significant correlations between original and new features
                    significant_correlations = []
                    for orig_col in numeric_cols:
                        for new_col in new_feature_names:
                            corr_val = corr_matrix.loc[orig_col, new_col]
                            if abs(corr_val) >= correlation_threshold:
                                significant_correlations.append({
                                    'original_feature': orig_col,
                                    'polynomial_feature': new_col,
                                    'correlation': float(corr_val)
                                })
                    
                    correlation_results = {
                        'significant_correlations': significant_correlations,
                        'threshold_used': correlation_threshold,
                        'total_comparisons': len(numeric_cols) * len(new_feature_names)
                    }
                    
                except Exception as e:
                    warning_msg = f"Correlation analysis failed: {str(e)}"
                    warnings_list.append(warning_msg)
                    logger.warning(warning_msg)
            
            # Determine output path
            if output_file_path is None:
                output_file_path = os.path.join(os.path.dirname(file_path), "polynomial_features.csv")
            
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
            except Exception as e:
                raise ToolError(f"Error saving processed data: {str(e)}")
            
            # Generate summary
            summary = f"Polynomial Features Creation completed successfully:\n"
            summary += f"- Input file: {file_path}\n"
            summary += f"- Output file: {output_file_path}\n"
            summary += f"- Columns used: {numeric_cols}\n"
            summary += f"- Polynomial degree: {degree}\n"
            summary += f"- Interaction only: {interaction_only}\n"
            summary += f"- Include bias: {include_bias}\n"
            summary += f"- Original data shape: {original_shape}\n"
            summary += f"- Final data shape: {final_shape}\n"
            summary += f"- Original features: {original_feature_count}\n"
            summary += f"- New polynomial features: {new_features_created}\n"
            summary += f"- Total features: {final_shape[1]}\n"
            summary += f"- Feature expansion ratio: {final_shape[1]/original_shape[1]:.2f}x\n"
            
            if warnings_list:
                summary += f"\nWARNINGS:\n"
                for warning in warnings_list:
                    summary += f"- {warning}\n"
            
            # Feature type breakdown
            summary += f"\nFEATURE TYPE BREAKDOWN:\n"
            for ftype, features in feature_types.items():
                if features:
                    summary += f"- {ftype.title()} features: {len(features)}\n"
                    if len(features) <= 10:
                        summary += f"  Examples: {features[:5]}\n"
                    else:
                        summary += f"  Examples: {features[:3]} ... (and {len(features)-3} more)\n"
            
            # Original feature statistics
            summary += f"\nORIGINAL FEATURE STATISTICS:\n"
            for col, stats in feature_stats.items():
                summary += f"- {col}:\n"
                summary += f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}] (span: {stats['range']:.3f})\n"
                summary += f"  Mean±Std: {stats['mean']:.3f}±{stats['std']:.3f}\n"
                if stats['missing_count'] > 0:
                    summary += f"  Missing values: {stats['missing_count']}\n"
            
            # Correlation analysis results
            if correlation_results and correlation_results['significant_correlations']:
                summary += f"\nSIGNIFICANT CORRELATIONS (threshold ≥ {correlation_threshold:.2f}):\n"
                top_correlations = sorted(
                    correlation_results['significant_correlations'], 
                    key=lambda x: abs(x['correlation']), 
                    reverse=True
                )[:10]  # Show top 10
                
                for corr in top_correlations:
                    summary += f"- {corr['original_feature']} ↔ {corr['polynomial_feature']}: {corr['correlation']:.3f}\n"
                
                if len(correlation_results['significant_correlations']) > 10:
                    summary += f"  ... and {len(correlation_results['significant_correlations']) - 10} more correlations\n"
            
            # Polynomial feature characteristics
            summary += f"\nPOLYNOMIAL FEATURE CHARACTERISTICS:\n"
            summary += f"- Degree {degree} captures relationships up to {degree}-way interactions\n"
            if interaction_only:
                summary += f"- Interaction-only mode: captures feature combinations without pure polynomials\n"
                summary += f"- Useful for: detecting synergistic effects between features\n"
            else:
                summary += f"- Full polynomial: includes both pure polynomial terms and interactions\n"
                summary += f"- Useful for: capturing both individual non-linearity and feature interactions\n"
            
            if include_bias:
                summary += f"- Bias term included: provides constant offset for regression models\n"
            
            # Model application recommendations
            summary += f"\nMODEL APPLICATION RECOMMENDATIONS:\n"
            summary += f"Linear Models:\n"
            summary += f"- Linear/Logistic Regression: Polynomial features enable non-linear learning\n"
            summary += f"- Benefits: Simple interpretation, efficient training\n"
            summary += f"- Considerations: Use regularization (Ridge/Lasso) to prevent overfitting\n"
            
            summary += f"\nTree-based Models:\n"
            summary += f"- Random Forest/XGBoost: May benefit from explicit interactions\n"
            summary += f"- Benefits: Can capture relationships not found by tree splits\n"
            summary += f"- Considerations: Trees already capture interactions, so benefit may be limited\n"
            
            summary += f"\nNeural Networks:\n"
            summary += f"- Benefits: Provides explicit feature engineering for shallow networks\n"
            summary += f"- Considerations: Deep networks can learn these relationships automatically\n"
            
            # Usage considerations
            summary += f"\nUSAGE CONSIDERATIONS:\n"
            summary += f"Overfitting Risk:\n"
            summary += f"- Feature count increased by {new_features_created} ({(new_features_created/original_feature_count)*100:.0f}%)\n"
            if degree > 2:
                summary += f"- High degree ({degree}) increases overfitting risk\n"
            if new_features_created > original_shape[0] / 10:
                summary += f"- Many features relative to samples may cause overfitting\n"
            
            summary += f"\nComputational Complexity:\n"
            if final_shape[1] > 100:
                summary += f"- {final_shape[1]} features may slow training\n"
            if final_shape[1] > 1000:
                summary += f"- Consider feature selection or dimensionality reduction\n"
            
            summary += f"\nRecommended Next Steps:\n"
            summary += f"- Apply feature scaling/normalization for algorithms sensitive to scale\n"
            summary += f"- Use regularization (L1/L2) with linear models\n"
            summary += f"- Consider feature selection to remove low-impact polynomial terms\n"
            summary += f"- Validate model performance with cross-validation\n"
            if degree > 2:
                summary += f"- Monitor for overfitting with validation curves\n"
            
            # Applicable situations
            summary += f"\nAPPLICABLE SITUATIONS:\n"
            summary += f"- Enhancing linear models for non-linear pattern recognition\n"
            summary += f"- Capturing feature interactions in regression problems\n"
            summary += f"- Improving performance when non-linear relationships are suspected\n"
            summary += f"- Feature engineering for competition machine learning\n"
            summary += f"- Creating interaction terms for interpretable models\n"
            summary += f"- Preprocessing for algorithms that don't handle non-linearity well\n"
            
            # Limitations
            summary += f"\nLIMITATIONS AND CONSIDERATIONS:\n"
            summary += f"- Exponential feature growth with degree and input features\n"
            summary += f"- High risk of overfitting, especially with small datasets\n"
            summary += f"- May create multicollinearity issues\n"
            summary += f"- Requires careful regularization and validation\n"
            summary += f"- Computational cost increases significantly with complexity\n"
            summary += f"- Feature interpretability decreases with higher degrees\n"
            
            if degree >= 3 and len(numeric_cols) >= 4:
                summary += f"\nHIGH COMPLEXITY WARNING:\n"
                summary += f"- Current configuration (degree={degree}, features={len(numeric_cols)}) is computationally intensive\n"
                summary += f"- Consider reducing degree or using feature selection\n"
                summary += f"- Monitor memory usage and training time\n"
            
            return ToolResult(output=summary)
            
        except ToolError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            error_msg = f"Unexpected error in polynomial features creation: {str(e)}"
            logger.error(error_msg)
            return ToolResult(error=error_msg)


# Usage example and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_tool():
        tool = PolynomialFeaturesTool()
        
        # Create sample data with both linear and non-linear relationships
        np.random.seed(42)
        n_samples = 150
        
        # Create base features
        x1 = np.random.normal(0, 1, n_samples)
        x2 = np.random.normal(2, 1.5, n_samples)
        x3 = np.random.uniform(-2, 2, n_samples)
        
        # Create target with polynomial relationships
        # Linear component
        target_linear = 2 * x1 + 1.5 * x2 - 0.8 * x3
        # Non-linear components (polynomial and interaction)
        target_nonlinear = 0.5 * x1**2 - 0.3 * x2**2 + 0.4 * x1 * x2 + 0.2 * x1 * x3
        # Final target with noise
        target = target_linear + target_nonlinear + np.random.normal(0, 0.5, n_samples)
        
        # Create classification target
        target_class = (target > np.median(target)).astype(int)
        
        sample_data = pd.DataFrame({
            'feature_1': x1,
            'feature_2': x2,
            'feature_3': x3,
            'categorical_col': np.random.choice(['A', 'B', 'C'], n_samples),  # Non-numeric for testing
            'target_continuous': target,
            'target_binary': target_class
        })
        
        # Save sample data
        sample_data.to_csv('test_polynomial.csv', index=False)
        
        # Test basic polynomial features
        result = await tool.execute(
            file_path='test_polynomial.csv',
            columns=['feature_1', 'feature_2', 'feature_3'],
            degree=2,
            interaction_only=False,
            include_bias=False,
            analyze_correlations=True,
            correlation_threshold=0.1
        )
        
        print(result.output if result.output else result.error)
        
        # Test interaction-only features
        print("\n" + "="*50)
        print("Testing interaction-only polynomial features:")
        
        interaction_result = await tool.execute(
            file_path='test_polynomial.csv',
            columns=['feature_1', 'feature_2'],
            degree=3,
            interaction_only=True,
            include_bias=True,
            analyze_correlations=True,
            correlation_threshold=0.2,
            output_file_path='test_polynomial_interactions.csv'
        )
        
        print(interaction_result.output if interaction_result.output else interaction_result.error)
        
        # Test high-degree warning
        print("\n" + "="*50)
        print("Testing high-degree polynomial (should trigger warnings):")
        
        high_degree_result = await tool.execute(
            file_path='test_polynomial.csv',
            columns=['feature_1', 'feature_2', 'feature_3'],
            degree=4,
            max_features_warning=50,
            analyze_correlations=False
        )
        
        print(high_degree_result.output if high_degree_result.output else high_degree_result.error)
        
        # Test error handling - non-numeric columns
        print("\n" + "="*50)
        print("Testing error handling with non-numeric columns:")
        
        error_result = await tool.execute(
            file_path='test_polynomial.csv',
            columns=['feature_1', 'categorical_col'],  # Mix of numeric and non-numeric
            degree=2
        )
        
        print(error_result.output if error_result.output else error_result.error)
    
    asyncio.run(test_tool())