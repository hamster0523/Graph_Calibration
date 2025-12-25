import pandas as pd
import numpy as np
import os
import warnings
from typing import Union, List, Optional
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

from app.tool.base import BaseTool, ToolResult
from app.exceptions import ToolError
from app.logger import logger

def create_feature_combinations(data: pd.DataFrame, 
                                columns: Union[str, List[str]], 
                                combination_type: str = 'multiplication', 
                                max_combination_size: int = 2) -> pd.DataFrame:
    """
    Create feature combinations from specified columns of a DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame.
        columns (str or List[str]): Column label or list of column labels to use for creating feature combinations.
        combination_type (str, optional): Type of combination to create. Options are 'multiplication' or 'addition'. Defaults to 'multiplication'.
        max_combination_size (int, optional): Maximum number of features to combine. Defaults to 2.

    Returns:
        pd.DataFrame: DataFrame with original and new combined features.

    Raises:
        ValueError: If specified columns are not numeric or if invalid parameters are provided.
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
                print(f"Warning: Duplicate identical columns found for '{col}'. Only one instance will be used for feature combinations.")
                unique_columns.append(col)
            else:
                raise ValueError(f"Duplicate non-identical columns found for '{col}'. Please resolve this before creating feature combinations.")
        else:
            unique_columns.append(col)

    # Check if all specified columns are numeric
    for col in unique_columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise ValueError(f"Column '{col}' is {data[col].dtype}, which is not numeric. Feature combinations require numeric data.")

    if max_combination_size < 2:
        raise ValueError("max_combination_size must be at least 2.")

    if combination_type not in ['multiplication', 'addition']:
        raise ValueError("combination_type must be either 'multiplication' or 'addition'.")

    result = data.copy()

    for r in range(2, min(len(unique_columns), max_combination_size) + 1):
        for combo in combinations(unique_columns, r):
            if combination_type == 'multiplication':
                new_col = result[list(combo)].prod(axis=1)
                new_col_name = ' * '.join(combo)
            else:  # addition
                new_col = result[list(combo)].sum(axis=1)
                new_col_name = ' + '.join(combo)
            
            result[new_col_name] = new_col

    if result.shape[1] > 1000:
        warnings.warn("The resulting DataFrame has over 1000 columns. "
                      "This may lead to computational issues and overfitting.")

    return result


class FeatureCombinationsTool(BaseTool):
    """
    Tool for creating feature combinations from numerical columns in CSV/Excel files.
    Useful for capturing interactions between features that may be important for the target variable.
    Supports both multiplication and addition combinations to discover complex patterns.
    """
    
    name: str = "create_feature_combinations"
    description: str = (
        "Create feature combinations from specified numerical columns of a DataFrame. This tool is "
        "essential for capturing interactions between features that may be crucial for target prediction. "
        "It generates multiplication or addition combinations that can help both linear and non-linear "
        "models discover complex patterns and relationships that individual features might not capture. "
        "Particularly valuable for revealing synergistic effects between variables."
    )
    
    parameters: dict = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the CSV/Excel file containing the dataset with numerical columns for feature combination"
            },
            "columns": {
                "type": ["string", "array"],
                "description": "Column label or list of column labels to use for creating feature combinations. All specified columns must be numerical.",
                "items": {"type": "string"}
            },
            "combination_type": {
                "type": "string",
                "description": "Type of combination to create. 'multiplication': captures non-linear interactions; 'addition': creates aggregate features.",
                "enum": ["multiplication", "addition"],
                "default": "multiplication"
            },
            "max_combination_size": {
                "type": "integer",
                "description": "Maximum number of features to combine. Higher values create more complex interactions but increase overfitting risk.",
                "default": 2,
                "minimum": 2,
                "maximum": 5
            },
            "target_column": {
                "type": "string",
                "description": "Optional target column name for analyzing combination importance and predictive power."
            },
            "analyze_importance": {
                "type": "boolean",
                "description": "Whether to analyze the importance of created combinations using mutual information or correlation.",
                "default": True
            },
            "importance_threshold": {
                "type": "number",
                "description": "Minimum importance score for highlighting significant feature combinations when analyze_importance is True.",
                "default": 0.05,
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
                "description": "Path where the data with feature combinations should be saved. If not provided, will create 'feature_combinations.csv' in the same directory as the input file."
            }
        },
        "required": ["file_path", "columns"]
    }

    async def execute(
        self,
        file_path: str,
        columns: Union[str, List[str]],
        combination_type: str = "multiplication",
        max_combination_size: int = 2,
        target_column: Optional[str] = None,
        analyze_importance: bool = True,
        importance_threshold: float = 0.05,
        max_features_warning: int = 1000,
        output_file_path: Optional[str] = None,
    ) -> ToolResult:
        """
        Execute feature combinations creation for capturing feature interactions.
        
        Creates mathematical combinations (multiplication or addition) of numerical
        features to reveal hidden patterns and interactions that individual features
        cannot capture. Particularly effective for enhancing model performance
        when feature interactions are important for the target variable.
        
        Args:
            file_path: Path to the input data file (CSV or Excel)
            columns: Column name(s) for feature combination (must be numerical)
            combination_type: Type of combination ('multiplication' or 'addition')
            max_combination_size: Maximum features per combination (2-5)
            target_column: Optional target for importance analysis
            analyze_importance: Whether to analyze combination importance
            importance_threshold: Minimum score for highlighting combinations
            max_features_warning: Warning threshold for total feature count
            output_file_path: Optional output file path (defaults to create new file)
            
        Returns:
            ToolResult with detailed feature combination analysis including
            combination statistics, importance rankings, interpretability insights,
            and usage recommendations for different model types
        """
        try:
            # Validate inputs
            if combination_type not in ["multiplication", "addition"]:
                raise ToolError("combination_type must be 'multiplication' or 'addition'")
            
            if max_combination_size < 2 or max_combination_size > 5:
                raise ToolError("max_combination_size must be between 2 and 5")
            
            if not (0.0 <= importance_threshold <= 1.0):
                raise ToolError("importance_threshold must be between 0.0 and 1.0")
            
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
                raise ToolError(f"Non-numeric columns found: {non_numeric_cols}. Feature combinations require numeric data.")
            
            if len(numeric_cols) < 2:
                raise ToolError(f"At least 2 numeric columns required for combinations. Found: {len(numeric_cols)}")
            
            # Validate target column if provided
            target_data = None
            target_is_classification = False
            if target_column:
                if target_column not in data.columns:
                    raise ToolError(f"Target column '{target_column}' not found in data")
                
                target_data = data[target_column]
                # Check if target is classification or regression
                if pd.api.types.is_numeric_dtype(target_data):
                    unique_targets = len(target_data.unique())
                    target_is_classification = unique_targets <= 10  # Heuristic for classification
                else:
                    target_is_classification = True
                    # Convert non-numeric target to numeric for analysis
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    target_data = pd.Series(le.fit_transform(target_data), index=target_data.index)
            
            # Store original data info
            original_shape = data.shape
            original_feature_count = len(numeric_cols)
            
            # Check for missing values
            missing_data = data[numeric_cols].isnull().sum()
            if missing_data.sum() > 0:
                warning_msg = f"Missing values found in features: {dict(missing_data[missing_data > 0])}"
                warnings_list.append(warning_msg)
                logger.warning(warning_msg)
            
            # Calculate expected feature combinations
            from math import comb
            expected_combinations = 0
            for r in range(2, min(len(numeric_cols), max_combination_size) + 1):
                expected_combinations += comb(len(numeric_cols), r)
            
            total_expected_features = original_shape[1] + expected_combinations
            
            # Warn about potential computational issues
            if total_expected_features > max_features_warning:
                warning_msg = f"Expected {total_expected_features} total features (>{max_features_warning}). Consider reducing max_combination_size or feature count."
                warnings_list.append(warning_msg)
                logger.warning(warning_msg)
            
            if max_combination_size > 3 and len(numeric_cols) > 5:
                warning_msg = f"High combination size ({max_combination_size}) with many features ({len(numeric_cols)}) may cause overfitting"
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
                    'zeros': int((col_data == 0).sum()),
                    'missing_count': int(data[col].isnull().sum())
                }
            
            # Create feature combinations
            try:
                result_data = data.copy()
                combination_details = []
                
                for r in range(2, min(len(numeric_cols), max_combination_size) + 1):
                    for combo in combinations(numeric_cols, r):
                        if combination_type == 'multiplication':
                            new_col = result_data[list(combo)].prod(axis=1)
                            new_col_name = ' * '.join(combo)
                            operation = 'product'
                        else:  # addition
                            new_col = result_data[list(combo)].sum(axis=1)
                            new_col_name = ' + '.join(combo)
                            operation = 'sum'
                        
                        result_data[new_col_name] = new_col
                        
                        # Store combination details
                        combination_details.append({
                            'name': new_col_name,
                            'features': list(combo),
                            'size': len(combo),
                            'operation': operation,
                            'mean': float(new_col.mean()),
                            'std': float(new_col.std()),
                            'min': float(new_col.min()),
                            'max': float(new_col.max()),
                            'zeros': int((new_col == 0).sum()),
                            'inf_values': int(np.isinf(new_col).sum()),
                            'nan_values': int(new_col.isnull().sum())
                        })
                
            except Exception as e:
                raise ToolError(f"Error creating feature combinations: {str(e)}")
            
            # Analyze combinations
            combinations_created = len(combination_details)
            final_shape = result_data.shape
            
            # Check for problematic values
            problematic_combinations = []
            for combo in combination_details:
                if combo['inf_values'] > 0 or combo['nan_values'] > 0:
                    problematic_combinations.append(combo['name'])
            
            if problematic_combinations:
                warning_msg = f"Combinations with infinite/NaN values: {problematic_combinations[:5]}"
                if len(problematic_combinations) > 5:
                    warning_msg += f" (and {len(problematic_combinations)-5} more)"
                warnings_list.append(warning_msg)
            
            # Importance analysis
            importance_results = {}
            if analyze_importance and target_column and target_data is not None:
                try:
                    # Get new combination columns
                    combination_names = [combo['name'] for combo in combination_details]
                    
                    if len(combination_names) > 0:
                        # Prepare data for importance analysis
                        X_combinations = result_data[combination_names].dropna()
                        y_clean = target_data.loc[X_combinations.index]
                        
                        if len(X_combinations) > 0:
                            # Calculate mutual information
                            if target_is_classification:
                                mi_scores = mutual_info_classif(X_combinations, y_clean, random_state=42)
                            else:
                                mi_scores = mutual_info_regression(X_combinations, y_clean, random_state=42)
                            
                            # Normalize scores
                            if mi_scores.max() > 0:
                                mi_scores_normalized = mi_scores / mi_scores.max()
                            else:
                                mi_scores_normalized = mi_scores
                            
                            # Create importance DataFrame
                            importance_df = pd.DataFrame({
                                'combination': combination_names,
                                'importance_score': mi_scores_normalized,
                                'raw_score': mi_scores
                            }).sort_values('importance_score', ascending=False)
                            
                            # Filter significant combinations
                            significant_combinations = importance_df[
                                importance_df['importance_score'] >= importance_threshold
                            ]
                            
                            importance_results = {
                                'method': 'mutual_information',
                                'target_type': 'classification' if target_is_classification else 'regression',
                                'total_combinations': len(combination_names),
                                'significant_combinations': len(significant_combinations),
                                'threshold_used': importance_threshold,
                                'top_combinations': significant_combinations.head(10).to_dict('records')
                            }
                        
                except Exception as e:
                    warning_msg = f"Importance analysis failed: {str(e)}"
                    warnings_list.append(warning_msg)
                    logger.warning(warning_msg)
            
            # Determine output path
            if output_file_path is None:
                output_file_path = os.path.join(os.path.dirname(file_path), "feature_combinations.csv")
            
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
            
            # Save combination details
            details_file_path = os.path.join(os.path.dirname(file_path), "combination_details.csv")
            try:
                pd.DataFrame(combination_details).to_csv(details_file_path, index=False)
            except Exception as e:
                warning_msg = f"Could not save combination details: {str(e)}"
                warnings_list.append(warning_msg)
            
            # Generate summary
            summary = f"Feature Combinations Creation completed successfully:\n"
            summary += f"- Input file: {file_path}\n"
            summary += f"- Output file: {output_file_path}\n"
            summary += f"- Details file: {details_file_path}\n"
            summary += f"- Columns used: {numeric_cols}\n"
            summary += f"- Combination type: {combination_type}\n"
            summary += f"- Max combination size: {max_combination_size}\n"
            summary += f"- Original data shape: {original_shape}\n"
            summary += f"- Final data shape: {final_shape}\n"
            summary += f"- Original features: {original_feature_count}\n"
            summary += f"- New combinations: {combinations_created}\n"
            summary += f"- Total features: {final_shape[1]}\n"
            summary += f"- Feature expansion ratio: {final_shape[1]/original_shape[1]:.2f}x\n"
            
            if target_column:
                summary += f"- Target column: {target_column} ({'classification' if target_is_classification else 'regression'})\n"
            
            if warnings_list:
                summary += f"\nWARNINGS:\n"
                for warning in warnings_list:
                    summary += f"- {warning}\n"
            
            # Combination breakdown by size
            summary += f"\nCOMBINATION BREAKDOWN:\n"
            size_counts = {}
            for combo in combination_details:
                size = combo['size']
                size_counts[size] = size_counts.get(size, 0) + 1
            
            for size in sorted(size_counts.keys()):
                summary += f"- {size}-way combinations: {size_counts[size]}\n"
            
            # Original feature statistics
            summary += f"\nORIGINAL FEATURE STATISTICS:\n"
            for col, stats in feature_stats.items():
                summary += f"- {col}:\n"
                summary += f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}] (span: {stats['range']:.3f})\n"
                summary += f"  Mean±Std: {stats['mean']:.3f}±{stats['std']:.3f}\n"
                if stats['zeros'] > 0:
                    summary += f"  Zero values: {stats['zeros']}\n"
                if stats['missing_count'] > 0:
                    summary += f"  Missing values: {stats['missing_count']}\n"
            
            # Combination statistics
            summary += f"\nCOMBINATION STATISTICS:\n"
            if combination_type == 'multiplication':
                summary += f"- Multiplication combinations capture non-linear interactions\n"
                summary += f"- Useful for: detecting synergistic effects, product features\n"
                summary += f"- Warning: Can create extreme values or zeros\n"
            else:
                summary += f"- Addition combinations create aggregate features\n"
                summary += f"- Useful for: creating composite scores, total features\n"
                summary += f"- More stable than multiplication combinations\n"
            
            # Show example combinations
            summary += f"\nEXAMPLE COMBINATIONS (showing first 10):\n"
            for i, combo in enumerate(combination_details[:10]):
                summary += f"  {i+1}. {combo['name']}\n"
                summary += f"     Range: [{combo['min']:.3f}, {combo['max']:.3f}], Mean: {combo['mean']:.3f}\n"
                if combo['zeros'] > 0:
                    summary += f"     Zero values: {combo['zeros']}\n"
                if combo['inf_values'] > 0 or combo['nan_values'] > 0:
                    summary += f"     Problematic values: {combo['inf_values']} inf, {combo['nan_values']} NaN\n"
            
            if len(combination_details) > 10:
                summary += f"  ... and {len(combination_details) - 10} more combinations\n"
            
            # Importance analysis results
            if importance_results:
                summary += f"\nIMPORTANCE ANALYSIS RESULTS:\n"
                summary += f"- Method: {importance_results['method']}\n"
                summary += f"- Target type: {importance_results['target_type']}\n"
                summary += f"- Total combinations analyzed: {importance_results['total_combinations']}\n"
                summary += f"- Significant combinations (≥{importance_threshold:.3f}): {importance_results['significant_combinations']}\n"
                
                if importance_results['top_combinations']:
                    summary += f"\nTOP COMBINATIONS BY IMPORTANCE:\n"
                    for i, combo in enumerate(importance_results['top_combinations'][:5]):
                        summary += f"  {i+1}. {combo['combination']}: {combo['importance_score']:.3f}\n"
                else:
                    summary += f"- No combinations exceeded importance threshold\n"
            
            # Model application recommendations
            summary += f"\nMODEL APPLICATION RECOMMENDATIONS:\n"
            
            summary += f"\nLinear Models:\n"
            if combination_type == 'multiplication':
                summary += f"- Multiplication combinations enable non-linear pattern capture\n"
                summary += f"- Particularly valuable for linear/logistic regression\n"
                summary += f"- Benefits: Explicit interaction terms, interpretable coefficients\n"
            else:
                summary += f"- Addition combinations create composite features\n"
                summary += f"- Useful for creating meaningful aggregate variables\n"
            summary += f"- Considerations: Use regularization to prevent overfitting\n"
            
            summary += f"\nTree-based Models:\n"
            summary += f"- May benefit from explicit interactions not found by splits\n"
            summary += f"- Addition combinations can improve performance\n"
            summary += f"- Multiplication combinations less critical (trees capture interactions)\n"
            
            summary += f"\nNeural Networks:\n"
            summary += f"- Explicit combinations can help shallow networks\n"
            summary += f"- Deep networks may learn these patterns automatically\n"
            summary += f"- Consider for interpretability or feature constraint scenarios\n"
            
            # Usage considerations
            summary += f"\nUSAGE CONSIDERATIONS:\n"
            
            summary += f"\nOverfitting Risk:\n"
            summary += f"- Feature count increased by {combinations_created} ({(combinations_created/original_feature_count)*100:.0f}%)\n"
            if combinations_created > original_shape[0] / 10:
                summary += f"- Many combinations relative to samples may cause overfitting\n"
            if max_combination_size > 2:
                summary += f"- High-order combinations ({max_combination_size}-way) increase overfitting risk\n"
            
            summary += f"\nComputational Complexity:\n"
            if final_shape[1] > 100:
                summary += f"- {final_shape[1]} total features may slow training\n"
            if final_shape[1] > 1000:
                summary += f"- Consider feature selection or dimensionality reduction\n"
            
            summary += f"\nInterpretability:\n"
            if combination_type == 'multiplication':
                summary += f"- Multiplication: Feature A * Feature B represents interaction strength\n"
                summary += f"- Higher-order: Increasingly complex to interpret\n"
            else:
                summary += f"- Addition: Feature A + Feature B represents combined effect\n"
                summary += f"- Generally more interpretable than multiplication\n"
            
            summary += f"\nRecommended Next Steps:\n"
            summary += f"- Apply feature scaling if using algorithms sensitive to scale\n"
            summary += f"- Use feature selection to identify most valuable combinations\n"
            summary += f"- Monitor for multicollinearity in linear models\n"
            summary += f"- Validate performance with cross-validation\n"
            
            if problematic_combinations:
                summary += f"- Handle infinite/NaN values in problematic combinations\n"
            
            if importance_results and importance_results['significant_combinations'] > 0:
                summary += f"- Focus on top {min(5, importance_results['significant_combinations'])} combinations for model development\n"
            
            # Applicable situations
            summary += f"\nAPPLICABLE SITUATIONS:\n"
            summary += f"- Capturing feature interactions for linear models\n"
            summary += f"- Creating composite features for domain-specific insights\n"
            summary += f"- Enhancing model performance through explicit interactions\n"
            summary += f"- Feature engineering for competition machine learning\n"
            summary += f"- Discovering synergistic effects between variables\n"
            summary += f"- Creating interpretable interaction terms\n"
            
            # Limitations
            summary += f"\nLIMITATIONS AND CONSIDERATIONS:\n"
            summary += f"- Combinatorial explosion with many features or high max_combination_size\n"
            summary += f"- Risk of overfitting, especially with small datasets\n"
            summary += f"- May create multicollinearity issues\n"
            summary += f"- Multiplication can produce extreme values or numerical instability\n"
            summary += f"- Interpretation becomes complex with high-order combinations\n"
            summary += f"- Computational cost grows rapidly with feature count\n"
            
            if combination_type == 'multiplication':
                summary += f"\nMULTIPLICATION-SPECIFIC CONSIDERATIONS:\n"
                summary += f"- Can create very large or very small values\n"
                summary += f"- Zero values in any component create zero combinations\n"
                summary += f"- Consider feature scaling before combination\n"
                summary += f"- Monitor for numerical overflow/underflow\n"
            
            return ToolResult(output=summary)
            
        except ToolError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            error_msg = f"Unexpected error in feature combinations creation: {str(e)}"
            logger.error(error_msg)
            return ToolResult(error=error_msg)


# Usage example and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_tool():
        tool = FeatureCombinationsTool()
        
        # Create sample data with potential interactions
        np.random.seed(42)
        n_samples = 200
        
        # Create base features with known interactions
        price = np.random.uniform(10, 100, n_samples)
        quantity = np.random.uniform(1, 20, n_samples)
        discount = np.random.uniform(0, 0.3, n_samples)
        
        # Create target with known interactions
        # Revenue = price * quantity * (1 - discount) + noise
        revenue = price * quantity * (1 - discount) + np.random.normal(0, 5, n_samples)
        
        # Create classification target
        high_revenue = (revenue > np.median(revenue)).astype(int)
        
        sample_data = pd.DataFrame({
            'price': price,
            'quantity': quantity,
            'discount': discount,
            'category': np.random.choice(['A', 'B', 'C'], n_samples),  # Non-numeric for testing
            'revenue': revenue,
            'high_revenue': high_revenue
        })
        
        # Save sample data
        sample_data.to_csv('test_combinations.csv', index=False)
        
        # Test multiplication combinations
        result = await tool.execute(
            file_path='test_combinations.csv',
            columns=['price', 'quantity', 'discount'],
            combination_type='multiplication',
            max_combination_size=3,
            target_column='revenue',
            analyze_importance=True,
            importance_threshold=0.1
        )
        
        print(result.output if result.output else result.error)
        
        # Test addition combinations
        print("\n" + "="*50)
        print("Testing addition combinations:")
        
        addition_result = await tool.execute(
            file_path='test_combinations.csv',
            columns=['price', 'quantity'],
            combination_type='addition',
            max_combination_size=2,
            target_column='high_revenue',
            analyze_importance=True,
            importance_threshold=0.05,
            output_file_path='test_combinations_addition.csv'
        )
        
        print(addition_result.output if addition_result.output else addition_result.error)
        
        # Test high combination size warning
        print("\n" + "="*50)
        print("Testing high combination size (should trigger warnings):")
        
        high_size_result = await tool.execute(
            file_path='test_combinations.csv',
            columns=['price', 'quantity', 'discount'],
            combination_type='multiplication',
            max_combination_size=4,
            max_features_warning=20,
            analyze_importance=False
        )
        
        print(high_size_result.output if high_size_result.output else high_size_result.error)
        
        # Test error handling - non-numeric columns
        print("\n" + "="*50)
        print("Testing error handling with non-numeric columns:")
        
        error_result = await tool.execute(
            file_path='test_combinations.csv',
            columns=['price', 'category'],  # Mix of numeric and non-numeric
            combination_type='multiplication'
        )
        
        print(error_result.output if error_result.output else error_result.error)
    
    asyncio.run(test_tool())