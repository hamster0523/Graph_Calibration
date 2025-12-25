import pandas as pd
import numpy as np
import os
import warnings
from typing import Union, List, Optional
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score

from app.tool.base import BaseTool, ToolResult
from app.exceptions import ToolError
from app.logger import logger

def perform_rfe(data: pd.DataFrame, 
                target: Union[str, pd.Series], 
                n_features_to_select: Union[int, float] = 0.5, 
                step: int = 1, 
                estimator: str = 'auto',
                columns: Union[str, List[str]] = None) -> pd.DataFrame:
    """
    Perform Recursive Feature Elimination (RFE) on the specified columns of a DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame containing features.
        target (str or pd.Series): The target variable. If string, it should be the name of the target column in data.
        n_features_to_select (int or float, optional): Number of features to select.
            If int, it represents the exact number of features.
            If float between 0 and 1, it represents the proportion of features to select.
            Defaults to 0.5 (50% of features).
        step (int, optional): Number of features to remove at each iteration. Defaults to 1.
        estimator (str, optional): The estimator to use for feature importance ranking.
            Options: 'auto', 'logistic', 'rf', 'linear', 'rf_regressor'.
            'auto' will automatically choose based on the target variable type.
            Defaults to 'auto'.
        columns (str or List[str], optional): Column label or sequence of labels to consider.
            If None, use all columns except the target (if target is a column name in data).
            Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with selected features
    """
    # Prepare the feature matrix and target vector
    if isinstance(target, str):
        y = data[target]
        X = data.drop(columns=[target])
    else:
        y = target
        X = data

    # Select columns if specified
    if columns:
        if isinstance(columns, str):
            columns = [columns]
        X = X[columns]

    # Determine the number of features to select
    if isinstance(n_features_to_select, float):
        n_features_to_select = max(1, int(n_features_to_select * X.shape[1]))

    # Determine if the target is continuous or discrete
    is_continuous = np.issubdtype(y.dtype, np.number) and len(np.unique(y)) > 10

    # Choose the estimator
    if estimator == 'auto':
        estimator = 'linear' if is_continuous else 'logistic'

    if estimator == 'logistic':
        est = LogisticRegression(random_state=42)
    elif estimator == 'rf':
        est = RandomForestClassifier(random_state=42)
    elif estimator == 'linear':
        est = LinearRegression()
    elif estimator == 'rf_regressor':
        est = RandomForestRegressor(random_state=42)
    else:
        raise ValueError("Invalid estimator. Choose 'auto', 'logistic', 'rf', 'linear', or 'rf_regressor'.")

    # Perform RFE
    rfe = RFE(estimator=est, n_features_to_select=n_features_to_select, step=step)
    rfe.fit(X, y)

    # Get selected features
    selected_features = X.columns[rfe.support_].tolist()

    return data[selected_features]


class RFETool(BaseTool):
    """
    Tool for performing Recursive Feature Elimination (RFE) on CSV/Excel files.
    Useful for feature selection, especially when dealing with high-dimensional data.
    RFE recursively eliminates features by training models and removing the least important features.
    """
    
    name: str = "perform_rfe"
    description: str = (
        "Perform Recursive Feature Elimination (RFE) on specified columns of a DataFrame. This tool is "
        "useful for feature selection, especially when dealing with high-dimensional data. RFE works by "
        "recursively training models and eliminating the least important features based on feature "
        "importance scores or coefficients. It's particularly effective for identifying the most relevant "
        "features for predictive modeling while reducing dimensionality."
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
                "description": "The name of the target column in the DataFrame. Must be present in the dataset."
            },
            "n_features_to_select": {
                "type": ["integer", "number"],
                "description": "Number of features to select. If int: exact number of features; If float (0-1): proportion of features to select.",
                "default": 0.5,
                "minimum": 0.01
            },
            "step": {
                "type": "integer",
                "description": "Number of features to remove at each iteration. Higher values speed up processing but may reduce selection quality.",
                "default": 1,
                "minimum": 1
            },
            "estimator": {
                "type": "string",
                "description": "The estimator to use for feature importance ranking. 'auto': automatically choose based on target type; 'logistic': LogisticRegression; 'rf': RandomForestClassifier; 'linear': LinearRegression; 'rf_regressor': RandomForestRegressor.",
                "enum": ["auto", "logistic", "rf", "linear", "rf_regressor"],
                "default": "auto"
            },
            "columns": {
                "type": ["string", "array", "null"],
                "description": "Column label or sequence of labels to consider for RFE. If None, all columns except target will be used.",
                "items": {"type": "string"},
                "default": None
            },
            "include_target": {
                "type": "boolean",
                "description": "Whether to include the target column in the output dataset.",
                "default": True
            },
            "cross_validate": {
                "type": "boolean",
                "description": "Whether to perform cross-validation to evaluate feature selection quality.",
                "default": True
            },
            "cv_folds": {
                "type": "integer",
                "description": "Number of cross-validation folds when cross_validate is True.",
                "default": 5,
                "minimum": 2,
                "maximum": 10
            },
            "output_file_path": {
                "type": "string",
                "description": "Path where the processed data should be saved. If not provided, will create 'rfe_selected.csv' in the same directory as the input file."
            }
        },
        "required": ["file_path", "target"]
    }

    async def execute(
        self,
        file_path: str,
        target: str,
        n_features_to_select: Union[int, float] = 0.5,
        step: int = 1,
        estimator: str = "auto",
        columns: Optional[Union[str, List[str]]] = None,
        include_target: bool = True,
        cross_validate: bool = True,
        cv_folds: int = 5,
        output_file_path: Optional[str] = None,
    ) -> ToolResult:
        """
        Execute Recursive Feature Elimination for feature selection.
        
        Uses RFE to iteratively train models and eliminate the least important features
        based on feature importance scores. Particularly effective for high-dimensional
        data where feature selection is crucial for model performance.
        
        Args:
            file_path: Path to the input data file (CSV or Excel)
            target: Name of the target column in the dataset
            n_features_to_select: Number or proportion of features to select
            step: Number of features to remove per iteration
            estimator: Model type for feature importance ('auto', 'logistic', 'rf', 'linear', 'rf_regressor')
            columns: Specific columns for RFE (None for all except target)
            include_target: Whether to include target in output dataset
            cross_validate: Whether to evaluate selection with cross-validation
            cv_folds: Number of cross-validation folds
            output_file_path: Optional output file path (defaults to create new file)
            
        Returns:
            ToolResult with detailed RFE analysis including selected features,
            feature rankings, model performance, and selection recommendations
        """
        try:
            # Validate inputs
            if isinstance(n_features_to_select, float):
                if not (0.01 <= n_features_to_select <= 1.0):
                    raise ToolError("When n_features_to_select is float, it must be between 0.01 and 1.0")
            elif isinstance(n_features_to_select, int):
                if n_features_to_select < 1:
                    raise ToolError("When n_features_to_select is int, it must be >= 1")
            
            if step < 1:
                raise ToolError("step must be >= 1")
            
            if not (2 <= cv_folds <= 10):
                raise ToolError("cv_folds must be between 2 and 10")
            
            if estimator not in ["auto", "logistic", "rf", "linear", "rf_regressor"]:
                raise ToolError("estimator must be one of: 'auto', 'logistic', 'rf', 'linear', 'rf_regressor'")
            
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
            
            # Store original data info
            original_shape = data.shape
            warnings_list = []
            
            # Prepare feature matrix
            y = data[target]
            
            # Determine feature columns
            if columns is None:
                # Use all columns except target
                feature_columns = [col for col in data.columns if col != target]
            else:
                # Use specified columns
                if isinstance(columns, str):
                    feature_columns = [columns]
                else:
                    feature_columns = list(columns)
                
                # Validate columns exist and don't include target
                missing_columns = [col for col in feature_columns if col not in data.columns]
                if missing_columns:
                    raise ToolError(f"Columns not found in data: {missing_columns}")
                
                if target in feature_columns:
                    raise ToolError(f"Target column '{target}' cannot be included in feature columns")
            
            if len(feature_columns) == 0:
                raise ToolError("No feature columns available for RFE")
            
            X = data[feature_columns]
            
            # Validate target variable
            target_null_count = y.isnull().sum()
            if target_null_count > 0:
                raise ToolError(f"Target column '{target}' has {target_null_count} missing values. RFE requires complete target data.")
            
            # Check for missing values in features
            feature_null_counts = X.isnull().sum()
            if feature_null_counts.sum() > 0:
                warning_msg = f"Missing values in features: {dict(feature_null_counts[feature_null_counts > 0])}"
                warnings_list.append(warning_msg)
                logger.warning(warning_msg)
                
                # Remove rows with missing values
                complete_indices = X.dropna().index
                X = X.loc[complete_indices]
                y = y.loc[complete_indices]
                
                if len(X) == 0:
                    raise ToolError("No complete cases remaining after removing missing values")
            
            # Validate numeric features
            non_numeric_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
            if non_numeric_cols:
                raise ToolError(f"Non-numeric feature columns found: {non_numeric_cols}. RFE requires numeric features.")
            
            # Determine problem type and validate target
            unique_targets = len(y.unique())
            is_continuous = pd.api.types.is_numeric_dtype(y) and unique_targets > 10
            problem_type = "regression" if is_continuous else "classification"
            
            # Determine actual number of features to select
            total_features = X.shape[1]
            if isinstance(n_features_to_select, float):
                n_features_actual = max(1, int(n_features_to_select * total_features))
            else:
                n_features_actual = min(n_features_to_select, total_features)
            
            if n_features_actual >= total_features:
                warning_msg = f"Requested {n_features_actual} features but only {total_features} available. Using all features."
                warnings_list.append(warning_msg)
                n_features_actual = total_features
            
            # Choose estimator automatically if needed
            estimator_actual = estimator
            if estimator == 'auto':
                estimator_actual = 'linear' if is_continuous else 'logistic'
            
            # Validate estimator choice against problem type
            if problem_type == "regression" and estimator_actual in ['logistic', 'rf']:
                warning_msg = f"Using {estimator_actual} for regression. Consider 'linear' or 'rf_regressor' instead."
                warnings_list.append(warning_msg)
            elif problem_type == "classification" and estimator_actual in ['linear', 'rf_regressor']:
                warning_msg = f"Using {estimator_actual} for classification. Consider 'logistic' or 'rf' instead."
                warnings_list.append(warning_msg)
            
            # Create estimator
            try:
                if estimator_actual == 'logistic':
                    est = LogisticRegression(random_state=42, max_iter=1000)
                elif estimator_actual == 'rf':
                    est = RandomForestClassifier(random_state=42, n_estimators=100)
                elif estimator_actual == 'linear':
                    est = LinearRegression()
                elif estimator_actual == 'rf_regressor':
                    est = RandomForestRegressor(random_state=42, n_estimators=100)
                else:
                    raise ToolError(f"Invalid estimator: {estimator_actual}")
            except Exception as e:
                raise ToolError(f"Error creating estimator: {str(e)}")
            
            # Perform RFE
            try:
                rfe = RFE(estimator=est, n_features_to_select=n_features_actual, step=step)
                rfe.fit(X, y)
            except Exception as e:
                raise ToolError(f"Error during RFE execution: {str(e)}")
            
            # Get results
            selected_features = X.columns[rfe.support_].tolist()
            rejected_features = X.columns[~rfe.support_].tolist()
            feature_rankings = rfe.ranking_
            
            # Create feature ranking DataFrame
            feature_ranking_df = pd.DataFrame({
                'feature': X.columns,
                'ranking': feature_rankings,
                'selected': rfe.support_
            }).sort_values('ranking')
            
            # Cross-validation evaluation
            cv_results = {}
            if cross_validate and len(selected_features) > 0:
                try:
                    X_selected = X[selected_features]
                    
                    # Evaluate full model
                    full_scores = cross_val_score(est, X, y, cv=cv_folds, scoring='neg_mean_squared_error' if is_continuous else 'accuracy')
                    
                    # Evaluate selected features model
                    selected_scores = cross_val_score(est, X_selected, y, cv=cv_folds, scoring='neg_mean_squared_error' if is_continuous else 'accuracy')
                    
                    cv_results = {
                        'full_model_mean': float(full_scores.mean()),
                        'full_model_std': float(full_scores.std()),
                        'selected_model_mean': float(selected_scores.mean()),
                        'selected_model_std': float(selected_scores.std()),
                        'performance_change': float(selected_scores.mean() - full_scores.mean())
                    }
                except Exception as e:
                    warning_msg = f"Cross-validation failed: {str(e)}"
                    warnings_list.append(warning_msg)
                    logger.warning(warning_msg)
            
            # Create output dataset
            output_columns = selected_features.copy()
            if include_target:
                output_columns.append(target)
            
            # Use complete cases for output
            result_data = data.loc[X.index, output_columns]
            
            # Determine output path
            if output_file_path is None:
                output_file_path = os.path.join(os.path.dirname(file_path), "rfe_selected.csv")
            
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
            
            # Save feature rankings
            rankings_file_path = os.path.join(os.path.dirname(file_path), "rfe_feature_rankings.csv")
            try:
                feature_ranking_df.to_csv(rankings_file_path, index=False)
            except Exception as e:
                warning_msg = f"Could not save feature rankings: {str(e)}"
                warnings_list.append(warning_msg)
            
            # Generate summary
            final_shape = result_data.shape
            features_removed = len(rejected_features)
            
            summary = f"Recursive Feature Elimination completed successfully:\n"
            summary += f"- Input file: {file_path}\n"
            summary += f"- Output file: {output_file_path}\n"
            summary += f"- Rankings file: {rankings_file_path}\n"
            summary += f"- Target column: {target}\n"
            summary += f"- Problem type: {problem_type}\n"
            summary += f"- Estimator: {estimator_actual}\n"
            summary += f"- Step size: {step}\n"
            summary += f"- Original data shape: {original_shape}\n"
            summary += f"- Final data shape: {final_shape}\n"
            summary += f"- Total features analyzed: {total_features}\n"
            summary += f"- Features selected: {len(selected_features)}\n"
            summary += f"- Features removed: {features_removed}\n"
            summary += f"- Selection ratio: {len(selected_features)/total_features:.1%}\n"
            
            if warnings_list:
                summary += f"\nWARNINGS:\n"
                for warning in warnings_list:
                    summary += f"- {warning}\n"
            
            summary += f"\nRFE CHARACTERISTICS:\n"
            summary += f"- Recursive elimination based on feature importance\n"
            summary += f"- Model-dependent feature selection\n"
            summary += f"- Considers feature interactions through model training\n"
            summary += f"- Computationally expensive for large feature sets\n"
            summary += f"- Step size affects speed vs. optimality trade-off\n"
            
            summary += f"\nESTIMATOR CHARACTERISTICS:\n"
            if estimator_actual == 'logistic':
                summary += f"- LogisticRegression: Linear decision boundaries, coefficient-based importance\n"
                summary += f"- Good for linearly separable classification problems\n"
            elif estimator_actual == 'rf':
                summary += f"- RandomForestClassifier: Non-linear, feature importance from tree splits\n"
                summary += f"- Handles feature interactions and non-linear relationships\n"
            elif estimator_actual == 'linear':
                summary += f"- LinearRegression: Linear relationships, coefficient-based importance\n"
                summary += f"- Assumes linear relationship between features and target\n"
            elif estimator_actual == 'rf_regressor':
                summary += f"- RandomForestRegressor: Non-linear regression, feature importance from splits\n"
                summary += f"- Captures complex non-linear relationships\n"
            
            if cv_results:
                metric_name = "MSE" if is_continuous else "Accuracy"
                summary += f"\nCROSS-VALIDATION RESULTS ({cv_folds}-fold):\n"
                summary += f"- Full model {metric_name}: {cv_results['full_model_mean']:.4f} (±{cv_results['full_model_std']:.4f})\n"
                summary += f"- Selected features {metric_name}: {cv_results['selected_model_mean']:.4f} (±{cv_results['selected_model_std']:.4f})\n"
                summary += f"- Performance change: {cv_results['performance_change']:.4f}\n"
                
                if cv_results['performance_change'] < 0:
                    summary += f"- WARNING: Performance decreased with feature selection\n"
                else:
                    summary += f"- Feature selection maintained/improved performance\n"
            
            summary += f"\nSELECTED FEATURES (ranking order):\n"
            selected_ranking = feature_ranking_df[feature_ranking_df['selected']]
            for _, row in selected_ranking.iterrows():
                summary += f"  {row['ranking']:2d}. {row['feature']}\n"
            
            if len(rejected_features) > 0:
                summary += f"\nREJECTED FEATURES (showing top 10 by ranking):\n"
                rejected_ranking = feature_ranking_df[~feature_ranking_df['selected']].head(10)
                for _, row in rejected_ranking.iterrows():
                    summary += f"  {row['ranking']:2d}. {row['feature']}\n"
                if len(rejected_features) > 10:
                    summary += f"  ... and {len(rejected_features) - 10} more features\n"
            
            summary += f"\nAPPLICABLE SITUATIONS:\n"
            summary += f"- High-dimensional datasets with many irrelevant features\n"
            summary += f"- Feature selection for predictive modeling\n"
            summary += f"- Dimensionality reduction with model-based importance\n"
            summary += f"- Identifying core features for interpretable models\n"
            summary += f"- Preprocessing for computationally expensive algorithms\n"
            
            summary += f"\nLIMITATIONS AND CONSIDERATIONS:\n"
            summary += f"- Computationally expensive for large feature sets\n"
            summary += f"- Results depend heavily on the chosen estimator\n"
            summary += f"- May not find optimal feature subset (greedy approach)\n"
            summary += f"- Doesn't account for feature interactions not captured by estimator\n"
            summary += f"- Step size affects both speed and selection quality\n"
            summary += f"- Consider cross-validation for robust selection\n"
            
            if step > 1:
                summary += f"\nSTEP SIZE CONSIDERATION:\n"
                summary += f"- Current step size: {step}\n"
                summary += f"- Larger steps speed up processing but may miss optimal features\n"
                summary += f"- Consider step=1 for more thorough feature evaluation\n"
            
            return ToolResult(output=summary)
            
        except ToolError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            error_msg = f"Unexpected error in RFE analysis: {str(e)}"
            logger.error(error_msg)
            return ToolResult(error=error_msg)


# Usage example and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_tool():
        tool = RFETool()
        
        # Create sample data with relevant and irrelevant features
        np.random.seed(42)
        n_samples = 200
        
        # Create target variable
        target_base = np.random.normal(0, 1, n_samples)
        
        # Create relevant features (correlated with target)
        relevant_feature1 = target_base + np.random.normal(0, 0.3, n_samples)
        relevant_feature2 = target_base * 1.5 + np.random.normal(0, 0.4, n_samples)
        relevant_feature3 = -target_base + np.random.normal(0, 0.2, n_samples)
        
        # Create irrelevant features (random noise)
        irrelevant_feature1 = np.random.normal(0, 1, n_samples)
        irrelevant_feature2 = np.random.normal(5, 2, n_samples)
        irrelevant_feature3 = np.random.uniform(-1, 1, n_samples)
        irrelevant_feature4 = np.random.exponential(1, n_samples)
        
        # Create binary target for classification
        target_classification = (target_base > 0).astype(int)
        
        # Create regression target
        target_regression = target_base + relevant_feature1 * 0.5 + np.random.normal(0, 0.1, n_samples)
        
        sample_data = pd.DataFrame({
            'relevant_1': relevant_feature1,
            'relevant_2': relevant_feature2,
            'relevant_3': relevant_feature3,
            'irrelevant_1': irrelevant_feature1,
            'irrelevant_2': irrelevant_feature2,
            'irrelevant_3': irrelevant_feature3,
            'irrelevant_4': irrelevant_feature4,
            'target_class': target_classification,
            'target_reg': target_regression
        })
        
        # Save sample data
        sample_data.to_csv('test_rfe.csv', index=False)
        
        # Test RFE for classification
        result = await tool.execute(
            file_path='test_rfe.csv',
            target='target_class',
            n_features_to_select=0.6,  # Select 60% of features
            step=1,
            estimator='auto',
            columns=None,
            include_target=True,
            cross_validate=True,
            cv_folds=5,
            output_file_path='test_rfe_classification.csv'
        )
        
        print(result.output if result.output else result.error)
        
        # Test RFE for regression with Random Forest
        print("\n" + "="*50)
        print("Testing RFE for regression with Random Forest:")
        
        regression_result = await tool.execute(
            file_path='test_rfe.csv',
            target='target_reg',
            n_features_to_select=4,  # Select exactly 4 features
            step=2,  # Faster elimination
            estimator='rf_regressor',
            cross_validate=True
        )
        
        print(regression_result.output if regression_result.output else regression_result.error)
        
        # Test with specific columns
        print("\n" + "="*50)
        print("Testing RFE with specific feature columns:")
        
        specific_result = await tool.execute(
            file_path='test_rfe.csv',
            target='target_class',
            columns=['relevant_1', 'relevant_2', 'irrelevant_1', 'irrelevant_2'],
            n_features_to_select=2,
            estimator='logistic',
            cross_validate=False
        )
        
        print(specific_result.output if specific_result.output else specific_result.error)
    
    asyncio.run(test_tool())