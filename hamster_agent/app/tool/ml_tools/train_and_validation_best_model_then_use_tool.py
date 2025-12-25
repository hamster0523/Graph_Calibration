import pandas as pd
import numpy as np
import os
import warnings
from typing import Union, List, Optional, Dict, Any
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (f1_score, mean_squared_error, accuracy_score, precision_score, 
                           recall_score, r2_score, mean_absolute_error, classification_report,
                           confusion_matrix, roc_auc_score)
from sklearn.preprocessing import LabelEncoder
import joblib

from app.tool.base import BaseTool, ToolResult
from app.exceptions import ToolError
from app.logger import logger

def train_and_validation_and_select_the_best_model(X, y, problem_type='binary', selected_models=['XGBoost', 'SVM', 'random forest']):
    """
    Train, validation and select the best machine learning model based on the training data and labels,
    and return the best performing model along with the performance scores of each model 
    with their best hyperparameters.

    This function is designed to automate the process of model training, model selection and hyperparameter tuning.
    It uses cross-validation to evaluate the performance of different models and selects the best one
    for the given problem type (binary classification, multiclass classification, or regression).
    
    Args:
        X (pd.DataFrame): Features for training.
        y (pd.Series): Labels for training.
        problem_type (str): Type of problem ('binary', 'multiclass', 'regression').
        selected_models (list, optional): List of model names to be considered for selection. 
                                          If None, a default set of models will be used.
                                          Default: ['XGBoost', 'SVM', 'random forest']
    
    Returns:
        best_model: The best performing model, trained on the train dataset.
    """
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models and their hyperparameter grids
    if problem_type in ['binary', 'multiclass']:
        models = {
            'logistic regression': (LogisticRegression(max_iter=1000), {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['saga'],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'l1_ratio': [0.5],
            }),
            'decision tree': (DecisionTreeClassifier(), {
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }),
            'random forest': (RandomForestClassifier(), {
                'n_estimators': [10, 50, 100],
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10]
            }),
            'XGBoost': (GradientBoostingClassifier(), {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5, 7]
            }),
            'SVM': (SVC(), {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }),
            'neural network': (MLPClassifier(max_iter=1000), {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            })
        }
        scoring = 'accuracy' if problem_type == 'binary' else 'f1_weighted'
    elif problem_type == 'regression':
        models = {
            'linear regression': (LinearRegression(), {
                'fit_intercept': [True, False],
                'copy_X': [True, False]
            }),
            'decision tree': (DecisionTreeRegressor(), {
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }),
            'random forest': (RandomForestRegressor(), {
                'n_estimators': [10, 50, 100],
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10]
            }),
            'XGBoost': (GradientBoostingRegressor(), {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5, 7]
            }),
            'SVM': (SVR(), {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }),
            'neural network': (MLPRegressor(max_iter=1000), {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            })
        }
        scoring = 'neg_mean_squared_error'
    else:
        raise ValueError("Invalid problem_type. Choose from 'binary', 'multiclass', or 'regression'.")

    best_model = None
    best_score = float('-inf') if problem_type in ['binary', 'multiclass'] else float('inf')
    results = {}

    models = {model_name: models[model_name] for model_name in selected_models}
    # Hyperparameter optimization
    for model_name, (model, param_grid) in models.items():
        optimizer = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring=scoring)
        optimizer.fit(X_train, y_train)
        print(f"Finished model training: {model_name}")
        
        # Evaluate the model on the validation set
        y_pred = optimizer.predict(X_val)
        if problem_type in ['binary', 'multiclass']:
            score = accuracy_score(y_val, y_pred) if problem_type == 'binary' else f1_score(y_val, y_pred, average='weighted')
        else:
            score = -mean_squared_error(y_val, y_pred)

        # Store the results
        results[model_name] = {
            'best_params': optimizer.best_params_,
            'score': score
        }

        if (problem_type in ['binary', 'multiclass'] and score > best_score) or \
           (problem_type == 'regression' and score < best_score):
            best_score = score
            best_model = optimizer.best_estimator_

    # Output results
    for model_name, result in results.items():
        print(f"Model: {model_name}, Best Params: {result['best_params']}, Score: {result['score']}")

    return best_model


class ModelTrainingAndPredictionTool(BaseTool):
    """
    Tool for automated model training, validation, selection, hyperparameter tuning, and prediction.
    This tool trains multiple ML models, selects the best performer, and applies it to make predictions
    on specified data. Supports binary/multiclass classification and regression tasks.
    """
    
    name: str = "train_and_validation_and_select_the_best_model"
    description: str = (
        "Automate model training, validation, selection, and hyperparameter tuning for various machine "
        "learning tasks. This tool trains multiple models with grid search optimization, selects the "
        "best performer based on cross-validation, and returns predictions along with comprehensive "
        "performance metrics. Ideal for automated machine learning workflows and model comparison."
    )
    
    parameters: dict = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the CSV/Excel file containing the dataset for model training"
            },
            "target_column": {
                "type": "string",
                "description": "Name of the target column in the dataset"
            },
            "feature_columns": {
                "type": ["array", "null"],
                "description": "List of feature column names to use for training. If None, all columns except target will be used.",
                "items": {"type": "string"}
            },
            "problem_type": {
                "type": "string",
                "description": "Type of machine learning problem",
                "enum": ["binary", "multiclass", "regression"],
                "default": "binary"
            },
            "selected_models": {
                "type": "array",
                "description": "List of model names to consider for selection and comparison",
                "items": {"type": "string"},
                "default": ["XGBoost", "SVM", "neural network"]
            },
            "test_size": {
                "type": "number",
                "description": "Proportion of dataset to use for testing (0.1-0.4)",
                "default": 0.2,
                "minimum": 0.1,
                "maximum": 0.4
            },
            "cv_folds": {
                "type": "integer",
                "description": "Number of cross-validation folds for model evaluation",
                "default": 5,
                "minimum": 3,
                "maximum": 10
            },
            "prediction_data_path": {
                "type": "string",
                "description": "Optional path to separate dataset for predictions. If not provided, predictions will be made on the test set."
            },
            "save_model": {
                "type": "boolean",
                "description": "Whether to save the best trained model to disk",
                "default": True
            },
            "output_predictions_path": {
                "type": "string",
                "description": "Path where predictions should be saved. If not provided, will create 'model_predictions.csv' in the same directory as input file."
            }
        },
        "required": ["file_path", "target_column"]
    }

    async def execute(
        self,
        file_path: str,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        problem_type: str = "binary",
        selected_models: List[str] = ["XGBoost", "SVM", "neural network"],
        test_size: float = 0.2,
        cv_folds: int = 5,
        prediction_data_path: Optional[str] = None,
        save_model: bool = True,
        output_predictions_path: Optional[str] = None,
    ) -> ToolResult:
        """
        Execute automated model training, selection, and prediction pipeline.
        
        Trains multiple machine learning models with hyperparameter optimization,
        selects the best performer through cross-validation, and applies it to
        make predictions. Provides comprehensive model comparison and performance
        analysis suitable for automated ML workflows.
        
        Args:
            file_path: Path to the training data file (CSV or Excel)
            target_column: Name of the target variable column
            feature_columns: List of feature columns (None for all except target)
            problem_type: Type of ML problem ('binary', 'multiclass', 'regression')
            selected_models: List of model names to train and compare
            test_size: Proportion of data for testing (0.1-0.4)
            cv_folds: Number of cross-validation folds (3-10)
            prediction_data_path: Optional separate prediction dataset
            save_model: Whether to save the best model to disk
            output_predictions_path: Path for saving predictions
            
        Returns:
            ToolResult with detailed model comparison, best model performance,
            predictions results, and comprehensive analysis including feature
            importance, model characteristics, and usage recommendations
        """
        try:
            # Validate inputs
            if problem_type not in ["binary", "multiclass", "regression"]:
                raise ToolError("problem_type must be 'binary', 'multiclass', or 'regression'")
            
            if not (0.1 <= test_size <= 0.4):
                raise ToolError("test_size must be between 0.1 and 0.4")
            
            if not (3 <= cv_folds <= 10):
                raise ToolError("cv_folds must be between 3 and 10")
            
            # Define available models per problem type
            available_models = {
                'binary': ["XGBoost", "SVM", "random forest", "decision tree", "logistic regression", "neural network"],
                'multiclass': ["XGBoost", "SVM", "random forest", "decision tree", "logistic regression", "neural network"],
                'regression': ["linear regression", "decision tree", "random forest", "XGBoost", "SVM", "neural network"]
            }
            
            # Validate selected models
            invalid_models = set(selected_models) - set(available_models[problem_type])
            if invalid_models:
                raise ToolError(f"Invalid models for {problem_type}: {list(invalid_models)}. Available: {available_models[problem_type]}")
            
            # Load training data
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
            if target_column not in data.columns:
                raise ToolError(f"Target column '{target_column}' not found in data")
            
            # Prepare features
            if feature_columns is None:
                feature_columns = [col for col in data.columns if col != target_column]
            else:
                missing_features = set(feature_columns) - set(data.columns)
                if missing_features:
                    raise ToolError(f"Feature columns not found: {list(missing_features)}")
            
            if len(feature_columns) == 0:
                raise ToolError("No feature columns available")
            
            # Prepare data
            X = data[feature_columns]
            y = data[target_column]
            
            warnings_list = []
            
            # Handle missing values
            missing_features = X.isnull().sum()
            missing_target = y.isnull().sum()
            
            if missing_features.sum() > 0 or missing_target > 0:
                warning_msg = f"Missing values found - Features: {dict(missing_features[missing_features > 0])}, Target: {missing_target}"
                warnings_list.append(warning_msg)
                
                # Remove rows with missing values
                complete_mask = X.notna().all(axis=1) & y.notna()
                X = X[complete_mask]
                y = y[complete_mask]
                
                if len(X) == 0:
                    raise ToolError("No complete cases remaining after removing missing values")
            
            # Validate and prepare target for problem type
            original_y = y.copy()
            label_encoder = None
            
            if problem_type in ['binary', 'multiclass']:
                unique_classes = len(y.unique())
                
                if problem_type == 'binary' and unique_classes != 2:
                    raise ToolError(f"Binary classification requires exactly 2 classes, found {unique_classes}")
                elif problem_type == 'multiclass' and unique_classes < 3:
                    raise ToolError(f"Multiclass classification requires 3+ classes, found {unique_classes}")
                
                # Encode non-numeric targets
                if not pd.api.types.is_numeric_dtype(y):
                    label_encoder = LabelEncoder()
                    y = pd.Series(label_encoder.fit_transform(y), index=y.index)
            
            else:  # regression
                if not pd.api.types.is_numeric_dtype(y):
                    raise ToolError("Regression requires numeric target variable")
            
            # Validate numeric features
            non_numeric_features = [col for col in feature_columns if not pd.api.types.is_numeric_dtype(X[col])]
            if non_numeric_features:
                warning_msg = f"Non-numeric features found: {non_numeric_features}. Consider encoding them first."
                warnings_list.append(warning_msg)
                
                # For simplicity, remove non-numeric features (could be enhanced with automatic encoding)
                X = X.select_dtypes(include=[np.number])
                feature_columns = list(X.columns)
                
                if len(feature_columns) == 0:
                    raise ToolError("No numeric features available after filtering")
            
            # Store data info
            original_shape = data.shape
            processed_shape = (len(X), len(feature_columns))
            
            # Define model configurations
            models_config = self._get_models_config(problem_type)
            
            # Filter selected models
            available_model_configs = {name: config for name, config in models_config.items() 
                                     if name in selected_models}
            
            if not available_model_configs:
                raise ToolError(f"No valid models selected from: {selected_models}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, 
                stratify=y if problem_type in ['binary', 'multiclass'] else None
            )
            
            # Train and evaluate models
            model_results = {}
            best_model = None
            best_score = float('-inf') if problem_type in ['binary', 'multiclass'] else float('inf')
            best_model_name = None
            
            scoring_metric = self._get_scoring_metric(problem_type)
            
            for model_name, (model_class, param_grid) in available_model_configs.items():
                try:
                    logger.info(f"Training {model_name}...")
                    
                    # Grid search with cross-validation
                    grid_search = GridSearchCV(
                        estimator=model_class, 
                        param_grid=param_grid, 
                        cv=cv_folds, 
                        scoring=scoring_metric,
                        n_jobs=-1
                    )
                    
                    grid_search.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = grid_search.predict(X_test)
                    
                    # Calculate metrics
                    metrics = self._calculate_metrics(y_test, y_pred, problem_type)
                    
                    # Cross-validation scores
                    cv_scores = cross_val_score(
                        grid_search.best_estimator_, X_train, y_train, 
                        cv=cv_folds, scoring=scoring_metric
                    )
                    
                    model_results[model_name] = {
                        'best_params': grid_search.best_params_,
                        'cv_score_mean': float(cv_scores.mean()),
                        'cv_score_std': float(cv_scores.std()),
                        'test_metrics': metrics,
                        'best_estimator': grid_search.best_estimator_
                    }
                    
                    # Select best model
                    current_score = metrics['primary_metric']
                    if self._is_better_score(current_score, best_score, problem_type):
                        best_score = current_score
                        best_model = grid_search.best_estimator_
                        best_model_name = model_name
                    
                    logger.info(f"Completed {model_name} - Score: {current_score:.4f}")
                    
                except Exception as e:
                    warning_msg = f"Failed to train {model_name}: {str(e)}"
                    warnings_list.append(warning_msg)
                    logger.warning(warning_msg)
            
            if best_model is None:
                raise ToolError("No models were successfully trained")
            
            # Load prediction data if provided
            prediction_results = {}
            if prediction_data_path:
                try:
                    if prediction_data_path.endswith('.csv'):
                        pred_data = pd.read_csv(prediction_data_path)
                    elif prediction_data_path.endswith(('.xlsx', '.xls')):
                        pred_data = pd.read_excel(prediction_data_path)
                    else:
                        raise ToolError("Prediction data must be CSV or Excel format")
                    
                    # Ensure same features are available
                    missing_pred_features = set(feature_columns) - set(pred_data.columns)
                    if missing_pred_features:
                        raise ToolError(f"Prediction data missing features: {list(missing_pred_features)}")
                    
                    X_pred = pred_data[feature_columns]
                    
                    # Handle missing values in prediction data
                    if X_pred.isnull().sum().sum() > 0:
                        warning_msg = "Missing values in prediction data. Rows with missing values will be skipped."
                        warnings_list.append(warning_msg)
                        
                        complete_pred_mask = X_pred.notna().all(axis=1)
                        X_pred_clean = X_pred[complete_pred_mask]
                        pred_indices = pred_data.index[complete_pred_mask]
                    else:
                        X_pred_clean = X_pred
                        pred_indices = pred_data.index
                    
                    if len(X_pred_clean) > 0:
                        predictions = best_model.predict(X_pred_clean)
                        
                        # Convert back to original labels if needed
                        if label_encoder is not None:
                            predictions = label_encoder.inverse_transform(predictions)
                        
                        prediction_results = {
                            'predictions': predictions.tolist(),
                            'indices': pred_indices.tolist(),
                            'total_predictions': len(predictions),
                            'skipped_rows': len(pred_data) - len(predictions)
                        }
                        
                        # Save predictions
                        pred_df = pred_data.copy()
                        pred_df['prediction'] = np.nan
                        pred_df.loc[pred_indices, 'prediction'] = predictions
                        
                        if output_predictions_path is None:
                            output_predictions_path = os.path.join(os.path.dirname(file_path), "model_predictions.csv")
                        
                        pred_df.to_csv(output_predictions_path, index=False)
                
                except Exception as e:
                    warning_msg = f"Error processing prediction data: {str(e)}"
                    warnings_list.append(warning_msg)
                    logger.warning(warning_msg)
            
            else:
                # Use test set for predictions
                test_predictions = best_model.predict(X_test)
                
                # Convert back to original labels if needed
                if label_encoder is not None:
                    test_predictions = label_encoder.inverse_transform(test_predictions)
                    original_test = label_encoder.inverse_transform(y_test)
                else:
                    original_test = y_test
                
                prediction_results = {
                    'predictions': test_predictions.tolist(),
                    'actual': original_test.tolist() if hasattr(original_test, 'tolist') else list(original_test),
                    'total_predictions': len(test_predictions)
                }
                
                # Save test predictions
                if output_predictions_path is None:
                    output_predictions_path = os.path.join(os.path.dirname(file_path), "model_predictions.csv")
                
                test_results_df = pd.DataFrame({
                    'actual': original_test,
                    'prediction': test_predictions
                })
                test_results_df.to_csv(output_predictions_path, index=False)
            
            # Save best model
            model_save_path = None
            if save_model:
                model_save_path = os.path.join(os.path.dirname(file_path), f"best_model_{best_model_name}.pkl")
                try:
                    joblib.dump(best_model, model_save_path)
                except Exception as e:
                    warning_msg = f"Could not save model: {str(e)}"
                    warnings_list.append(warning_msg)
                    model_save_path = None
            
            # Feature importance (if available)
            feature_importance = self._get_feature_importance(best_model, feature_columns)
            
            # Generate comprehensive summary
            summary = self._generate_summary(
                file_path, target_column, problem_type, selected_models,
                original_shape, processed_shape, model_results, best_model_name,
                prediction_results, feature_importance, warnings_list,
                output_predictions_path, model_save_path, label_encoder
            )
            
            return ToolResult(output=summary)
            
        except ToolError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            error_msg = f"Unexpected error in model training and prediction: {str(e)}"
            logger.error(error_msg)
            return ToolResult(error=error_msg)
    
    def _get_models_config(self, problem_type: str) -> Dict[str, tuple]:
        """Get model configurations for the specified problem type."""
        if problem_type in ['binary', 'multiclass']:
            return {
                'logistic regression': (LogisticRegression(max_iter=1000), {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'lbfgs'],
                    'penalty': ['l1', 'l2']
                }),
                'decision tree': (DecisionTreeClassifier(random_state=42), {
                    'max_depth': [None, 5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }),
                'random forest': (RandomForestClassifier(random_state=42), {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }),
                'XGBoost': (GradientBoostingClassifier(random_state=42), {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }),
                'SVM': (SVC(random_state=42), {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }),
                'neural network': (MLPClassifier(max_iter=1000, random_state=42), {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                })
            }
        else:  # regression
            return {
                'linear regression': (LinearRegression(), {
                    'fit_intercept': [True, False]
                }),
                'decision tree': (DecisionTreeRegressor(random_state=42), {
                    'max_depth': [None, 5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }),
                'random forest': (RandomForestRegressor(random_state=42), {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }),
                'XGBoost': (GradientBoostingRegressor(random_state=42), {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }),
                'SVM': (SVR(), {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }),
                'neural network': (MLPRegressor(max_iter=1000, random_state=42), {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                })
            }
    
    def _get_scoring_metric(self, problem_type: str) -> str:
        """Get the primary scoring metric for the problem type."""
        if problem_type == 'binary':
            return 'accuracy'
        elif problem_type == 'multiclass':
            return 'f1_weighted'
        else:  # regression
            return 'neg_mean_squared_error'
    
    def _calculate_metrics(self, y_true, y_pred, problem_type: str) -> Dict[str, float]:
        """Calculate comprehensive metrics for the given problem type."""
        if problem_type in ['binary', 'multiclass']:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'primary_metric': float(accuracy if problem_type == 'binary' else f1)
            }
            
            # Add AUC for binary classification
            if problem_type == 'binary':
                try:
                    auc = roc_auc_score(y_true, y_pred)
                    metrics['auc'] = float(auc)
                except:
                    pass
        
        else:  # regression
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            metrics = {
                'mse': float(mse),
                'rmse': float(np.sqrt(mse)),
                'mae': float(mae),
                'r2': float(r2),
                'primary_metric': float(-mse)  # Negative MSE for comparison
            }
        
        return metrics
    
    def _is_better_score(self, current_score: float, best_score: float, problem_type: str) -> bool:
        """Determine if current score is better than best score."""
        if problem_type in ['binary', 'multiclass']:
            return current_score > best_score
        else:  # regression (using negative MSE)
            return current_score > best_score
    
    def _get_feature_importance(self, model, feature_columns: List[str]) -> Optional[Dict[str, float]]:
        """Extract feature importance if available."""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_).flatten()
            else:
                return None
            
            # Normalize importance
            if importance.max() > 0:
                importance = importance / importance.max()
            
            return dict(zip(feature_columns, importance.astype(float)))
        except:
            return None
    
    def _generate_summary(self, file_path: str, target_column: str, problem_type: str, 
                         selected_models: List[str], original_shape: tuple, processed_shape: tuple,
                         model_results: Dict, best_model_name: str, prediction_results: Dict,
                         feature_importance: Optional[Dict], warnings_list: List[str],
                         output_predictions_path: str, model_save_path: Optional[str],
                         label_encoder) -> str:
        """Generate comprehensive summary report."""
        
        summary = f"Model Training, Validation and Prediction completed successfully:\n"
        summary += f"- Input file: {file_path}\n"
        summary += f"- Target column: {target_column}\n"
        summary += f"- Problem type: {problem_type}\n"
        summary += f"- Models evaluated: {selected_models}\n"
        summary += f"- Original data shape: {original_shape}\n"
        summary += f"- Processed data shape: {processed_shape}\n"
        summary += f"- Best model: {best_model_name}\n"
        summary += f"- Predictions saved to: {output_predictions_path}\n"
        
        if model_save_path:
            summary += f"- Model saved to: {model_save_path}\n"
        
        if warnings_list:
            summary += f"\nWARNINGS:\n"
            for warning in warnings_list:
                summary += f"- {warning}\n"
        
        # Model comparison
        summary += f"\nMODEL COMPARISON RESULTS:\n"
        sorted_results = sorted(
            model_results.items(), 
            key=lambda x: x[1]['test_metrics']['primary_metric'], 
            reverse=problem_type in ['binary', 'multiclass']
        )
        
        for rank, (model_name, results) in enumerate(sorted_results, 1):
            summary += f"{rank}. {model_name}:\n"
            summary += f"   Cross-validation: {results['cv_score_mean']:.4f} (±{results['cv_score_std']:.4f})\n"
            
            metrics = results['test_metrics']
            if problem_type in ['binary', 'multiclass']:
                summary += f"   Test Accuracy: {metrics['accuracy']:.4f}\n"
                summary += f"   Test F1-Score: {metrics['f1_score']:.4f}\n"
                summary += f"   Test Precision: {metrics['precision']:.4f}\n"
                summary += f"   Test Recall: {metrics['recall']:.4f}\n"
                if 'auc' in metrics:
                    summary += f"   Test AUC: {metrics['auc']:.4f}\n"
            else:
                summary += f"   Test RMSE: {metrics['rmse']:.4f}\n"
                summary += f"   Test MAE: {metrics['mae']:.4f}\n"
                summary += f"   Test R²: {metrics['r2']:.4f}\n"
            
            summary += f"   Best Parameters: {results['best_params']}\n\n"
        
        # Best model details
        best_results = model_results[best_model_name]
        summary += f"BEST MODEL DETAILS ({best_model_name}):\n"
        summary += f"- Cross-validation score: {best_results['cv_score_mean']:.4f} (±{best_results['cv_score_std']:.4f})\n"
        summary += f"- Optimal hyperparameters: {best_results['best_params']}\n"
        
        # Prediction results
        summary += f"\nPREDICTION RESULTS:\n"
        if 'actual' in prediction_results:
            summary += f"- Test set predictions: {prediction_results['total_predictions']} samples\n"
            summary += f"- Predictions vs actual values saved to output file\n"
        else:
            summary += f"- Total predictions made: {prediction_results.get('total_predictions', 0)}\n"
            if prediction_results.get('skipped_rows', 0) > 0:
                summary += f"- Skipped rows (missing values): {prediction_results['skipped_rows']}\n"
        
        # Feature importance
        if feature_importance:
            summary += f"\nFEATURE IMPORTANCE (Top 10):\n"
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            for i, (feature, importance) in enumerate(sorted_importance[:10], 1):
                summary += f"  {i:2d}. {feature}: {importance:.4f}\n"
        
        # Model characteristics
        summary += f"\nMODEL CHARACTERISTICS:\n"
        model_info = {
            'logistic regression': "Linear decision boundary, fast training, interpretable coefficients",
            'decision tree': "Non-linear, interpretable rules, prone to overfitting",
            'random forest': "Ensemble method, robust, handles non-linearity well",
            'XGBoost': "Gradient boosting, high performance, handles complex patterns",
            'SVM': "Margin-based, works well with high dimensions, kernel trick",
            'neural network': "Universal approximator, handles complex patterns, requires more data"
        }
        
        summary += f"- {best_model_name}: {model_info.get(best_model_name, 'Advanced ML algorithm')}\n"
        
        # Usage recommendations
        summary += f"\nUSAGE RECOMMENDATIONS:\n"
        
        if problem_type in ['binary', 'multiclass']:
            best_accuracy = best_results['test_metrics']['accuracy']
            if best_accuracy >= 0.9:
                summary += f"- Excellent model performance (accuracy: {best_accuracy:.3f})\n"
                summary += f"- Model is ready for production use\n"
            elif best_accuracy >= 0.8:
                summary += f"- Good model performance (accuracy: {best_accuracy:.3f})\n"
                summary += f"- Consider additional feature engineering or more data\n"
            else:
                summary += f"- Moderate performance (accuracy: {best_accuracy:.3f})\n"
                summary += f"- Significant improvement needed - check data quality and features\n"
        else:
            best_r2 = best_results['test_metrics']['r2']
            if best_r2 >= 0.8:
                summary += f"- Excellent model performance (R²: {best_r2:.3f})\n"
                summary += f"- Model explains most variance in target variable\n"
            elif best_r2 >= 0.6:
                summary += f"- Good model performance (R²: {best_r2:.3f})\n"
                summary += f"- Model captures important patterns\n"
            else:
                summary += f"- Moderate performance (R²: {best_r2:.3f})\n"
                summary += f"- Consider feature engineering or alternative approaches\n"
        
        summary += f"\nNEXT STEPS:\n"
        summary += f"- Validate predictions on new, unseen data\n"
        summary += f"- Monitor model performance over time\n"
        summary += f"- Consider ensemble methods if single model performance is insufficient\n"
        summary += f"- Retrain periodically with new data\n"
        
        if feature_importance:
            summary += f"- Focus on top features for further analysis\n"
            summary += f"- Consider removing low-importance features for model simplification\n"
        
        # Applicable situations
        summary += f"\nAPPLICABLE SITUATIONS:\n"
        summary += f"- Automated model selection and hyperparameter tuning\n"
        summary += f"- Comparing multiple algorithms for optimal performance\n"
        summary += f"- Generating predictions for new datasets\n"
        summary += f"- Establishing baseline models for ML projects\n"
        summary += f"- Model performance benchmarking\n"
        
        # Limitations
        summary += f"\nLIMITATIONS AND CONSIDERATIONS:\n"
        summary += f"- Grid search may not find global optimum\n"
        summary += f"- Limited hyperparameter ranges for computational efficiency\n"
        summary += f"- Requires numeric features (categorical encoding needed)\n"
        summary += f"- Performance depends on data quality and feature relevance\n"
        summary += f"- Model interpretability varies by algorithm choice\n"
        
        if label_encoder is not None:
            summary += f"- Target labels were encoded: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}\n"
        
        return summary


# Usage example and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_tool():
        tool = ModelTrainingAndPredictionTool()
        
        # Create sample classification dataset
        np.random.seed(42)
        n_samples = 300
        
        # Generate features
        feature1 = np.random.normal(0, 1, n_samples)
        feature2 = np.random.normal(2, 1.5, n_samples)
        feature3 = np.random.uniform(-2, 2, n_samples)
        
        # Create target with some relationship to features
        linear_combination = 0.5 * feature1 + 0.3 * feature2 - 0.2 * feature3
        probabilities = 1 / (1 + np.exp(-linear_combination))
        binary_target = (probabilities > 0.5).astype(int)
        
        # Create regression target
        regression_target = linear_combination + np.random.normal(0, 0.5, n_samples)
        
        # Create sample data
        sample_data = pd.DataFrame({
            'feature1': feature1,
            'feature2': feature2,
            'feature3': feature3,
            'categorical_feature': np.random.choice(['A', 'B', 'C'], n_samples),
            'binary_target': binary_target,
            'regression_target': regression_target
        })
        
        # Save sample data
        sample_data.to_csv('test_model_training.csv', index=False)
        
        # Test binary classification
        result = await tool.execute(
            file_path='test_model_training.csv',
            target_column='binary_target',
            feature_columns=['feature1', 'feature2', 'feature3'],
            problem_type='binary',
            selected_models=['XGBoost', 'SVM', 'random forest'],
            test_size=0.2,
            cv_folds=5,
            save_model=True
        )
        
        print(result.output if result.output else result.error)
        
        # Test regression
        print("\n" + "="*70)
        print("Testing regression problem:")
        
        regression_result = await tool.execute(
            file_path='test_model_training.csv',
            target_column='regression_target',
            problem_type='regression',
            selected_models=['linear regression', 'random forest', 'XGBoost'],
            output_predictions_path='regression_predictions.csv'
        )
        
        print(regression_result.output if regression_result.output else regression_result.error)
        
        # Test with separate prediction data
        print("\n" + "="*70)
        print("Testing with separate prediction dataset:")
        
        # Create separate prediction data
        pred_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 50),
            'feature2': np.random.normal(2, 1.5, 50),
            'feature3': np.random.uniform(-2, 2, 50),
            'categorical_feature': np.random.choice(['A', 'B', 'C'], 50)
        })
        pred_data.to_csv('prediction_data.csv', index=False)
        
        prediction_result = await tool.execute(
            file_path='test_model_training.csv',
            target_column='binary_target',
            feature_columns=['feature1', 'feature2', 'feature3'],
            problem_type='binary',
            selected_models=['XGBoost'],
            prediction_data_path='prediction_data.csv',
            output_predictions_path='separate_predictions.csv'
        )
        
        print(prediction_result.output if prediction_result.output else prediction_result.error)
    
    asyncio.run(test_tool())
