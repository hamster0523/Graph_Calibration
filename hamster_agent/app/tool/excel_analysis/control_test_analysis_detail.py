import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from app.config import config
from app.exceptions import ToolError
from app.logger import logger
from app.tool.base import BaseTool, ToolResult


class ControlTestAnalysisTool(BaseTool):

    name: str = "control_test_analysis_tool"
    description: str = "Analyze performance differences between control and test groups by device model. This tool compares performance metrics across different device models, identifies devices with performance degradation, generates visualizations, and saves analysis results including bar charts and categorized CSV data."
    parameters: dict = {
        "type": "object",
        "properties": {
            "csv_file_path": {
                "type": "string",
                "description": "Path to the CSV file containing control and test group data with devicemodel column",
            },
            "group_column": {
                "type": "string",
                "description": "Column name that identifies control/test groups",
                "default": "param_value"
            },
            "control_value": {
                "type": "string",
                "description": "Value in group_column that represents the control group",
                "default": "control"
            },
            "test_value": {
                "type": "string",
                "description": "Value in group_column that represents the test group",
                "default": "test"
            },
            "degradation_threshold": {
                "type": "number",
                "description": "Percentage threshold for determining performance degradation (e.g., 1.0 for 1% degradation)",
                "default": 1.0
            },
            "confidence_alpha": {
                "type": "number",
                "description": "Alpha parameter for confidence score calculation (0.5-2.0, where 1.0 is linear, >1.0 is conservative)",
                "default": 1.0
            },
            "visualization_top_percent": {
                "type": "number",
                "description": "Percentage of top degraded devices with highest confidence to show in charts (0-100)",
                "default": 20.0
            },
            "output_dir": {
                "type": "string",
                "description": "Directory to save analysis results. If not provided, creates 'control_test_analysis' directory in same location as CSV file",
            }
        },
        "required": ["csv_file_path"]
    }

    async def execute(
        self,
        csv_file_path: str,
        group_column: str = "param_value",
        control_value: str = "control",
        test_value: str = "test",
        performance_metrics: List[str] = [],
        degradation_threshold: float = 1.0,
        confidence_alpha: float = 1.0,
        visualization_top_percent: float = 20.0,
        output_dir: str = ""
    ) -> ToolResult:
        """
        Analyze control vs test group performance by device model.

        Args:
            csv_file_path: Path to CSV file with control/test data
            group_column: Column identifying control/test groups
            control_value: Value representing control group
            test_value: Value representing test group
            performance_metrics: List of metric columns to analyze
            degradation_threshold: Percentage threshold for degradation
            confidence_alpha: Alpha parameter for confidence score (0.5-2.0)
            visualization_top_percent: Top % of degraded devices to show in charts
            output_dir: Directory to save results

        Returns:
            ToolResult with analysis summary and degraded devices
        """
        try:
            # Validate input file
            if not os.path.exists(csv_file_path):
                raise ToolError(f"CSV file not found: {csv_file_path}")

            # Read CSV data
            try:
                df = pd.read_csv(csv_file_path)
                # logger.info(f"Successfully loaded CSV file with shape: {df.shape}")
            except Exception as e:
                raise ToolError(f"Error reading CSV file: {str(e)}")

            # Validate required columns
            if 'devicemodel' not in df.columns:
                raise ToolError("CSV file must contain 'devicemodel' column")

            if group_column not in df.columns:
                raise ToolError(f"Group column '{group_column}' not found in CSV")

            # Filter control and test groups
            control_data = df[df[group_column] == control_value].copy()
            test_data = df[df[group_column] == test_value].copy()

            if control_data.empty:
                raise ToolError(f"No data found for control group ('{control_value}')")
            if test_data.empty:
                raise ToolError(f"No data found for test group ('{test_value}')")

            # logger.info(f"Control group: {len(control_data)} records, Test group: {len(test_data)} records")

            # Auto-detect performance metrics if not provided - analyze ALL numeric columns
            if not performance_metrics:
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                exclude_columns = ['devicemodel', group_column]
                performance_metrics = [col for col in numeric_columns if col not in exclude_columns]
                logger.info(f"Auto-detected performance metrics: {performance_metrics}")
            else:
                # If metrics provided, still validate they exist and are numeric
                valid_metrics = []
                for metric in performance_metrics:
                    if metric in df.columns:
                        # Check if column can be converted to numeric
                        try:
                            pd.to_numeric(df[metric], errors='coerce')
                            valid_metrics.append(metric)
                        except:
                            logger.warning(f"Metric {metric} is not numeric, skipping")
                    else:
                        logger.warning(f"Metric {metric} not found in CSV, skipping")
                performance_metrics = valid_metrics
                logger.info(f"Valid performance metrics: {performance_metrics}")

            if not performance_metrics:
                raise ToolError("No valid numeric performance metrics found for analysis")

            # Setup output directory
            if not output_dir:
                csv_dir = os.path.dirname(csv_file_path)
                output_dir = os.path.join(csv_dir, 'control_test_analysis')

            os.makedirs(output_dir, exist_ok=True)
            # logger.info(f"Output directory: {output_dir}")

            # Perform overall analysis first (test vs control for all metrics)
            overall_analysis = await self._analyze_overall_performance(
                control_data, test_data, performance_metrics, degradation_threshold
            )

            # Perform analysis by device model
            analysis_results = await self._analyze_by_device_model(
                control_data, test_data, performance_metrics, degradation_threshold, confidence_alpha
            )

            # Generate visualizations (only show top degraded devices with high confidence)
            self._create_visualizations(
                analysis_results, performance_metrics, output_dir, visualization_top_percent
            )

            # Create overall analysis visualization
            self._create_overall_analysis_visualization(overall_analysis, output_dir)

            # Save analysis data
            self._save_analysis_data(
                analysis_results, control_data, test_data, output_dir
            )

            # Save overall analysis data
            self._save_overall_analysis_data(overall_analysis, output_dir)

            # Generate summary report
            summary = self._generate_summary_report(
                analysis_results, overall_analysis, degradation_threshold, output_dir,
                len(control_data), len(test_data), performance_metrics,
                confidence_alpha, visualization_top_percent
            )

            return ToolResult(output=summary)

        except ToolError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            error_msg = f"Unexpected error in control test analysis: {str(e)}"
            logger.error(error_msg)
            return ToolResult(error=error_msg)

    async def _analyze_overall_performance(
        self,
        control_data: pd.DataFrame,
        test_data: pd.DataFrame,
        performance_metrics: List[str],
        degradation_threshold: float
    ) -> Dict:
        """分析test组和control组的总体性能对比，不区分设备型号。"""

        # logger.info("开始进行总体性能分析（不区分设备型号）")

        overall_results = {
            'metrics_comparison': {},
            'summary_stats': {
                'total_control_records': len(control_data),
                'total_test_records': len(test_data),
                'total_metrics_analyzed': 0,
                'degraded_metrics': 0,
                'improved_metrics': 0,
                'stable_metrics': 0,
                'no_data_metrics': 0
            }
        }

        for metric in performance_metrics:
            # logger.info(f"分析指标: {metric}")

            # 提取control组和test组的有效数值
            control_values = pd.to_numeric(control_data[metric], errors='coerce').dropna()
            test_values = pd.to_numeric(test_data[metric], errors='coerce').dropna()

            metric_result = {
                'metric_name': metric,
                'control_count': len(control_values),
                'test_count': len(test_values),
                'control_mean': np.nan,
                'control_std': np.nan,
                'control_median': np.nan,
                'test_mean': np.nan,
                'test_std': np.nan,
                'test_median': np.nan,
                'change_percent': 0.0,
                'absolute_change': 0.0,
                'status': 'no_data',
                'is_higher_better': True
            }

            # 计算统计指标
            if not control_values.empty:
                metric_result['control_mean'] = control_values.mean()
                metric_result['control_std'] = control_values.std()
                metric_result['control_median'] = control_values.median()

            if not test_values.empty:
                metric_result['test_mean'] = test_values.mean()
                metric_result['test_std'] = test_values.std()
                metric_result['test_median'] = test_values.median()

            # 计算变化百分比
            if not np.isnan(metric_result['control_mean']) and not np.isnan(metric_result['test_mean']):
                if metric_result['control_mean'] != 0:
                    # 计算变化百分比: (test_mean - control_mean) / control_mean * 100
                    change_percent = ((metric_result['test_mean'] - metric_result['control_mean']) /
                                    metric_result['control_mean']) * 100
                    metric_result['change_percent'] = change_percent
                    metric_result['absolute_change'] = metric_result['test_mean'] - metric_result['control_mean']

                    # 判断是否更高更好
                    is_higher_better = await self._is_higher_better(metric)
                    metric_result['is_higher_better'] = is_higher_better

                    # 确定状态
                    if abs(change_percent) >= degradation_threshold:
                        if change_percent > 0:
                            # 正向变化：test组比control组高
                            metric_result['status'] = 'improved' if is_higher_better else 'degraded'
                        else:
                            # 负向变化：test组比control组低
                            metric_result['status'] = 'degraded' if is_higher_better else 'improved'
                    else:
                        metric_result['status'] = 'stable'

                    overall_results['summary_stats']['total_metrics_analyzed'] += 1

                    if metric_result['status'] == 'degraded':
                        overall_results['summary_stats']['degraded_metrics'] += 1
                    elif metric_result['status'] == 'improved':
                        overall_results['summary_stats']['improved_metrics'] += 1
                    else:
                        overall_results['summary_stats']['stable_metrics'] += 1
                else:
                    metric_result['status'] = 'stable'
                    overall_results['summary_stats']['total_metrics_analyzed'] += 1
                    overall_results['summary_stats']['stable_metrics'] += 1
            else:
                overall_results['summary_stats']['no_data_metrics'] += 1

            overall_results['metrics_comparison'][metric] = metric_result

            logger.info(f"指标 {metric}: 状态={metric_result['status']}, 变化={metric_result['change_percent']:.2f}%")

        logger.info(f"总体分析完成: {overall_results['summary_stats']['total_metrics_analyzed']} 个指标, "
                   f"恶化={overall_results['summary_stats']['degraded_metrics']}, "
                   f"改善={overall_results['summary_stats']['improved_metrics']}, "
                   f"稳定={overall_results['summary_stats']['stable_metrics']}")

        return overall_results

    async def _analyze_by_device_model(
        self,
        control_data: pd.DataFrame,
        test_data: pd.DataFrame,
        performance_metrics: List[str],
        degradation_threshold: float,
        confidence_alpha: float
    ) -> Dict:
        """Analyze performance differences by device model for ALL columns."""

        results = {
            'device_analysis': {},
            'degraded_devices': [],
            'improved_devices': [],
            'stable_devices': [],
            'column_analysis': {}  # Add detailed column analysis
        }

        # Get unique device models from both groups
        control_devices = set(control_data['devicemodel'].unique())
        test_devices = set(test_data['devicemodel'].unique())
        all_devices = control_devices.union(test_devices)

        # Calculate total sample size for reference
        total_control_count = len(control_data)
        total_test_count = len(test_data)
        total_samples = total_control_count + total_test_count

        # logger.info(f"Analyzing {len(all_devices)} unique device models across {len(performance_metrics)} metrics")
        # logger.info(f"Total sample size: {total_samples} ({total_control_count} control + {total_test_count} test)")

        # Initialize column analysis tracking
        for metric in performance_metrics:
            results['column_analysis'][metric] = {
                'total_devices': 0,
                'degraded_devices': [],
                'improved_devices': [],
                'stable_devices': [],
                'no_data_devices': []
            }

        for device in all_devices:
            device_control = control_data[control_data['devicemodel'] == device]
            device_test = test_data[test_data['devicemodel'] == device]

            device_result = {
                'device_model': device,
                'control_count': len(device_control),
                'test_count': len(device_test),
                'total_device_count': len(device_control) + len(device_test),
                'sample_proportion': 0.0,  # Add sample proportion
                'balance_score': 0.0,  # Add balance score
                'confidence_score': 0.0,  # Updated confidence score
                'metrics_analysis': {},
                'overall_status': 'stable',
                'degradation_score': 0.0,
                'total_metrics_analyzed': 0,
                'degraded_metrics_count': 0,
                'improved_metrics_count': 0,
                'stable_metrics_count': 0
            }

            # Calculate enhanced confidence score
            nc = len(device_control)
            nt = len(device_test)
            device_total = nc + nt

            if nc > 0 and nt > 0 and total_samples > 0:
                # 1. Sample balance score (how balanced control vs test)
                balance_ratio = min(nc, nt) / max(nc, nt)
                balance_score = balance_ratio ** confidence_alpha

                # 2. Sample proportion score (how representative in total dataset)
                sample_proportion = device_total / total_samples

                # Apply logarithmic scaling to prevent very small devices from having zero confidence
                # but still heavily penalize devices with very few samples
                proportion_score = min(1.0, np.log10(device_total * 10) / 2.0) # Scales from ~0 to 1

                # 3. Combined confidence score
                # Base score of 10, weighted by both balance and proportion
                confidence_score = 10.0 * balance_score * proportion_score

                device_result['sample_proportion'] = sample_proportion
                device_result['balance_score'] = balance_score
                device_result['confidence_score'] = confidence_score
            else:
                device_result['confidence_score'] = 0.0
                device_result['sample_proportion'] = 0.0
                device_result['balance_score'] = 0.0

            degradation_count = 0
            improvement_count = 0
            stable_count = 0
            total_degradation = 0.0
            valid_metrics = 0

            # Analyze EVERY performance metric for this device
            for metric in performance_metrics:
                metric_analysis = await self._analyze_metric_performance(
                    device_control, device_test, metric, degradation_threshold
                )
                device_result['metrics_analysis'][metric] = metric_analysis

                # Track column-level statistics
                results['column_analysis'][metric]['total_devices'] += 1

                if metric_analysis['status'] == 'degraded':
                    degradation_count += 1
                    total_degradation += abs(metric_analysis['change_percent'])
                    results['column_analysis'][metric]['degraded_devices'].append(device)
                elif metric_analysis['status'] == 'improved':
                    improvement_count += 1
                    results['column_analysis'][metric]['improved_devices'].append(device)
                elif metric_analysis['status'] == 'stable':
                    stable_count += 1
                    results['column_analysis'][metric]['stable_devices'].append(device)
                else:  # no_data
                    results['column_analysis'][metric]['no_data_devices'].append(device)

                # Only count metrics with valid data for degradation score
                if metric_analysis['status'] != 'no_data':
                    valid_metrics += 1

            # Update device statistics
            device_result['total_metrics_analyzed'] = valid_metrics
            device_result['degraded_metrics_count'] = degradation_count
            device_result['improved_metrics_count'] = improvement_count
            device_result['stable_metrics_count'] = stable_count

            # Calculate degradation score only for valid metrics
            if valid_metrics > 0:
                device_result['degradation_score'] = total_degradation / valid_metrics

                # Determine overall device status based on majority of metrics
                if degradation_count > improvement_count and degradation_count > stable_count:
                    device_result['overall_status'] = 'degraded'
                    results['degraded_devices'].append(device)
                elif improvement_count > degradation_count and improvement_count > stable_count:
                    device_result['overall_status'] = 'improved'
                    results['improved_devices'].append(device)
                else:
                    device_result['overall_status'] = 'stable'
                    results['stable_devices'].append(device)
            else:
                device_result['degradation_score'] = 0.0
                device_result['overall_status'] = 'no_data'
                results['stable_devices'].append(device)  # Treat no-data as stable

            results['device_analysis'][device] = device_result

        return results

    async def _analyze_metric_performance(
        self,
        control_data: pd.DataFrame,
        test_data: pd.DataFrame,
        metric: str,
        threshold: float
    ) -> Dict:
        """Analyze performance change for a specific metric."""

        result = {
            'metric': metric,
            'control_mean': np.nan,
            'test_mean': np.nan,
            'control_std': np.nan,
            'test_std': np.nan,
            'change_percent': 0.0,
            'status': 'no_data'
        }

        if not control_data.empty and metric in control_data.columns:
            control_values = pd.to_numeric(control_data[metric], errors='coerce').dropna()
            if not control_values.empty:
                result['control_mean'] = control_values.mean()
                result['control_std'] = control_values.std()

        if not test_data.empty and metric in test_data.columns:
            test_values = pd.to_numeric(test_data[metric], errors='coerce').dropna()
            if not test_values.empty:
                result['test_mean'] = test_values.mean()
                result['test_std'] = test_values.std()

        # Calculate performance change
        if not np.isnan(result['control_mean']) and not np.isnan(result['test_mean']):
            if result['control_mean'] != 0:
                change = ((result['test_mean'] - result['control_mean']) / result['control_mean']) * 100
                result['change_percent'] = change

                # Determine status based on change
                if abs(change) >= threshold:
                    if change > 0:
                        # For performance metrics, positive change might be good or bad
                        # Use LLM to determine if higher is better
                        is_higher_better = await self._is_higher_better(metric)
                        result['status'] = 'improved' if is_higher_better else 'degraded'
                    else:
                        is_higher_better = await self._is_higher_better(metric)
                        result['status'] = 'degraded' if is_higher_better else 'improved'
                else:
                    result['status'] = 'stable'
            else:
                result['status'] = 'stable'

        return result

    async def _is_higher_better(self, metric: str) -> bool:
        """Determine if higher values are better for a metric using LLM and metadata."""
        try:
            # Special handling for jank-related metrics (lower is always better for jank)
            # metric_lower = metric.lower()
            # jank_indicators = ['jank', 'lag']
            # if any(indicator in metric_lower for indicator in jank_indicators):
            #     logger.info(f"Detected jank-related metric '{metric}', setting lower is better")
            #     return False

            # Get metadata file path from config
            if not config.run_flow_config or not config.run_flow_config.meta_data_name:
                logger.warning("No meta_data_name configured, using default (higher is better)")
                return True

            meta_data_name = config.run_flow_config.meta_data_name
            meta_data_path = os.path.join(config.workspace_root, meta_data_name)

            # Load metadata
            if not os.path.exists(meta_data_path):
                logger.warning(f"Metadata file not found: {meta_data_path}, using default (higher is better)")
                return True

            with open(meta_data_path, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)

            # Search for metric in metadata
            metric_info = None
            if isinstance(meta_data, list):
                # If metadata is a list of metric objects
                for item in meta_data:
                    if isinstance(item, dict) and item.get("Field") == metric:
                        metric_info = item
                        break
            elif isinstance(meta_data, dict):
                # If metadata is a dict with metrics as keys or contains a metrics field
                if "metrics" in meta_data and isinstance(meta_data["metrics"], list):
                    for item in meta_data["metrics"]:
                        if isinstance(item, dict) and item.get("Field") == metric:
                            metric_info = item
                            break
                elif metric in meta_data:
                    metric_info = meta_data[metric]

            # If metric found in metadata, return its HigherIsBetter value
            if metric_info:
                #logger.info(f"Found metric '{metric}' in metadata, using metadata to determine if higher is better")

                is_higher_better = metric_info.get("is_high_better", None)

                if is_higher_better is not None:
                    if isinstance(is_higher_better, str):
                        is_higher_better = is_higher_better.lower() == 'true'
                    elif isinstance(is_higher_better, bool):
                        is_higher_better = is_higher_better
                    else:
                        logger.warning(f"Invalid type for is_higher_better: {type(is_higher_better)}, using default (higher is better)")
                        is_higher_better = True

                #logger.info(f"Metric '{metric}' found in metadata with is_higher_better={is_higher_better}")
                return is_higher_better if is_higher_better is not None else True  # Default to True if not specified
            else:
                #logger.info(f"Metric '{metric}' not found in metadata, using default (higher is better)")
                return True

        except Exception as e:
            logger.error(f"Error processing metadata for metric '{metric}': {str(e)}, using default (higher is better)")
            return True

    def _get_high_confidence_degraded_devices(
        self,
        analysis_results: Dict,
        top_percent: float = 20.0
    ) -> List[str]:
        """Get top % of degraded devices with highest confidence scores."""

        # Get all degraded devices with their confidence scores
        degraded_devices_with_confidence = []
        for device in analysis_results['degraded_devices']:
            if device in analysis_results['device_analysis']:
                confidence = analysis_results['device_analysis'][device]['confidence_score']
                degraded_devices_with_confidence.append((device, confidence))

        # Sort by confidence score (highest first)
        degraded_devices_with_confidence.sort(key=lambda x: x[1], reverse=True)

        # Get top percentage
        if degraded_devices_with_confidence:
            top_count = max(1, int(len(degraded_devices_with_confidence) * top_percent / 100.0))
            return [device for device, _ in degraded_devices_with_confidence[:top_count]]
        else:
            return []

    def _create_visualizations(
        self,
        analysis_results: Dict,
        performance_metrics: List[str],
        output_dir: str,
        visualization_top_percent: float = 20.0
    ) -> None:
        """Create visualization charts for the analysis."""

        # Use default fonts to avoid font issues and ensure English text rendering
        plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        # Create performance comparison bar chart (filtered by confidence)
        self._create_performance_bar_chart(analysis_results, performance_metrics, output_dir, visualization_top_percent)

        # Create degradation score chart (filtered by confidence)
        self._create_degradation_score_chart(analysis_results, output_dir, visualization_top_percent)

        # Create metric-wise comparison charts (filtered by confidence)
        self._create_metric_comparison_charts(analysis_results, performance_metrics, output_dir, visualization_top_percent)

    def _create_performance_bar_chart(
        self,
        analysis_results: Dict,
        performance_metrics: List[str],
        output_dir: str,
        visualization_top_percent: float = 20.0
    ) -> None:
        """Create bar chart showing overall performance by device (filtered by confidence). Split into multiple charts if too many devices."""

        # Get high confidence degraded devices for visualization
        high_confidence_degraded = self._get_high_confidence_degraded_devices(
            analysis_results, visualization_top_percent
        )

        # Include all improved and stable devices, plus high-confidence degraded ones
        devices_to_show = set(high_confidence_degraded)
        devices_to_show.update(analysis_results['improved_devices'])
        devices_to_show.update(analysis_results['stable_devices'])

        # If no devices to show, show all
        if not devices_to_show:
            devices_to_show = set(analysis_results['device_analysis'].keys())

        devices = list(devices_to_show)

        # Sort devices for consistent display order (degraded first, then by degradation score)
        device_sort_data = []
        for device in devices:
            status = analysis_results['device_analysis'][device]['overall_status']
            degradation_score = analysis_results['device_analysis'][device]['degradation_score']
            confidence = analysis_results['device_analysis'][device]['confidence_score']

            # Assign priority: degraded=1, improved=2, stable=3
            priority = 1 if status == 'degraded' else (2 if status == 'improved' else 3)
            device_sort_data.append((device, priority, -degradation_score, -confidence))

        # Sort by priority, then by degradation score (descending), then by confidence (descending)
        device_sort_data.sort(key=lambda x: (x[1], x[2], x[3]))
        devices = [item[0] for item in device_sort_data]

        # Split devices into batches of 30
        max_devices_per_chart = 30
        device_batches = []
        for i in range(0, len(devices), max_devices_per_chart):
            batch = devices[i:i + max_devices_per_chart]
            device_batches.append(batch)

        total_charts = len(device_batches)

        # Create charts for each batch
        for batch_idx, device_batch in enumerate(device_batches, 1):
            degradation_scores = [
                analysis_results['device_analysis'][device]['degradation_score']
                for device in device_batch
            ]
            confidence_scores = [
                analysis_results['device_analysis'][device]['confidence_score']
                for device in device_batch
            ]

            # Color code by status and confidence
            colors = []
            for device in device_batch:
                status = analysis_results['device_analysis'][device]['overall_status']
                confidence = analysis_results['device_analysis'][device]['confidence_score']
                if status == 'degraded':
                    # Use darker red for higher confidence degraded devices
                    if device in high_confidence_degraded:
                        colors.append('darkred')
                    else:
                        colors.append('red')
                elif status == 'improved':
                    colors.append('green')
                else:
                    colors.append('blue')

            plt.figure(figsize=(15, 8))
            bars = plt.bar(range(len(device_batch)), degradation_scores, color=colors, alpha=0.7)

            # Title includes batch information if multiple charts
            if total_charts > 1:
                title = f'Device Performance Degradation Comparison (Part {batch_idx}/{total_charts})'
            else:
                title = 'Device Performance Degradation Comparison'

            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel('Device Model', fontsize=12)
            plt.ylabel('Average Degradation Percentage (%)', fontsize=12)
            plt.xticks(range(len(device_batch)), device_batch, rotation=45, ha='right')

            # Add value labels on bars with confidence scores
            for i, (bar, score, confidence) in enumerate(zip(bars, degradation_scores, confidence_scores)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{score:.1f}%\n(C:{confidence:.1f})', ha='center', va='bottom', fontsize=9)

            legend_elements = [
                Patch(facecolor='darkred', alpha=0.7, label='High Confidence Degraded'),
                Patch(facecolor='red', alpha=0.7, label='Low Confidence Degraded'),
                Patch(facecolor='green', alpha=0.7, label='Improved'),
                Patch(facecolor='blue', alpha=0.7, label='Stable')
            ]
            plt.legend(handles=legend_elements, loc='upper right')

            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()

            # Save with batch suffix if multiple charts
            if total_charts > 1:
                chart_path = os.path.join(output_dir, f'device_performance_comparison_part{batch_idx}.png')
            else:
                chart_path = os.path.join(output_dir, 'device_performance_comparison.png')

            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            # logger.info(f"Saved performance comparison chart part {batch_idx}/{total_charts}: {chart_path} (showing {len(device_batch)} devices)")

        # Log summary
        if total_charts > 1:
            logger.info(f"Created {total_charts} performance comparison charts for {len(devices)} total devices")

    def _create_degradation_score_chart(
        self,
        analysis_results: Dict,
        output_dir: str,
        visualization_top_percent: float = 20.0
    ) -> None:
        """Create horizontal bar chart for degradation scores (filtered by confidence)."""

        # Get high confidence degraded devices
        high_confidence_degraded = self._get_high_confidence_degraded_devices(
            analysis_results, visualization_top_percent
        )

        # Only show degraded devices with their confidence scores
        device_data = []
        for device in high_confidence_degraded:
            if device in analysis_results['device_analysis']:
                data = analysis_results['device_analysis'][device]
                device_data.append({
                    'device': device,
                    'score': data['degradation_score'],
                    'confidence': data['confidence_score']
                })

        if not device_data:
            logger.info("No high confidence degraded devices to display in ranking chart")
            return

        # Sort by degradation score (highest first)
        device_data.sort(key=lambda x: x['score'], reverse=True)

        devices = [item['device'] for item in device_data]
        scores = [item['score'] for item in device_data]
        confidences = [item['confidence'] for item in device_data]

        plt.figure(figsize=(12, max(8, len(devices) * 0.5)))
        bars = plt.barh(range(len(devices)), scores,
                       color=['darkred' if conf > 7.5 else 'red' if conf > 5.0 else 'orange'
                              for conf in confidences], alpha=0.7)

        plt.title('High Confidence Degraded Devices Ranking', fontsize=16, fontweight='bold')
        plt.xlabel('Average Performance Degradation (%)', fontsize=12)
        plt.ylabel('Device Model', fontsize=12)
        plt.yticks(range(len(devices)), devices)

        # Add value labels with confidence scores
        for i, (bar, score, confidence) in enumerate(zip(bars, scores, confidences)):
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{score:.1f}% (C:{confidence:.1f})', ha='left', va='center', fontsize=10)

        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        chart_path = os.path.join(output_dir, 'degradation_ranking.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        # logger.info(f"Saved degradation ranking chart: {chart_path}")

    def _create_metric_comparison_charts(
        self,
        analysis_results: Dict,
        performance_metrics: List[str],
        output_dir: str,
        visualization_top_percent: float = 20.0
    ) -> None:
        """Create individual metric comparison charts (filtered by confidence)."""

        for metric in performance_metrics:
            self._create_single_metric_chart(analysis_results, metric, output_dir, visualization_top_percent)

    def _create_single_metric_chart(
        self,
        analysis_results: Dict,
        metric: str,
        output_dir: str,
        visualization_top_percent: float = 20.0
    ) -> None:
        """Create comparison chart for a single metric (only degraded devices, sorted by confidence)."""

        # Only show degraded devices sorted by confidence score
        degraded_devices = analysis_results['degraded_devices']
        if not degraded_devices:
            logger.info(f"No degraded devices found for metric {metric}, skipping chart")
            return

        # Get degraded devices with their confidence scores and metric data
        device_data = []
        for device in degraded_devices:
            if device in analysis_results['device_analysis']:
                device_analysis = analysis_results['device_analysis'][device]
                if metric in device_analysis['metrics_analysis']:
                    metric_data = device_analysis['metrics_analysis'][metric]
                    # Only include devices with valid metric data and degraded status for this metric
                    if (not np.isnan(metric_data['control_mean']) and
                        not np.isnan(metric_data['test_mean']) and
                        metric_data['status'] == 'degraded'):
                        device_data.append({
                            'device': device,
                            'control_mean': metric_data['control_mean'],
                            'test_mean': metric_data['test_mean'],
                            'confidence': device_analysis['confidence_score'],
                            'change_percent': metric_data['change_percent']
                        })

        if not device_data:
            logger.info(f"No degraded devices with valid data for metric {metric}, skipping chart")
            return

        # Sort by confidence score (highest first)
        device_data.sort(key=lambda x: x['confidence'], reverse=True)

        # Extract data for plotting
        devices = [item['device'] for item in device_data]
        control_means = [item['control_mean'] for item in device_data]
        test_means = [item['test_mean'] for item in device_data]
        confidences = [item['confidence'] for item in device_data]
        change_percents = [item['change_percent'] for item in device_data]

        # Limit to reasonable number of devices for readability
        max_devices = 15  # Show top 15 degraded devices by confidence
        if len(devices) > max_devices:
            devices = devices[:max_devices]
            control_means = control_means[:max_devices]
            test_means = test_means[:max_devices]
            confidences = confidences[:max_devices]
            change_percents = change_percents[:max_devices]

        x = np.arange(len(devices))
        width = 0.35

        plt.figure(figsize=(15, 8))
        bars1 = plt.bar(x - width/2, control_means, width, label='Control Group', alpha=0.8, color='skyblue')
        bars2 = plt.bar(x + width/2, test_means, width, label='Test Group', alpha=0.8, color='orange')

        plt.title(f'{metric} - Degraded Devices Comparison (Sorted by Confidence)', fontsize=16, fontweight='bold')
        plt.xlabel('Device Models (Sorted by Confidence Score)', fontsize=12)
        plt.ylabel(f'{metric} Value', fontsize=12)
        plt.xticks(x, devices, rotation=45, ha='right')
        plt.legend()

        # Add value labels with change percentage and confidence
        for i, (bar1, bar2, conf, change) in enumerate(zip(bars1, bars2, confidences, change_percents)):
            # Control group label
            plt.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height(),
                    f'{control_means[i]:.2f}', ha='center', va='bottom', fontsize=8)
            # Test group label with change percentage
            plt.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height(),
                    f'{test_means[i]:.2f}\n({change:+.1f}%)', ha='center', va='bottom', fontsize=8)

        # Add confidence scores as text at the bottom
        for i, conf in enumerate(confidences):
            plt.text(x[i], min(min(control_means), min(test_means)) * 0.95,
                    f'C:{conf:.1f}', ha='center', va='top', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))

        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        safe_metric_name = metric.replace('/', '_').replace('\\', '_').replace(':', '_')
        chart_path = os.path.join(output_dir, f'metric_{safe_metric_name}_degraded_devices.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        # logger.info(f"Saved {metric} degraded devices chart: {chart_path} (showing {len(devices)} devices)")

    def _create_overall_analysis_visualization(
        self,
        overall_analysis: Dict,
        output_dir: str
    ) -> None:
        """创建总体分析的可视化图表。"""

        metrics_data = overall_analysis['metrics_comparison']

        # 准备数据用于可视化
        metrics = []
        change_percents = []
        statuses = []
        colors = []

        for metric, data in metrics_data.items():
            if data['status'] != 'no_data':
                metrics.append(metric)
                change_percents.append(data['change_percent'])
                statuses.append(data['status'])

                # 根据状态设置颜色
                if data['status'] == 'degraded':
                    colors.append('red')
                elif data['status'] == 'improved':
                    colors.append('green')
                else:
                    colors.append('blue')

        if not metrics:
            logger.info("无有效数据用于总体分析可视化")
            return

        # 创建总体性能变化图表
        plt.figure(figsize=(15, 8))
        bars = plt.bar(range(len(metrics)), change_percents, color=colors, alpha=0.7)

        plt.title('Overall Performance Change: Test vs Control Groups', fontsize=16, fontweight='bold')
        plt.xlabel('Performance Metrics', fontsize=12)
        plt.ylabel('Performance Change (%)', fontsize=12)
        plt.xticks(range(len(metrics)), metrics, rotation=45, ha='right')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # 添加数值标签
        for i, (bar, change, status) in enumerate(zip(bars, change_percents, statuses)):
            plt.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.5 if bar.get_height() >= 0 else -1.5),
                    f'{change:.1f}%\n({status})',
                    ha='center', va='bottom' if bar.get_height() >= 0 else 'top',
                    fontsize=9)

        # 添加图例
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Degraded'),
            Patch(facecolor='green', alpha=0.7, label='Improved'),
            Patch(facecolor='blue', alpha=0.7, label='Stable')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        chart_path = os.path.join(output_dir, 'overall_performance_comparison.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        # logger.info(f"保存总体性能对比图表: {chart_path}")

        # 创建性能改善/恶化排名图
        self._create_overall_ranking_chart(overall_analysis, output_dir)

    def _create_overall_ranking_chart(
        self,
        overall_analysis: Dict,
        output_dir: str
    ) -> None:
        """创建按变化幅度排序的总体性能排名图表。"""

        metrics_data = overall_analysis['metrics_comparison']

        # 准备数据
        valid_metrics = []
        for metric, data in metrics_data.items():
            if data['status'] != 'no_data':
                valid_metrics.append({
                    'metric': metric,
                    'change_percent': data['change_percent'],
                    'status': data['status']
                })

        if not valid_metrics:
            return

        # 按变化幅度排序（绝对值）
        valid_metrics.sort(key=lambda x: abs(x['change_percent']), reverse=True)

        metrics = [item['metric'] for item in valid_metrics]
        change_percents = [item['change_percent'] for item in valid_metrics]
        statuses = [item['status'] for item in valid_metrics]

        # 设置颜色
        colors = []
        for status in statuses:
            if status == 'degraded':
                colors.append('red')
            elif status == 'improved':
                colors.append('green')
            else:
                colors.append('blue')

        plt.figure(figsize=(12, max(8, len(metrics) * 0.4)))
        bars = plt.barh(range(len(metrics)), change_percents, color=colors, alpha=0.7)

        plt.title('Overall Performance Ranking by Change Magnitude', fontsize=16, fontweight='bold')
        plt.xlabel('Performance Change (%)', fontsize=12)
        plt.ylabel('Performance Metrics', fontsize=12)
        plt.yticks(range(len(metrics)), metrics)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

        # 添加数值标签
        for i, (bar, change, status) in enumerate(zip(bars, change_percents, statuses)):
            plt.text(bar.get_width() + (0.5 if bar.get_width() >= 0 else -0.5),
                    bar.get_y() + bar.get_height()/2,
                    f'{change:.1f}% ({status})',
                    ha='left' if bar.get_width() >= 0 else 'right', va='center', fontsize=10)

        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        chart_path = os.path.join(output_dir, 'overall_performance_ranking.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        # logger.info(f"保存总体性能排名图表: {chart_path}")

    def _save_analysis_data(
        self,
        analysis_results: Dict,
        control_data: pd.DataFrame,
        test_data: pd.DataFrame,
        output_dir: str
    ) -> None:
        """Save analysis data to CSV files."""

        # Save device analysis summary
        summary_data = []
        for device, data in analysis_results['device_analysis'].items():
            row = {
                'device_model': device,
                'overall_status': data['overall_status'],
                'degradation_score': data['degradation_score'],
                'confidence_score': data['confidence_score'],  # Add confidence score
                'control_count': data['control_count'],
                'test_count': data['test_count'],
                'sample_ratio': data['control_count'] / max(data['test_count'], 1),  # Add sample ratio
                'total_metrics_analyzed': data.get('total_metrics_analyzed', 0),
                'degraded_metrics_count': data.get('degraded_metrics_count', 0),
                'improved_metrics_count': data.get('improved_metrics_count', 0),
                'stable_metrics_count': data.get('stable_metrics_count', 0)
            }

            # Add metric analysis details
            for metric, metric_data in data['metrics_analysis'].items():
                row[f'{metric}_control_mean'] = metric_data['control_mean']
                row[f'{metric}_test_mean'] = metric_data['test_mean']
                row[f'{metric}_change_percent'] = metric_data['change_percent']
                row[f'{metric}_status'] = metric_data['status']

            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, 'device_analysis_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8')
        # logger.info(f"Saved analysis summary: {summary_path}")

        # Save only degraded devices list
        degraded_devices = analysis_results['degraded_devices']
        if degraded_devices:
            degraded_df = pd.DataFrame({'device_model': degraded_devices})
            degraded_path = os.path.join(output_dir, 'degraded_devices.csv')
            degraded_df.to_csv(degraded_path, index=False, encoding='utf-8')
            # logger.info(f"Saved degraded devices: {degraded_path}")

        # Save detailed device-metric breakdown
        detailed_breakdown = []
        for device, data in analysis_results['device_analysis'].items():
            for metric, metric_data in data['metrics_analysis'].items():
                breakdown_row = {
                    'device_model': device,
                    'metric_name': metric,
                    'control_mean': metric_data['control_mean'],
                    'test_mean': metric_data['test_mean'],
                    'control_std': metric_data.get('control_std', np.nan),
                    'test_std': metric_data.get('test_std', np.nan),
                    'change_percent': metric_data['change_percent'],
                    'status': metric_data['status'],
                    'control_count': data['control_count'],
                    'test_count': data['test_count'],
                    'confidence_score': data['confidence_score']  # Add confidence score
                }
                detailed_breakdown.append(breakdown_row)

        breakdown_df = pd.DataFrame(detailed_breakdown)
        breakdown_path = os.path.join(output_dir, 'detailed_device_metric_breakdown.csv')
        breakdown_df.to_csv(breakdown_path, index=False, encoding='utf-8')
        # logger.info(f"Saved detailed breakdown: {breakdown_path}")

    def _save_overall_analysis_data(
        self,
        overall_analysis: Dict,
        output_dir: str
    ) -> None:
        """保存总体分析数据到CSV文件。"""

        # 保存总体性能对比数据
        overall_data = []
        for metric, data in overall_analysis['metrics_comparison'].items():
            row = {
                'metric_name': metric,
                'control_count': data['control_count'],
                'test_count': data['test_count'],
                'control_mean': data['control_mean'],
                'control_std': data['control_std'],
                'control_median': data['control_median'],
                'test_mean': data['test_mean'],
                'test_std': data['test_std'],
                'test_median': data['test_median'],
                'change_percent': data['change_percent'],
                'absolute_change': data['absolute_change'],
                'status': data['status'],
                'is_higher_better': data['is_higher_better']
            }
            overall_data.append(row)

        overall_df = pd.DataFrame(overall_data)
        overall_path = os.path.join(output_dir, 'overall_performance_analysis.csv')
        overall_df.to_csv(overall_path, index=False, encoding='utf-8')
        # logger.info(f"保存总体性能分析数据: {overall_path}")

        # 保存汇总统计信息
        summary_stats = overall_analysis['summary_stats']
        stats_data = [
            {'statistic': 'total_control_records', 'value': summary_stats['total_control_records']},
            {'statistic': 'total_test_records', 'value': summary_stats['total_test_records']},
            {'statistic': 'total_metrics_analyzed', 'value': summary_stats['total_metrics_analyzed']},
            {'statistic': 'degraded_metrics', 'value': summary_stats['degraded_metrics']},
            {'statistic': 'improved_metrics', 'value': summary_stats['improved_metrics']},
            {'statistic': 'stable_metrics', 'value': summary_stats['stable_metrics']},
            {'statistic': 'no_data_metrics', 'value': summary_stats['no_data_metrics']}
        ]

        stats_df = pd.DataFrame(stats_data)
        stats_path = os.path.join(output_dir, 'overall_analysis_summary_stats.csv')
        stats_df.to_csv(stats_path, index=False, encoding='utf-8')
        # logger.info(f"保存总体分析汇总统计: {stats_path}")

    def _generate_summary_report(
        self,
        analysis_results: Dict,
        overall_analysis: Dict,
        degradation_threshold: float,
        output_dir: str,
        control_count: int,
        test_count: int,
        performance_metrics: List[str],
        confidence_alpha: float,
        visualization_top_percent: float
    ) -> str:
        """Generate comprehensive analysis summary report."""

        degraded_devices = analysis_results['degraded_devices']
        improved_devices = analysis_results['improved_devices']
        stable_devices = analysis_results['stable_devices']

        total_devices = len(analysis_results['device_analysis'])

        summary = "Control vs Test 性能对比分析报告\n"
        summary += "=" * 50 + "\n\n"

        summary += f"**数据概览:**\n"
        summary += f"- Control组数据量: {control_count} 条记录\n"
        summary += f"- Test组数据量: {test_count} 条记录\n"
        summary += f"- 分析设备型号数量: {total_devices} 个\n"
        summary += f"- 性能指标数量: {len(performance_metrics)} 个\n"
        summary += f"- 恶化阈值: {degradation_threshold}%\n"
        summary += f"- 置信度参数α: {confidence_alpha}\n"
        summary += f"- 可视化显示比例: 恶化设备中置信度最高的{visualization_top_percent}%\n\n"

        # 添加总体分析结果
        summary += f"**总体性能分析结果（不区分设备）:**\n"
        overall_stats = overall_analysis['summary_stats']
        summary += f"📊 分析指标总数: {overall_stats['total_metrics_analyzed']} 个\n"
        summary += f"🔴 恶化指标: {overall_stats['degraded_metrics']} 个\n"
        summary += f"🟢 改善指标: {overall_stats['improved_metrics']} 个\n"
        summary += f"🔵 稳定指标: {overall_stats['stable_metrics']} 个\n"
        summary += f"⚪ 无数据指标: {overall_stats['no_data_metrics']} 个\n\n"

        # 显示所有变化指标的详细情况
        metrics_data = overall_analysis['metrics_comparison']
        all_changes = []
        for metric, data in metrics_data.items():
            if data['status'] != 'no_data':
                all_changes.append((metric, data['change_percent'], data['status']))

        if all_changes:
            # 按变化幅度排序（绝对值）
            all_changes.sort(key=lambda x: abs(x[1]), reverse=True)
            summary += f"**总体所有指标变化详情（按变化幅度排序）:**\n"
            for i, (metric, change, status) in enumerate(all_changes, 1):
                status_icon = "🔴" if status == "degraded" else "🟢" if status == "improved" else "🔵"
                summary += f"{i:2d}. {metric}: {change:+.2f}% {status_icon}\n"
            summary += "\n"

            # 按状态分类显示
            degraded_metrics = [(m, c) for m, c, s in all_changes if s == 'degraded']
            improved_metrics = [(m, c) for m, c, s in all_changes if s == 'improved']
            stable_metrics = [(m, c) for m, c, s in all_changes if s == 'stable']

            if degraded_metrics:
                summary += f"**🔴 恶化指标明细 ({len(degraded_metrics)} 个):**\n"
                for i, (metric, change) in enumerate(degraded_metrics, 1):
                    summary += f"   {i}. {metric}: {change:+.2f}%\n"
                summary += "\n"

            if improved_metrics:
                summary += f"**🟢 改善指标明细 ({len(improved_metrics)} 个):**\n"
                for i, (metric, change) in enumerate(improved_metrics, 1):
                    summary += f"   {i}. {metric}: {change:+.2f}%\n"
                summary += "\n"

            if stable_metrics:
                summary += f"**🔵 稳定指标明细 ({len(stable_metrics)} 个):**\n"
                for i, (metric, change) in enumerate(stable_metrics, 1):
                    summary += f"   {i}. {metric}: {change:+.2f}%\n"
                summary += "\n"

        summary += f"**设备级性能分析结果:**\n"
        summary += f"🔴 **性能恶化设备: {len(degraded_devices)} 个**\n"
        if degraded_devices:
            summary += f"   恶化设备列表: {', '.join(degraded_devices)}\n"

        summary += f"🟢 **性能改善设备: {len(improved_devices)} 个**\n"
        if improved_devices:
            summary += f"   改善设备列表: {', '.join(improved_devices)}\n"

        summary += f"🔵 **性能稳定设备: {len(stable_devices)} 个**\n"
        if stable_devices:
            summary += f"   稳定设备列表: {', '.join(stable_devices)}\n\n"

        # Top degraded devices with scores and confidence
        if degraded_devices:
            summary += f"**性能恶化TOP设备详情（按置信度排序）:**\n"
            # Sort degraded devices by confidence score
            degraded_with_confidence = [
                (device, analysis_results['device_analysis'][device]['degradation_score'],
                 analysis_results['device_analysis'][device]['confidence_score'])
                for device in degraded_devices
            ]
            degraded_with_confidence.sort(key=lambda x: x[2], reverse=True)  # Sort by confidence

            for i, (device, score, confidence) in enumerate(degraded_with_confidence[:5], 1):
                summary += f"{i}. {device}: 平均恶化 {score:.2f}%, 置信度 {confidence:.2f}\n"
            summary += "\n"

            # Show high confidence degraded devices
            high_confidence_degraded = self._get_high_confidence_degraded_devices(
                analysis_results, visualization_top_percent
            )
            summary += f"**高置信度恶化设备 (Top {visualization_top_percent}%):**\n"
            summary += f"   设备列表: {', '.join(high_confidence_degraded)}\n\n"

        # Performance metrics analysis
        summary += f"**指标分析详情:**\n"
        for metric in performance_metrics:
            degraded_count = 0
            improved_count = 0

            for device_data in analysis_results['device_analysis'].values():
                if metric in device_data['metrics_analysis']:
                    status = device_data['metrics_analysis'][metric]['status']
                    if status == 'degraded':
                        degraded_count += 1
                    elif status == 'improved':
                        improved_count += 1

            summary += f"- {metric}: {degraded_count} 个设备恶化, {improved_count} 个设备改善\n"

        summary += f"\n**生成文件:**\n"

        # Calculate how many device performance comparison charts were generated
        total_devices = len(analysis_results['device_analysis'])
        max_devices_per_chart = 30
        total_charts = (total_devices + max_devices_per_chart - 1) // max_devices_per_chart

        if total_charts > 1:
            summary += f"📊 设备性能对比图 ({total_charts} 张): {output_dir}/device_performance_comparison_part[1-{total_charts}].png\n"
        else:
            summary += f"📊 设备性能对比图: {output_dir}/device_performance_comparison.png\n"

        summary += f"📈 恶化排名图: {output_dir}/degradation_ranking.png\n"
        summary += f"� 总体性能对比图: {output_dir}/overall_performance_comparison.png\n"
        summary += f"📊 总体性能排名图: {output_dir}/overall_performance_ranking.png\n"
        summary += f"📋 总体性能分析数据: {output_dir}/overall_performance_analysis.csv\n"
        summary += f"📋 总体分析汇总统计: {output_dir}/overall_analysis_summary_stats.csv\n"
        summary += f"�📋 设备分析汇总数据: {output_dir}/device_analysis_summary.csv\n"
        summary += f"📋 详细设备指标明细: {output_dir}/detailed_device_metric_breakdown.csv\n"
        summary += f"📁 分类设备列表: {output_dir}/[degraded|improved|stable]_devices.csv\n"

        for metric in performance_metrics:
            safe_name = metric.replace('/', '_').replace('\\', '_').replace(':', '_')
            summary += f"📊 {metric} Degraded Devices Chart: {output_dir}/metric_{safe_name}_degraded_devices.png\n"

        # Add column-wise insights
        if 'column_analysis' in analysis_results:
            summary += f"\n**列指标分析洞察:**\n"
            column_analysis = analysis_results['column_analysis']
            for metric, column_data in column_analysis.items():
                degraded_count = len(column_data['degraded_devices'])
                improved_count = len(column_data['improved_devices'])
                stable_count = len(column_data['stable_devices'])
                total_valid = degraded_count + improved_count + stable_count

                if total_valid > 0:
                    degraded_pct = (degraded_count / total_valid) * 100
                    summary += f"- {metric}: {degraded_pct:.1f}% 设备恶化 ({degraded_count}/{total_valid})\n"
                    if degraded_count > 0:
                        top_degraded = column_data['degraded_devices'][:3]
                        summary += f"  └─ 主要恶化设备: {', '.join(top_degraded)}{'等' if len(column_data['degraded_devices']) > 3 else ''}\n"

        summary += f"\n**关键发现:**\n"

        # 总体分析洞察
        overall_stats = overall_analysis['summary_stats']
        degraded_pct = (overall_stats['degraded_metrics'] / max(overall_stats['total_metrics_analyzed'], 1)) * 100
        improved_pct = (overall_stats['improved_metrics'] / max(overall_stats['total_metrics_analyzed'], 1)) * 100

        summary += f"📊 **总体性能趋势**:\n"
        summary += f"   - {degraded_pct:.1f}% 的指标出现恶化 ({overall_stats['degraded_metrics']}/{overall_stats['total_metrics_analyzed']})\n"
        summary += f"   - {improved_pct:.1f}% 的指标出现改善 ({overall_stats['improved_metrics']}/{overall_stats['total_metrics_analyzed']})\n"

        if overall_stats['degraded_metrics'] > overall_stats['improved_metrics']:
            summary += f"   ⚠️ 总体趋势：性能恶化趋势明显\n"
        elif overall_stats['improved_metrics'] > overall_stats['degraded_metrics']:
            summary += f"   ✅ 总体趋势：性能改善趋势明显\n"
        else:
            summary += f"   🔵 总体趋势：性能表现相对稳定\n"

        summary += f"\n🔧 **设备级分析**:\n"
        if degraded_devices:
            summary += f"⚠️  共发现 {len(degraded_devices)} 个设备存在性能恶化问题\n"
            summary += f"⚠️  建议重点关注: {', '.join(degraded_devices[:3])}{'等设备' if len(degraded_devices) > 3 else ''}\n"
        else:
            summary += f"✅ 未发现明显的设备性能恶化问题\n"

        if improved_devices:
            summary += f"✅ {len(improved_devices)} 个设备显示性能改善\n"

        summary += f"\n**建议行动:**\n"
        summary += f"1. 针对恶化设备进行深入调查，分析恶化原因\n"
        summary += f"2. 检查test组变更是否对特定设备型号产生负面影响\n"
        summary += f"3. 考虑对恶化设备进行专门优化或回退test组变更\n"

        return summary

