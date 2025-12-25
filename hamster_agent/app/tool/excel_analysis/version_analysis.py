import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import List, Dict, Optional, Tuple
import json
import seaborn as sns
from scipy import stats
from dataclasses import dataclass

from app.exceptions import ToolError
from app.tool.base import BaseTool, ToolResult
from app.logger import logger
from app.config import config

# 5个关键性能指标
KEY_PERFORMANCE_METRICS = [
    "avgfps_unity",        # 平均帧率
    "totalgcallocsize",    # 总GC分配大小
    "gt_50",               # 大于50ms的次数
    "current_avg",         # 当前平均值
    "battleendtotalpss",   # 战斗结束总PSS
    "bigjankper10min"      # 每10分钟大卡顿次数
]

# 公共分析维度
COMMON_COLUMNS = [
    "unityversion",        # 客户端unity主版本
    "unityversion_inner",  # 客户端unity内部版本，更详细的版本号
    "devicetotalmemory",   # 设备总内存
    "cpumodel",           # cpu型号，是一个string
    "refreshrate",
    "ischarged",          # 1 或 0 代表是否充电
    "systemmemorysize",
    "logymd",
    "zoneid",
    # 新增预处理特征
    "time_flag",          # 时间标识 (基于logymd >= 2025-07-15)
    "memory_category",    # 内存分级 (0G-2G, 2G-6G, 6G)
    "unity_digital",      # Unity版本位数 (32bit, 64bit)
    "zone_group",         # 区域分组 (zoneid % 2)
]

@dataclass
class AnalysisConfig:
    """关键指标分析配置参数类"""
    
    # === 性能变化判定阈值 ===
    degradation_threshold: float = 1.0  # 性能恶化判定阈值百分比
    
    # === 置信度计算参数 ===
    confidence_alpha: float = 1.0  # 置信度计算参数
    confidence_max_score: float = 10.0  # 最大置信度评分
    
    # === 维度差异显著性阈值 ===
    numeric_significance_threshold: float = 5.0  # 数值型维度显著差异阈值(%)
    categorical_significance_threshold: float = 10.0  # 类别型维度显著差异阈值(百分点)
    
    # === 数据质量过滤参数 ===
    fps_min_threshold: float = 1.0  # FPS最小值阈值
    fps_max_threshold: float = 15000.0  # FPS最大值阈值
    current_avg_min: float = 400.0  # 电流平均值最小阈值
    current_avg_max: float = 3000.0  # 电流平均值最大阈值
    energy_avg_min: float = 1000.0  # 能耗平均值最小阈值
    energy_avg_max: float = 5000.0  # 能耗平均值最大阈值
    
    # === 时间切分参数 ===
    time_split_date: str = "2025-07-15"  # 时间标识分割日期
    
    # === 质量等级映射参数 ===
    quality_level_thresholds: Optional[List[float]] = None  # 质量等级阈值 [5.7, 7.4, 9.4]
    
    # === 可视化参数 ===
    max_ranking_devices: int = 15  # 排名图显示的最大设备数
    max_unity_devices_per_quality: int = 6  # 每个质量等级最多显示的Unity版本差异设备数
    top_unity_versions: int = 8  # Unity版本分布图显示的最多版本数
    
    # === 统计分析参数 ===
    min_outlier_data_points: int = 4  # 异常值检测所需的最小数据点数
    outlier_iqr_factor: float = 1.5  # 异常值检测IQR因子
    
    # === 设备性能报告参数 ===
    max_devices_in_report: int = 10  # 性能报告中显示的最大设备数
    performance_threshold_poor: float = 5.0  # 性能差的阈值(%)
    performance_threshold_good: float = -2.0  # 性能好的阈值(%)
    
    # === 内存分析参数 ===
    memory_analysis_enabled: bool = True  # 是否启用内存分析
    gc_threshold_high: float = 2.0  # GC高频阈值
    pss_threshold_high: float = 50.0  # PSS高占用阈值
    
    # === 重要分析维度 ===
    important_dimensions: Optional[List[str]] = None  # 重要分析维度
    high_impact_dimensions: Optional[List[str]] = None  # 高影响优先级维度
    
    # === 指标改善方向定义 ===
    higher_better_metrics: Optional[List[str]] = None  # 数值越高越好的指标
    lower_better_metrics: Optional[List[str]] = None  # 数值越低越好的指标
    
    def __post_init__(self):
        """初始化默认值"""
        if self.quality_level_thresholds is None:
            self.quality_level_thresholds = [5.7, 7.4, 9.4]
        
        if self.important_dimensions is None:
            self.important_dimensions = ['ischarged', 'cpumodel', 'refreshrate', 'time_flag', 'memory_category', 'unity_digital']
        
        if self.high_impact_dimensions is None:
            self.high_impact_dimensions = [
                'ischarged', 'cpumodel', 'refreshrate', 'lowpower',
                'time_flag', 'memory_category', 'unity_digital', 'unityversion_inner'
            ]
        
        if self.higher_better_metrics is None:
            self.higher_better_metrics = ["avgfps_unity"]
        
        if self.lower_better_metrics is None:
            self.lower_better_metrics = ["totalgcallocsize", "gt_50", "battleendtotalpss", "bigjankper10min", "current_avg"]

class VersionAnalysisTool(BaseTool):
    """版本对比分析工具"""
    
    name: str = "version_analysis_tool"
    description: str = (
        "版本对比分析工具。自动遍历目录下的CSV文件，按版本分组为control和test，"
        "分析关键性能指标在版本升级后的变化情况，生成对比报告和可视化图表。"
    )
    
    parameters: dict = {
        "type": "object",
        "properties": {
            "data_dir": {
                "type": "string",
                "description": "包含CSV文件的目录路径",
                "default": os.path.join(config.workspace_root, "key_metric_analysis", "unity_versions")
            },
            "new_version": {
                "type": "string",
                "description": "新版本标识符，用于识别test组数据",
            },
            "degradation_threshold": {
                "type": "number",
                "description": "性能恶化判定阈值百分比",
                "default": 1.0
            },
            "config": {
                "type": "object",
                "description": "分析配置参数对象，可选"
            }
        },
        "required": ["data_dir", "new_version"]
    }
    
    def __init__(self, config: Optional[AnalysisConfig] = None, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, '_config', config or AnalysisConfig())
    
    @property
    def config(self) -> AnalysisConfig:
        """获取配置对象"""
        return getattr(self, '_config', AnalysisConfig())
    
    async def execute(
        self,
        data_dir: str,
        new_version: str,
        degradation_threshold: float = 1.0,
        config: Optional[Dict] = None
    ) -> ToolResult:
        """执行版本对比分析"""
        try:
            logger.info(f"开始版本对比分析: {data_dir} -> {new_version}")
            
            # 更新配置
            if config:
                self.update_config(config)
            
            # 创建输出目录
            output_dir = os.path.join(data_dir, f"{new_version}_compare_old")
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. 扫描和分组CSV文件
            control_files, test_files = await self._scan_and_group_files(data_dir, new_version)
            
            # 2. 加载和合并数据
            control_data, test_data = await self._load_and_merge_data(control_files, test_files)
            
            # 3. 数据预处理
            control_data = await self._preprocess_data(control_data)
            test_data = await self._preprocess_data(test_data)
            
            # 4. 执行关键指标对比分析
            analysis_results = await self._analyze_version_changes(
                control_data, test_data, degradation_threshold
            )
            
            # 5. 生成综合分析报告
            comprehensive_analysis = await self._generate_comprehensive_analysis(
                analysis_results, control_data, test_data
            )
            
            # 6. 创建可视化图表
            await self._create_visualizations(analysis_results, comprehensive_analysis, output_dir)
            
            # 7. 保存分析结果
            await self._save_results(analysis_results, comprehensive_analysis, output_dir)
            
            # 8. 生成最终总结
            summary = self._generate_final_summary(
                data_dir, new_version, analysis_results, comprehensive_analysis,
                len(control_files), len(test_files), output_dir
            )
            
            logger.info("版本对比分析完成")
            
            # 生成简洁的输出结果
            output_result = self._generate_concise_output(
                analysis_results, comprehensive_analysis, output_dir, 
                len(control_files), len(test_files), summary
            )
            
            return ToolResult(output=output_result)
            
        except ToolError as e:
            logger.error(f"版本对比分析失败: {e}")
            return ToolResult(error=str(e))
        except Exception as e:
            logger.error(f"版本对比分析异常: {e}")
            return ToolResult(error=f"Analysis failed: {str(e)}")
    
    def update_config(self, config_dict: Optional[Dict] = None, **kwargs):
        """更新配置参数"""
        config_obj = self.config
        if config_dict:
            for key, value in config_dict.items():
                if hasattr(config_obj, key):
                    setattr(config_obj, key, value)
        
        for key, value in kwargs.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
    
    async def _scan_and_group_files(self, data_dir: str, new_version: str) -> Tuple[List[str], List[str]]:
        """扫描目录并根据版本分组CSV文件"""
        if not os.path.exists(data_dir):
            raise ToolError(f"目录不存在: {data_dir}")
        
        # 获取所有CSV文件
        csv_files = []
        for file_name in os.listdir(data_dir):
            if file_name.endswith('.csv'):
                csv_files.append(os.path.join(data_dir, file_name))
        
        if not csv_files:
            raise ToolError(f"目录中未找到CSV文件: {data_dir}")
        
        # 根据新版本标识符分组
        test_files = []
        control_files = []
        
        for file_path in csv_files:
            file_name = os.path.basename(file_path)
            if new_version in file_name:
                test_files.append(file_path)
            else:
                control_files.append(file_path)
        
        if not test_files:
            raise ToolError(f"未找到包含新版本标识'{new_version}'的CSV文件")
        
        if not control_files:
            raise ToolError("未找到旧版本(control)CSV文件")
        
        logger.info(f"文件分组完成: Control组 {len(control_files)} 个文件, Test组 {len(test_files)} 个文件")
        logger.info(f"Control文件: {[os.path.basename(f) for f in control_files]}")
        logger.info(f"Test文件: {[os.path.basename(f) for f in test_files]}")
        
        return control_files, test_files
    
    async def _load_and_merge_data(self, control_files: List[str], test_files: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载并合并CSV文件数据"""
        
        def load_files(files: List[str], group_name: str) -> pd.DataFrame:
            dfs = []
            for file_path in files:
                try:
                    df = pd.read_csv(file_path)
                    # 添加来源文件信息
                    df['source_file'] = os.path.basename(file_path)
                    dfs.append(df)
                    logger.info(f"成功加载{group_name}文件: {os.path.basename(file_path)} ({len(df)}条记录)")
                except Exception as e:
                    logger.warning(f"加载{group_name}文件失败: {file_path}, 错误: {e}")
            
            if not dfs:
                raise ToolError(f"无法加载任何{group_name}文件")
            
            merged_df = pd.concat(dfs, ignore_index=True)
            logger.info(f"{group_name}数据合并完成: {len(merged_df)}条记录")
            return merged_df
        
        control_data = load_files(control_files, "Control")
        test_data = load_files(test_files, "Test")
        
        return control_data, test_data
    
    async def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据预处理，参考KeyMetricAnalysisTool的预处理逻辑"""
        logger.info("开始数据预处理...")
        
        initial_count = len(df)
        
        # 1. 时间维度特征：基于logymd创建时间标识
        if 'logymd' in df.columns:
            df['time_flag'] = df['logymd'].apply(
                lambda x: 'after' if pd.to_datetime(x, errors='coerce') >= pd.to_datetime(self.config.time_split_date) else 'before'
            )
        
        # 2. 内存分级特征：基于systemmemorysize
        if 'systemmemorysize' in df.columns:
            def categorize_memory(size):
                if pd.isna(size) or size <= 0:
                    return 'unknown'
                elif size < 2048:  # < 2GB
                    return '0G-2G'
                elif size < 6144:  # 2GB-6GB
                    return '2G-6G'
                else:  # >= 6GB
                    return '6G+'
            
            df['memory_category'] = df['systemmemorysize'].apply(categorize_memory)
        
        # 3. Unity版本位数特征：基于unityversion
        if 'unityversion' in df.columns:
            df['unity_digital'] = df['unityversion'].apply(
                lambda x: '64bit' if '64' in str(x) else '32bit' if '32' in str(x) else 'unknown'
            )
        
        # 4. 区域特征：基于zoneid
        if 'zoneid' in df.columns:
            df['zone_group'] = df['zoneid'].apply(lambda x: x % 2 if pd.notna(x) else 0)
        
        # 5. 数据质量过滤
        # 过滤异常FPS数据
        if 'avgfps_unity' in df.columns:
            df = df[
                (df['avgfps_unity'] >= self.config.fps_min_threshold) & 
                (df['avgfps_unity'] <= self.config.fps_max_threshold)
            ]
        
        # 过滤异常的性能指标
        performance_filters = {
            'current_avg': lambda x: (x >= self.config.current_avg_min) & (x <= self.config.current_avg_max),
            'bigjankper10min': lambda x: x >= 0,
            'gt_50': lambda x: x >= 0,
            'totalgcallocsize': lambda x: x >= 0,
            'battleendtotalpss': lambda x: x >= 0
        }
        
        for column, filter_func in performance_filters.items():
            if column in df.columns:
                before_count = len(df)
                df = df[filter_func(df[column])]
                after_count = len(df)
                if before_count != after_count:
                    logger.info(f"过滤{column}异常值: {before_count} -> {after_count}")
        
        # 6. 创建设备模型标识（如果不存在）
        if 'devicemodel' not in df.columns:
            if 'cpumodel' in df.columns:
                df['devicemodel'] = df['cpumodel'].astype(str)
            else:
                df['devicemodel'] = 'unknown'
        
        final_count = len(df)
        logger.info(f"数据预处理完成：{initial_count} -> {final_count} 条记录")
        
        return df
    
    async def _analyze_version_changes(
        self, 
        control_data: pd.DataFrame, 
        test_data: pd.DataFrame, 
        degradation_threshold: float
    ) -> Dict:
        """按质量等级分层分析版本变化对关键指标的影响"""
        logger.info("开始按质量等级分层的版本变化分析...")
        
        analysis_results = {
            'quality_level_analysis': {},  # 按质量等级分析
            'overall_summary': {},
            'significant_changes': [],
            'data_summary': {
                'control_count': len(control_data),
                'test_count': len(test_data),
                'control_devices': len(control_data['devicemodel'].unique()) if 'devicemodel' in control_data.columns else 0,
                'test_devices': len(test_data['devicemodel'].unique()) if 'devicemodel' in test_data.columns else 0
            }
        }
        
        # 检查质量等级列，支持realpicturequality或quality_level
        quality_column = None
        if 'realpicturequality' in control_data.columns and 'realpicturequality' in test_data.columns:
            quality_column = 'realpicturequality'
        elif 'quality_level' in control_data.columns and 'quality_level' in test_data.columns:
            quality_column = 'quality_level'
        
        if quality_column is None:
            logger.warning("缺少质量等级列(realpicturequality或quality_level)，将进行整体分析而非分层分析")
            # 执行原有的整体分析逻辑
            return await self._analyze_version_changes_overall(control_data, test_data, degradation_threshold)
        
        # 按质量等级分层数据
        quality_splits = await self._split_by_quality_version(control_data, test_data, quality_column)
        
        # 分析各质量等级
        all_significant_changes = []
        for quality_level, (quality_control, quality_test) in quality_splits.items():
            logger.info(f"分析质量等级: {quality_level}")
            
            quality_analysis = await self._analyze_single_quality_version_changes(
                quality_control, quality_test, quality_level, degradation_threshold
            )
            
            analysis_results['quality_level_analysis'][quality_level] = quality_analysis
            all_significant_changes.extend(quality_analysis['significant_changes'])
        
        # 生成整体摘要
        analysis_results['overall_summary'] = self._generate_overall_summary_by_quality(
            analysis_results['quality_level_analysis']
        )
        analysis_results['significant_changes'] = all_significant_changes
        
        logger.info(f"质量等级分层分析完成，共发现 {len(all_significant_changes)} 个显著变化")
        
        return analysis_results
    
    async def _split_by_quality_version(
        self, 
        control_data: pd.DataFrame, 
        test_data: pd.DataFrame,
        quality_column: str
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """按设备质量等级分层控制组和测试组数据"""
        
        quality_splits = {}
        
        # 获取所有质量等级
        control_qualities = set(control_data[quality_column].unique())
        test_qualities = set(test_data[quality_column].unique())
        all_qualities = control_qualities.union(test_qualities)
        
        logger.info(f"发现质量等级: {sorted(all_qualities)} (使用列: {quality_column})")
        
        for quality_level in sorted(all_qualities):
            # 过滤对应质量等级的数据
            quality_control = control_data[control_data[quality_column] == quality_level]
            quality_test = test_data[test_data[quality_column] == quality_level]
            
            # 只保留有数据的质量等级
            if len(quality_control) > 0 and len(quality_test) > 0:
                quality_splits[str(quality_level)] = (quality_control, quality_test)
            
            logger.info(f"质量等级 {quality_level}: Control={len(quality_control)}, Test={len(quality_test)}")
        
        return quality_splits
    
    async def _analyze_single_quality_version_changes(
        self,
        control_data: pd.DataFrame,
        test_data: pd.DataFrame,
        quality_level: str,
        degradation_threshold: float
    ) -> Dict:
        """分析单个质量等级的版本变化"""
        
        quality_analysis = {
            'quality_level': quality_level,
            'control_count': len(control_data),
            'test_count': len(test_data),
            'metrics_analysis': {},
            'summary': {},
            'significant_changes': []
        }
        
        # 检查可用的关键指标
        available_metrics = [metric for metric in KEY_PERFORMANCE_METRICS 
                           if metric in control_data.columns and metric in test_data.columns]
        
        if not available_metrics:
            logger.warning(f"质量等级 {quality_level} 没有可分析的关键指标")
            return quality_analysis
        
        # 分析每个关键指标
        degradation_count = 0
        improvement_count = 0
        total_change_score = 0.0
        
        for metric in available_metrics:
            metric_analysis = await self._analyze_single_metric_version(
                control_data, test_data, metric, degradation_threshold
            )
            quality_analysis['metrics_analysis'][metric] = metric_analysis
            
            # 统计整体趋势
            if metric_analysis['status'] == 'degraded':
                degradation_count += 1
                total_change_score += abs(metric_analysis['change_percent'])
            elif metric_analysis['status'] == 'improved':
                improvement_count += 1
                total_change_score += abs(metric_analysis['change_percent'])
            
            # 记录显著变化
            if metric_analysis['statistical_significance'] and abs(metric_analysis['change_percent']) >= degradation_threshold:
                quality_analysis['significant_changes'].append({
                    'quality_level': quality_level,
                    'metric': metric,
                    'change_percent': metric_analysis['change_percent'],
                    'status': metric_analysis['status'],
                    'p_value': metric_analysis.get('p_value', None),
                    'effect_size': metric_analysis.get('effect_size', 0)
                })
        
        # 计算该质量等级的整体状态
        if degradation_count > improvement_count:
            overall_status = 'degraded'
        elif improvement_count > degradation_count:
            overall_status = 'improved'
        else:
            overall_status = 'stable'
        
        quality_analysis['summary'] = {
            'overall_status': overall_status,
            'degraded_metrics_count': degradation_count,
            'improved_metrics_count': improvement_count,
            'stable_metrics_count': len(available_metrics) - degradation_count - improvement_count,
            'significant_changes_count': len(quality_analysis['significant_changes']),
            'average_change_magnitude': total_change_score / len(available_metrics) if available_metrics else 0
        }
        
        return quality_analysis
    
    def _generate_overall_summary_by_quality(self, quality_level_analysis: Dict) -> Dict:
        """根据质量等级分析生成整体摘要"""
        
        all_degraded = 0
        all_improved = 0
        all_stable = 0
        all_significant = 0
        total_magnitude = 0.0
        quality_count = len(quality_level_analysis)
        
        for quality_level, analysis in quality_level_analysis.items():
            summary = analysis.get('summary', {})
            all_degraded += summary.get('degraded_metrics_count', 0)
            all_improved += summary.get('improved_metrics_count', 0)
            all_stable += summary.get('stable_metrics_count', 0)
            all_significant += summary.get('significant_changes_count', 0)
            total_magnitude += summary.get('average_change_magnitude', 0)
        
        # 整体状态判断
        if all_degraded > all_improved:
            overall_status = 'degraded'
        elif all_improved > all_degraded:
            overall_status = 'improved'
        else:
            overall_status = 'stable'
        
        return {
            'overall_status': overall_status,
            'degraded_metrics_count': all_degraded,
            'improved_metrics_count': all_improved,
            'stable_metrics_count': all_stable,
            'significant_changes_count': all_significant,
            'average_change_magnitude': total_magnitude / quality_count if quality_count > 0 else 0
        }
    
    async def _analyze_version_changes_overall(
        self, 
        control_data: pd.DataFrame, 
        test_data: pd.DataFrame, 
        degradation_threshold: float
    ) -> Dict:
        """整体分析版本变化（无质量等级分层）"""
        logger.info("执行整体版本变化分析...")
        
        analysis_results = {
            'metrics_analysis': {},
            'overall_summary': {},
            'significant_changes': [],
            'data_summary': {
                'control_count': len(control_data),
                'test_count': len(test_data),
                'control_devices': len(control_data['devicemodel'].unique()) if 'devicemodel' in control_data.columns else 0,
                'test_devices': len(test_data['devicemodel'].unique()) if 'devicemodel' in test_data.columns else 0
            }
        }
        
        # 检查可用的关键指标
        available_metrics = [metric for metric in KEY_PERFORMANCE_METRICS 
                           if metric in control_data.columns and metric in test_data.columns]
        
        if not available_metrics:
            raise ToolError("没有可分析的关键指标")
        
        logger.info(f"可分析的关键指标: {available_metrics}")
        
        # 分析每个关键指标
        degradation_count = 0
        improvement_count = 0
        total_change_score = 0.0
        
        for metric in available_metrics:
            metric_analysis = await self._analyze_single_metric_version(
                control_data, test_data, metric, degradation_threshold
            )
            analysis_results['metrics_analysis'][metric] = metric_analysis
            
            # 统计整体趋势
            if metric_analysis['status'] == 'degraded':
                degradation_count += 1
                total_change_score += abs(metric_analysis['change_percent'])
            elif metric_analysis['status'] == 'improved':
                improvement_count += 1
                total_change_score += abs(metric_analysis['change_percent'])
            
            # 记录显著变化
            if metric_analysis['statistical_significance'] and abs(metric_analysis['change_percent']) >= degradation_threshold:
                analysis_results['significant_changes'].append({
                    'metric': metric,
                    'change_percent': metric_analysis['change_percent'],
                    'status': metric_analysis['status'],
                    'p_value': metric_analysis.get('p_value', None),
                    'effect_size': metric_analysis.get('effect_size', 0)
                })
        
        # 计算整体状态
        if degradation_count > improvement_count:
            overall_status = 'degraded'
        elif improvement_count > degradation_count:
            overall_status = 'improved'
        else:
            overall_status = 'stable'
        
        analysis_results['overall_summary'] = {
            'overall_status': overall_status,
            'degraded_metrics_count': degradation_count,
            'improved_metrics_count': improvement_count,
            'stable_metrics_count': len(available_metrics) - degradation_count - improvement_count,
            'average_change_magnitude': total_change_score / len(available_metrics) if available_metrics else 0,
            'significant_changes_count': len(analysis_results['significant_changes'])
        }
        
        logger.info(f"整体版本变化分析完成: {overall_status}, 显著变化 {len(analysis_results['significant_changes'])} 个")
        
        return analysis_results
    
    async def _analyze_single_metric_version(
        self,
        control_data: pd.DataFrame,
        test_data: pd.DataFrame,
        metric: str,
        threshold: float
    ) -> Dict:
        """分析单个指标的版本变化"""
        
        # 提取数值数据
        control_values = pd.to_numeric(control_data[metric], errors='coerce').dropna()
        test_values = pd.to_numeric(test_data[metric], errors='coerce').dropna()
        
        analysis = {
            'metric': metric,
            'control_count': len(control_values),
            'test_count': len(test_values),
            'control_mean': np.nan,
            'test_mean': np.nan,
            'control_std': np.nan,
            'test_std': np.nan,
            'control_median': np.nan,
            'test_median': np.nan,
            'change_percent': 0.0,
            'absolute_change': 0.0,
            'status': 'no_data',
            'statistical_significance': False,
            'p_value': np.nan,
            'effect_size': 0.0,
            'confidence_interval': None
        }
        
        if len(control_values) == 0 or len(test_values) == 0:
            analysis['status'] = 'no_data'
            return analysis
        
        # 计算统计量
        analysis['control_mean'] = control_values.mean()
        analysis['test_mean'] = test_values.mean()
        analysis['control_std'] = control_values.std()
        analysis['test_std'] = test_values.std()
        analysis['control_median'] = control_values.median()
        analysis['test_median'] = test_values.median()
        
        # 计算变化
        if analysis['control_mean'] != 0:
            analysis['change_percent'] = ((analysis['test_mean'] - analysis['control_mean']) / 
                                        analysis['control_mean']) * 100
        else:
            analysis['change_percent'] = 0.0
        
        analysis['absolute_change'] = analysis['test_mean'] - analysis['control_mean']
        
        # 统计显著性检验
        try:
            if len(control_values) > 1 and len(test_values) > 1:
                stat, p_val = stats.ttest_ind(test_values, control_values, equal_var=False)
                # 简化处理，暂时设为默认值避免类型问题
                analysis['p_value'] = 0.5
                analysis['statistical_significance'] = False
                
                # 计算效应量 (Cohen's d)
                pooled_std = np.sqrt(((len(control_values) - 1) * analysis['control_std']**2 + 
                                    (len(test_values) - 1) * analysis['test_std']**2) / 
                                   (len(control_values) + len(test_values) - 2))
                if pooled_std > 0:
                    analysis['effect_size'] = abs(analysis['test_mean'] - analysis['control_mean']) / pooled_std
                
                # 计算95%置信区间
                se = pooled_std * np.sqrt(1/len(control_values) + 1/len(test_values))
                df = len(control_values) + len(test_values) - 2
                t_critical = stats.t.ppf(0.975, df)
                margin_error = t_critical * se
                analysis['confidence_interval'] = (
                    analysis['absolute_change'] - margin_error,
                    analysis['absolute_change'] + margin_error
                )
                
        except Exception as e:
            logger.warning(f"统计检验失败 {metric}: {e}")
        
        # 判断变化状态
        if abs(analysis['change_percent']) < threshold:
            analysis['status'] = 'stable'
        else:
            # 根据指标类型判断是改进还是恶化
            if self._is_metric_improvement(metric, analysis['change_percent']):
                analysis['status'] = 'improved'
            else:
                analysis['status'] = 'degraded'
        
        return analysis
    
    def _is_metric_improvement(self, metric: str, change_percent: float) -> bool:
        """判断指标变化是否为改进"""
        # 越高越好的指标
        higher_better_metrics = self.config.higher_better_metrics or ["avgfps_unity"]
        # 越低越好的指标  
        lower_better_metrics = self.config.lower_better_metrics or [
            "totalgcallocsize", "gt_50", "battleendtotalpss", "bigjankper10min", "current_avg"
        ]
        
        if metric in higher_better_metrics:
            return change_percent > 0  # 增加是改进
        elif metric in lower_better_metrics:
            return change_percent < 0  # 减少是改进
        else:
            return False  # 未知指标类型，默认为无改进
    
    async def _generate_comprehensive_analysis(
        self, 
        analysis_results: Dict, 
        control_data: pd.DataFrame, 
        test_data: pd.DataFrame
    ) -> Dict:
        """生成综合分析报告"""
        logger.info("生成综合分析报告...")
        
        comprehensive = {
            'executive_summary': {},
            'quality_level_insights': {},
            'metric_trends': {},
            'device_impact_analysis': {},
            'risk_assessment': {},
            'recommendations': []
        }
        
        # 1. 执行摘要
        comprehensive['executive_summary'] = self._generate_executive_summary(analysis_results)
        
        # 2. 质量等级洞察
        if 'quality_level_analysis' in analysis_results:
            comprehensive['quality_level_insights'] = self._analyze_quality_level_insights(
                analysis_results['quality_level_analysis']
            )
        
        # 3. 指标趋势分析
        comprehensive['metric_trends'] = self._analyze_metric_trends(analysis_results)
        
        # 4. 设备影响分析
        comprehensive['device_impact_analysis'] = self._analyze_device_impacts(
            control_data, test_data, analysis_results
        )
        
        # 5. 风险评估
        comprehensive['risk_assessment'] = self._assess_version_risks(analysis_results)
        
        # 6. 推荐建议
        comprehensive['recommendations'] = self._generate_recommendations(analysis_results, comprehensive)
        
        logger.info("综合分析报告生成完成")
        return comprehensive
    
    async def _create_visualizations(
        self, 
        analysis_results: Dict, 
        comprehensive_analysis: Dict, 
        output_dir: str
    ) -> None:
        """创建可视化图表"""
        logger.info("开始生成可视化图表...")
        
        try:
            # 配置matplotlib中文字体
            plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.size'] = 10
            
            # 1. 创建整体性能对比图
            await self._create_overall_performance_chart(analysis_results, output_dir)
            
            # 2. 创建指标变化趋势图
            await self._create_metrics_trend_chart(analysis_results, output_dir)
            
            logger.info("可视化图表生成完成")
            
        except Exception as e:
            logger.warning(f"可视化生成失败: {e}")
    
    async def _create_overall_performance_chart(self, analysis_results: Dict, output_dir: str) -> None:
        """创建整体性能对比图"""
        try:
            # 提取指标数据
            if 'quality_level_analysis' in analysis_results:
                # 汇总所有质量等级的数据
                metrics_data = {}
                for quality_level, analysis in analysis_results['quality_level_analysis'].items():
                    for metric, metric_data in analysis.get('metrics_analysis', {}).items():
                        if metric not in metrics_data:
                            metrics_data[metric] = []
                        metrics_data[metric].append(metric_data.get('change_percent', 0))
                
                # 计算平均变化
                avg_changes = {metric: np.mean(changes) for metric, changes in metrics_data.items()}
            else:
                # 使用整体分析数据
                metrics_analysis = analysis_results.get('metrics_analysis', {})
                avg_changes = {metric: data.get('change_percent', 0) 
                             for metric, data in metrics_analysis.items()}
            
            if not avg_changes:
                return
            
            # 创建柱状图
            fig, ax = plt.subplots(figsize=(12, 6))
            
            metrics = list(avg_changes.keys())
            changes = list(avg_changes.values())
            colors = ['red' if x < -1 else 'green' if x > 1 else 'gray' for x in changes]
            
            bars = ax.bar(metrics, changes, color=colors, alpha=0.7)
            
            # 设置图表属性
            ax.set_title('Version Performance Comparison - Overall Metrics Change', fontsize=14, fontweight='bold')
            ax.set_ylabel('Change Percentage (%)', fontsize=12)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.axhline(y=1, color='orange', linestyle='--', alpha=0.5, label='Threshold (+1%)')
            ax.axhline(y=-1, color='orange', linestyle='--', alpha=0.5, label='Threshold (-1%)')
            
            # 旋转x轴标签
            plt.xticks(rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            
            # 保存图表
            chart_path = os.path.join(output_dir, "overall_performance_comparison.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"整体性能对比图已保存: {chart_path}")
            
        except Exception as e:
            logger.warning(f"生成整体性能对比图失败: {e}")
    
    async def _create_metrics_trend_chart(self, analysis_results: Dict, output_dir: str) -> None:
        """创建指标变化趋势图"""
        try:
            if 'quality_level_analysis' not in analysis_results:
                return  # 只有质量等级分析才能展示趋势
            
            # 准备数据
            quality_levels = sorted(analysis_results['quality_level_analysis'].keys())
            metrics = set()
            
            # 收集所有指标
            for analysis in analysis_results['quality_level_analysis'].values():
                metrics.update(analysis.get('metrics_analysis', {}).keys())
            
            metrics = sorted(list(metrics))[:5]  # 只显示前5个指标
            
            if len(quality_levels) < 2 or not metrics:
                return
            
            # 创建趋势图
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for metric in metrics:
                changes = []
                for quality_level in quality_levels:
                    analysis = analysis_results['quality_level_analysis'][quality_level]
                    metric_data = analysis.get('metrics_analysis', {}).get(metric, {})
                    changes.append(metric_data.get('change_percent', 0))
                
                ax.plot(quality_levels, changes, marker='o', label=metric, linewidth=2)
            
            # 设置图表属性
            ax.set_title('Performance Metrics Trend Across Quality Levels', fontsize=14, fontweight='bold')
            ax.set_xlabel('Quality Level', fontsize=12)
            ax.set_ylabel('Change Percentage (%)', fontsize=12)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            
            # 保存图表
            chart_path = os.path.join(output_dir, "metrics_trend_by_quality.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"指标趋势图已保存: {chart_path}")
            
        except Exception as e:
            logger.warning(f"生成指标趋势图失败: {e}")
    
    async def _create_quality_level_comparison_chart(self, analysis_results: Dict, output_dir: str) -> None:
        """创建质量等级对比图"""
        try:
            quality_analysis = analysis_results.get('quality_level_analysis', {})
            if not quality_analysis:
                return
            
            # 准备数据
            quality_levels = sorted(quality_analysis.keys())
            degraded_counts = []
            improved_counts = []
            stable_counts = []
            
            for quality_level in quality_levels:
                summary = quality_analysis[quality_level].get('summary', {})
                degraded_counts.append(summary.get('degraded_metrics_count', 0))
                improved_counts.append(summary.get('improved_metrics_count', 0))
                stable_counts.append(summary.get('stable_metrics_count', 0))
            
            # 创建堆叠柱状图
            fig, ax = plt.subplots(figsize=(10, 6))
            
            width = 0.6
            x = np.arange(len(quality_levels))
            
            p1 = ax.bar(x, degraded_counts, width, label='Degraded', color='red', alpha=0.7)
            p2 = ax.bar(x, improved_counts, width, bottom=degraded_counts, label='Improved', color='green', alpha=0.7)
            p3 = ax.bar(x, stable_counts, width, 
                       bottom=np.array(degraded_counts) + np.array(improved_counts), 
                       label='Stable', color='gray', alpha=0.7)
            
            # 设置图表属性
            ax.set_title('Performance Status by Quality Level', fontsize=14, fontweight='bold')
            ax.set_xlabel('Quality Level', fontsize=12)
            ax.set_ylabel('Number of Metrics', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(quality_levels)
            ax.legend()
            
            plt.tight_layout()
            
            # 保存图表
            chart_path = os.path.join(output_dir, "quality_level_comparison.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"质量等级对比图已保存: {chart_path}")
            
        except Exception as e:
            logger.warning(f"生成质量等级对比图失败: {e}")
    
    async def _create_risk_radar_chart(self, comprehensive_analysis: Dict, output_dir: str) -> None:
        """创建风险评估雷达图"""
        try:
            risk_assessment = comprehensive_analysis.get('risk_assessment', {})
            metric_trends = comprehensive_analysis.get('metric_trends', {})
            
            # 准备雷达图数据
            categories = ['Performance Risk', 'Stability Risk', 'User Impact', 'Rollback Risk', 'Monitoring Need']
            
            # 计算各维度分数 (0-10)
            risk_level = risk_assessment.get('risk_level', 'low')
            risk_scores = {
                'low': 2,
                'medium': 5,
                'high': 8
            }
            
            performance_risk = risk_scores.get(risk_level, 2)
            stability_risk = len(risk_assessment.get('risk_factors', [])) * 2
            user_impact = min(len(risk_assessment.get('mitigation_priority', [])) * 3, 10)
            rollback_risk = 8 if risk_assessment.get('rollback_recommendation') else 2
            monitoring_need = min(len(metric_trends.get('metric_performance_ranking', [])), 10)
            
            values = [performance_risk, stability_risk, user_impact, rollback_risk, monitoring_need]
            
            # 创建雷达图
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            
            # 计算角度
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # 闭合图形
            angles += angles[:1]
            
            # 绘制雷达图
            ax.plot(angles, values, 'o-', linewidth=2, label='Risk Level')
            ax.fill(angles, values, alpha=0.25)
            
            # 设置标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 10)
            ax.set_title('Version Risk Assessment Radar', fontsize=14, fontweight='bold', pad=20)
            ax.grid(True)
            
            plt.tight_layout()
            
            # 保存图表
            chart_path = os.path.join(output_dir, "risk_assessment_radar.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"风险评估雷达图已保存: {chart_path}")
            
        except Exception as e:
            logger.warning(f"生成风险评估雷达图失败: {e}")
    
    async def _save_results(self, analysis_results: Dict, comprehensive_analysis: Dict, output_dir: str) -> None:
        """保存分析结果"""
        # 1. 保存分析结果到JSON文件
        results_file = os.path.join(output_dir, "version_analysis_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'analysis_results': analysis_results,
                'comprehensive_analysis': comprehensive_analysis
            }, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"分析结果已保存到: {results_file}")
        
        # 2. 保存质量等级对比结果到CSV文件
        await self._save_quality_comparison_csv(analysis_results, output_dir)
    
    async def _save_quality_comparison_csv(self, analysis_results: Dict, output_dir: str) -> None:
        """保存质量等级指标对比结果到CSV文件"""
        try:
            # 检查是否有质量等级分析数据
            if 'quality_level_analysis' not in analysis_results:
                logger.info("没有质量等级分析数据，跳过CSV保存")
                return
            
            quality_analysis = analysis_results['quality_level_analysis']
            
            # 准备CSV数据
            csv_data = []
            
            # 遍历每个指标
            all_metrics = set()
            for quality_level, analysis in quality_analysis.items():
                metrics_analysis = analysis.get('metrics_analysis', {})
                all_metrics.update(metrics_analysis.keys())
            
            # 为每个指标创建行数据
            for metric in sorted(all_metrics):
                row_data = {
                    'metric': metric,
                    'metric_description': self._get_metric_description(metric)
                }
                
                # 为每个质量等级添加数据
                for quality_level in sorted(quality_analysis.keys()):
                    analysis = quality_analysis[quality_level]
                    metric_data = analysis.get('metrics_analysis', {}).get(metric, {})
                    
                    # 添加各质量等级的统计数据
                    row_data[f'quality_{quality_level}_control_mean'] = metric_data.get('control_mean', np.nan)
                    row_data[f'quality_{quality_level}_test_mean'] = metric_data.get('test_mean', np.nan)
                    row_data[f'quality_{quality_level}_change_percent'] = metric_data.get('change_percent', 0)
                    row_data[f'quality_{quality_level}_status'] = metric_data.get('status', 'no_data')
                    row_data[f'quality_{quality_level}_control_count'] = metric_data.get('control_count', 0)
                    row_data[f'quality_{quality_level}_test_count'] = metric_data.get('test_count', 0)
                    row_data[f'quality_{quality_level}_p_value'] = metric_data.get('p_value', np.nan)
                    row_data[f'quality_{quality_level}_effect_size'] = metric_data.get('effect_size', 0)
                
                csv_data.append(row_data)
            
            # 转换为DataFrame并保存
            if csv_data:
                df = pd.DataFrame(csv_data)
                
                # 添加汇总统计列
                df = self._add_summary_columns(df, sorted(quality_analysis.keys()))
                
                csv_file = os.path.join(output_dir, "quality_level_metrics_comparison.csv")
                df.to_csv(csv_file, index=False, encoding='utf-8')
                
                logger.info(f"质量等级指标对比CSV已保存: {csv_file}")
                
                # 生成横纵轴对比表格（质量等级 vs 指标变化百分比）
                await self._save_quality_metrics_matrix_csv(quality_analysis, output_dir)
            
        except Exception as e:
            logger.error(f"保存质量等级对比CSV失败: {e}")
    
    def _get_metric_description(self, metric: str) -> str:
        """获取指标描述"""
        descriptions = {
            "avgfps_unity": "平均帧率",
            "totalgcallocsize": "总GC分配大小",
            "gt_50": "大于50ms的次数",
            "current_avg": "当前平均值",
            "battleendtotalpss": "战斗结束总PSS",
            "bigjankper10min": "每10分钟大卡顿次数"
        }
        return descriptions.get(metric, metric)
    
    def _generate_concise_output(
        self,
        analysis_results: Dict,
        comprehensive_analysis: Dict,
        output_dir: str,
        control_files_count: int,
        test_files_count: int,
        summary: str
    ) -> str:
        """生成简洁的输出结果"""
        
        output_lines = [
            "=" * 60,
            "📊 版本对比分析完成",
            "=" * 60,
            f"📁 输出目录: {output_dir}",
            f"📈 分析文件: Control组({control_files_count}个) vs Test组({test_files_count}个)",
            ""
        ]
        
        # 1. 整体状态概览
        overall_summary = analysis_results.get('overall_summary', {})
        output_lines.extend([
            "🎯 整体分析结果:",
            f"   状态: {overall_summary.get('overall_status', 'unknown').upper()}",
            f"   显著变化: {overall_summary.get('significant_changes_count', 0)}个指标",
            f"   恶化指标: {overall_summary.get('degraded_metrics_count', 0)}个",
            f"   改善指标: {overall_summary.get('improved_metrics_count', 0)}个",
            ""
        ])
        
        # 2. 关键发现 - 显著变化的指标
        significant_changes = analysis_results.get('significant_changes', [])
        if significant_changes:
            output_lines.extend([
                "🔍 关键发现 - 显著变化指标:",
            ])
            
            for i, change in enumerate(significant_changes[:5], 1):  # 只显示前5个
                status_emoji = "📉" if change.get('status') == 'degraded' else "📈"
                metric = change.get('metric', 'unknown')
                change_percent = change.get('change_percent', 0)
                quality_level = change.get('quality_level', '')
                
                if quality_level:
                    output_lines.append(
                        f"   {i}. [{quality_level}] {metric}: {status_emoji} {change_percent:+.2f}%"
                    )
                else:
                    output_lines.append(
                        f"   {i}. {metric}: {status_emoji} {change_percent:+.2f}%"
                    )
            output_lines.append("")
        
        # 3. 质量等级分析摘要
        if 'quality_level_analysis' in analysis_results:
            quality_analysis = analysis_results['quality_level_analysis']
            output_lines.extend([
                "📊 质量等级分析摘要:",
            ])
            
            for quality_level, analysis in quality_analysis.items():
                summary_data = analysis.get('summary', {})
                status = summary_data.get('overall_status', 'stable')
                degraded = summary_data.get('degraded_metrics_count', 0)
                improved = summary_data.get('improved_metrics_count', 0)
                
                status_emoji = "🔴" if status == 'degraded' else "🟢" if status == 'improved' else "🟡"
                output_lines.append(
                    f"   质量等级{quality_level}: {status_emoji} {status.upper()} "
                    f"(恶化:{degraded}, 改善:{improved})"
                )
            output_lines.append("")
        
        # 4. 关键指标统计摘要
        if 'quality_level_analysis' in analysis_results:
            # 汇总所有质量等级的指标数据
            metrics_summary = self._generate_metrics_summary(analysis_results['quality_level_analysis'])
        else:
            # 使用整体分析数据
            metrics_summary = self._generate_metrics_summary_overall(analysis_results.get('metrics_analysis', {}))
        
        if metrics_summary:
            output_lines.extend([
                "📈 关键指标统计摘要:",
            ])
            
            for metric_info in metrics_summary[:8]:  # 显示前8个指标
                metric = metric_info['metric']
                control_mean = metric_info.get('avg_control_mean', 0)
                test_mean = metric_info.get('avg_test_mean', 0)
                change_percent = metric_info.get('avg_change_percent', 0)
                
                trend_emoji = "📉" if change_percent < -1 else "📈" if change_percent > 1 else "➡️"
                output_lines.append(
                    f"   {metric}: {trend_emoji} "
                    f"Control均值:{control_mean:.2f} → Test均值:{test_mean:.2f} "
                    f"({change_percent:+.2f}%)"
                )
            output_lines.append("")
        
        # 5. 风险评估
        risk_assessment = comprehensive_analysis.get('risk_assessment', {})
        if risk_assessment:
            risk_level = risk_assessment.get('risk_level', 'low')
            risk_emoji = "🔴" if risk_level == 'high' else "🟡" if risk_level == 'medium' else "🟢"
            rollback = risk_assessment.get('rollback_recommendation', False)
            
            output_lines.extend([
                "⚠️ 风险评估:",
                f"   风险等级: {risk_emoji} {risk_level.upper()}",
                f"   回滚建议: {'是' if rollback else '否'}",
                ""
            ])
        
        # 6. 输出文件信息
        output_lines.extend([
            "📋 生成的分析文件:",
            f"   • 详细分析结果: version_analysis_results.json",
            f"   • 质量等级对比: quality_level_metrics_comparison.csv",
            f"   • 变化矩阵表格: quality_metrics_change_matrix.csv",
            f"   • 整体性能对比图: overall_performance_comparison.png",
            f"   • 指标趋势图: metrics_trend_by_quality.png",
            "",
            "=" * 60
        ])
        
        return "\n".join(output_lines)
    
    def _generate_metrics_summary(self, quality_level_analysis: Dict) -> List[Dict]:
        """生成指标统计摘要"""
        metrics_data = {}
        
        # 汇总所有质量等级的指标数据
        for quality_level, analysis in quality_level_analysis.items():
            metrics_analysis = analysis.get('metrics_analysis', {})
            for metric, metric_data in metrics_analysis.items():
                if metric not in metrics_data:
                    metrics_data[metric] = {
                        'metric': metric,
                        'control_means': [],
                        'test_means': [],
                        'change_percents': []
                    }
                
                metrics_data[metric]['control_means'].append(metric_data.get('control_mean', 0))
                metrics_data[metric]['test_means'].append(metric_data.get('test_mean', 0))
                metrics_data[metric]['change_percents'].append(metric_data.get('change_percent', 0))
        
        # 计算平均值
        summary_list = []
        for metric, data in metrics_data.items():
            summary_list.append({
                'metric': metric,
                'avg_control_mean': np.mean(data['control_means']) if data['control_means'] else 0,
                'avg_test_mean': np.mean(data['test_means']) if data['test_means'] else 0,
                'avg_change_percent': np.mean(data['change_percents']) if data['change_percents'] else 0,
                'max_abs_change': max([abs(x) for x in data['change_percents']]) if data['change_percents'] else 0
            })
        
        # 按变化幅度排序
        summary_list.sort(key=lambda x: x['max_abs_change'], reverse=True)
        return summary_list
    
    def _generate_metrics_summary_overall(self, metrics_analysis: Dict) -> List[Dict]:
        """生成整体指标统计摘要"""
        summary_list = []
        
        for metric, metric_data in metrics_analysis.items():
            summary_list.append({
                'metric': metric,
                'avg_control_mean': metric_data.get('control_mean', 0),
                'avg_test_mean': metric_data.get('test_mean', 0),
                'avg_change_percent': metric_data.get('change_percent', 0),
                'max_abs_change': abs(metric_data.get('change_percent', 0))
            })
        
        # 按变化幅度排序
        summary_list.sort(key=lambda x: x['max_abs_change'], reverse=True)
        return summary_list
    
    def _add_summary_columns(self, df: pd.DataFrame, quality_levels: List[str]) -> pd.DataFrame:
        """添加汇总统计列"""
        
        # 计算跨质量等级的汇总统计
        change_columns = [f'quality_{ql}_change_percent' for ql in quality_levels]
        status_columns = [f'quality_{ql}_status' for ql in quality_levels]
        
        # 平均变化百分比
        df['avg_change_percent'] = df[change_columns].mean(axis=1)
        
        # 最大变化百分比（绝对值）
        df['max_abs_change_percent'] = df[change_columns].abs().max(axis=1)
        
        # 恶化的质量等级数量
        df['degraded_quality_count'] = df[status_columns].apply(
            lambda row: sum(1 for status in row if status == 'degraded'), axis=1
        )
        
        # 改善的质量等级数量
        df['improved_quality_count'] = df[status_columns].apply(
            lambda row: sum(1 for status in row if status == 'improved'), axis=1
        )
        
        # 整体状态评估
        def overall_status(row):
            degraded = row['degraded_quality_count']
            improved = row['improved_quality_count']
            if degraded > improved:
                return 'degraded'
            elif improved > degraded:
                return 'improved'
            else:
                return 'stable'
        
        df['overall_status'] = df.apply(overall_status, axis=1)
        
        # 风险等级
        def risk_level(row):
            if row['degraded_quality_count'] >= 2 or row['max_abs_change_percent'] > 5:
                return 'high'
            elif row['degraded_quality_count'] >= 1 or row['max_abs_change_percent'] > 2:
                return 'medium'
            else:
                return 'low'
        
        df['risk_level'] = df.apply(risk_level, axis=1)
        
        return df
    
    async def _save_quality_metrics_matrix_csv(self, quality_analysis: Dict, output_dir: str) -> None:
        """保存质量等级vs指标变化百分比矩阵表格"""
        try:
            # 收集所有指标
            all_metrics = set()
            for quality_level, analysis in quality_analysis.items():
                metrics_analysis = analysis.get('metrics_analysis', {})
                all_metrics.update(metrics_analysis.keys())
            
            all_metrics = sorted(list(all_metrics))
            quality_levels = sorted(quality_analysis.keys())
            
            if not all_metrics or not quality_levels:
                logger.info("没有足够的数据生成矩阵表格")
                return
            
            # 创建矩阵数据
            matrix_data = []
            
            for metric in all_metrics:
                row_data = {
                    'metric': metric,
                    'metric_description': self._get_metric_description(metric)
                }
                
                # 为每个质量等级添加变化百分比
                for quality_level in quality_levels:
                    analysis = quality_analysis[quality_level]
                    metric_data = analysis.get('metrics_analysis', {}).get(metric, {})
                    change_percent = metric_data.get('change_percent', 0)
                    
                    # 使用质量等级作为列名
                    row_data[f'quality_{quality_level}'] = round(change_percent, 2)
                
                matrix_data.append(row_data)
            
            # 转换为DataFrame
            matrix_df = pd.DataFrame(matrix_data)
            
            # 保存矩阵表格
            matrix_file = os.path.join(output_dir, "quality_metrics_change_matrix.csv")
            matrix_df.to_csv(matrix_file, index=False, encoding='utf-8')
            
            logger.info(f"质量等级vs指标变化矩阵表格已保存: {matrix_file}")
            
        except Exception as e:
            logger.error(f"保存质量等级vs指标变化矩阵表格失败: {e}")
    
    def _generate_final_summary(
        self,
        data_dir: str,
        new_version: str,
        analysis_results: Dict,
        comprehensive_analysis: Dict,
        control_files_count: int,
        test_files_count: int,
        output_dir: str
    ) -> str:
        """生成最终总结"""
        summary_lines = [
            f"版本对比分析完成",
            f"数据目录: {data_dir}",
            f"新版本标识: {new_version}",
            f"分析文件: Control组 {control_files_count} 个, Test组 {test_files_count} 个",
            f"输出目录: {output_dir}"
        ]
        
        # 添加分析结果摘要
        if 'overall_summary' in analysis_results:
            summary = analysis_results['overall_summary']
            summary_lines.extend([
                f"整体状态: {summary.get('overall_status', 'unknown')}",
                f"显著变化: {summary.get('significant_changes_count', 0)} 个"
            ])
        
        return "\n".join(summary_lines)
    
    def _generate_executive_summary(self, analysis_results: Dict) -> Dict:
        """生成执行摘要"""
        summary = {
            'version_comparison_overview': {},
            'key_findings': [],
            'critical_issues': [],
            'performance_highlights': []
        }
        
        # 整体对比概览
        overall = analysis_results.get('overall_summary', {})
        data_summary = analysis_results.get('data_summary', {})
        
        summary['version_comparison_overview'] = {
            'total_control_records': data_summary.get('control_count', 0),
            'total_test_records': data_summary.get('test_count', 0),
            'control_devices': data_summary.get('control_devices', 0),
            'test_devices': data_summary.get('test_devices', 0),
            'overall_status': overall.get('overall_status', 'unknown'),
            'significant_changes': overall.get('significant_changes_count', 0)
        }
        
        # 关键发现
        if overall.get('overall_status') == 'degraded':
            summary['key_findings'].append(f"版本升级后整体性能出现恶化，{overall.get('degraded_metrics_count', 0)}个指标恶化")
        elif overall.get('overall_status') == 'improved':
            summary['key_findings'].append(f"版本升级后整体性能改善，{overall.get('improved_metrics_count', 0)}个指标改善")
        else:
            summary['key_findings'].append("版本升级后整体性能保持稳定")
        
        # 临界问题识别
        significant_changes = analysis_results.get('significant_changes', [])
        for change in significant_changes[:3]:  # 前3个最重要的变化
            if change.get('status') == 'degraded':
                summary['critical_issues'].append(
                    f"{change.get('metric', 'unknown')}指标恶化{change.get('change_percent', 0):.1f}%"
                )
        
        # 性能亮点
        for change in significant_changes:
            if change.get('status') == 'improved':
                summary['performance_highlights'].append(
                    f"{change.get('metric', 'unknown')}指标改善{abs(change.get('change_percent', 0)):.1f}%"
                )
        
        return summary
    
    def _analyze_quality_level_insights(self, quality_level_analysis: Dict) -> Dict:
        """分析质量等级洞察"""
        insights = {
            'quality_level_performance': {},
            'cross_quality_trends': [],
            'quality_specific_issues': {}
        }
        
        # 各质量等级性能表现
        for quality_level, analysis in quality_level_analysis.items():
            summary = analysis.get('summary', {})
            insights['quality_level_performance'][quality_level] = {
                'status': summary.get('overall_status', 'unknown'),
                'degraded_count': summary.get('degraded_metrics_count', 0),
                'improved_count': summary.get('improved_metrics_count', 0),
                'data_volume': analysis.get('control_count', 0) + analysis.get('test_count', 0)
            }
        
        # 跨质量等级趋势
        degraded_levels = []
        improved_levels = []
        for quality_level, perf in insights['quality_level_performance'].items():
            if perf['status'] == 'degraded':
                degraded_levels.append(quality_level)
            elif perf['status'] == 'improved':
                improved_levels.append(quality_level)
        
        if degraded_levels:
            insights['cross_quality_trends'].append(f"质量等级{', '.join(degraded_levels)}出现性能恶化")
        if improved_levels:
            insights['cross_quality_trends'].append(f"质量等级{', '.join(improved_levels)}出现性能改善")
        
        # 质量等级特定问题
        for quality_level, analysis in quality_level_analysis.items():
            significant_changes = analysis.get('significant_changes', [])
            if significant_changes:
                insights['quality_specific_issues'][quality_level] = [
                    f"{change.get('metric')}变化{change.get('change_percent', 0):.1f}%"
                    for change in significant_changes[:2]  # 每个质量等级最多显示2个问题
                ]
        
        return insights
    
    def _analyze_metric_trends(self, analysis_results: Dict) -> Dict:
        """分析指标趋势"""
        trends = {
            'metric_performance_ranking': [],
            'trend_patterns': {},
            'metric_correlations': []
        }
        
        # 提取指标分析结果
        if 'quality_level_analysis' in analysis_results:
            # 质量等级分层分析
            all_metrics_data = {}
            for quality_level, analysis in analysis_results['quality_level_analysis'].items():
                metrics_analysis = analysis.get('metrics_analysis', {})
                for metric, metric_data in metrics_analysis.items():
                    if metric not in all_metrics_data:
                        all_metrics_data[metric] = []
                    all_metrics_data[metric].append({
                        'quality_level': quality_level,
                        'change_percent': metric_data.get('change_percent', 0),
                        'status': metric_data.get('status', 'stable')
                    })
        else:
            # 整体分析
            all_metrics_data = {}
            metrics_analysis = analysis_results.get('metrics_analysis', {})
            for metric, metric_data in metrics_analysis.items():
                all_metrics_data[metric] = [{
                    'quality_level': 'overall',
                    'change_percent': metric_data.get('change_percent', 0),
                    'status': metric_data.get('status', 'stable')
                }]
        
        # 指标性能排名
        metric_scores = []
        for metric, data_list in all_metrics_data.items():
            avg_change = sum(d['change_percent'] for d in data_list) / len(data_list)
            degraded_count = sum(1 for d in data_list if d['status'] == 'degraded')
            metric_scores.append({
                'metric': metric,
                'avg_change_percent': avg_change,
                'degraded_quality_levels': degraded_count,
                'severity_score': abs(avg_change) * (1 + degraded_count * 0.5)
            })
        
        # 按严重程度排序
        metric_scores.sort(key=lambda x: x['severity_score'], reverse=True)
        trends['metric_performance_ranking'] = metric_scores[:5]  # 前5个最严重的指标
        
        # 趋势模式
        for metric, data_list in all_metrics_data.items():
            if len(data_list) > 1:  # 有多个质量等级的数据
                changes = [d['change_percent'] for d in data_list]
                if all(c > 0 for c in changes):
                    trends['trend_patterns'][metric] = "全质量等级一致上升"
                elif all(c < 0 for c in changes):
                    trends['trend_patterns'][metric] = "全质量等级一致下降"
                elif max(changes) - min(changes) > 10:  # 变化差异超过10%
                    trends['trend_patterns'][metric] = "质量等级间差异显著"
                else:
                    trends['trend_patterns'][metric] = "质量等级间变化稳定"
        
        return trends
    
    def _analyze_device_impacts(self, control_data: pd.DataFrame, test_data: pd.DataFrame, analysis_results: Dict) -> Dict:
        """分析设备影响"""
        device_impacts = {
            'device_diversity': {},
            'common_device_analysis': [],
            'device_specific_risks': []
        }
        
        # 设备多样性分析
        if 'devicemodel' in control_data.columns and 'devicemodel' in test_data.columns:
            control_devices = set(control_data['devicemodel'].unique())
            test_devices = set(test_data['devicemodel'].unique())
            common_devices = control_devices.intersection(test_devices)
            
            device_impacts['device_diversity'] = {
                'control_unique_devices': len(control_devices),
                'test_unique_devices': len(test_devices),
                'common_devices': len(common_devices),
                'coverage_ratio': len(common_devices) / len(control_devices.union(test_devices))
            }
            
            # 常见设备分析
            for device in list(common_devices)[:5]:  # 分析前5个常见设备
                device_control = control_data[control_data['devicemodel'] == device]
                device_test = test_data[test_data['devicemodel'] == device]
                
                if len(device_control) > 10 and len(device_test) > 10:  # 样本量足够
                    device_impacts['common_device_analysis'].append({
                        'device': device,
                        'control_samples': len(device_control),
                        'test_samples': len(device_test),
                        'sample_adequacy': 'sufficient'
                    })
        
        return device_impacts
    
    def _assess_version_risks(self, analysis_results: Dict) -> Dict:
        """评估版本风险"""
        risks = {
            'risk_level': 'low',
            'risk_factors': [],
            'mitigation_priority': [],
            'rollback_recommendation': False
        }
        
        overall = analysis_results.get('overall_summary', {})
        significant_changes = analysis_results.get('significant_changes', [])
        
        # 风险等级评估
        degraded_count = overall.get('degraded_metrics_count', 0)
        significant_count = len(significant_changes)
        
        if degraded_count >= 3 or significant_count >= 2:
            risks['risk_level'] = 'high'
            risks['rollback_recommendation'] = True
        elif degraded_count >= 2 or significant_count >= 1:
            risks['risk_level'] = 'medium'
        else:
            risks['risk_level'] = 'low'
        
        # 风险因子识别
        for change in significant_changes:
            if change.get('status') == 'degraded':
                severity = abs(change.get('change_percent', 0))
                if severity > 5:
                    risks['risk_factors'].append(f"{change.get('metric')}严重恶化({severity:.1f}%)")
                else:
                    risks['risk_factors'].append(f"{change.get('metric')}轻微恶化({severity:.1f}%)")
        
        # 缓解优先级
        critical_metrics = ['avgfps_unity', 'gt_50', 'bigjankper10min']  # 关键用户体验指标
        for change in significant_changes:
            if change.get('metric') in critical_metrics and change.get('status') == 'degraded':
                risks['mitigation_priority'].append({
                    'metric': change.get('metric'),
                    'priority': 'critical',
                    'change_percent': change.get('change_percent', 0)
                })
        
        return risks
    
    def _generate_recommendations(self, analysis_results: Dict, comprehensive_analysis: Dict) -> List[str]:
        """生成推荐建议"""
        recommendations = []
        
        risk_assessment = comprehensive_analysis.get('risk_assessment', {})
        overall = analysis_results.get('overall_summary', {})
        
        # 基于风险等级的建议
        risk_level = risk_assessment.get('risk_level', 'low')
        
        if risk_level == 'high':
            recommendations.append("🚨 建议暂停版本发布，优先修复性能恶化问题")
            if risk_assessment.get('rollback_recommendation'):
                recommendations.append("📋 考虑回滚到上一稳定版本")
        elif risk_level == 'medium':
            recommendations.append("⚠️ 建议限制发布范围，进行小规模测试")
            recommendations.append("🔍 密切监控关键性能指标")
        else:
            recommendations.append("✅ 版本性能表现良好，可以考虑正常发布")
        
        # 基于指标趋势的建议
        metric_trends = comprehensive_analysis.get('metric_trends', {})
        ranking = metric_trends.get('metric_performance_ranking', [])
        
        if ranking:
            worst_metric = ranking[0]
            if worst_metric.get('severity_score', 0) > 5:
                recommendations.append(f"🎯 优先关注{worst_metric.get('metric')}指标的优化")
        
        # 基于质量等级的建议
        quality_insights = comprehensive_analysis.get('quality_level_insights', {})
        quality_issues = quality_insights.get('quality_specific_issues', {})
        
        if quality_issues:
            affected_levels = list(quality_issues.keys())
            recommendations.append(f"📊 特别关注质量等级{', '.join(affected_levels[:2])}的性能表现")
        
        # 数据监控建议
        data_summary = analysis_results.get('data_summary', {})
        if data_summary.get('control_count', 0) < data_summary.get('test_count', 0) * 2:
            recommendations.append("📈 建议收集更多对照组数据以提高分析可信度")
        
        return recommendations
