import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
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
        "unityversion",  # 客户端unity主版本
        "unityversion_inner", # 客户端unity内部版本，更详细的版本号
        "devicetotalmemory", # 设备总内存
        "cpumodel", # cpu型号，是一个string
        "refreshrate",
        "ischarged", # 1 或 0 代表是否充电
        "systemmemorysize",
        "logymd",
        "zoneid",
        # 新增预处理特征
        "time_flag",  # 时间标识 (基于logymd >= 2025-07-15)
        "memory_category",  # 内存分级 (0G-2G, 2G-6G, 6G)
        "unity_digital",  # Unity版本位数 (32bit, 64bit)
        "zone_group",  # 区域分组 (zoneid % 2)
    ]

@dataclass
class AnalysisConfig:
    """关键指标分析配置参数类

    使用示例:
        # 创建自定义配置
        custom_config = AnalysisConfig(
            degradation_threshold=2.0,  # 提高恶化阈值到2%
            numeric_significance_threshold=3.0,  # 降低数值显著性阈值到3%
            max_ranking_devices=20,  # 显示更多设备排名
            confidence_max_score=5.0  # 降低最大置信度评分
        )

        # 使用配置
        analyzer = KeyMetricAnalysisTool(config=custom_config)
        results = await analyzer.execute(csv_file_path, output_dir)
    """

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

    # === 设备质量等级映射参数 ===
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

    def get_quality_level_conditions(self):
        """获取质量等级映射条件"""
        thresholds = self.quality_level_thresholds or [5.7, 7.4, 9.4]
        return [
            (0, thresholds[0]),      # 0 < score < 5.7 -> level 1
            (thresholds[0], thresholds[1]),  # 5.7 <= score < 7.4 -> level 2
            (thresholds[1], thresholds[2]),  # 7.4 <= score < 9.4 -> level 3
            (thresholds[2], float('inf'))  # score >= 9.4 -> level 4
        ]

class KeyMetricAnalysisTool(BaseTool):

    name: str = "key_metric_analysis_tool"
    description: str = (
        "关键指标深度分析工具。按照设备质量等级(realpicturequality)和设备型号(devicemodel)进行层级分析，"
        "对5个关键性能指标在不同公共维度下的control vs test组变化进行深入挖掘，"
        "包括充电状态、CPU型号、刷新率等维度对性能的影响分析。"
    )

    parameters: dict = {
        "type": "object",
        "properties": {
            "csv_file_path": {
                "type": "string",
                "description": "包含关键指标数据的CSV文件路径",
                "default": os.path.join(config.workspace_root, "key_metric.csv")
            },
            "group_column": {
                "type": "string",
                "description": "标识control/test组的列名",
                "default": "param_value"
            },
            "control_value": {
                "type": "string",
                "description": "control组的标识值",
                "default": "control"
            },
            "test_value": {
                "type": "string",
                "description": "test组的标识值",
                "default": "test"
            },
            "degradation_threshold": {
                "type": "number",
                "description": "性能恶化判定阈值百分比",
                "default": 1.0
            },
            "confidence_alpha": {
                "type": "number",
                "description": "置信度计算参数",
                "default": 1.0
            },
            "config": {
                "type": "object",
                "description": "分析配置参数对象，可选，用于覆盖默认配置"
            }
        },
        "required": ["csv_file_path"]
    }

    def __init__(self, config: Optional[AnalysisConfig] = None, **kwargs):
        super().__init__(**kwargs)
        # 使用 object.__setattr__ 绕过 Pydantic 的字段检查
        object.__setattr__(self, '_config', config or AnalysisConfig())

    @property
    def config(self) -> AnalysisConfig:
        """获取配置对象"""
        return getattr(self, '_config', AnalysisConfig())

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

    def _get_available_key_metrics(self):
        """获取可用的关键指标"""
        return getattr(self, 'available_key_metrics', KEY_PERFORMANCE_METRICS)

    def _get_available_common_columns(self):
        """获取可用的公共列"""
        return getattr(self, 'available_common_columns', COMMON_COLUMNS)

    async def execute(
        self,
        csv_file_path: str,
        group_column: str = "param_value",
        control_value: str = "control",
        test_value: str = "test",
        degradation_threshold: float = 1.0,
        confidence_alpha: float = 1.0,
        output_dir = None,
        config: Optional[Dict] = None
    ) -> ToolResult:
        """
        执行关键指标层级分析流程

        Args:
            csv_file_path: CSV数据文件路径
            group_column: 组别标识列名
            control_value: 对照组值
            test_value: 测试组值
            degradation_threshold: 恶化阈值
            confidence_alpha: 置信度参数
            output_dir: 输出目录
            config: 配置参数字典，用于覆盖默认配置

        Returns:
            ToolResult: 分析结果
        """
        try:
            # 更新配置参数
            if config:
                self.update_config(config)

            # 使用配置参数覆盖传入的参数
            degradation_threshold = self.config.degradation_threshold
            confidence_alpha = self.config.confidence_alpha

            # 1. 数据加载和验证
            df, available_common_columns, available_key_metrics = await self._load_and_validate_data(csv_file_path, group_column)

            # 动态设置实例属性用于其他方法访问
            object.__setattr__(self, 'available_common_columns', available_common_columns)
            object.__setattr__(self, 'available_key_metrics', available_key_metrics)
            object.__setattr__(self, 'group_column', group_column)

            # 2. 设置输出目录
            if output_dir is None:
                csv_dir = os.path.dirname(csv_file_path)
                output_dir = os.path.join(csv_dir, "key_metric_analysis")
            os.makedirs(output_dir, exist_ok=True)

            # 3. 按realpicturequality拆分数据
            quality_splits = await self._split_by_quality(df, group_column, control_value, test_value)

            # 4. 对每个质量等级进行设备级分析
            all_results = {}

            for quality_level, quality_data in quality_splits.items():
                logger.info(f"开始分析质量等级: {quality_level}")

                # 按devicemodel进一步拆分
                device_splits = await self._split_by_device_model(
                    quality_data, quality_level, group_column, control_value, test_value
                )

                # 分析每个设备型号
                quality_results = {}
                for device_model, device_data in device_splits.items():
                    logger.info(f"分析设备: {device_model} (质量等级: {quality_level})")

                    device_results = await self._analyze_device_metrics(
                        device_data, device_model, quality_level,
                        degradation_threshold, confidence_alpha
                    )

                    quality_results[device_model] = device_results

                all_results[quality_level] = quality_results

            # 5. 生成综合分析报告
            comprehensive_analysis = await self._generate_comprehensive_analysis(all_results)

            # 6. 创建可视化图表
            await self._create_comprehensive_visualizations(all_results, output_dir)

            # 7. 保存分析数据
            await self._save_analysis_results(all_results, comprehensive_analysis, output_dir)

            # 8. 按Unity版本拆分数据
            # split_path_dir = os.path.join(os.path.dirname(output_dir), "unity_version_splits")
            await self._split_data_by_unity_version(all_results, output_dir)

            # 9. 生成设备性能报告
            device_performance_report = self._generate_device_performance_report(all_results, output_dir)

            # 10. 生成总结报告
            summary = self._generate_final_summary(
                csv_file_path, all_results, comprehensive_analysis,
                quality_splits, output_dir, degradation_threshold, confidence_alpha
            )

            # 将设备性能报告添加到摘要中
            summary += "\n\n" + "="*60 + "\n"
            summary += "📊 详细设备性能报告已生成: device_performance_report.md\n"
            summary += "="*60 + "\n"

            return ToolResult(output=summary)

        except ToolError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            error_msg = f"关键指标分析过程中发生错误: {str(e)}"
            logger.error(error_msg)
            return ToolResult(error=error_msg)

    async def _load_and_validate_data(self, csv_file_path: str, group_column: str) -> tuple[pd.DataFrame, list, list]:
        """加载和验证数据，返回数据框和可用的列信息"""

        if not os.path.exists(csv_file_path):
            raise ToolError(f"CSV文件不存在: {csv_file_path}")

        try:
            df = pd.read_csv(csv_file_path)
            logger.info(f"成功加载CSV文件，数据形状: {df.shape}")
        except Exception as e:
            raise ToolError(f"读取CSV文件失败: {str(e)}")

        # 应用SQL逻辑进行数据预处理
        df = await self._apply_sql_preprocessing(df)

        # 验证基础必需列
        base_required_columns = [group_column, "devicemodel", "realpicturequality"]
        missing_base_columns = [col for col in base_required_columns if col not in df.columns]

        if missing_base_columns:
            raise ToolError(f"缺少基础必需列: {missing_base_columns}")

        # 检查关键指标列的可用性
        available_key_metrics = [col for col in KEY_PERFORMANCE_METRICS if col in df.columns]
        missing_key_metrics = [col for col in KEY_PERFORMANCE_METRICS if col not in df.columns]

        if missing_key_metrics:
            logger.warning(f"缺少部分关键指标列: {missing_key_metrics}")

        if not available_key_metrics:
            raise ToolError(f"所有关键指标列都缺失，无法进行分析: {KEY_PERFORMANCE_METRICS}")

        logger.info(f"可用关键指标: {available_key_metrics} ({len(available_key_metrics)}/{len(KEY_PERFORMANCE_METRICS)})")

        # 验证公共列
        available_common_columns = [col for col in COMMON_COLUMNS if col in df.columns]
        missing_common_columns = [col for col in COMMON_COLUMNS if col not in df.columns]

        if missing_common_columns:
            logger.warning(f"缺少部分公共列: {missing_common_columns}")

        logger.info(f"可用公共分析维度: {available_common_columns}")

        # 更新可用的列，并确保包含group_column用于区分control/test
        available_common_columns = available_common_columns
        available_key_metrics = available_key_metrics  # 保存可用的关键指标
        group_column = group_column  # 保存group_column以便后续使用

        return df, available_common_columns, available_key_metrics

    async def _apply_sql_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用SQL逻辑进行数据预处理和特征工程"""

        logger.info("开始应用SQL预处理逻辑...")

        # 1. 时间维度特征：基于logymd创建时间标识
        if 'logymd' in df.columns:
            df['logymd'] = pd.to_datetime(df['logymd'], format='%Y-%m-%d', errors='coerce')
            # 创建时间标识：使用配置的时间分割日期
            split_date = pd.to_datetime(self.config.time_split_date)
            df['time_flag'] = (df['logymd'] >= split_date).astype(int)
            logger.info(f"创建时间标识特征 time_flag (分割日期: {self.config.time_split_date})")

        # 2. 内存分级特征：基于systemmemorysize
        if 'systemmemorysize' in df.columns:
            def categorize_memory(memory_size):
                if pd.isna(memory_size):
                    return 'unknown'
                memory_gb = memory_size / 1024.0
                if memory_gb <= 2:
                    return '0G-2G'
                elif memory_gb < 6:
                    return '2G-6G'
                elif memory_gb >= 6:
                    return '6G'
                else:
                    return 'else'

            df['memory_category'] = df['systemmemorysize'].apply(categorize_memory)
            logger.info("创建内存分级特征 memory_category")

        # 3. Unity版本位数特征：基于unityversion
        if 'unityversion' in df.columns:
            def extract_digital_version(unity_version):
                if pd.isna(unity_version):
                    return 'unknown'
                if '64' in str(unity_version):
                    return '64bit'
                else:
                    return '32bit'

            df['unity_digital'] = df['unityversion'].apply(extract_digital_version)
            logger.info("创建Unity版本位数特征 unity_digital")

        # 4. 区域特征：基于zoneid
        if 'zoneid' in df.columns:
            df['zone_group'] = df['zoneid'] % 2
            logger.info("创建区域分组特征 zone_group")

        # 5. 数据质量过滤（基于SQL的WHERE条件）
        initial_count = len(df)

        # 过滤异常FPS数据
        if 'avgfps_unity' in df.columns:
            df = df[(df['avgfps_unity'] >= self.config.fps_min_threshold) &
                   (df['avgfps_unity'] <= self.config.fps_max_threshold)]

        # 过滤异常的性能指标
        performance_filters = {
            'avgfps': lambda x: x > 0,
            'jankper10min': lambda x: x >= 0,
            'bigjankper10min': lambda x: x >= 0,
            'current_avg': lambda x: (x >= self.config.current_avg_min) & (x <= self.config.current_avg_max) if 'current_avg' in df.columns else True,
        }

        for column, filter_func in performance_filters.items():
            if column in df.columns:
                before_count = len(df)
                df = df[filter_func(df[column])]
                after_count = len(df)
                if before_count != after_count:
                    logger.info(f"过滤 {column} 异常值：{before_count} -> {after_count}")

        # 6. 设备质量等级映射（基于fixed_score）
        if 'fixed_score' in df.columns and 'nowpicturequality' in df.columns:
            # 转换为数值类型
            df['fixed_score'] = pd.to_numeric(df['fixed_score'], errors='coerce')
            df['nowpicturequality'] = pd.to_numeric(df['nowpicturequality'], errors='coerce')

            # 基于配置的阈值确定期望的画质等级
            thresholds = self.config.quality_level_thresholds or [5.7, 7.4, 9.4]
            conditions = [
                (df['fixed_score'] > 0) & (df['fixed_score'] < thresholds[0]),
                (df['fixed_score'] >= thresholds[0]) & (df['fixed_score'] < thresholds[1]),
                (df['fixed_score'] >= thresholds[1]) & (df['fixed_score'] < thresholds[2]),
                (df['fixed_score'] >= thresholds[2])
            ]
            choices = [1, 2, 3, 4]

            df['expected_quality'] = np.select(conditions, choices, default=0)

            # 只保留画质等级与设备能力匹配的数据
            df = df[df['nowpicturequality'] == df['expected_quality']]

            # 使用期望的质量等级作为realpicturequality
            df['realpicturequality'] = df['expected_quality']
            logger.info(f"应用设备质量等级映射，保留 {len(df)} 条匹配记录")

        # 7. 创建设备模型标识（如果不存在）
        if 'devicemodel' not in df.columns:
            # 基于内存和Unity版本创建复合设备模型
            device_features = []
            if 'memory_category' in df.columns:
                device_features.append('memory_category')
            if 'unity_digital' in df.columns:
                device_features.append('unity_digital')
            if 'cpumodel' in df.columns:
                device_features.append('cpumodel')

            if device_features:
                df['devicemodel'] = df[device_features].apply(
                    lambda x: '_'.join([str(v) for v in x if pd.notna(v)]), axis=1
                )
                logger.info(f"基于 {device_features} 创建复合设备模型标识")
            else:
                df['devicemodel'] = 'unknown_device'
                logger.warning("无法创建设备模型标识，使用默认值")

        # 8. 规范化指标名称（确保与KEY_PERFORMANCE_METRICS一致）
        metric_mapping = {
            'avgfps_unity': 'avgfps_unity',
            'battleendtotalpss': 'battleendtotalpss',
            'current_avg': 'current_avg',
            'gt_50': 'gt_50',  # GC次数
            'totalgcallocsize': 'totalgcallocsize',  # GC分配
            'bigjankper10min': 'bigjankper10min'
        }

        # 重命名列以匹配预期的指标名称
        for old_name, new_name in metric_mapping.items():
            if old_name in df.columns and old_name != new_name:
                df = df.rename(columns={old_name: new_name})
                logger.info(f"重命名指标列：{old_name} -> {new_name}")

        final_count = len(df)
        logger.info(f"SQL预处理完成：{initial_count} -> {final_count} 条记录")

        # 9. 添加预处理后的统计信息
        if final_count > 0:
            logger.info("预处理后的数据统计：")
            if 'memory_category' in df.columns:
                logger.info(f"内存分布：\n{df['memory_category'].value_counts()}")
            if 'unity_digital' in df.columns:
                logger.info(f"Unity版本分布：\n{df['unity_digital'].value_counts()}")
            if 'realpicturequality' in df.columns:
                logger.info(f"质量等级分布：\n{df['realpicturequality'].value_counts()}")

        return df

    async def _split_by_quality(
        self,
        df: pd.DataFrame,
        group_column: str,
        control_value: str,
        test_value: str
    ) -> Dict[str, Dict]:
        """按realpicturequality拆分数据"""

        quality_levels = sorted(df['realpicturequality'].unique())
        logger.info(f"发现 {len(quality_levels)} 个质量等级: {quality_levels}")

        quality_splits = {}

        for quality_level in quality_levels:
            quality_df = df[df['realpicturequality'] == quality_level].copy()

            control_data = quality_df[quality_df[group_column] == control_value].copy()
            test_data = quality_df[quality_df[group_column] == test_value].copy()

            if control_data.empty or test_data.empty:
                logger.warning(f"质量等级 {quality_level} 缺少control或test数据，跳过")
                continue

            quality_splits[quality_level] = {
                'control': control_data,
                'test': test_data,
                'total_records': len(quality_df),
                'control_records': len(control_data),
                'test_records': len(test_data)
            }

            logger.info(f"质量等级 {quality_level}: Control={len(control_data)}, Test={len(test_data)}")

        return quality_splits

    async def _split_by_device_model(
        self,
        quality_data: Dict,
        quality_level: str,
        group_column: str,
        control_value: str,
        test_value: str
    ) -> Dict[str, Dict]:
        """在质量等级内按devicemodel拆分"""

        control_data = quality_data['control']
        test_data = quality_data['test']

        # 获取所有设备型号
        control_devices = set(control_data['devicemodel'].unique())
        test_devices = set(test_data['devicemodel'].unique())
        all_devices = control_devices.union(test_devices)

        logger.info(f"质量等级 {quality_level} 包含 {len(all_devices)} 个设备型号")

        device_splits = {}

        for device_model in all_devices:
            device_control = control_data[control_data['devicemodel'] == device_model].copy()
            device_test = test_data[test_data['devicemodel'] == device_model].copy()

            if device_control.empty or device_test.empty:
                logger.warning(f"设备 {device_model} (质量{quality_level}) 缺少control或test数据，跳过")
                continue

            # 保留可用的关键指标、公共列和group_column
            available_key_metrics = getattr(self, 'available_key_metrics', KEY_PERFORMANCE_METRICS)
            available_common_columns = getattr(self, 'available_common_columns', COMMON_COLUMNS)
            group_column = getattr(self, 'group_column', group_column)

            final_columns = available_key_metrics + available_common_columns + [group_column]
            # 去重并确保列存在
            final_columns = list(set(final_columns))
            final_columns = [col for col in final_columns if col in device_control.columns]

            device_splits[device_model] = {
                'control': device_control[final_columns].copy(),
                'test': device_test[final_columns].copy(),
                'control_count': len(device_control),
                'test_count': len(device_test),
                'quality_level': quality_level
            }

            logger.info(f"设备 {device_model} (质量{quality_level}): Control={len(device_control)}, Test={len(device_test)}")

        return device_splits

    async def _analyze_device_metrics(
        self,
        device_data: Dict,
        device_model: str,
        quality_level: str,
        degradation_threshold: float,
        confidence_alpha: float
    ) -> Dict:
        """分析单个设备的关键指标"""

        control_data = device_data['control']
        test_data = device_data['test']

        analysis_result = {
            'device_model': device_model,
            'quality_level': quality_level,
            'control_count': device_data['control_count'],
            'test_count': device_data['test_count'],
            'confidence_score': 0.0,
            'key_metrics_analysis': {},
            'overall_status': 'stable',
            'degradation_score': 0.0,
            'performance_change_explanations': {},  # 改为性能变化解释
            'insights': [],
            # 保存原始数据供后续图表分析使用
            'control_data': control_data.copy(),
            'test_data': test_data.copy()
        }

        # 计算置信度分数
        nc, nt = device_data['control_count'], device_data['test_count']
        if nc > 0 and nt > 0:
            balance_ratio = min(nc, nt) / max(nc, nt)
            sample_size_factor = min(1.0, np.log10((nc + nt) * 10) / 2.0)
            analysis_result['confidence_score'] = self.config.confidence_max_score * (balance_ratio ** confidence_alpha) * sample_size_factor

        # 分析可用的关键指标
        degradation_count = 0
        improvement_count = 0
        total_degradation = 0.0
        significant_changed_metrics = []  # 记录有显著变化的指标

        available_key_metrics = getattr(self, 'available_key_metrics', KEY_PERFORMANCE_METRICS)

        for metric in available_key_metrics:
            if metric in control_data.columns and metric in test_data.columns:
                metric_analysis = await self._analyze_single_metric(
                    control_data, test_data, metric, degradation_threshold
                )
                analysis_result['key_metrics_analysis'][metric] = metric_analysis

                if metric_analysis['status'] == 'degraded':
                    degradation_count += 1
                    total_degradation += abs(metric_analysis['change_percent'])
                    significant_changed_metrics.append(metric)
                elif metric_analysis['status'] == 'improved':
                    improvement_count += 1
                    significant_changed_metrics.append(metric)

        # 计算整体状态
        if degradation_count > improvement_count:
            analysis_result['overall_status'] = 'degraded'
        elif improvement_count > degradation_count:
            analysis_result['overall_status'] = 'improved'

        analysis_result['degradation_score'] = total_degradation / max(len(available_key_metrics), 1)

        # 注释掉原因分析 - 只有当关键指标有显著变化时，才分析common_columns来寻找原因
        # if significant_changed_metrics:
        #     logger.info(f"设备 {device_model} 有显著性能变化，分析可能原因...")
        #     analysis_result['performance_change_explanations'] = await self._explain_performance_changes(
        #         control_data, test_data, significant_changed_metrics
        #     )

        # 生成洞察
        analysis_result['insights'] = await self._generate_device_insights(analysis_result)

        return analysis_result

    async def _analyze_single_metric(
        self,
        control_data: pd.DataFrame,
        test_data: pd.DataFrame,
        metric: str,
        threshold: float
    ) -> Dict:
        """分析单个关键指标"""

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
            'effect_size': 0.0
        }

        if len(control_values) > 0:
            analysis['control_mean'] = control_values.mean()
            analysis['control_std'] = control_values.std()
            analysis['control_median'] = control_values.median()

        if len(test_values) > 0:
            analysis['test_mean'] = test_values.mean()
            analysis['test_std'] = test_values.std()
            analysis['test_median'] = test_values.median()

        # 计算变化
        if not np.isnan(analysis['control_mean']) and not np.isnan(analysis['test_mean']):
            if analysis['control_mean'] != 0:
                change_percent = ((analysis['test_mean'] - analysis['control_mean']) /
                                analysis['control_mean']) * 100
                analysis['change_percent'] = change_percent
                analysis['absolute_change'] = analysis['test_mean'] - analysis['control_mean']

                # 判断是否为改善 (针对不同指标)
                is_improvement = self._is_metric_improvement(metric, change_percent)

                if abs(change_percent) >= threshold:
                    analysis['status'] = 'improved' if is_improvement else 'degraded'
                else:
                    analysis['status'] = 'stable'

                # 统计显著性检验 (暂时简化)
                analysis['statistical_significance'] = False
                analysis['effect_size'] = 0.0
            else:
                analysis['status'] = 'stable'

        return analysis

    def _is_metric_improvement(self, metric: str, change_percent: float) -> bool:
        """判断指标变化是否为改善"""

        # 对于不同指标定义改善方向
        higher_better_metrics = ["avgfps_unity", "current_avg"]  # 越高越好
        lower_better_metrics = ["totalgcallocsize", "gt_50", "battleendtotalpss", "bigjankper10min"]  # 越低越好

        if metric in higher_better_metrics:
            return change_percent > 0  # 增加是改善
        elif metric in lower_better_metrics:
            return change_percent < 0  # 减少是改善
        else:
            return False  # 未知指标，保守处理

    async def _explain_performance_changes(
        self,
        control_data: pd.DataFrame,
        test_data: pd.DataFrame,
        changed_metrics: List[str]
    ) -> Dict:
        """分析性能变化的可能原因"""

        explanations = {
            'potential_causes': [],
            'dimension_differences': {},
            'key_findings': []
        }

        logger.info(f"分析维度差异来解释指标变化: {changed_metrics}")

        available_common_columns = getattr(self, 'available_common_columns', COMMON_COLUMNS)

        # 分析每个common_column，寻找可能导致性能变化的因素
        for column in available_common_columns:
            if column in control_data.columns and column in test_data.columns:
                dimension_analysis = await self._analyze_dimension_for_explanation(
                    control_data, test_data, column
                )

                if dimension_analysis['has_significant_difference']:
                    explanations['dimension_differences'][column] = dimension_analysis
                    explanations['potential_causes'].extend(dimension_analysis['explanations'])

        # 生成关键发现
        explanations['key_findings'] = await self._generate_causal_insights(
            explanations['dimension_differences'], changed_metrics
        )

        return explanations

    async def _analyze_dimension_for_explanation(
        self,
        control_data: pd.DataFrame,
        test_data: pd.DataFrame,
        column: str
    ) -> Dict:
        """分析单个维度是否能解释性能变化"""

        analysis = {
            'dimension': column,
            'has_significant_difference': False,
            'explanations': [],
            'difference_details': {}
        }

        # 获取非空值
        control_values = control_data[column].dropna()
        test_values = test_data[column].dropna()

        if len(control_values) == 0 or len(test_values) == 0:
            return analysis

        # 明确定义哪些列应该按类别型处理
        categorical_columns = [
            'unityversion_inner',  # 版本号字符串 (如: 2019-2.01.0050.10-6dca93b4...)
            'unityversion',        # 版本名称 (如: UI20_64_2019)
            'cpumodel',           # CPU型号
            'ischarged',          # 充电状态（0/1）
            # 新增预处理特征
            'time_flag',          # 时间标识 (0/1)
            'memory_category',    # 内存分级 (0G-2G, 2G-6G, 6G)
            'unity_digital',      # Unity版本位数 (32bit, 64bit)
            'zone_group',         # 区域分组 (0/1)
            'logymd',            # 日期
            'zoneid'             # 区域ID
        ]

        # 明确定义哪些列应该按数值型处理
        numeric_columns = [
            'devicetotalmemory',  # 设备内存
            'refreshrate',        # 刷新率
            'systemmemorysize'    # 系统内存大小
        ]

        if column in categorical_columns:
            # 类别型数据处理
            logger.debug(f"将 {column} 按类别型数据处理")
            analysis = await self._analyze_categorical_explanation(
                control_values, test_values, analysis
            )
        elif column in numeric_columns:
            # 数值型数据处理
            logger.debug(f"将 {column} 按数值型数据处理")
            analysis = await self._analyze_numeric_explanation(
                control_values, test_values, analysis
            )
        else:
            # 自动判断：尝试转换为数值，如果失败则按类别处理
            try:
                pd.to_numeric(control_values.iloc[:5])  # 测试前5个值
                logger.debug(f"自动检测 {column} 为数值型数据")
                analysis = await self._analyze_numeric_explanation(
                    control_values, test_values, analysis
                )
            except (ValueError, TypeError):
                logger.debug(f"自动检测 {column} 为类别型数据")
                analysis = await self._analyze_categorical_explanation(
                    control_values, test_values, analysis
                )

        return analysis

    async def _analyze_numeric_explanation(
        self,
        control_values: pd.Series,
        test_values: pd.Series,
        analysis: Dict
    ) -> Dict:
        """分析数值型维度的解释性差异"""

        control_mean = control_values.mean()
        test_mean = test_values.mean()

        if control_mean != 0:
            diff_percent = ((test_mean - control_mean) / control_mean) * 100
        else:
            diff_percent = 0

        analysis['difference_details'] = {
            'control_mean': control_mean,
            'test_mean': test_mean,
            'difference_percent': diff_percent,
            'absolute_difference': test_mean - control_mean
        }

        # 设定显著差异阈值：数值差异>5%（降低阈值以检测更多差异）
        if abs(diff_percent) > 5:
            analysis['has_significant_difference'] = True

            direction = "更高" if diff_percent > 0 else "更低"
            explanation = f"Test组在{analysis['dimension']}维度的平均值{direction} {abs(diff_percent):.1f}%"

            # 特定维度的性能影响分析
            if analysis['dimension'] == 'refreshrate':
                if diff_percent > 0:
                    explanation += "，更高的刷新率可能导致更大的性能负载"
                else:
                    explanation += "，更低的刷新率可能减少了性能压力"
            elif analysis['dimension'] == 'devicetotalmemory':
                if diff_percent > 0:
                    explanation += "，更大的内存可能影响内存管理策略"
                else:
                    explanation += "，更小的内存可能造成内存压力"

            analysis['explanations'].append(explanation)

        return analysis

    async def _analyze_categorical_explanation(
        self,
        control_values: pd.Series,
        test_values: pd.Series,
        analysis: Dict
    ) -> Dict:
        """分析类别型维度的解释性差异"""

        # 计算分布
        control_dist = control_values.value_counts(normalize=True).to_dict()
        test_dist = test_values.value_counts(normalize=True).to_dict()

        analysis['difference_details'] = {
            'control_distribution': control_dist,
            'test_distribution': test_dist
        }

        # 找出显著差异的类别
        all_categories = set(control_dist.keys()).union(set(test_dist.keys()))
        significant_differences = []

        for category in all_categories:
            control_pct = control_dist.get(category, 0) * 100
            test_pct = test_dist.get(category, 0) * 100
            diff = test_pct - control_pct

            # 设定显著差异阈值：分布差异>10个百分点（降低阈值以检测更多差异）
            if abs(diff) > 10:
                significant_differences.append({
                    'category': category,
                    'control_percent': control_pct,
                    'test_percent': test_pct,
                    'difference': diff
                })

        if significant_differences:
            analysis['has_significant_difference'] = True

            for diff in significant_differences:
                explanation = f"Test组中{analysis['dimension']}='{diff['category']}'的比例" + \
                             f"{'更高' if diff['difference'] > 0 else '更低'}" + \
                             f" {abs(diff['difference']):.1f}个百分点"

                # 特定维度的性能影响分析
                if analysis['dimension'] == 'ischarged':
                    if diff['category'] == 1 and diff['difference'] > 0:
                        explanation += " (充电状态可能影响CPU频率和发热，进而影响性能)"
                    elif diff['category'] == 0 and diff['difference'] > 0:
                        explanation += " (非充电状态可能有更稳定的性能表现)"
                elif analysis['dimension'] == 'cpumodel':
                    explanation += f" (不同CPU型号的性能特征差异可能影响测试结果)"
                elif analysis['dimension'] == 'lowpower':
                    if diff['category'] == 1 and diff['difference'] > 0:
                        explanation += " (更多设备处于低功耗模式，可能限制了性能)"
                # 新增预处理特征的解释
                elif analysis['dimension'] == 'time_flag':
                    if diff['category'] == 1 and diff['difference'] > 0:
                        explanation += " (7月15日后的数据更多，可能反映了版本更新或环境变化)"
                    elif diff['category'] == 0 and diff['difference'] > 0:
                        explanation += " (7月15日前的数据更多，可能是基线性能表现)"
                elif analysis['dimension'] == 'memory_category':
                    if 'G' in str(diff['category']):
                        explanation += f" (不同内存规格的设备性能特征不同)"
                elif analysis['dimension'] == 'unity_digital':
                    if diff['category'] == '64bit' and diff['difference'] > 0:
                        explanation += " (64位Unity引擎版本更多，可能有不同的性能特征)"
                    elif diff['category'] == '32bit' and diff['difference'] > 0:
                        explanation += " (32位Unity引擎版本更多，可能影响内存和性能表现)"
                elif analysis['dimension'] == 'zone_group':
                    explanation += f" (不同服务器区域的网络和负载条件可能不同)"

                analysis['explanations'].append(explanation)

        return analysis

    async def _generate_causal_insights(
        self,
        dimension_differences: Dict,
        changed_metrics: List[str]
    ) -> List[str]:
        """生成因果关系洞察"""

        insights = []

        # 优先级维度：对性能影响最大的维度
        high_impact_dimensions = [
            'ischarged', 'cpumodel', 'refreshrate', 'lowpower',
            # 新增关键预处理特征
            'time_flag', 'memory_category', 'unity_digital', 'unityversion_inner'
        ]

        # 按优先级生成洞察
        for dimension in high_impact_dimensions:
            if dimension in dimension_differences:
                diff_analysis = dimension_differences[dimension]

                # 生成针对性的洞察
                if dimension == 'ischarged' and diff_analysis['has_significant_difference']:
                    insights.append(f"🔋 充电状态差异可能是影响 {', '.join(changed_metrics)} 性能的关键因素")

                elif dimension == 'cpumodel' and diff_analysis['has_significant_difference']:
                    insights.append(f"💻 CPU型号分布差异可能导致了 {', '.join(changed_metrics)} 的性能变化")

                elif dimension == 'refreshrate' and diff_analysis['has_significant_difference']:
                    insights.append(f"📱 刷新率差异可能影响了帧率和卡顿相关指标的表现")

                elif dimension == 'lowpower' and diff_analysis['has_significant_difference']:
                    insights.append(f"⚡ 功耗模式差异可能是性能变化的重要原因")

                # 新增预处理特征的洞察
                elif dimension == 'time_flag' and diff_analysis['has_significant_difference']:
                    insights.append(f"📅 测试时间分布差异(7月15日前后)可能影响了 {', '.join(changed_metrics)} 的表现")

                elif dimension == 'memory_category' and diff_analysis['has_significant_difference']:
                    insights.append(f"💾 设备内存规格分布差异可能是 {', '.join(changed_metrics)} 变化的原因")

                elif dimension == 'unity_digital' and diff_analysis['has_significant_difference']:
                    insights.append(f"🎮 Unity引擎位数(32/64位)分布差异可能影响了性能表现")

                elif dimension == 'unityversion_inner' and diff_analysis['has_significant_difference']:
                    insights.append(f"🔧 Unity详细版本分布差异可能是性能变化的关键因素")

        # 如果没有明显的高优先级原因，检查其他维度
        if not insights:
            for dimension, diff_analysis in dimension_differences.items():
                if diff_analysis['has_significant_difference']:
                    insights.append(f"📊 {dimension}维度的差异可能与性能变化相关")

        # 如果仍然没有明显原因
        if not insights:
            insights.append("📝 未发现明显的环境因素差异，性能变化可能由代码逻辑或其他未监控因素引起")

        return insights

    async def _generate_device_insights(self, analysis_result: Dict) -> List[str]:
        """为设备分析结果生成洞察"""

        insights = []
        device_model = analysis_result['device_model']
        quality_level = analysis_result['quality_level']

        # 整体状态洞察
        if analysis_result['overall_status'] == 'degraded':
            insights.append(f"🔴 设备 {device_model} (质量{quality_level}) 整体性能恶化")
        elif analysis_result['overall_status'] == 'improved':
            insights.append(f"🟢 设备 {device_model} (质量{quality_level}) 整体性能改善")

        # 关键指标洞察
        degraded_metrics = []
        improved_metrics = []

        for metric, metric_analysis in analysis_result['key_metrics_analysis'].items():
            if metric_analysis['status'] == 'degraded':
                degraded_metrics.append(f"{metric} ({metric_analysis['change_percent']:.1f}%)")
            elif metric_analysis['status'] == 'improved':
                improved_metrics.append(f"{metric} ({metric_analysis['change_percent']:.1f}%)")

        if degraded_metrics:
            insights.append(f"📉 恶化指标: {', '.join(degraded_metrics)}")

        if improved_metrics:
            insights.append(f"📈 改善指标: {', '.join(improved_metrics)}")

        # 注释掉性能变化原因分析
        # if 'performance_change_explanations' in analysis_result:
        #     explanations = analysis_result['performance_change_explanations']
        #
        #     if explanations['key_findings']:
        #         insights.append("🔍 可能的变化原因:")
        #         insights.extend([f"   • {finding}" for finding in explanations['key_findings']])
        #
        #     if explanations['potential_causes']:
        #         insights.append("📋 环境因素差异:")
        #         insights.extend([f"   • {cause}" for cause in explanations['potential_causes'][:3]])  # 只显示前3个

        return insights

    def _configure_chart_fonts(self) -> str:
        """配置图表字体并返回选中的字体名称"""
        import matplotlib.font_manager as fm

        # 获取系统可用字体列表
        available_fonts = [f.name for f in fm.fontManager.ttflist]

        # 定义优先使用的英文字体列表
        preferred_fonts = [
            'DejaVu Sans', 'Arial', 'Helvetica',
            'Verdana', 'Tahoma', 'Calibri', 'Segoe UI', 'Ubuntu',
            'Droid Sans', 'Noto Sans', 'sans-serif'
        ]

        # 找到第一个可用的字体
        selected_font = 'sans-serif'  # 默认字体
        for font in preferred_fonts:
            if font in available_fonts:
                selected_font = font
                break

        # 设置字体配置
        plt.rcParams['font.family'] = [selected_font]
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 10

        return selected_font

    async def _generate_comprehensive_analysis(self, all_results: Dict) -> Dict:
        """生成综合分析结果"""

        comprehensive = {
            'quality_level_summary': {},
            'device_ranking': [],
            'metric_impact_analysis': {},
            'dimension_impact_analysis': {},
            'overall_insights': []
        }

        available_key_metrics = getattr(self, 'available_key_metrics', KEY_PERFORMANCE_METRICS)
        available_common_columns = getattr(self, 'available_common_columns', COMMON_COLUMNS)

        # 质量等级汇总
        for quality_level, quality_results in all_results.items():
            # 确保quality_level是原生Python类型
            quality_level_key = int(quality_level) if hasattr(quality_level, 'item') else quality_level

            quality_summary = {
                'total_devices': len(quality_results),
                'degraded_devices': 0,
                'improved_devices': 0,
                'stable_devices': 0,
                'avg_confidence': 0.0
            }

            total_confidence = 0.0
            for device_model, device_analysis in quality_results.items():
                status = device_analysis['overall_status']
                if status == 'degraded':
                    quality_summary['degraded_devices'] += 1
                elif status == 'improved':
                    quality_summary['improved_devices'] += 1
                else:
                    quality_summary['stable_devices'] += 1

                total_confidence += device_analysis['confidence_score']

            quality_summary['avg_confidence'] = total_confidence / len(quality_results) if quality_results else 0
            comprehensive['quality_level_summary'][quality_level_key] = quality_summary

        # 设备排名
        all_devices = []
        for quality_level, quality_results in all_results.items():
            for device_model, device_analysis in quality_results.items():
                # 确保所有数值类型都是Python原生类型
                all_devices.append({
                    'device_model': str(device_model),
                    'quality_level': int(quality_level) if hasattr(quality_level, 'item') else quality_level,
                    'overall_status': str(device_analysis['overall_status']),
                    'degradation_score': float(device_analysis['degradation_score']),
                    'confidence_score': float(device_analysis['confidence_score'])
                })

        # 按恶化程度和置信度排序
        degraded_devices = [d for d in all_devices if d['overall_status'] == 'degraded']
        degraded_devices.sort(key=lambda x: (x['degradation_score'], x['confidence_score']), reverse=True)
        comprehensive['device_ranking'] = degraded_devices[:10]  # Top 10 恶化设备

        # 3. Generate metric impact analysis
        comprehensive['metric_impact_analysis'] = self._analyze_metric_impacts(all_results)

        # 4. Generate dimension impact analysis
        comprehensive['dimension_impact_analysis'] = self._analyze_dimension_impacts(
            all_results, available_common_columns)

        # 5. Generate overall insights
        comprehensive['overall_insights'] = self._generate_overall_insights(
            comprehensive, available_key_metrics, available_common_columns)

        return comprehensive

    def _analyze_metric_impacts(self, all_results: Dict) -> Dict:
        """分析指标影响程度"""
        metric_impacts = {}
        available_key_metrics = getattr(self, 'available_key_metrics', KEY_PERFORMANCE_METRICS)

        for metric in available_key_metrics:
            metric_data = {
                'total_devices_analyzed': 0,
                'devices_with_degradation': 0,
                'avg_degradation_magnitude': 0,
                'max_degradation': 0,
                'affected_quality_levels': [],
                'top_affected_devices': []
            }

            all_degradations = []
            affected_devices = []

            for quality_level, quality_results in all_results.items():
                quality_has_degradation = False

                for device_model, device_analysis in quality_results.items():
                    metric_data['total_devices_analyzed'] += 1

                    # 检查该指标的变化
                    if 'key_metrics_analysis' in device_analysis:
                        metric_analysis = device_analysis['key_metrics_analysis'].get(metric, {})

                        if metric_analysis.get('change_percent', 0) > 0:  # 恶化
                            metric_data['devices_with_degradation'] += 1
                            quality_has_degradation = True

                            degradation = metric_analysis['change_percent']
                            all_degradations.append(degradation)

                            affected_devices.append({
                                'device_model': device_model,
                                'quality_level': quality_level,
                                'degradation_percentage': float(degradation)
                            })

                if quality_has_degradation:
                    metric_data['affected_quality_levels'].append(quality_level)

            # 计算统计信息
            if all_degradations:
                metric_data['avg_degradation_magnitude'] = float(np.mean(all_degradations))
                metric_data['max_degradation'] = float(np.max(all_degradations))

                # 获取最受影响的设备（前5个）
                affected_devices.sort(key=lambda x: x['degradation_percentage'], reverse=True)
                metric_data['top_affected_devices'] = affected_devices[:5]

            metric_impacts[metric] = metric_data

        return metric_impacts

    def _analyze_dimension_impacts(self, all_results: Dict, available_common_columns: List[str]) -> Dict:
        """分析维度影响程度"""
        dimension_impacts = {}

        for dimension in available_common_columns:
            dimension_data = {
                'total_analysis_count': 0,
                'significant_differences_found': 0,
                'avg_impact_magnitude': 0,
                'top_impacted_scenarios': [],
                'quality_level_distribution': {}
            }

            all_impacts = []
            impacted_scenarios = []

            for quality_level, quality_results in all_results.items():
                quality_impacts = 0

                for device_model, device_analysis in quality_results.items():
                    # 注释掉检查维度分析结果
                    # explanations = device_analysis.get('performance_change_explanations', {})
                    # dimension_differences = explanations.get('dimension_differences', {})
                    #
                    # if dimension in dimension_differences:
                    #     dimension_analysis = dimension_differences[dimension]
                        dimension_data['total_analysis_count'] += 1

                        # 注释掉维度差异分析
                        # if dimension_analysis.get('has_significant_difference', False):
                        #     dimension_data['significant_differences_found'] += 1
                        #     quality_impacts += 1
                        #
                        #     # 计算影响程度（基于差异的显著性）
                        #     difference_details = dimension_analysis.get('difference_details', {})
                        #     impact_magnitude = 1.0 if dimension_analysis.get('has_significant_difference') else 0.0
                        #     all_impacts.append(impact_magnitude)
                        #
                        #     impacted_scenarios.append({
                        #         'device_model': device_model,
                        #         'quality_level': quality_level,
                        #         'dimension_value': dimension,
                        #         'difference_summary': dimension_analysis.get('summary', ''),
                        #         'impact_magnitude': float(impact_magnitude)
                        #     })

                dimension_data['quality_level_distribution'][str(quality_level)] = quality_impacts

            # 计算平均影响程度
            if all_impacts:
                dimension_data['avg_impact_magnitude'] = float(np.mean(all_impacts))

                # 获取最受影响的场景（前5个）
                impacted_scenarios.sort(key=lambda x: x['impact_magnitude'], reverse=True)
                dimension_data['top_impacted_scenarios'] = impacted_scenarios[:5]

            dimension_impacts[dimension] = dimension_data

        return dimension_impacts

    def _generate_overall_insights(self, comprehensive: Dict, available_key_metrics: List[str],
                                 available_common_columns: List[str]) -> List[str]:
        """生成整体洞察"""
        insights = []

        # 1. 设备排名洞察
        if comprehensive['device_ranking']:
            top_degraded = comprehensive['device_ranking'][0]
            insights.append(f"🚨 最需要关注的设备: {top_degraded['device_model']} "
                          f"(质量等级{top_degraded['quality_level']})，恶化评分: "
                          f"{top_degraded['degradation_score']:.2f}")

            total_degraded = len(comprehensive['device_ranking'])
            insights.append(f"📊 共发现 {total_degraded} 个性能恶化的设备型号组合")

        # 2. 指标影响洞察
        metric_impacts = comprehensive.get('metric_impact_analysis', {})
        if metric_impacts:
            # 找到影响最严重的指标
            most_impacted_metric = None
            max_affected_devices = 0

            for metric, data in metric_impacts.items():
                if data['devices_with_degradation'] > max_affected_devices:
                    max_affected_devices = data['devices_with_degradation']
                    most_impacted_metric = metric

            if most_impacted_metric:
                metric_data = metric_impacts[most_impacted_metric]
                insights.append(f"⚠️ 最受影响的性能指标: {most_impacted_metric}，"
                              f"影响了 {metric_data['devices_with_degradation']} 个设备，"
                              f"平均恶化程度: {metric_data['avg_degradation_magnitude']:.2f}%")

        # 3. 维度影响洞察
        dimension_impacts = comprehensive.get('dimension_impact_analysis', {})
        if dimension_impacts:
            # 找到影响最显著的维度
            most_significant_dimension = None
            max_significance_rate = 0

            for dimension, data in dimension_impacts.items():
                if data['total_analysis_count'] > 0:
                    significance_rate = data['significant_differences_found'] / data['total_analysis_count']
                    if significance_rate > max_significance_rate:
                        max_significance_rate = significance_rate
                        most_significant_dimension = dimension

            if most_significant_dimension:
                dimension_data = dimension_impacts[most_significant_dimension]
                insights.append(f"🔍 最关键的环境维度: {most_significant_dimension}，"
                              f"在 {dimension_data['significant_differences_found']} 个场景中发现显著差异，"
                              f"显著性比例: {max_significance_rate:.1%}")

        # 4. 质量等级洞察
        quality_summary = comprehensive.get('quality_level_summary', {})
        if quality_summary:
            total_quality_levels = len(quality_summary)
            affected_levels = len([level for level, data in quality_summary.items()
                                 if data['degraded_devices'] > 0])

            insights.append(f"📈 质量等级分布: 共分析 {total_quality_levels} 个质量等级，"
                          f"其中 {affected_levels} 个等级存在性能恶化设备")

        # 5. 数据完整性洞察
        available_metrics_count = len(available_key_metrics)
        available_dimensions_count = len(available_common_columns)

        insights.append(f"📋 数据完整性: 可分析 {available_metrics_count}/{len(KEY_PERFORMANCE_METRICS)} 个关键指标，"
                       f"{available_dimensions_count}/{len(COMMON_COLUMNS)} 个环境维度")

        # 6. 推荐建议
        if comprehensive['device_ranking']:
            insights.append("💡 建议优先优化排名前3的恶化设备，重点关注最受影响的性能指标")

        if dimension_impacts:
            insights.append("🔧 建议深入分析关键环境维度的配置差异，识别潜在的环境因素影响")

        return insights

    async def _create_comprehensive_visualizations(self, all_results: Dict, output_dir: str) -> None:
        """创建综合可视化图表"""

        # 检测并设置可用的英文字体
        selected_font = self._configure_chart_fonts()

        logger.info(f"Using font: {selected_font} for chart generation")

        logger.info("Generating comprehensive visualization charts...")

        # 1. 按质量等级分别生成概览图
        await self._create_quality_level_overview_separate(all_results, output_dir)

        # 2. 按质量等级分别生成设备指标热力图
        await self._create_metrics_heatmap_by_quality(all_results, output_dir)

        # 3. 生成关键指标相关性分析
        await self._create_metrics_correlation_analysis(all_results, output_dir)

        # 4. 暂时跳过异常值检测（原始数据未保存在分析结果中）
        logger.info("Skipping outlier detection - raw data not preserved in analysis results")
        # await self._create_outlier_detection_charts(all_results, output_dir)

        # 5. 公共维度影响分析
        await self._create_dimension_impact_charts(all_results, output_dir)

        # 6. 设备性能排名图
        await self._create_device_ranking_chart(all_results, output_dir)

        # 7. Unity版本分布可视化
        await self._create_unity_version_distribution_charts(all_results, output_dir)

    async def _create_quality_level_overview(self, all_results: Dict, output_dir: str) -> None:
        """创建质量等级概览图"""

        available_key_metrics = getattr(self, 'available_key_metrics', KEY_PERFORMANCE_METRICS)

        quality_levels = list(all_results.keys())
        metrics_data = {metric: [] for metric in available_key_metrics}

        for quality_level in quality_levels:
            quality_results = all_results[quality_level]

            for metric in available_key_metrics:
                # 计算该质量等级下该指标的平均变化
                metric_changes = []
                for device_model, device_analysis in quality_results.items():
                    if metric in device_analysis['key_metrics_analysis']:
                        change = device_analysis['key_metrics_analysis'][metric]['change_percent']
                        if not np.isnan(change):
                            metric_changes.append(change)

                avg_change = np.mean(metric_changes) if metric_changes else 0
                metrics_data[metric].append(avg_change)

        # 创建热力图
        plt.figure(figsize=(12, 8))
        heatmap_data = [metrics_data[metric] for metric in available_key_metrics]

        im = plt.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')

        plt.title('Key Metrics Performance by Quality Level', fontsize=16, fontweight='bold')
        plt.xlabel('Quality Levels', fontsize=12)
        plt.ylabel('Key Metrics', fontsize=12)

        plt.xticks(range(len(quality_levels)), quality_levels)
        plt.yticks(range(len(available_key_metrics)), available_key_metrics)

        # 添加数值标签
        for i in range(len(available_key_metrics)):
            for j in range(len(quality_levels)):
                text = plt.text(j, i, f'{heatmap_data[i][j]:.1f}%',
                              ha="center", va="center", color="black" if abs(heatmap_data[i][j]) < 2 else "white")

        plt.colorbar(im, label='Performance Change (%)')
        plt.tight_layout()

        chart_path = os.path.join(output_dir, 'quality_level_performance_overview.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved quality level performance overview: {chart_path}")

    async def _create_quality_level_overview_separate(self, all_results: Dict, output_dir: str) -> None:
        """Create overview charts by quality level with pagination (50 devices max per chart)"""

        # Configure fonts for proper device name handling
        selected_font = self._configure_chart_fonts()

        available_key_metrics = getattr(self, 'available_key_metrics', KEY_PERFORMANCE_METRICS)

        for quality_level, quality_results in all_results.items():
            if not quality_results:
                continue

            # Collect device data for this quality level
            device_names = []
            for device_model in quality_results.keys():
                # Clean device name to ensure proper display
                clean_device_name = str(device_model).encode('ascii', 'ignore').decode('ascii')
                if not clean_device_name.strip():
                    clean_device_name = f"Device_{len(device_names) + 1}"
                device_names.append(clean_device_name)

            metrics_data = {metric: [] for metric in available_key_metrics}

            for device_model, device_analysis in quality_results.items():
                for metric in available_key_metrics:
                    if metric in device_analysis['key_metrics_analysis']:
                        change = device_analysis['key_metrics_analysis'][metric]['change_percent']
                        metrics_data[metric].append(change if not np.isnan(change) else 0)
                    else:
                        metrics_data[metric].append(0)

            # Split into chunks of 50 devices maximum
            chunk_size = 50
            total_devices = len(device_names)

            for chunk_idx in range(0, total_devices, chunk_size):
                end_idx = min(chunk_idx + chunk_size, total_devices)

                chunk_device_names = device_names[chunk_idx:end_idx]
                chunk_metrics_data = {metric: values[chunk_idx:end_idx] for metric, values in metrics_data.items()}

                # Create heatmap for this chunk
                plt.figure(figsize=(max(10, len(chunk_device_names) * 0.8), max(6, len(available_key_metrics) * 0.6)))
                heatmap_data = [chunk_metrics_data[metric] for metric in available_key_metrics]

                im = plt.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')

                # Determine chart title with page info if multiple chunks
                if total_devices > chunk_size:
                    page_num = (chunk_idx // chunk_size) + 1
                    total_pages = (total_devices + chunk_size - 1) // chunk_size
                    title = f'Quality Level {quality_level} - Key Metrics Performance (Page {page_num}/{total_pages})'
                else:
                    title = f'Quality Level {quality_level} - Key Metrics Performance'

                plt.title(title, fontsize=16, fontweight='bold')
                plt.xlabel('Device Models', fontsize=12)
                plt.ylabel('Key Metrics', fontsize=12)

                plt.xticks(range(len(chunk_device_names)), chunk_device_names, rotation=45, ha='right')
                plt.yticks(range(len(available_key_metrics)), available_key_metrics)

                # Add value labels
                for i in range(len(available_key_metrics)):
                    for j in range(len(chunk_device_names)):
                        if j < len(heatmap_data[i]):
                            text = plt.text(j, i, f'{heatmap_data[i][j]:.1f}%',
                                          ha="center", va="center",
                                          color="black" if abs(heatmap_data[i][j]) < 2 else "white",
                                          fontsize=8)

                plt.colorbar(im, label='Performance Change (%)')
                plt.tight_layout()

                # Generate filename with page number if multiple chunks
                if total_devices > chunk_size:
                    page_num = (chunk_idx // chunk_size) + 1
                    chart_path = os.path.join(output_dir, f'quality_level_{quality_level}_performance_overview_page_{page_num}.png')
                else:
                    chart_path = os.path.join(output_dir, f'quality_level_{quality_level}_performance_overview.png')

                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()

                logger.info(f"Saved quality level {quality_level} performance overview: {chart_path}")

    async def _create_metrics_heatmap_by_quality(self, all_results: Dict, output_dir: str) -> None:
        """Create device metrics heatmaps by quality level with pagination (50 devices max per chart)"""

        # Configure fonts for proper device name handling
        selected_font = self._configure_chart_fonts()

        available_key_metrics = getattr(self, 'available_key_metrics', KEY_PERFORMANCE_METRICS)

        for quality_level, quality_results in all_results.items():
            if not quality_results:
                continue

            # Collect device data for this quality level
            device_metric_data = []
            device_labels = []

            for device_model, device_analysis in quality_results.items():
                # Clean device name to ensure proper display
                clean_device_name = str(device_model).encode('ascii', 'ignore').decode('ascii')
                if not clean_device_name.strip():
                    clean_device_name = f"Device_{len(device_labels) + 1}"

                device_labels.append(clean_device_name)

                metric_row = []
                for metric in available_key_metrics:
                    if metric in device_analysis['key_metrics_analysis']:
                        change = device_analysis['key_metrics_analysis'][metric]['change_percent']
                        metric_row.append(change if not np.isnan(change) else 0)
                    else:
                        metric_row.append(0)

                device_metric_data.append(metric_row)

            if not device_metric_data:
                continue

            # Split into chunks of 50 devices maximum
            chunk_size = 50
            total_devices = len(device_labels)

            for chunk_idx in range(0, total_devices, chunk_size):
                end_idx = min(chunk_idx + chunk_size, total_devices)

                chunk_device_labels = device_labels[chunk_idx:end_idx]
                chunk_device_data = device_metric_data[chunk_idx:end_idx]

                # Create heatmap for this chunk
                plt.figure(figsize=(max(10, len(available_key_metrics) * 1.2), max(8, len(chunk_device_labels) * 0.5)))

                im = plt.imshow(chunk_device_data, cmap='RdYlBu_r', aspect='auto')

                # Determine chart title with page info if multiple chunks
                if total_devices > chunk_size:
                    page_num = (chunk_idx // chunk_size) + 1
                    total_pages = (total_devices + chunk_size - 1) // chunk_size
                    title = f'Quality Level {quality_level} - Device Performance Heatmap (Page {page_num}/{total_pages})'
                else:
                    title = f'Quality Level {quality_level} - Device Performance Heatmap'

                plt.title(title, fontsize=16, fontweight='bold')
                plt.xlabel('Key Performance Metrics', fontsize=12)
                plt.ylabel('Device Models', fontsize=12)

                plt.xticks(range(len(available_key_metrics)), available_key_metrics, rotation=45, ha='right')
                plt.yticks(range(len(chunk_device_labels)), chunk_device_labels, fontsize=10)

                plt.colorbar(im, label='Performance Change (%)')
                plt.tight_layout()

                # Generate filename with page number if multiple chunks
                if total_devices > chunk_size:
                    page_num = (chunk_idx // chunk_size) + 1
                    chart_path = os.path.join(output_dir, f'quality_level_{quality_level}_device_metrics_heatmap_page_{page_num}.png')
                else:
                    chart_path = os.path.join(output_dir, f'quality_level_{quality_level}_device_metrics_heatmap.png')

                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()

                logger.info(f"Saved quality level {quality_level} device metrics heatmap: {chart_path}")

    async def _create_metrics_correlation_analysis(self, all_results: Dict, output_dir: str) -> None:
        """创建关键指标之间的关联分析"""

        available_key_metrics = getattr(self, 'available_key_metrics', KEY_PERFORMANCE_METRICS)

        for quality_level, quality_results in all_results.items():
            if not quality_results:
                continue

            # 收集该质量等级所有设备的指标数据
            metrics_data = {metric: [] for metric in available_key_metrics}

            for device_model, device_analysis in quality_results.items():
                for metric in available_key_metrics:
                    if metric in device_analysis['key_metrics_analysis']:
                        change = device_analysis['key_metrics_analysis'][metric]['change_percent']
                        metrics_data[metric].append(change if not np.isnan(change) else 0)
                    else:
                        metrics_data[metric].append(0)

            # 创建数据框计算相关性
            if all(len(values) > 1 for values in metrics_data.values()):
                df = pd.DataFrame(metrics_data)
                correlation_matrix = df.corr()

                # 创建相关性热力图
                plt.figure(figsize=(10, 8))
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

                im = plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')

                plt.title(f'Quality Level {quality_level} - Metrics Correlation Analysis', fontsize=16, fontweight='bold')
                plt.xlabel('Key Performance Metrics', fontsize=12)
                plt.ylabel('Key Performance Metrics', fontsize=12)

                plt.xticks(range(len(available_key_metrics)), available_key_metrics, rotation=45, ha='right')
                plt.yticks(range(len(available_key_metrics)), available_key_metrics)

                # 添加相关系数标签
                for i in range(len(available_key_metrics)):
                    for j in range(len(available_key_metrics)):
                        corr_value = correlation_matrix.iloc[i, j]
                        if not pd.isna(corr_value):
                            try:
                                # 转换为字符串再转回浮点数
                                corr_str = str(corr_value)
                                corr_val = float(corr_str)
                                text = plt.text(j, i, f'{corr_val:.2f}',
                                              ha="center", va="center",
                                              color="black" if abs(corr_val) < 0.5 else "white",
                                              fontsize=8)
                            except (ValueError, TypeError):
                                # 如果转换失败，只显示文本
                                text = plt.text(j, i, "N/A",
                                              ha="center", va="center",
                                              color="gray",
                                              fontsize=8)

                plt.colorbar(im, label='Correlation Coefficient')
                plt.tight_layout()

                chart_path = os.path.join(output_dir, f'quality_level_{quality_level}_metrics_correlation.png')
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()

                logger.info(f"Saved quality level {quality_level} metrics correlation: {chart_path}")

    async def _create_outlier_detection_charts(self, all_results: Dict, output_dir: str) -> None:
        """对同一质量等级和设备型号中的control和test组进行异常值检测"""

        available_key_metrics = getattr(self, 'available_key_metrics', KEY_PERFORMANCE_METRICS)

        for quality_level, quality_results in all_results.items():
            if not quality_results:
                continue

            for device_model, device_analysis in quality_results.items():
                control_data = device_analysis['control_data']
                test_data = device_analysis['test_data']

                # 为每个指标检测异常值
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle(f'Quality {quality_level} - {device_model} - Outlier Detection', fontsize=16, fontweight='bold')

                axes = axes.flatten()

                for idx, metric in enumerate(available_key_metrics[:4]):  # 最多显示4个指标
                    if idx >= len(axes):
                        break

                    # 获取control和test组的指标数据
                    control_values = []
                    test_values = []

                    # 从DataFrame行中提取数据
                    for _, row in control_data.iterrows():
                        if metric in row and pd.notna(row[metric]):
                            value = row[metric]
                            if not pd.isna(value):
                                control_values.append(float(value))

                    for _, row in test_data.iterrows():
                        if metric in row and pd.notna(row[metric]):
                            value = row[metric]
                            if not pd.isna(value):
                                test_values.append(float(value))

                    # 计算异常值
                    all_values = control_values + test_values
                    if len(all_values) > self.config.min_outlier_data_points:  # 需要足够的数据点
                        Q1 = np.percentile(all_values, 25)
                        Q3 = np.percentile(all_values, 75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - self.config.outlier_iqr_factor * IQR
                        upper_bound = Q3 + self.config.outlier_iqr_factor * IQR

                        # 识别异常值
                        control_outliers = [x for x in control_values if x < lower_bound or x > upper_bound]
                        test_outliers = [x for x in test_values if x < lower_bound or x > upper_bound]

                        # 绘制箱线图
                        ax = axes[idx]
                        box_data = [control_values, test_values]
                        box_labels = ['Control', 'Test']

                        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
                        bp['boxes'][0].set_facecolor('lightblue')
                        bp['boxes'][1].set_facecolor('lightcoral')

                        ax.set_title(f'{metric}\nControl outliers: {len(control_outliers)}, Test outliers: {len(test_outliers)}')
                        ax.set_ylabel('Value')
                        ax.grid(True, alpha=0.3)

                        # 标记异常值范围
                        ax.axhline(y=upper_bound, color='red', linestyle='--', alpha=0.7, label='Upper threshold')
                        ax.axhline(y=lower_bound, color='red', linestyle='--', alpha=0.7, label='Lower threshold')

                # 隐藏多余的子图
                for idx in range(len(available_key_metrics), len(axes)):
                    axes[idx].set_visible(False)

                plt.tight_layout()
                chart_path = os.path.join(output_dir, f'quality_{quality_level}_{device_model}_outlier_detection.png')
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()

                logger.info(f"Saved outlier detection for quality {quality_level}, device {device_model}: {chart_path}")

    async def _create_metrics_heatmap(self, all_results: Dict, output_dir: str) -> None:
        """创建指标热力图"""

        available_key_metrics = getattr(self, 'available_key_metrics', KEY_PERFORMANCE_METRICS)

        # 收集所有设备的指标数据
        device_metric_data = []
        device_labels = []

        for quality_level, quality_results in all_results.items():
            for device_model, device_analysis in quality_results.items():
                device_label = f"{device_model}\n(Q{quality_level})"
                device_labels.append(device_label)

                metric_row = []
                for metric in available_key_metrics:
                    if metric in device_analysis['key_metrics_analysis']:
                        change = device_analysis['key_metrics_analysis'][metric]['change_percent']
                        metric_row.append(change if not np.isnan(change) else 0)
                    else:
                        metric_row.append(0)

                device_metric_data.append(metric_row)

        if not device_metric_data:
            return

        # 创建热力图
        plt.figure(figsize=(15, max(8, len(device_labels) * 0.3)))

        im = plt.imshow(device_metric_data, cmap='RdYlBu_r', aspect='auto')

        plt.title('Device Performance Heatmap - Key Metrics Change (%)', fontsize=16, fontweight='bold')
        plt.xlabel('Key Performance Metrics', fontsize=12)
        plt.ylabel('Devices (Quality Level)', fontsize=12)

        plt.xticks(range(len(available_key_metrics)), available_key_metrics, rotation=45, ha='right')
        plt.yticks(range(len(device_labels)), device_labels, fontsize=10)

        plt.colorbar(im, label='Performance Change (%)')
        plt.tight_layout()

        chart_path = os.path.join(output_dir, 'device_metrics_heatmap.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved device metrics heatmap: {chart_path}")

    async def _create_dimension_impact_charts(self, all_results: Dict, output_dir: str) -> None:
        """创建公共维度影响分析图"""

        available_common_columns = getattr(self, 'available_common_columns', COMMON_COLUMNS)
        logger.info(f"可用的公共维度列: {available_common_columns}")

        # 为每个重要维度创建分析图，包括新增的预处理特征
        important_dimensions = ['ischarged', 'cpumodel', 'refreshrate', 'time_flag', 'memory_category', 'unity_digital']

        for dimension in important_dimensions:
            logger.info(f"检查维度: {dimension}")

            if dimension in available_common_columns:
                logger.info(f"Dimension {dimension} exists in data, starting chart generation...")
                await self._create_single_dimension_chart(all_results, dimension, output_dir)
            else:
                logger.warning(f"维度 {dimension} 不存在于可用列中，跳过图表生成")
                logger.info(f"当前可用维度: {available_common_columns}")

    async def _create_single_dimension_chart(self, all_results: Dict, dimension: str, output_dir: str) -> None:
        """Create single dimension impact analysis chart"""

        logger.info(f"Starting to collect analysis data for dimension {dimension}...")

        # Collect analysis data for this dimension
        dimension_insights = []

        for quality_level, quality_results in all_results.items():
            for device_model, device_analysis in quality_results.items():
                # Comment out dimension analysis related visualization data collection
                # explanations = device_analysis.get('performance_change_explanations', {})
                # dimension_differences = explanations.get('dimension_differences', {})
                #
                # if dimension in dimension_differences:
                #     dim_analysis = dimension_differences[dimension]
                #     logger.info(f"设备 {device_model} 在维度 {dimension} 有分析数据")
                #
                #     for insight in dim_analysis.get('explanations', []):
                #         dimension_insights.append({
                #             'device': f"{device_model} (Q{quality_level})",
                #             'insight': insight,
                #             'quality_level': quality_level
                #         })
                # else:
                #     logger.debug(f"设备 {device_model} 在维度 {dimension} 没有显著差异")
                pass

        logger.info(f"Dimension {dimension} collected {len(dimension_insights)} insights")

        if not dimension_insights:
            logger.warning(f"Dimension {dimension} has no analysis data, skipping chart generation")
            logger.info(f"Possible reasons: 1) This dimension has no significant differences across all devices; 2) No devices have performance changes; 3) Data quality issues")
            return

        # Create text chart
        plt.figure(figsize=(14, max(6, len(dimension_insights) * 0.4)))

        y_positions = list(range(len(dimension_insights)))
        colors = ['red' if 'higher' in insight['insight'] or 'more' in insight['insight'] else 'blue'
                 for insight in dimension_insights]

        plt.barh(y_positions, [1]*len(dimension_insights), color=colors, alpha=0.3)

        plt.title(f'{dimension.capitalize()} Impact Analysis', fontsize=16, fontweight='bold')
        plt.xlabel('Impact Direction', fontsize=12)
        plt.ylabel('Devices', fontsize=12)

        # Add insight text
        for i, insight_data in enumerate(dimension_insights):
            plt.text(0.5, i, insight_data['insight'], ha='center', va='center',
                    fontsize=10, wrap=True)

        plt.yticks(y_positions, [insight['device'] for insight in dimension_insights])
        plt.xlim(0, 1)
        plt.tight_layout()

        chart_path = os.path.join(output_dir, f'{dimension}_impact_analysis.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved {dimension} dimension impact analysis chart: {chart_path}")

    async def _create_device_ranking_chart(self, all_results: Dict, output_dir: str) -> None:
        """Create device performance ranking charts with pagination (50 devices max per chart)"""

        # Configure fonts for proper device name handling
        selected_font = self._configure_chart_fonts()

        # Collect all degraded devices
        degraded_devices = []

        for quality_level, quality_results in all_results.items():
            for device_model, device_analysis in quality_results.items():
                if device_analysis['overall_status'] == 'degraded':
                    # Clean device name to ensure proper display
                    clean_device_name = str(device_model).encode('ascii', 'ignore').decode('ascii')
                    if not clean_device_name.strip():
                        clean_device_name = f"Device_{len(degraded_devices) + 1}"

                    degraded_devices.append({
                        'device': f"{clean_device_name} (Q{quality_level})",
                        'degradation_score': device_analysis['degradation_score'],
                        'confidence_score': device_analysis['confidence_score'],
                        'quality_level': quality_level
                    })

        if not degraded_devices:
            logger.info("No degraded devices found, skipping ranking chart generation")
            return

        # Sort by degradation level
        degraded_devices.sort(key=lambda x: x['degradation_score'], reverse=True)

        # Split into chunks of 50 devices maximum
        chunk_size = 50
        total_devices = len(degraded_devices)

        for chunk_idx in range(0, total_devices, chunk_size):
            end_idx = min(chunk_idx + chunk_size, total_devices)

            chunk_devices = degraded_devices[chunk_idx:end_idx]

            devices = [d['device'] for d in chunk_devices]
            scores = [d['degradation_score'] for d in chunk_devices]
            confidences = [d['confidence_score'] for d in chunk_devices]

            plt.figure(figsize=(12, max(8, len(devices) * 0.5)))

            # Set colors based on confidence
            colors = ['darkred' if conf > 7.5 else 'red' if conf > 5.0 else 'orange'
                     for conf in confidences]

            bars = plt.barh(range(len(devices)), scores, color=colors, alpha=0.7)

            # Determine chart title with page info if multiple chunks
            if total_devices > chunk_size:
                page_num = (chunk_idx // chunk_size) + 1
                total_pages = (total_devices + chunk_size - 1) // chunk_size
                title = f'Top Degraded Devices Ranking (Page {page_num}/{total_pages})'
            else:
                title = 'Top Degraded Devices Ranking'

            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel('Average Degradation Score (%)', fontsize=12)
            plt.ylabel('Devices (Quality Level)', fontsize=12)

            plt.yticks(range(len(devices)), devices)

            # Add confidence labels
            for i, (bar, score, confidence) in enumerate(zip(bars, scores, confidences)):
                plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                        f'{score:.1f}% (C:{confidence:.1f})',
                        va='center', fontsize=10)

            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()

            # Generate filename with page number if multiple chunks
            if total_devices > chunk_size:
                page_num = (chunk_idx // chunk_size) + 1
                chart_path = os.path.join(output_dir, f'degraded_devices_ranking_page_{page_num}.png')
            else:
                chart_path = os.path.join(output_dir, 'degraded_devices_ranking.png')

            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Saved degraded devices ranking chart: {chart_path}")

    async def _create_unity_version_distribution_charts(self, all_results: Dict, output_dir: str) -> None:
        """Create Unity version distribution charts with pagination (50 devices max per chart)"""

        # Configure fonts for proper device name handling
        selected_font = self._configure_chart_fonts()

        available_common_columns = getattr(self, 'available_common_columns', COMMON_COLUMNS)

        # Check if unityversion_inner column exists
        if 'unityversion_inner' not in available_common_columns:
            logger.warning("unityversion_inner column does not exist, skipping Unity version distribution chart generation")
            return

        for quality_level, quality_results in all_results.items():
            if not quality_results:
                continue

            # Include all devices for Unity version distribution analysis
            devices_with_unity_data = []

            for device_model, device_analysis in quality_results.items():
                # Check if device has Unity version data
                control_data = device_analysis.get('control_data', pd.DataFrame())
                test_data = device_analysis.get('test_data', pd.DataFrame())

                if (not control_data.empty and not test_data.empty and
                    'unityversion_inner' in control_data.columns and
                    'unityversion_inner' in test_data.columns):
                    devices_with_unity_data.append((device_model, device_analysis))

            # If no devices have Unity version data, skip this quality level
            if not devices_with_unity_data:
                logger.info(f"Quality level {quality_level} has no devices with Unity version data, skipping chart generation")
                continue

            # Apply pagination: maximum 50 devices, split into multiple charts
            chunk_size = 50
            total_devices = len(devices_with_unity_data)

            for chunk_idx in range(0, total_devices, chunk_size):
                end_idx = min(chunk_idx + chunk_size, total_devices)
                chunk_devices = devices_with_unity_data[chunk_idx:end_idx]

                # Create charts for this chunk of devices
                device_count = len(chunk_devices)
                cols = min(3, device_count)  # Maximum 3 devices per row
                rows = (device_count + cols - 1) // cols

                fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))

                # Determine chart title with page info if multiple chunks
                if total_devices > chunk_size:
                    page_num = (chunk_idx // chunk_size) + 1
                    total_pages = (total_devices + chunk_size - 1) // chunk_size
                    title = f'Quality Level {quality_level} - Unity Version Distribution by Device (Page {page_num}/{total_pages})\n(Control vs Test Group Comparison)'
                else:
                    title = f'Quality Level {quality_level} - Unity Version Distribution by Device\n(Control vs Test Group Comparison)'

                fig.suptitle(title, fontsize=14, fontweight='bold')

                # Ensure axes is a 2D array
                if rows == 1 and cols == 1:
                    axes = [[axes]]
                elif rows == 1:
                    axes = [axes]
                elif cols == 1:
                    axes = [[ax] for ax in axes]

                device_idx = 0
                for device_model, device_analysis in chunk_devices:
                    if device_idx >= rows * cols:
                        break

                    row = device_idx // cols
                    col = device_idx % cols
                    ax = axes[row][col]

                    # Clean device name to ensure proper display
                    clean_device_name = str(device_model).encode('ascii', 'ignore').decode('ascii')
                    if not clean_device_name.strip():
                        clean_device_name = f"Device_{device_idx + 1}"

                    # Get Unity version distribution data from control and test groups
                    control_data = device_analysis.get('control_data', pd.DataFrame())
                    test_data = device_analysis.get('test_data', pd.DataFrame())

                    if not control_data.empty and not test_data.empty and 'unityversion_inner' in control_data.columns and 'unityversion_inner' in test_data.columns:
                        # Function to simplify unity version strings
                        def simplify_unity_version(version_str):
                            """Extract version number from unity_inner string"""
                            version_str = str(version_str)
                            # Split by '-' and take the first 4 parts to get version number
                            parts = version_str.split('-')
                            return '-'.join(parts[:2])  # e.g., "2019-2.01.0052.03"

                        # Simplify unity versions for simplboth control and test data
                        control_simplified = control_data['unityversion_inner'].apply(simplify_unity_version)
                        test_simplified = test_data['unityversion_inner'].apply(simplify_unity_version)

                        # Count simplified unity_inner values for control and test separately
                        control_version_counts = control_simplified.value_counts().to_dict()
                        test_version_counts = test_simplified.value_counts().to_dict()

                        # Get all versions from both groups
                        all_versions = set(control_version_counts.keys()).union(set(test_version_counts.keys()))

                        # Get top 6 most common versions by total count
                        version_total_counts = {}
                        for version in all_versions:
                            version_total_counts[version] = control_version_counts.get(version, 0) + test_version_counts.get(version, 0)

                        top_versions = sorted(version_total_counts.items(), key=lambda x: x[1], reverse=True)[:6]
                        top_version_names = [v[0] for v in top_versions]

                        # Prepare data for bar chart - control vs test comparison
                        control_values = [control_version_counts.get(v, 0) for v in top_version_names]
                        test_values = [test_version_counts.get(v, 0) for v in top_version_names]

                        # Calculate total samples for each group
                        control_total = len(control_data)
                        test_total = len(test_data)

                        # Create bar chart for version distribution comparison
                        x = range(len(top_version_names))
                        width = 0.35

                        ax.bar([i - width/2 for i in x], control_values, width, label=f'Control (n={control_total})',
                               alpha=0.8, color='lightblue')
                        ax.bar([i + width/2 for i in x], test_values, width, label=f'Test (n={test_total})',
                               alpha=0.8, color='lightcoral')

                        ax.set_title(f'{clean_device_name}', fontsize=11, fontweight='bold')
                        ax.set_ylabel('Sample Count', fontsize=9)
                        ax.set_xlabel('Unity Version', fontsize=9)
                        ax.set_xticks(x)
                        ax.set_xticklabels(top_version_names, rotation=45, ha='right', fontsize=8)
                        ax.legend(fontsize=8)
                        ax.grid(True, alpha=0.3)

                        # Add count labels on top of bars
                        max_value = max(control_values + test_values) if (control_values + test_values) else 1
                        for i, (control_count, test_count) in enumerate(zip(control_values, test_values)):
                            if control_count > 0:
                                ax.text(i - width/2, control_count + max_value * 0.01,
                                       str(control_count), ha='center', va='bottom', fontsize=8, fontweight='bold')
                            if test_count > 0:
                                ax.text(i + width/2, test_count + max_value * 0.01,
                                       str(test_count), ha='center', va='bottom', fontsize=8, fontweight='bold')

                        # Mark significant differences based on absolute count differences
                        for i, version in enumerate(top_version_names):
                            control_count = control_version_counts.get(version, 0)
                            test_count = test_version_counts.get(version, 0)
                            diff = abs(test_count - control_count)
                            total_samples = control_count + test_count
                            # Consider significant if difference is more than 20% of total or more than 5 samples
                            if diff > max(5, total_samples * 0.2):
                                ax.text(i, max(control_count, test_count) + max_value * 0.1,
                                       f'Δ{diff}', ha='center', va='bottom', fontsize=7, color='red', fontweight='bold')
                    else:
                        # If no Unity version data available, show message
                        ax.text(0.5, 0.5, f"Device: {clean_device_name}\n(No Unity version data)",
                               ha='center', va='center', transform=ax.transAxes,
                               fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
                        ax.set_title(f'{clean_device_name}', fontsize=11, fontweight='bold')
                        ax.set_xticks([])
                        ax.set_yticks([])

                    device_idx += 1

                # Hide extra subplots
                for idx in range(device_idx, rows * cols):
                    row = idx // cols
                    col = idx % cols
                    axes[row][col].set_visible(False)

                plt.tight_layout()

                # Generate filename with page number if multiple chunks
                if total_devices > chunk_size:
                    page_num = (chunk_idx // chunk_size) + 1
                    chart_path = os.path.join(output_dir, f'unity_version_distribution_quality_{quality_level}_page_{page_num}.png')
                else:
                    chart_path = os.path.join(output_dir, f'unity_version_distribution_quality_{quality_level}.png')

                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()

                logger.info(f"Saved Unity version distribution chart for quality level {quality_level}: {chart_path}")

    async def _save_analysis_results(self, all_results: Dict, comprehensive_analysis: Dict, output_dir: str) -> None:
        """保存分析结果到文件"""

        # 1. 保存详细分析结果
        detailed_results = []

        for quality_level, quality_results in all_results.items():
            for device_model, device_analysis in quality_results.items():

                # 基本信息
                row = {
                    'quality_level': quality_level,
                    'device_model': device_model,
                    'control_count': device_analysis['control_count'],
                    'test_count': device_analysis['test_count'],
                    'confidence_score': device_analysis['confidence_score'],
                    'overall_status': device_analysis['overall_status'],
                    'degradation_score': device_analysis['degradation_score']
                }

                # 关键指标分析 - 只保存可用的指标
                available_key_metrics = getattr(self, 'available_key_metrics', KEY_PERFORMANCE_METRICS)
                for metric in available_key_metrics:
                    if metric in device_analysis['key_metrics_analysis']:
                        metric_data = device_analysis['key_metrics_analysis'][metric]
                        row[f'{metric}_change_percent'] = metric_data['change_percent']
                        row[f'{metric}_status'] = metric_data['status']
                        row[f'{metric}_control_mean'] = metric_data['control_mean']
                        row[f'{metric}_test_mean'] = metric_data['test_mean']

                # 洞察
                row['insights'] = '; '.join(device_analysis['insights'])

                detailed_results.append(row)

        detailed_df = pd.DataFrame(detailed_results)
        detailed_path = os.path.join(output_dir, 'detailed_analysis_results.csv')
        detailed_df.to_csv(detailed_path, index=False, encoding='utf-8')
        logger.info(f"Saved detailed analysis results: {detailed_path}")

        # 2. 注释掉保存公共维度分析
        # dimension_results = []
        #
        # for quality_level, quality_results in all_results.items():
        #     for device_model, device_analysis in quality_results.items():
        #         if 'performance_change_explanations' in device_analysis:
        #             explanations = device_analysis['performance_change_explanations']
        #             if 'dimension_differences' in explanations:
        #                 for dimension, dim_analysis in explanations['dimension_differences'].items():
        #                     dim_row = {
        #                         'quality_level': quality_level,
        #                         'device_model': device_model,
        #                         'dimension': dimension,
        #                         'has_significant_difference': dim_analysis.get('has_significant_difference', False),
        #                         'explanations': '; '.join(dim_analysis.get('explanations', []))
        #                     }
        #
        #                     # 添加差异详情
        #                     if 'difference_details' in dim_analysis:
        #                         details = dim_analysis['difference_details']
        #                         if isinstance(details, dict) and 'control_mean' in details:
        #                             dim_row.update({
        #                                 'control_mean': details.get('control_mean'),
        #                                 'test_mean': details.get('test_mean'),
        #                                 'difference_percent': details.get('difference_percent')
        #                             })
        #
        #                     dimension_results.append(dim_row)
        #
        # if dimension_results:
        #     dimension_df = pd.DataFrame(dimension_results)
        #     dimension_path = os.path.join(output_dir, 'dimension_analysis_results.csv')
        #     dimension_df.to_csv(dimension_path, index=False, encoding='utf-8')
        #     logger.info(f"保存维度分析结果: {dimension_path}")

        # 3. 保存综合分析
        # 转换numpy类型以避免JSON序列化错误
        def convert_numpy_types(obj):
            """递归转换numpy类型为Python原生类型"""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif hasattr(obj, 'item') and hasattr(obj, 'dtype'):  # numpy标量
                return obj.item()
            else:
                return obj

        comprehensive_analysis_cleaned = convert_numpy_types(comprehensive_analysis)

        with open(os.path.join(output_dir, 'comprehensive_analysis.json'), 'w', encoding='utf-8') as f:
            json.dump(comprehensive_analysis_cleaned, f, indent=2, ensure_ascii=False)

    async def _split_data_by_unity_version(self, all_results: Dict, output_dir: str) -> None:
        """Split data by unity_inner version and save to separate CSV files"""

        logger.info("Starting to split data by Unity version...")

        def simplify_unity_version(version_str):
            """Extract version number from unity_inner string"""
            version_str = str(version_str)
            # Split by '-' and take the first 2 parts to get simplified version
            parts = version_str.split('-')
            return '-'.join(parts[:2])  # e.g., "2019-2"

        # Create unity_versions directory
        unity_versions_dir = os.path.join(output_dir, 'unity_versions')
        os.makedirs(unity_versions_dir, exist_ok=True)

        # Dictionary to store data by simplified version
        version_data = {}
        version_counts = {}

        # Collect all data and group by simplified unity version
        for quality_level, quality_results in all_results.items():
            for device_model, device_analysis in quality_results.items():
                control_data = device_analysis.get('control_data', pd.DataFrame())
                test_data = device_analysis.get('test_data', pd.DataFrame())

                # Process control data
                if not control_data.empty and 'unityversion_inner' in control_data.columns:
                    control_copy = control_data.copy()
                    control_copy['group_type'] = 'control'
                    control_copy['quality_level'] = quality_level
                    control_copy['device_model'] = device_model
                    control_copy['simplified_unity_version'] = control_copy['unityversion_inner'].apply(simplify_unity_version)

                    for _, row in control_copy.iterrows():
                        simplified_version = row['simplified_unity_version']
                        if simplified_version not in version_data:
                            version_data[simplified_version] = []
                            version_counts[simplified_version] = {'control': 0, 'test': 0, 'total': 0}

                        version_data[simplified_version].append(row.to_dict())
                        version_counts[simplified_version]['control'] += 1
                        version_counts[simplified_version]['total'] += 1

                # Process test data
                if not test_data.empty and 'unityversion_inner' in test_data.columns:
                    test_copy = test_data.copy()
                    test_copy['group_type'] = 'test'
                    test_copy['quality_level'] = quality_level
                    test_copy['device_model'] = device_model
                    test_copy['simplified_unity_version'] = test_copy['unityversion_inner'].apply(simplify_unity_version)

                    for _, row in test_copy.iterrows():
                        simplified_version = row['simplified_unity_version']
                        if simplified_version not in version_data:
                            version_data[simplified_version] = []
                            version_counts[simplified_version] = {'control': 0, 'test': 0, 'total': 0}

                        version_data[simplified_version].append(row.to_dict())
                        version_counts[simplified_version]['test'] += 1
                        version_counts[simplified_version]['total'] += 1

        # Save each version's data to separate CSV files
        for simplified_version, rows in version_data.items():
            if rows:  # Only save if there's data
                version_df = pd.DataFrame(rows)

                # Clean version name for filename (replace invalid characters)
                safe_version_name = simplified_version.replace('/', '_').replace('\\', '_').replace(':', '_')
                csv_filename = f"unity_version_{safe_version_name}.csv"
                csv_path = os.path.join(unity_versions_dir, csv_filename)

                # Sort by quality_level, device_model, group_type for better organization
                version_df = version_df.sort_values(['quality_level', 'device_model', 'group_type'])

                # Save to CSV
                version_df.to_csv(csv_path, index=False, encoding='utf-8')

                count_info = version_counts[simplified_version]
                logger.info(f"Saved Unity version {simplified_version}: {csv_path}")
                logger.info(f"  - Total records: {count_info['total']}")
                logger.info(f"  - Control: {count_info['control']}, Test: {count_info['test']}")

        # Create a summary file with version statistics
        summary_data = []
        for version, counts in version_counts.items():
            summary_data.append({
                'unity_version': version,
                'control_records': counts['control'],
                'test_records': counts['test'],
                'total_records': counts['total']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('total_records', ascending=False)
        summary_path = os.path.join(unity_versions_dir, 'unity_versions_summary.csv')
        #summary_df.to_csv(summary_path, index=False, encoding='utf-8')

        logger.info(f"Created Unity versions summary: {summary_path}")
        logger.info(f"Total Unity versions found: {len(version_counts)}")
        logger.info(f"Unity versions data saved in directory: {unity_versions_dir}")

    def _generate_device_performance_report(
        self,
        all_results: Dict,
        output_dir: str
    ) -> str:
        """生成类似用户描述的设备性能分析报告"""

        logger.info("生成设备性能分析报告...")

        report_lines = []
        report_lines.append("# 设备性能分析报告")
        report_lines.append("=" * 60)
        report_lines.append("")

        # （一）卡顿、帧率、功耗数据分析
        report_lines.append("## （一）卡顿、帧率、功耗数据")
        report_lines.append("")

        # 收集所有设备的性能数据
        device_performance_data = []
        device_counter = 1

        for quality_level, quality_results in all_results.items():
            for device_model, device_analysis in quality_results.items():
                if device_counter > self.config.max_devices_in_report:
                    break

                metrics = device_analysis.get('key_metrics_analysis', {})

                # 提取关键性能指标
                perf_data = {
                    'device_id': device_counter,
                    'device_model': device_model,
                    'quality_level': quality_level,
                    'bigrate': None,  # 卡顿率 - 使用bigjankper10min
                    'fps': None,      # 帧率 - 使用avgfps_unity
                    'current_avg': None,  # 电流平均值
                    'gc_count': None, # GC次数 - 使用gt_50
                    'gc_alloc': None, # GC分配 - 使用totalgcallocsize
                    'overall_status': device_analysis.get('overall_status', 'stable'),
                    'confidence': device_analysis.get('confidence_score', 0)
                }

                # 填充数据
                if 'bigjankper10min' in metrics:
                    perf_data['bigrate'] = metrics['bigjankper10min'].get('test_mean', 0)

                if 'avgfps_unity' in metrics:
                    perf_data['fps'] = metrics['avgfps_unity'].get('test_mean', 0)

                if 'current_avg' in metrics:
                    perf_data['current_avg'] = metrics['current_avg'].get('test_mean', 0)

                if 'gt_50' in metrics:
                    perf_data['gc_count'] = metrics['gt_50'].get('test_mean', 0)

                if 'totalgcallocsize' in metrics:
                    perf_data['gc_alloc'] = metrics['totalgcallocsize'].get('test_mean', 0)

                device_performance_data.append(perf_data)
                device_counter += 1

        # 生成设备性能表格
        if device_performance_data:
            report_lines.append("记录了{}台设备的性能指标：".format(len(device_performance_data)))
            report_lines.append("")

            for data in device_performance_data:
                device_desc = f"**设备{data['device_id']}** ({data['device_model']}, 质量等级{data['quality_level']})："
                report_lines.append(device_desc)

                # 卡顿率分析
                if data['bigrate'] is not None:
                    bigrate_pct = data['bigrate']
                    bigrate_status = "高" if bigrate_pct > self.config.performance_threshold_poor else "正常"
                    report_lines.append(f"- **卡顿率**：{bigrate_pct:.2f}%，卡顿{bigrate_status}")

                # 帧率分析
                if data['fps'] is not None:
                    fps_val = data['fps']
                    fps_status = "低，画面易不流畅" if fps_val < 30 else "正常"
                    report_lines.append(f"- **帧率**：{fps_val:.2f} FPS，{fps_status}")

                # 电流分析
                if data['current_avg'] is not None:
                    current_val = data['current_avg']
                    current_status = "功耗表现较差" if abs(current_val) > 1000 else "功耗正常"
                    report_lines.append(f"- **电流平均值**：{current_val:.2f}，{current_status}")

                # GC分析
                if data['gc_count'] is not None:
                    gc_count_val = data['gc_count']
                    report_lines.append(f"- **GC次数**：{gc_count_val:.2f}")

                if data['gc_alloc'] is not None:
                    gc_alloc_val = data['gc_alloc']
                    gc_status = "较高" if gc_alloc_val > self.config.gc_threshold_high else "正常"
                    report_lines.append(f"- **GC分配**：{gc_alloc_val:.2f}，{gc_status}")

                # 整体状态
                status_emoji = {"degraded": "🔴", "improved": "🟢", "stable": "🔵"}
                status_desc = {"degraded": "性能恶化", "improved": "性能改善", "stable": "性能稳定"}
                emoji = status_emoji.get(data['overall_status'], "⚪")
                desc = status_desc.get(data['overall_status'], "未知")
                report_lines.append(f"- **整体状态**：{emoji} {desc} (置信度: {data['confidence']:.2f})")

                report_lines.append("")

        # （二）内存数据分析
        report_lines.append("## （二）内存数据分析")
        report_lines.append("")

        if self.config.memory_analysis_enabled:
            memory_analysis = self._analyze_memory_by_categories(all_results)

            report_lines.append("按内存区间和Unity版本位数分析totalpss内存占用：")
            report_lines.append("")

            for mem_category, mem_data in memory_analysis.items():
                report_lines.append(f"**{mem_category} 内存区间**：")

                for unity_digital, pss_data in mem_data.items():
                    avg_pss = pss_data.get('avg_pss', 0)
                    device_count = pss_data.get('device_count', 0)
                    pss_status = "内存占用较高" if avg_pss > self.config.pss_threshold_high else "内存占用正常"

                    report_lines.append(f"- Unity{unity_digital}位版本：totalpss平均 {avg_pss:.2f}，"
                                      f"{pss_status} (基于{device_count}个设备)")

                report_lines.append("")

        # 性能优化建议
        report_lines.append("## 性能优化建议")
        report_lines.append("")

        # 分析最需要关注的问题
        high_priority_issues = []
        optimization_suggestions = []

        for data in device_performance_data:
            if data['overall_status'] == 'degraded':
                issues = []
                if data['bigrate'] and data['bigrate'] > self.config.performance_threshold_poor:
                    issues.append("卡顿率高")
                if data['fps'] and data['fps'] < 30:
                    issues.append("帧率低")
                if data['current_avg'] and abs(data['current_avg']) > 1000:
                    issues.append("功耗异常")

                if issues:
                    high_priority_issues.append(f"设备{data['device_id']}：{', '.join(issues)}")

        if high_priority_issues:
            report_lines.append("🚨 **需要优先处理的问题**：")
            for issue in high_priority_issues[:5]:  # 显示前5个
                report_lines.append(f"- {issue}")
            report_lines.append("")

        # 通用优化建议
        report_lines.append("💡 **通用优化建议**：")
        report_lines.append("- 针对卡顿率高的设备，检查渲染管线和GC触发频率")
        report_lines.append("- 针对帧率低的设备，考虑降低渲染质量或优化GPU使用")
        report_lines.append("- 针对功耗异常的设备，分析CPU使用模式和充电状态影响")
        report_lines.append("- 定期监控内存使用，特别关注GC分配频率较高的设备")
        report_lines.append("")

        # 数据质量说明
        report_lines.append("## 数据说明")
        report_lines.append("")
        report_lines.append("本报告基于A/B测试数据分析生成，用于评估设备运行性能表现、")
        report_lines.append("内存利用效率等，可辅助优化设备体验、排查性能问题。")
        report_lines.append("")
        report_lines.append(f"- 分析设备数：{len(device_performance_data)}")
        report_lines.append(f"- 性能恶化设备：{len([d for d in device_performance_data if d['overall_status'] == 'degraded'])}")
        report_lines.append(f"- 性能改善设备：{len([d for d in device_performance_data if d['overall_status'] == 'improved'])}")
        report_lines.append(f"- 性能稳定设备：{len([d for d in device_performance_data if d['overall_status'] == 'stable'])}")

        # 保存报告
        report_content = "\n".join(report_lines)
        report_path = os.path.join(output_dir, 'device_performance_report.md')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"设备性能报告已保存到: {report_path}")
        return report_content

    def _analyze_memory_by_categories(self, all_results: Dict) -> Dict:
        """按内存类别和Unity版本分析内存使用情况"""

        memory_analysis = {}

        for quality_level, quality_results in all_results.items():
            for device_model, device_analysis in quality_results.items():

                # 注释掉获取维度分析中的内存和Unity版本信息
                # explanations = device_analysis.get('performance_change_explanations', {})
                # dimension_diffs = explanations.get('dimension_differences', {})

                # 提取内存类别信息
                memory_category = "未知"
                unity_digital = "未知"

                # 注释掉从维度分析中提取内存和Unity版本信息
                # if 'memory_category' in dimension_diffs:
                #     mem_details = dimension_diffs['memory_category'].get('difference_details', {})
                #     test_dist = mem_details.get('test_distribution', {})
                #     if test_dist:
                #         # 取占比最高的内存类别
                #         memory_category = max(test_dist.items(), key=lambda x: x[1])[0]
                #
                # if 'unity_digital' in dimension_diffs:
                #     unity_details = dimension_diffs['unity_digital'].get('difference_details', {})
                #     test_dist = unity_details.get('test_distribution', {})
                #     if test_dist:
                #         # 取占比最高的Unity版本位数
                #         unity_digital = max(test_dist.items(), key=lambda x: x[1])[0]

                # 直接从设备模型名称中尝试提取内存和Unity版本信息
                device_model_str = str(device_model)
                if '0G-2G' in device_model_str:
                    memory_category = '0G-2G'
                elif '2G-6G' in device_model_str:
                    memory_category = '2G-6G'
                elif '6G' in device_model_str:
                    memory_category = '6G'

                if '32bit' in device_model_str:
                    unity_digital = '32bit'
                elif '64bit' in device_model_str:
                    unity_digital = '64bit'

                # 获取totalpss数据（如果有的话，使用battleendtotalpss作为近似）
                metrics = device_analysis.get('key_metrics_analysis', {})
                totalpss = 0
                if 'battleendtotalpss' in metrics:
                    totalpss = metrics['battleendtotalpss'].get('test_mean', 0)

                # 组织数据结构
                if memory_category not in memory_analysis:
                    memory_analysis[memory_category] = {}

                if unity_digital not in memory_analysis[memory_category]:
                    memory_analysis[memory_category][unity_digital] = {
                        'total_pss': 0,
                        'device_count': 0
                    }

                memory_analysis[memory_category][unity_digital]['total_pss'] += totalpss
                memory_analysis[memory_category][unity_digital]['device_count'] += 1

        # 计算平均值
        for mem_category in memory_analysis:
            for unity_digital in memory_analysis[mem_category]:
                data = memory_analysis[mem_category][unity_digital]
                if data['device_count'] > 0:
                    data['avg_pss'] = data['total_pss'] / data['device_count']
                else:
                    data['avg_pss'] = 0

        return memory_analysis

    def _generate_final_summary(
        self,
        csv_file_path: str,
        all_results: Dict,
        comprehensive_analysis: Dict,
        quality_splits: Dict,
        output_dir: str,
        degradation_threshold: float,
        confidence_alpha: float
    ) -> str:
        """生成最终总结报告"""

        summary = "关键指标层级分析报告\n"
        summary += "=" * 60 + "\n\n"

        available_common_columns = getattr(self, 'available_common_columns', COMMON_COLUMNS)
        available_key_metrics = getattr(self, 'available_key_metrics', KEY_PERFORMANCE_METRICS)

        summary += f"**输入数据信息:**\n"
        summary += f"- 数据文件: {csv_file_path}\n"
        summary += f"- 分析维度: {len(available_common_columns)} 个公共维度\n"
        summary += f"- 关键指标: {len(available_key_metrics)}/{len(KEY_PERFORMANCE_METRICS)} 个核心指标 (可用/总计)\n"
        summary += f"- 可用指标: {', '.join(available_key_metrics)}\n"
        if len(available_key_metrics) < len(KEY_PERFORMANCE_METRICS):
            missing_metrics = [m for m in KEY_PERFORMANCE_METRICS if m not in available_key_metrics]
            summary += f"- 缺失指标: {', '.join(missing_metrics)}\n"
        summary += f"- 恶化阈值: {degradation_threshold}%\n"
        summary += f"- 置信度参数: {confidence_alpha}\n\n"

        summary += f"**数据分布概览:**\n"
        total_quality_levels = len(quality_splits)
        total_devices = sum(len(quality_results) for quality_results in all_results.values())

        summary += f"- 质量等级数量: {total_quality_levels} 个\n"
        summary += f"- 分析设备型号总数: {total_devices} 个\n"

        for quality_level, quality_data in quality_splits.items():
            device_count = len(all_results.get(quality_level, {}))
            summary += f"  * 质量等级 {quality_level}: {device_count} 个设备型号 "
            summary += f"(Control: {quality_data['control_records']}, Test: {quality_data['test_records']})\n"

        summary += "\n**质量等级分析结果:**\n"
        for quality_level, quality_summary in comprehensive_analysis['quality_level_summary'].items():
            summary += f"📊 **质量等级 {quality_level}**:\n"
            summary += f"   - 总设备数: {quality_summary['total_devices']}\n"
            summary += f"   - 🔴 恶化设备: {quality_summary['degraded_devices']}\n"
            summary += f"   - 🟢 改善设备: {quality_summary['improved_devices']}\n"
            summary += f"   - 🔵 稳定设备: {quality_summary['stable_devices']}\n"
            summary += f"   - 平均置信度: {quality_summary['avg_confidence']:.1f}\n\n"

        summary += f"**核心发现:**\n"

        # Top 恶化设备
        if comprehensive_analysis['device_ranking']:
            summary += f"🔴 **性能恶化最严重的设备:**\n"
            for i, device in enumerate(comprehensive_analysis['device_ranking'][:5], 1):
                summary += f"{i}. {device['device_model']} (质量{device['quality_level']}) - "
                summary += f"恶化程度: {device['degradation_score']:.1f}%, 置信度: {device['confidence_score']:.1f}\n"
            summary += "\n"

        # 关键指标整体表现 - 只分析可用的指标
        summary += f"📈 **关键指标整体表现:**\n"

        for metric in available_key_metrics:
            metric_changes = []

            for quality_level, quality_results in all_results.items():
                for device_model, device_analysis in quality_results.items():
                    if metric in device_analysis['key_metrics_analysis']:
                        change = device_analysis['key_metrics_analysis'][metric]['change_percent']
                        if not np.isnan(change):
                            metric_changes.append(change)

            if metric_changes:
                avg_change = np.mean(metric_changes)
                degraded_count = sum(1 for change in metric_changes if abs(change) >= degradation_threshold and
                                   not self._is_metric_improvement(metric, change))

                summary += f"- {metric}: 平均变化 {avg_change:.1f}%, {degraded_count} 个设备恶化\n"

        summary += f"\n**维度洞察:**\n"

        # 统计各维度的关键发现
        dimension_insights_count = {}
        for quality_level, quality_results in all_results.items():
            for device_model, device_analysis in quality_results.items():
                if 'performance_change_explanations' in device_analysis:
                    explanations = device_analysis['performance_change_explanations']
                    if 'dimension_differences' in explanations:
                        for dimension, dim_analysis in explanations['dimension_differences'].items():
                            if dim_analysis.get('has_significant_difference', False):
                                dimension_insights_count[dimension] = dimension_insights_count.get(dimension, 0) + 1

        for dimension, count in sorted(dimension_insights_count.items(), key=lambda x: x[1], reverse=True):
            summary += f"- {dimension}: {count} 个设备发现显著差异\n"

        summary += f"\n**生成文件:**\n"
        summary += f"📊 质量等级概览图: {output_dir}/quality_level_performance_overview.png\n"
        summary += f"🔥 设备指标热力图: {output_dir}/device_metrics_heatmap.png\n"
        summary += f"📈 恶化设备排名: {output_dir}/degraded_devices_ranking.png\n"
        summary += f"📋 详细分析结果: {output_dir}/detailed_analysis_results.csv\n"
        summary += f"📋 维度分析结果: {output_dir}/dimension_analysis_results.csv\n"
        summary += f"📁 综合分析数据: {output_dir}/comprehensive_analysis.json\n"

        # 添加Unity版本分割后的CSV文件路径信息
        unity_versions_dir = os.path.join(output_dir, 'unity_versions')
        if os.path.exists(unity_versions_dir):
            summary += f"\n**Unity版本分割数据文件:**\n"
            summary += f"📁 Unity版本数据目录: {unity_versions_dir}\n"

            # 列出所有生成的Unity版本CSV文件
            try:
                unity_csv_files = [f for f in os.listdir(unity_versions_dir) if f.endswith('.csv') and f.startswith('unity_version_')]
                unity_csv_files.sort()  # 按文件名排序

                if unity_csv_files:
                    summary += f"🔢 生成的Unity版本数据文件共 {len(unity_csv_files)} 个:\n"
                    for csv_file in unity_csv_files:
                        file_path = os.path.join(unity_versions_dir, csv_file)
                        # 从文件名提取版本信息
                        version_name = csv_file.replace('unity_version_', '').replace('.csv', '').replace('_', '-')
                        summary += f"   • {version_name}: {file_path}\n"

                    # 添加摘要文件
                    summary_file = os.path.join(unity_versions_dir, 'unity_versions_summary.csv')
                    if os.path.exists(summary_file):
                        summary += f"📊 Unity版本统计摘要: {summary_file}\n"
                else:
                    summary += f"⚠️ 未找到Unity版本分割的CSV文件\n"

            except Exception as e:
                summary += f"⚠️ 读取Unity版本目录时出错: {e}\n"

        for dimension in ['ischarged', 'cpumodel', 'refreshrate', 'time_flag', 'memory_category', 'unity_digital']:
            if dimension in available_common_columns:
                summary += f"📊 {dimension} 影响分析: {output_dir}/{dimension}_impact_analysis.png\n"

        summary += f"\n**关键建议:**\n"
        summary += f"1. 重点关注性能恶化严重且置信度高的设备型号\n"
        summary += f"2. 分析不同质量等级设备的性能差异模式\n"
        summary += f"3. 考虑充电状态、CPU型号等维度对性能的影响\n"
        summary += f"4. 针对不同质量等级制定差异化的优化策略\n"
        summary += f"5. 关注样本量充足且具有统计意义的对比结果\n"

        return summary
