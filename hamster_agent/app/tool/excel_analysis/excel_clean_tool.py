import os
from typing import List

import pandas as pd

from app.exceptions import ToolError
from app.tool.base import BaseTool


class ExcelCleanTool(BaseTool):
    """
    A tool for cleaning Excel files (CSV/TSV) by removing columns that are entirely empty
    or contain identical values throughout the column.
    在group_column列中，控制组和实验组的值分别为"Control"和"Test"，用于区分两组数据。
    在同一个devicemodel下，如果control组和test组中有一个组的数目为零，也需要进行剔除
    """

    name: str = "excel_clean_tool"
    description: str = (
        "Clean Excel files (CSV/TSV) by removing empty columns and columns with identical values."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the CSV or TSV file to clean",
            },
            "null_threshold": {
                "type": "number",
                "description": "Threshold for null values ratio (0.0-1.0). Columns with null ratio >= threshold will be removed",
                "default": 1.0,
            },
            "remove_constant_columns": {
                "type": "boolean",
                "description": "Whether to remove columns with identical non-null values",
                "default": True,
            },
            "group_column": {
                "type": "string",
                "description": "Column name that identifies control/test groups",
                "default": "param_value",
            },
            "control_value": {
                "type": "string",
                "description": "Value in group_column that represents the control group",
                "default": "control",
            },
            "test_value": {
                "type": "string",
                "description": "Value in group_column that represents the test group",
                "default": "test",
            },
            "encoding": {
                "type": "string",
                "description": "File encoding to use (default: utf-8)",
                "default": "utf-8",
            },
        },
        "required": ["file_path"],
    }

    async def execute(
        self,
        file_path: str,
        null_threshold: float = 1.0,
        remove_constant_columns: bool = True,
        group_column: str = "param_value",
        control_value: str = "control",
        test_value: str = "test",
        encoding: str = "utf-8",
    ) -> str:
        """
        Clean Excel file by removing columns that are entirely empty or contain identical values.
        """
        try:
            # Validate input file path
            if not os.path.exists(file_path):
                raise ToolError(f"File not found: {file_path}")

            # Determine file type and separator
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == ".csv":
                sep = ","
            elif file_ext == ".tsv":
                sep = "\t"
            else:
                raise ToolError(
                    f"Unsupported file extension: {file_ext}. Supported: .csv, .tsv"
                )

            # Read the input file
            try:
                df = pd.read_csv(
                    file_path, sep=sep, encoding=encoding, low_memory=False
                )
            except Exception as e:
                raise ToolError(f"Error reading file: {str(e)}")

            # 🔧 处理重复列名问题
            original_columns = df.columns.tolist()
            if len(original_columns) != len(set(original_columns)):
                # 发现重复列名，进行处理
                seen = {}
                new_columns = []
                for col in original_columns:
                    if col in seen:
                        seen[col] += 1
                        new_col = f"{col}_{seen[col]}"
                        new_columns.append(new_col)
                        print(f"⚠️  重复列名 '{col}' 重命名为 '{new_col}'")
                    else:
                        seen[col] = 0
                        new_columns.append(col)

                df.columns = new_columns
                print(
                    f"✅ 处理了 {len(original_columns) - len(set(original_columns))} 个重复列名"
                )

            # 🔧 重置索引以避免重复索引问题
            df = df.reset_index(drop=True)

            # Store original shape for reporting
            original_rows, original_cols = df.shape

            # Track dropped columns and reasons
            columns_dropped = []
            columns_dropped_reasons = {}

            # 🔧 验证必要的列是否存在
            if group_column not in df.columns:
                raise ToolError(
                    f"Group column '{group_column}' not found in the file. Available columns: {list(df.columns)}"
                )

            if "devicemodel" not in df.columns:
                raise ToolError(
                    f"'devicemodel' column not found in the file. Available columns: {list(df.columns)}"
                )

            # Identify columns to drop
            columns_to_drop = []

            # 1. Check for columns with null values exceeding the threshold
            for col in df.columns:
                null_ratio = df[col].isna().mean()
                if null_ratio >= null_threshold:
                    columns_to_drop.append(col)
                    columns_dropped_reasons[col] = f"Empty ratio: {null_ratio:.2%}"

            # 2. Check for columns with identical values (if enabled)
            if remove_constant_columns:
                for col in df.columns:
                    if col not in columns_to_drop:  # Skip already identified columns
                        # Check if all non-null values are identical
                        unique_values = df[col].dropna().unique()
                        if len(unique_values) <= 1:
                            columns_to_drop.append(col)
                            if len(unique_values) == 0:
                                columns_dropped_reasons[col] = "All values are null"
                            else:
                                columns_dropped_reasons[col] = (
                                    f"All values are identical: {unique_values[0]}"
                                )

            # Drop the identified columns
            df_cleaned = df.drop(columns=columns_to_drop)
            columns_dropped = columns_to_drop

            # 🔧 检查分组数据前，先确保数据类型正确
            try:
                # 确保group_column的值是字符串类型，便于比较
                df_cleaned[group_column] = (
                    df_cleaned[group_column].astype(str).str.lower()
                )
                control_value_lower = control_value.lower()
                test_value_lower = test_value.lower()

                # check for group_column and count every devicemodel control and test size
                control_data = df_cleaned[
                    df_cleaned[group_column] == control_value_lower
                ].copy()
                test_data = df_cleaned[
                    df_cleaned[group_column] == test_value_lower
                ].copy()

                print(
                    f"📊 Control组记录数: {len(control_data)}, Test组记录数: {len(test_data)}"
                )

                if len(control_data) == 0:
                    raise ToolError(
                        f"No records found for control group value '{control_value}' in column '{group_column}'"
                    )
                if len(test_data) == 0:
                    raise ToolError(
                        f"No records found for test group value '{test_value}' in column '{group_column}'"
                    )

            except Exception as e:
                raise ToolError(f"Error processing group data: {str(e)}")

            # device model set
            control_devices = set(control_data["devicemodel"].dropna().unique())
            test_devices = set(test_data["devicemodel"].dropna().unique())
            all_devices = control_devices.union(test_devices)

            print(
                f"📱 发现设备型号: Control={len(control_devices)}, Test={len(test_devices)}, 总计={len(all_devices)}"
            )

            # 记录需要删除的设备模型
            devices_to_remove = set()
            for device in all_devices:
                control_count = control_data[
                    control_data["devicemodel"] == device
                ].shape[0]
                test_count = test_data[test_data["devicemodel"] == device].shape[0]

                # 如果控制组或测试组的数量为零，则添加到删除列表
                if control_count == 0 or test_count == 0:
                    devices_to_remove.add(device)
                    print(
                        f"🗑️  将删除设备 {device}: Control={control_count}, Test={test_count}"
                    )

            # 如果有需要删除的设备模型，执行删除操作
            if devices_to_remove:
                # 记录被删除的设备模型及其原因
                for device in devices_to_remove:
                    control_count = control_data[
                        control_data["devicemodel"] == device
                    ].shape[0]
                    test_count = test_data[test_data["devicemodel"] == device].shape[0]
                    reason = f"Control count: {control_count}, Test count: {test_count}"
                    columns_dropped_reasons[f"device_{device}"] = reason

                # 从cleaned数据中删除这些设备模型的所有记录
                df_cleaned = df_cleaned[
                    ~df_cleaned["devicemodel"].isin(devices_to_remove)
                ]
                print(f"✅ 删除了 {len(devices_to_remove)} 个设备型号的数据")

            # 🔧 最终清理：重置索引并确保没有重复
            df_cleaned = df_cleaned.reset_index(drop=True)

            # Generate output file path
            output_path = self._generate_output_path(file_path)

            # Save the cleaned dataframe
            df_cleaned.to_csv(output_path, sep=sep, index=False, encoding=encoding)

            # Prepare result message
            cleaned_rows, cleaned_cols = df_cleaned.shape
            removed_cols = original_cols - cleaned_cols
            removed_rows = original_rows - cleaned_rows

            print(
                f"✅ 清理完成: {original_rows}×{original_cols} → {cleaned_rows}×{cleaned_cols}"
            )

            # Generate detailed report
            result_message = self._generate_result_message(
                file_path,
                output_path,
                original_rows,
                original_cols,
                cleaned_rows,
                cleaned_cols,
                columns_dropped,
                columns_dropped_reasons,
                removed_rows,
            )

            return result_message

        except ToolError as e:
            return str(e)
        except Exception as e:
            error_msg = f"Unexpected error while cleaning file: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

    def _generate_output_path(self, file_path: str) -> str:
        """Generate output file path by adding '_cleaned' before the extension"""
        base_name, extension = os.path.splitext(file_path)
        return f"{base_name}_cleaned{extension}"

    def _generate_result_message(
        self,
        input_path: str,
        output_path: str,
        original_rows: int,
        original_cols: int,
        cleaned_rows: int,
        cleaned_cols: int,
        columns_dropped: List[str],
        columns_dropped_reasons: dict,
        removed_rows: int = 0,
    ) -> str:
        """Generate a detailed result message"""

        # Basic statistics
        removed_cols = original_cols - cleaned_cols
        col_removed_pct = (
            (removed_cols / original_cols * 100) if original_cols > 0 else 0
        )
        row_removed_pct = (
            (removed_rows / original_rows * 100) if original_rows > 0 else 0
        )

        message = [
            "✅ Excel File Cleaning Complete!",
            "",
            "📊 Cleaning Summary:",
            f"• Input File: {input_path}",
            f"• Output File: {output_path}",
            f"• Original Shape: {original_rows} rows × {original_cols} columns",
            f"• Cleaned Shape: {cleaned_rows} rows × {cleaned_cols} columns",
            f"• Columns Removed: {removed_cols} ({col_removed_pct:.1f}% of total)",
            f"• Rows Removed: {removed_rows} ({row_removed_pct:.1f}% of total)",
            "",
        ]

        # Add detailed info about removed columns
        if columns_dropped:
            message.append("🗑️ Removed Columns:")
            # Show at most 10 columns with reasons
            for i, col in enumerate(columns_dropped[:10]):
                reason = columns_dropped_reasons.get(col, "Unknown reason")
                message.append(f"• {col}: {reason}")

            if len(columns_dropped) > 10:
                message.append(f"• ... and {len(columns_dropped) - 10} more columns")
        else:
            message.append("ℹ️ No columns were removed.")

        # 添加被删除的设备模型信息
        device_removed = [k for k in columns_dropped_reasons if k.startswith("device_")]
        if device_removed:
            message.append("\n📱 Removed Device Models (Control/Test count mismatch):")
            for i, device_key in enumerate(device_removed[:10]):
                device = device_key.replace("device_", "")
                reason = columns_dropped_reasons.get(device_key, "Unknown reason")
                message.append(f"• {device}: {reason}")

            if len(device_removed) > 10:
                message.append(
                    f"• ... and {len(device_removed) - 10} more device models"
                )

        return "\n".join(message)
