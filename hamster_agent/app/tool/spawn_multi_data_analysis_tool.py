import asyncio
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import json
from torch import AwaitType
# from zmq import Message

from app.config import PROJECT_ROOT, config
from app.exceptions import ToolError
from app.llm import LLM
from app.logger import logger
from app.prompt.game_data_analysis import AGGREGATION_REPORT_PROMPT, RUN_PROMPT_CH
from app.schema import AgentMemoryManager, Message
from app.tool.base import BaseTool, ToolResult


class SpawnMultiDataAnalysisTool(BaseTool):
    """
    一个用于在目录中找到所有CSV/TSV文件并并行运行数据分析代理的工具。
    """

    name: str = "spawn_multi_data_analysis_tool"
    description: str = (
        "并行分析指定目录中的所有CSV和TSV文件。"
        "这里，目标目录指的是存储多个子表（即多个CSV/TSV文件）的文件夹。"
        "对于该目录中的每个文件，它会生成一个data_analysis代理，该代理将其结果"
        "保存到与输入文件路径对应的专用子文件夹中。"
        "支持根据analysis_aspect配置选择性分析特定类型的子表。"
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "directory_path": {
                "type": "string",
                "description": "包含多个子表（CSV/TSV文件）的目录的绝对路径。",
            },
            "max_concurrent_agents": {
                "type": "integer",
                "description": "并行运行的代理的最大数量。",
                "default": 4,
            },
            "max_retries_per_file": {
                "type": "integer",
                "description": "如果分析失败，单个文件的最大重试次数。",
                "default": 1,
            },
            "max_retry_rounds": {
                "type": "integer",
                "description": "失败任务的最大重试轮数。",
                "default": 2,
            },
            "analysis_aspect": {
                "type": "object",
                "description": "分析方面配置，指定要分析的子表类型。例如: {'fps' : False, 'lag' : True, 'memory' : True, 'power' : False, 'temperature' : False}",
                "default": None,
            },
        },
        "required": ["directory_path"],
    }

    async def execute(
        self,
        directory_path: str,
        max_concurrent_agents: int = 4,
        max_retries_per_file: int = 1,
        max_retry_rounds: int = 2,
        analysis_aspect: Optional[Dict[str, bool]] = None,
    ) -> ToolResult:
        """
        执行给定目录中文件的并行分析。
        """
        if not os.path.isdir(directory_path):
            raise ToolError(f"Directory not found: {directory_path}")

        parent_directory = os.path.dirname(os.path.abspath(directory_path))
        logger.info(f"The parent directory of {directory_path} is {parent_directory}")

        if config.run_flow_config.meta_data_name is None:
            raise ToolError(f"Metadata file name is not configured in run_flow_config.")
        # Find the metadata file in the parent directory
        metadata_file_path = os.path.join(
            PROJECT_ROOT, "workspace", config.run_flow_config.meta_data_name
        )

        files_to_analyze = self._list_files(directory_path, analysis_aspect)
        if not files_to_analyze:
            if analysis_aspect:
                enabled_aspects = [k for k, v in analysis_aspect.items() if v]
                return ToolResult(
                    output=f"No CSV or TSV files found matching the enabled analysis aspects: {enabled_aspects}. "
                    f"Checked directory: {directory_path}"
                )
            else:
                return ToolResult(
                    output="No CSV or TSV files found in the directory to analyze."
                )

        task_queue = asyncio.Queue()
        for file_path in files_to_analyze:
            await task_queue.put((file_path, metadata_file_path))

        results = []

        # Create worker tasks
        worker_tasks = []
        for i in range(max_concurrent_agents):
            task = asyncio.create_task(
                self._worker(f"worker-{i+1}", task_queue, max_retries_per_file, results)
            )
            worker_tasks.append(task)

        # Wait for all items in the queue to be processed
        await task_queue.join()

        # Cancel worker tasks
        for task in worker_tasks:
            task.cancel()

        # Wait for tasks to finish cancellation
        await asyncio.gather(*worker_tasks, return_exceptions=True)

        # 统计成功和失败的任务
        successful_tasks = [r for r in results if r["success"]]
        failed_tasks = [r for r in results if not r["success"]]

        # 重试机制：为失败的任务重新分配worker
        retry_round = 1

        while failed_tasks and retry_round <= max_retry_rounds:
            logger.info(
                f"开始第 {retry_round} 轮重试，处理 {len(failed_tasks)} 个失败任务"
            )

            # 为失败的任务创建新的队列
            retry_queue = asyncio.Queue()
            for failed_task in failed_tasks:
                await retry_queue.put((failed_task["file_path"], metadata_file_path))

            # 清空results中的失败任务记录，准备重新记录
            results = [r for r in results if r["success"]]

            # 创建重试worker任务（使用较少的并发数）
            retry_workers = min(max_concurrent_agents, len(failed_tasks))
            retry_worker_tasks = []
            for i in range(retry_workers):
                task = asyncio.create_task(
                    self._worker(
                        f"retry-worker-{retry_round}-{i+1}",
                        retry_queue,
                        max_retries_per_file,
                        results,
                    )
                )
                retry_worker_tasks.append(task)

            # 等待重试队列处理完成
            await retry_queue.join()

            # 取消重试worker任务
            for task in retry_worker_tasks:
                task.cancel()

            # 等待任务完成取消
            await asyncio.gather(*retry_worker_tasks, return_exceptions=True)

            # 重新统计成功和失败的任务
            successful_tasks = [r for r in results if r["success"]]
            failed_tasks = [r for r in results if not r["success"]]

            logger.info(
                f"第 {retry_round} 轮重试完成，成功 {len([r for r in results if r['success'] and f'retry-worker-{retry_round}' in r['worker_name']])} 个，失败 {len([r for r in results if not r['success'] and f'retry-worker-{retry_round}' in r['worker_name']])} 个"
            )
            retry_round += 1

        # 构建详细的结果消息
        total_successful = len(successful_tasks)
        total_failed = len(failed_tasks)

        # 添加分析方面的信息
        result_message = ""
        if analysis_aspect:
            enabled_aspects = [k for k, v in analysis_aspect.items() if v]
            disabled_aspects = [k for k, v in analysis_aspect.items() if not v]
            result_message += f"📊 分析配置:\n"
            result_message += f"- 启用的分析方面: {enabled_aspects}\n"
            result_message += f"- 禁用的分析方面: {disabled_aspects}\n"
            result_message += f"- 符合条件的文件数: {len(files_to_analyze)}\n\n"

        result_message += f"分析完成！处理了 {len(files_to_analyze)} 个文件，最终成功 {total_successful} 个，失败 {total_failed} 个。"

        if retry_round > 1:
            result_message += f"（经过 {retry_round-1} 轮重试）"
        result_message += "\n\n"

        # 统计各轮次的成功情况
        initial_successful = len(
            [
                r
                for r in successful_tasks
                if not r["worker_name"].startswith("retry-worker")
            ]
        )
        retry_successful = total_successful - initial_successful

        if retry_round > 1:
            result_message += f"详细统计:\n"
            result_message += f"- 初始轮成功: {initial_successful} 个\n"
            result_message += f"- 重试轮成功: {retry_successful} 个\n"
            result_message += f"- 最终失败: {total_failed} 个\n\n"

        if successful_tasks:
            result_message += "成功的任务:\n"
            for task in successful_tasks:
                result_message += f"- 文件: {task['file_path']}\n"
                result_message += f"  输出目录: {task['output_dir']}\n"
                result_message += f"  内存路径: {task['memory_path']}\n"
                if task.get("copied_report_path"):
                    result_message += f"  复制的报告: {task['copied_report_path']}\n"
                result_message += f"  处理worker: {task['worker_name']}\n\n"

        if failed_tasks:
            result_message += "失败的任务:\n"
            for task in failed_tasks:
                result_message += f"- 文件: {task['file_path']}\n"
                result_message += f"  输出目录: {task['output_dir']}\n"
                result_message += f"  处理worker: {task['worker_name']}\n\n"

        classify_result = self._classify_all_success_result(results)

        # 为每个分类生成综合报告
        generated_summary_reports = await self._generate_category_summary_reports(
            classify_result
        )

        # 在结果消息中添加分类汇总信息
        if generated_summary_reports:
            result_message += "\n生成的分类汇总报告:\n"
            for category, report_path in generated_summary_reports.items():
                if report_path:
                    result_message += f"- {category.upper()} 分类报告: {report_path}\n"

        return ToolResult(output=result_message)

    def _find_metadata_file(self, directory: str) -> str | None:
        """在给定目录中查找元数据文件（例如，*metadata.json或*column_analysis.json）。"""
        try:
            for filename in os.listdir(directory):
                if filename.lower().endswith(("metadata.json", "column_analysis.json")):
                    return os.path.join(directory, filename)
        except OSError as e:
            logger.error(f"Error scanning directory {directory} for metadata file: {e}")
        return None

    def _list_files(
        self, directory_path: str, analysis_aspect: Optional[Dict[str, bool]] = None
    ) -> list[str]:
        """
        递归列出目录及其所有子目录中的 CSV 和 TSV 文件路径。

        Args:
            directory_path: 目录路径
            analysis_aspect: 分析方面配置，如果提供则只返回匹配的文件

        Returns:
            匹配条件的CSV/TSV文件路径列表
        """
        try:
            matched_files = []
            for root, _, files in os.walk(directory_path):
                for f in files:
                    if f.lower().endswith((".csv", ".tsv")):
                        file_path = os.path.join(root, f)

                        # 如果没有指定analysis_aspect，包含所有文件
                        if not analysis_aspect:
                            matched_files.append(file_path)
                            continue

                        # 检查文件名是否匹配启用的分析方面
                        filename_lower = f.lower()
                        should_include = False

                        for aspect, enabled in analysis_aspect.items():
                            if enabled:
                                # 支持多种匹配模式
                                aspect_lower = aspect.lower()

                                # 直接包含匹配
                                if aspect_lower in filename_lower:
                                    should_include = True
                                    logger.info(
                                        f"包含文件 {f}: 匹配启用的分析方面 '{aspect}' (直接匹配)"
                                    )
                                    break

                                # 支持常见的变体匹配
                                aspect_variants = self._get_aspect_variants(
                                    aspect_lower
                                )
                                for variant in aspect_variants:
                                    if variant in filename_lower:
                                        should_include = True
                                        logger.info(
                                            f"包含文件 {f}: 匹配启用的分析方面 '{aspect}' (变体匹配: {variant})"
                                        )
                                        break

                                if should_include:
                                    break

                        if should_include:
                            matched_files.append(file_path)
                        else:
                            logger.debug(f"跳过文件 {f}: 不匹配任何启用的分析方面")

            # 记录过滤结果
            if analysis_aspect:
                enabled_aspects = [k for k, v in analysis_aspect.items() if v]
                logger.info(
                    f"根据分析方面 {enabled_aspects} 过滤，找到 {len(matched_files)} 个匹配文件"
                )

            return matched_files
        except OSError as e:
            raise ToolError(f"Error reading directory {directory_path}: {e}")

    def _get_aspect_variants(self, aspect: str) -> List[str]:
        """
        获取分析方面的常见变体，用于更灵活的文件名匹配

        Args:
            aspect: 分析方面名称 (如 'fps', 'lag', 'memory' 等)

        Returns:
            包含各种可能变体的列表
        """
        variants = [aspect]  # 包含原始名称

        # 定义常见的变体映射
        variant_mapping = {
            "fps": ["fps", "framerate", "frame_rate", "frame", "render"],
            "lag": ["lag", "latency", "delay", "ping", "network"],
            "memory": ["memory", "mem", "ram", "heap", "storage"],
            "power": ["power", "battery", "energy", "consumption", "watt"],
            "temperature": ["temperature", "temp", "thermal", "heat", "celsius"],
            "cpu": ["cpu", "processor", "compute", "core"],
            "gpu": ["gpu", "graphics", "render", "shader"],
            "disk": ["disk", "storage", "io", "read", "write"],
            "network": ["network", "net", "wifi", "connection", "bandwidth"],
        }

        # 如果找到映射，使用映射的变体
        if aspect in variant_mapping:
            variants.extend(variant_mapping[aspect])

        # 移除重复项并保持原始顺序
        seen = set()
        unique_variants = []
        for variant in variants:
            if variant not in seen:
                seen.add(variant)
                unique_variants.append(variant)

        return unique_variants

    def _detect_file_analysis_aspect(self, file_path: str) -> Optional[str]:
        """
        根据文件名检测该文件属于哪个分析方面

        Args:
            file_path: CSV/TSV文件路径

        Returns:
            匹配的分析方面名称，如果没有匹配则返回None
        """
        filename_lower = os.path.basename(file_path).lower()

        # 定义所有可能的分析方面
        all_aspects = [
            "fps",
            "lag",
            "memory",
            "power",
            "temperature",
            "cpu",
            "gpu",
            "disk",
            "network",
        ]

        for aspect in all_aspects:
            aspect_variants = self._get_aspect_variants(aspect)
            for variant in aspect_variants:
                if variant in filename_lower:
                    logger.debug(
                        f"文件 {os.path.basename(file_path)} 匹配分析方面 '{aspect}' (变体: {variant})"
                    )
                    return aspect

        logger.debug(f"文件 {os.path.basename(file_path)} 未匹配任何已知的分析方面")
        return None

    async def _worker(
        self, name: str, queue: asyncio.Queue, max_retries: int, results: list
    ):
        """带有重试机制的工作函数，用于处理队列中的文件。"""
        # post import
        from app.agent.DataAnalysisExpert import DataAnalysisExpert
        from app.agent.game_data_analysis import GameDataAnalysisAgent

        while True:
            try:
                file_path, metadata_path = await queue.get()
                logger.info(f"[{name}] picked up task: {file_path}")

                filename = os.path.basename(file_path)
                parent_dir = os.path.dirname(file_path)
                output_dir_name = os.path.splitext(filename)[0]
                output_dir = os.path.join(parent_dir, output_dir_name)
                total_file_path = (
                    list(Path(config.workspace_root).glob("*_cleaned.csv"))[0]
                    if config.workspace_root
                    else None
                )

                # 从config.workspace_root读取additional_request.json
                additional_request_path = os.path.join(
                    config.workspace_root, "additional_request.json"
                )
                additional_request = {}
                try:
                    with open(additional_request_path, "r", encoding="utf-8") as f:
                        additional_request = json.load(f)
                    logger.info(
                        f"[{name}] 成功读取额外请求配置: {additional_request_path}"
                    )
                except FileNotFoundError:
                    logger.debug(
                        f"[{name}] 未找到额外请求配置文件: {additional_request_path}"
                    )
                except Exception as e:
                    logger.warning(f"[{name}] 读取额外请求配置文件失败: {e}")

                # 检测当前文件属于哪个分析方面
                file_aspect = self._detect_file_analysis_aspect(file_path)
                current_additional_request = None
                if file_aspect and additional_request.get(file_aspect):
                    current_additional_request = additional_request[file_aspect]
                    logger.info(
                        f"[{name}] 文件 {filename} 属于 '{file_aspect}' 分析方面，找到额外请求"
                    )
                else:
                    logger.debug(
                        f"[{name}] 文件 {filename} 无对应的额外请求 (方面: {file_aspect})"
                    )

                retry_count = 0
                task_succeeded = False
                memory_path = None

                while not task_succeeded and retry_count < max_retries:
                    if retry_count > 0:
                        logger.warning(
                            f"[{name}] Retrying task for {file_path} (Attempt {retry_count + 1}/{max_retries})"
                        )
                        # Clean up directory for a fresh start
                        if os.path.exists(output_dir):
                            shutil.rmtree(output_dir)

                    os.makedirs(output_dir, exist_ok=True)

                    prompt = RUN_PROMPT_CH.format(
                        total_file_path=total_file_path,
                        csv_file_path=file_path,
                        output_folder_path=output_dir,
                        metadata_file_path=metadata_path if metadata_path else "N/A",
                    )

                    # 如果有额外请求，附加到prompt中
                    if current_additional_request:
                        prompt += f"\n\n## 额外分析要求 ({file_aspect}):\n{current_additional_request}"
                        logger.info(f"[{name}] 已将额外请求添加到prompt中")

                    logger.info(
                        f"[{name}] Running agent for {file_path} with prompt:\n{prompt}"
                    )

                    try:
                        # agent = GameDataAnalysisAgent()
                        agent = DataAnalysisExpert()
                        await agent.update_system_prompt(csv_file_path=file_path)
                        result = await agent.run(prompt)
                        memory_manager = AgentMemoryManager()

                        # --- Post-verification Logic ---
                        llm_verified = await self._verify_completion_with_llm(result)
                        rules_verified = self._verify_completion_with_rules(output_dir)

                        if llm_verified or rules_verified:
                            logger.info(
                                f"[{name}] Task for {file_path} completed and verified successfully."
                            )
                            # Save the final successful result
                            output_md_path = os.path.join(
                                output_dir, "analysis_summary.md"
                            )
                            with open(output_md_path, "w", encoding="utf-8") as f:
                                f.write(str(result))
                            memory_path = await memory_manager.save_memory(agent)

                            # 复制并修复Final_Report.md文件
                            copied_report_path = self._copy_and_fix_report(output_dir)
                            if copied_report_path:
                                logger.info(
                                    f"[{name}] 报告已复制并修复到: {copied_report_path}"
                                )
                            else:
                                logger.warning(f"[{name}] 报告复制失败: {output_dir}")

                            task_succeeded = True
                        else:
                            logger.warning(
                                f"[{name}] Verification failed for {file_path}. LLM: {llm_verified}, Rules: {rules_verified}."
                            )
                            retry_count += 1

                    except Exception as e:
                        logger.error(
                            f"[{name}] Agent execution failed for {file_path}: {e}",
                            exc_info=True,
                        )
                        retry_count += 1

                # 记录任务结果
                copied_report_path = None
                if task_succeeded:
                    # 如果任务成功，尝试获取复制的报告路径
                    parent_dir = os.path.dirname(output_dir)
                    dir_name = os.path.basename(output_dir)
                    potential_report_path = os.path.join(
                        parent_dir, f"{dir_name}_Final_Report.md"
                    )
                    if os.path.exists(potential_report_path):
                        copied_report_path = potential_report_path

                task_result = {
                    "file_path": file_path,
                    "output_dir": output_dir,
                    "memory_path": memory_path,
                    "copied_report_path": copied_report_path,
                    "success": task_succeeded,
                    "worker_name": name,
                }
                results.append(task_result)

                if not task_succeeded:
                    logger.error(
                        f"[{name}] Task for {file_path} failed permanently after {max_retries} attempts."
                    )

            except asyncio.CancelledError:
                logger.info(f"[{name}] was cancelled.")
                break
            except Exception as e:
                logger.error(
                    f"[{name}] encountered an unexpected error: {e}", exc_info=True
                )
                break
            finally:
                # This must be called for each item gotten from the queue
                queue.task_done()

    async def _verify_completion_with_llm(self, agent_result: str) -> bool:
        """通过询问LLM评估代理的结果来验证任务完成情况。"""
        if not agent_result or not isinstance(agent_result, str):
            return False

        verification_prompt = f"""
        Based on the following output from a data analysis agent, did the agent successfully complete its analysis task and generate a meaningful summary?
        The task was to analyze a CSV file and produce findings. A successful completion should contain clear conclusions, not just code or error messages.

        Agent's final output:
        ---
        {agent_result[-1000:]}
        ---

        Based on this output, was the task successfully completed? Please answer with only "true" or "false".
        """
        try:
            llm = LLM()  # Assumes a default LLM configuration
            response = await llm.ask(
                messages=[{"role": "user", "content": verification_prompt}]
            )
            return (
                response.strip().lower() == "true" or "true" in response.strip().lower()
            )
        except Exception as e:
            logger.error(f"LLM verification failed: {e}")
            return False  # Default to failure if LLM call fails

    def _verify_completion_with_rules(self, output_dir: str) -> bool:
        """通过检查输出目录是否非空以及是否存在Final_Report.md文件来验证任务完成情况。"""
        if not os.path.isdir(output_dir):
            return False

        # Check if there are any files or subdirectories in the output directory
        if not os.listdir(output_dir):
            return False

        # Check if Final_Report.md exists in the output directory
        final_report_path = os.path.join(output_dir, "Final_Report.md")
        if not os.path.exists(final_report_path):
            logger.warning(
                f"任务验证失败：在输出目录 {output_dir} 中未找到 Final_Report.md 文件"
            )
            return False

        # Additional check: ensure Final_Report.md is not empty
        try:
            with open(final_report_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    logger.warning(
                        f"任务验证失败：Final_Report.md 文件为空 {final_report_path}"
                    )
                    return False
        except Exception as e:
            logger.warning(
                f"任务验证失败：无法读取 Final_Report.md 文件 {final_report_path}: {e}"
            )
            return False

        logger.info(
            f"任务验证成功：在输出目录 {output_dir} 中找到有效的 Final_Report.md 文件"
        )
        return True

    def _copy_and_fix_report(self, output_dir: str) -> str | None:
        """
        复制Final_Report.md文件到上一级目录并修复图片引用路径

        Args:
            output_dir: 输出目录路径，包含Final_Report.md文件

        Returns:
            复制后的报告文件路径，如果失败则返回None
        """
        try:
            # 构建源文件路径
            source_report_path = os.path.join(output_dir, "Final_Report.md")
            if not os.path.exists(source_report_path):
                logger.warning(f"源报告文件不存在: {source_report_path}")
                return None

            # 获取目录信息
            parent_dir = os.path.dirname(output_dir)
            dir_name = os.path.basename(output_dir)

            # 构建目标文件路径
            target_report_name = f"{dir_name}_Final_Report.md"
            target_report_path = os.path.join(parent_dir, target_report_name)

            # 读取源文件内容
            with open(source_report_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 修复图片引用路径
            fixed_content = self._fix_image_references(content, output_dir)

            # 写入目标文件
            with open(target_report_path, "w", encoding="utf-8") as f:
                f.write(fixed_content)

            logger.info(f"报告复制成功: {source_report_path} -> {target_report_path}")
            return target_report_path

        except Exception as e:
            logger.error(f"复制并修复报告失败 {output_dir}: {e}", exc_info=True)
            return None

    def _fix_image_references(self, content: str, output_dir: str) -> str:
        """
        修复Markdown内容中的图片引用路径

        Args:
            content: 原始Markdown内容
            output_dir: 输出目录的绝对路径

        Returns:
            修复后的Markdown内容
        """
        import re

        # 匹配Markdown图片语法: ![alt_text](image_path)
        image_pattern = r"!\[([^\]]*)\]\(([^)]+)\)"

        def replace_image_path(match):
            alt_text = match.group(1)
            image_path = match.group(2)

            # 如果已经是绝对路径，则不修改
            if os.path.isabs(image_path):
                return match.group(0)

            # 构建绝对路径
            absolute_image_path = os.path.join(output_dir, image_path)

            # 检查文件是否存在
            if os.path.exists(absolute_image_path):
                return f"![{alt_text}]({absolute_image_path})"
            else:
                logger.warning(f"图片文件不存在: {absolute_image_path}")
                return match.group(0)  # 保持原样

        # 替换所有图片引用
        fixed_content = re.sub(image_pattern, replace_image_path, content)
        return fixed_content

    def _classify_all_success_result(
        self, result: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:

        classify_result = {
            "fps": [],
            "lag": [],
            "memory": [],
            "other": [],
            "power": [],
            "temperature": [],
        }

        """
        {
                    "file_path": file_path,
                    "output_dir": output_dir,
                    "memory_path": memory_path,
                    "copied_report_path": copied_report_path,
                    "success": task_succeeded,
                    "worker_name": name,
                }
        """

        for task in result:
            task_file_path = task.get("file_path", "")
            task_file_name = os.path.basename(task_file_path)
            for key in classify_result.keys():
                if key in task_file_name.lower():
                    classify_result[key].append(task)
                    break

        return classify_result

    async def _generate_category_summary_reports(
        self, classify_result: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Optional[str]]:
        """
        为每个分类生成综合汇总报告

        Args:
            classify_result: 分类后的任务结果

        Returns:
            每个分类生成的报告文件路径字典
        """
        generated_reports = {}

        for category, tasks in classify_result.items():
            if not tasks:  # 跳过空分类
                generated_reports[category] = None
                continue

            # 只处理成功的任务
            successful_tasks = [task for task in tasks if task.get("success", False)]
            if not successful_tasks:
                generated_reports[category] = None
                continue

            logger.info(
                f"开始为 {category.upper()} 分类生成汇总报告，包含 {len(successful_tasks)} 个成功任务"
            )

            try:
                report_path = await self._create_category_summary_report(
                    category, successful_tasks
                )
                generated_reports[category] = report_path
                if report_path:
                    logger.info(
                        f"{category.upper()} 分类汇总报告生成成功: {report_path}"
                    )

                    # 使用LLM对生成的汇总报告进行深度分析
                    aggregation_success = await self._aggragete_categoty_report(
                        report_path
                    )
                    if aggregation_success:
                        logger.info(f"{category.upper()} 分类报告LLM汇总分析完成")
                    else:
                        logger.warning(f"{category.upper()} 分类报告LLM汇总分析失败")
                else:
                    logger.warning(f"{category.upper()} 分类汇总报告生成失败")
            except Exception as e:
                logger.error(
                    f"生成 {category.upper()} 分类汇总报告时发生错误: {e}",
                    exc_info=True,
                )
                generated_reports[category] = None

        return generated_reports

    async def _create_category_summary_report(
        self, category: str, tasks: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        为特定分类创建汇总报告

        Args:
            category: 分类名称 (fps, lag, memory, etc.)
            tasks: 该分类下的成功任务列表

        Returns:
            生成的汇总报告文件路径
        """
        try:
            # 收集所有报告内容
            device_reports = {}

            for task in tasks:
                copied_report_path = task.get("copied_report_path")
                if not copied_report_path or not os.path.exists(copied_report_path):
                    continue

                # 从文件名提取设备型号信息
                report_filename = os.path.basename(copied_report_path)
                # 例如：fps_1_subtable_Final_Report.md -> 设备型号: 1
                device_id = self._extract_device_id_from_filename(report_filename)

                # 读取报告内容
                with open(copied_report_path, "r", encoding="utf-8") as f:
                    report_content = f.read()

                device_reports[device_id] = {
                    "content": report_content,
                    "file_path": task.get("file_path", ""),
                    "output_dir": task.get("output_dir", ""),
                    "report_path": copied_report_path,
                }

            if not device_reports:
                logger.warning(f"没有找到 {category.upper()} 分类的有效报告文件")
                return None

            # 生成汇总报告
            summary_content = await self._generate_summary_content(
                category, device_reports
            )

            # 确定汇总报告的保存位置
            first_task = tasks[0]
            base_dir = os.path.dirname(
                os.path.dirname(first_task.get("output_dir", ""))
            )
            summary_report_path = os.path.join(
                base_dir, f"{category.upper()}_Summary_Report.md"
            )

            # 写入汇总报告
            with open(summary_report_path, "w", encoding="utf-8") as f:
                f.write(summary_content)

            return summary_report_path

        except Exception as e:
            logger.error(
                f"创建 {category.upper()} 分类汇总报告失败: {e}", exc_info=True
            )
            return None

    def _extract_device_id_from_filename(self, filename: str) -> str:
        """
        从文件名中提取设备ID

        Args:
            filename: 报告文件名，如 fps_1_subtable_Final_Report.md

        Returns:
            设备ID，如 "1"
        """
        import re

        # 匹配模式：分类_数字_subtable
        pattern = r"(\w+)_(\d+)_subtable"
        match = re.search(pattern, filename.lower())

        if match:
            return match.group(2)  # 返回数字部分
        else:
            # 如果无法提取，使用文件名作为标识
            return filename.replace("_Final_Report.md", "").replace("_", "-")

    async def _generate_summary_content(
        self, category: str, device_reports: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        生成分类汇总报告内容

        Args:
            category: 分类名称
            device_reports: 设备报告字典

        Returns:
            汇总报告的Markdown内容
        """
        from datetime import datetime

        # 报告头部
        summary_lines = [
            f"# {category.upper()} 性能分析汇总报告",
            "",
            f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**分析分类:** {category.upper()}",
            f"**设备数量:** {len(device_reports)}",
            "",
            "## 报告概述",
            "",
            f"本报告汇总了 {len(device_reports)} 个设备的 {category.upper()} 性能分析结果。",
            "每个设备的详细分析如下：",
            "",
            "---",
            "",
        ]

        # 为每个设备添加报告内容
        for device_id, report_data in sorted(device_reports.items()):
            summary_lines.extend(
                [
                    f"## 设备 {device_id} - {category.upper()} 性能分析",
                    "",
                    f"**原始文件:** `{os.path.basename(report_data['file_path'])}`",
                    f"**输出目录:** `{report_data['output_dir']}`",
                    f"**详细报告:** `{report_data['report_path']}`",
                    "",
                    "### 分析结果",
                    "",
                ]
            )

            # 添加原始报告内容，但去除重复的标题
            content = report_data["content"]
            content = self._clean_report_content_for_summary(content, device_id)

            summary_lines.append(content)
            summary_lines.extend(["", "---", ""])

        # 添加汇总结论部分
        summary_lines.extend(
            [
                "## 总体结论",
                "",
                f"通过对 {len(device_reports)} 个设备的 {category.upper()} 性能分析，我们得到了各设备的详细性能表现。",
                "具体的性能对比和建议请参考各设备的详细分析结果。",
                "",
                "### 设备列表",
                "",
            ]
        )

        # 添加设备列表
        for device_id, report_data in sorted(device_reports.items()):
            summary_lines.append(
                f"- **设备 {device_id}**: {os.path.basename(report_data['file_path'])}"
            )

        summary_lines.extend(
            ["", f"*报告生成完成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"]
        )

        return "\n".join(summary_lines)

    def _clean_report_content_for_summary(self, content: str, device_id: str) -> str:
        """
        清理报告内容，使其适合包含在汇总报告中

        Args:
            content: 原始报告内容
            device_id: 设备ID

        Returns:
            清理后的内容
        """
        lines = content.split("\n")
        cleaned_lines = []

        skip_main_title = True  # 跳过主标题

        for line in lines:
            # 跳过顶级标题（通常是 # 开头的）
            if line.startswith("# ") and skip_main_title:
                skip_main_title = False
                continue

            # 调整标题级别（将 ## 改为 ###，### 改为 ####）
            if line.startswith("## "):
                line = "###" + line[2:]
            elif line.startswith("### "):
                line = "####" + line[3:]

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    async def _aggragete_categoty_report(self, report_path: str) -> bool:
        """
        使用LLM对分类汇总报告进行深度分析，并将结论追加到报告末尾

        Args:
            report_path: 分类汇总报告的文件路径

        Returns:
            是否成功追加汇总结论
        """
        try:
            if not os.path.exists(report_path):
                logger.warning(f"报告文件不存在: {report_path}")
                return False

            # 读取现有报告内容
            with open(report_path, "r", encoding="utf-8") as f:
                current_content = f.read()

            # 提取分类名称
            report_filename = os.path.basename(report_path)
            category = report_filename.replace("_Summary_Report.md", "").lower()

            logger.info(f"开始使用LLM分析 {category.upper()} 分类汇总报告")

            # 使用LLM进行汇总分析
            aggregation_result = await self._perform_llm_aggregation(
                current_content, category
            )

            if not aggregation_result:
                logger.warning(f"LLM汇总分析失败: {report_path}")
                return False

            # 构建要追加的内容
            from datetime import datetime

            append_content = [
                "",
                "---",
                "",
                "## 🤖 AI智能汇总分析",
                "",
                f"**分析时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**分析对象:** {category.upper()} 性能数据",
                "",
                "### 综合分析结论",
                "",
                aggregation_result,
                "",
                "---",
                "",
                "*本分析由AI系统基于以上数据自动生成，旨在提供客观的性能评估和建议。*",
            ]

            # 将汇总结论追加到报告末尾
            updated_content = current_content + "\n".join(append_content)

            # 写回文件
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(updated_content)

            logger.info(f"LLM汇总分析已追加到报告: {report_path}")
            return True

        except Exception as e:
            logger.error(f"LLM汇总分析失败 {report_path}: {e}", exc_info=True)
            return False

    async def _perform_llm_aggregation(
        self, report_content: str, category: str
    ) -> Optional[str]:
        """
        使用LLM对报告内容进行汇总分析

        Args:
            report_content: 完整的汇总报告内容
            category: 分类名称 (fps, lag, memory, etc.)

        Returns:
            LLM生成的汇总分析结论
        """
        try:
            # 截取报告内容以避免超出token限制
            content_for_analysis = report_content

            # 构建专门的汇总提示词
            aggregation_prompt = AGGREGATION_REPORT_PROMPT.format(
                category=category.upper(), report_content=content_for_analysis
            )

            llm = LLM()
            response = await llm.ask(
                messages=[Message.user_message(aggregation_prompt)]
            )

            if not response or len(response.strip()) < 50:
                logger.warning(f"LLM返回的汇总分析过短或为空: {category}")
                return None

            logger.info(f"LLM成功生成 {category.upper()} 分类的汇总分析")
            return response.strip()

        except Exception as e:
            logger.error(f"LLM汇总分析执行失败 {category}: {e}", exc_info=True)
            return None
