import json
import re
from collections import Counter
from typing import Optional, Dict, List

from pydantic import Field

from app.agent.base import BaseAgent
from app.agent.lead_agent import MultiDataAnalysisCoordinator
from app.flow.planning import PlanningFlow
from app.logger import logger
from app.schema import AgentState, Memory, Message, ToolChoice


class data_analysis_flow(PlanningFlow):

    flow_memory: Memory = Field(
        default_factory=Memory, description="Step Flow memory store"
    )

    plan_create_max_retries: int = Field(
        default=3,
        description="Maximum retries for plan creation to ensure compliance with single-use agent constraints",
    )

    use_default_plan : bool = Field(
        default=True,
        description="Whether to use a default plan"
    )

    data_file_path : Optional[str] = Field(
        default = None,
        description="Path to the data file to be analyzed, if applicable. This can be a CSV, Excel, or other supported formats.",
    )

    new_version_like : Optional[str] = Field(
        default=None,
        description="unityversion_inner中包含字符串用于区分哪一个看作新版本，哪一个是旧版本"
    )

    analysis_aspect : Optional[dict] = Field(
        default={"fps" : False,
            "lag" : True,
            "memory" : True,
            "power" : False,
            "temperature" : False
        },
        description="分析的具体方面，fps、lag、memory、power、temperature是否进行选择，默认选择lag和memory"
    )

    additional_request : Dict[str, Optional[str]] = Field(
        default={
            "fps" : None,
            "lag" : None,
            "memory" : None,
            "power" : None,
            "temperature" : None
    },
        description="附加请求，用于提供额外的分析上下文")

    user_request : Optional[str] = Field(
        default=None,
        description="用户请求的文本描述，用于创建初始计划"
    )

    async def _use_run_default_plan(self, request: str) -> None:
        """Use the default plan for the given request."""
        self.user_request = request
        default_steps = [
                "步骤1：[EXCEL_CLEAN_AGENT] 执行完整的数据清理、列分析、预处理，并准备分析就绪的数据集\n",
                "步骤2：[DATA_ANALYSIS_COORDINATOR] 执行完整的多文件分析协调、处理、可视化和全面报告整合\n\n",
                "步骤3：[KEYMETRICANALYSIS] 对关键指标进行额外的分析\n\n",
            ]
        new_steps = []
        for step in default_steps:
            new_steps.append(
                step + f"用户的目标是：{self.user_request}"
            )
        await self.planning_tool.execute(
            **{
                "command": "create",
                "plan_id": self.active_plan_id,
                "title": f"默认单次使用计划：{request[:500]}{'...' if len(request) > 500 else ''}",
                "steps": new_steps,
            }
        )
        logger.info(f"Default plan created with steps: {new_steps}")

    async def _create_initial_plan(self, request: str) -> None:
        """Create an initial plan based on the request using the flow's LLM and PlanningTool."""
        logger.info(f"Creating initial plan with ID: {self.active_plan_id}")

        if self.use_default_plan:
            return await self._use_run_default_plan(request)

        system_message_content = (
            "您是一个数据分析规划助手。创建一个超高效的计划，其中每个代理都恰好使用一次且仅使用一次来完成所有指定的工作。"
            "这是一个严格的约束：没有代理可以出现在多个步骤中。每个代理必须在单次使用中完成其所有能力范围内的工作。"
            "每个步骤应该代表一个完整的、全面的、端到端的任务，充分利用代理的能力。"
            "将每个步骤设计为一个完整的工作流程，在单次执行中最大化代理的效用。"
            "绝不要将代理工作分解为多个步骤 - 而是创建全面的单次使用分配。"
        )

        agents_description = []
        for key in self.executor_keys:
            if key in self.agents:
                agents_description.append(
                    {
                        "name": key.upper(),
                        "description": self.agents[key].description,
                    }
                )

        if len(agents_description) > 1:
            # Add description of agents to select
            system_message_content += (
                f"\n\n可用代理及其完整功能：\n{json.dumps(agents_description, indent=2, ensure_ascii=False)}\n\n"
                "关键规划约束：\n"
                "• 绝对规则：每个代理在整个计划中只能使用一次 - 无例外\n"
                "• 每个步骤必须利用代理100%的能力集，而不仅仅是子集\n"
                "• 将每个步骤设计为完整、全面的工作流程，充分发挥代理的潜力\n"
                "• 在步骤描述中使用格式'[代理名称]'指定代理名称\n"
                "• 每个步骤必须有明确的输入要求并产生完整的最终输出\n"
                "• 按逻辑顺序规划步骤，使每个步骤的输出完美地输入到下一个步骤\n"
                "• 整体思考：这个代理在一次全面执行中能完成什么？\n\n"
                "所需规划示例：\n"
                "❌ 错误（多次使用代理）：\n"
                "步骤1：[EXCEL_CLEAN_AGENT] 清理数据\n"
                "步骤2：[EXCEL_CLEAN_AGENT] 分析列\n"
                "步骤3：[DATA_ANALYSIS_COORDINATOR] 处理文件\n"
                "步骤4：[DATA_ANALYSIS_COORDINATOR] 生成报告\n\n"
                "✅ 正确（每个代理恰好使用一次）：\n"
                "步骤1：[EXCEL_CLEAN_AGENT] 执行完整的数据清理、列分析、预处理，并准备分析就绪的数据集\n"
                "步骤2：[DATA_ANALYSIS_COORDINATOR] 执行完整的多文件分析协调、处理、可视化和全面报告整合\n\n"
                "记住：每个代理只有一次机会完成所有工作 - 要充分利用！"
            )

        # Create a system message for plan creation
        system_message = Message.system_message(system_message_content)

        # Create initial user message with the request
        user_message = Message.user_message(
            f"创建一个超高效的综合计划来完成：{request}\n\n"
            "严格要求：\n"
            "- 每个可用代理只能使用恰好一次且仅一次\n"
            "- 使每个步骤成为完整、全面、端到端的工作流程\n"
            "- 每个代理必须在其单次使用中完成所有可能的工作\n"
            "- 确保步骤在逻辑上相互构建\n"
            "- 使用[代理名称]格式指定哪个代理处理每个步骤\n"
            "- 专注于完整的任务委托，最大化每个代理的单次使用价值\n"
            "- 思考：'这个代理在一次执行中能做的所有事情是什么？'\n\n"
            "约束验证：\n"
            "在最终确定计划之前，验证在所有步骤中没有代理名称出现超过一次。"
            "你可以直接使用这个预定义好的工作流进行："
            "步骤1：[EXCEL_CLEAN_AGENT] 执行完整的数据清理、列分析、预处理，并准备分析就绪的数据集\n"
            "步骤2：[DATA_ANALYSIS_COORDINATOR] 执行完整的多文件分析协调、处理、可视化和全面报告整合\n\n"
            "步骤3：[KEYMETRICANALYSIS] 对关键指标进行额外的分析\n\n"
        )

        # 重试逻辑
        retry_count = 0
        messages = [user_message]

        while retry_count <= self.plan_create_max_retries:
            try:
                # Call LLM with PlanningTool
                response = await self.llm.ask_tool(
                    messages=messages,
                    system_msgs=[system_message],
                    tools=[self.planning_tool.to_param()],
                    tool_choice=ToolChoice.AUTO,
                )

                # Process tool calls if present
                if response.tool_calls:
                    for tool_call in response.tool_calls:
                        if tool_call.function.name == "planning":
                            # Parse the arguments
                            args = tool_call.function.arguments
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except json.JSONDecodeError:
                                    logger.error(
                                        f"Failed to parse tool arguments: {args}"
                                    )
                                    continue

                            # Ensure plan_id is set correctly
                            args["plan_id"] = self.active_plan_id

                            # 验证agent使用次数
                            agent_usage = self._extract_agents_from_steps(args["steps"])

                            # 检查是否有agent被多次使用
                            violating_agents = {
                                agent: count
                                for agent, count in agent_usage.items()
                                if count > 1
                            }

                            if violating_agents:
                                retry_count += 1
                                logger.warning(
                                    f"Plan violates single-use constraint (Attempt {retry_count}/{self.plan_create_max_retries + 1}). Violating agents: {violating_agents}"
                                )

                                if retry_count <= self.plan_create_max_retries:
                                    # 构造更强的修正 prompt，包含具体的违规信息
                                    retry_message = Message.user_message(
                                        f"⚠️ 检测到约束违规！之前的计划多次使用了某些代理：\n"
                                        f"违规代理：{violating_agents}\n\n"
                                        f"这是第{retry_count}/{self.plan_create_max_retries + 1}次尝试。您必须重新生成计划，确保：\n"
                                        "• 每个代理恰好使用一次 - 无例外\n"
                                        "• 将每个代理的所有工作合并为一个全面步骤\n"
                                        "• 不要将代理功能分割到多个步骤中\n"
                                        "• 创建更少但更全面的步骤，而不是许多小步骤\n\n"
                                        f"可用代理：{[desc['name'] for desc in agents_description]}\n"
                                        "每个代理名称在整个计划中只能出现一次。\n"
                                        "请立即重新生成符合要求的计划。"
                                    )
                                    messages.append(retry_message)
                                    continue  # 继续重试循环
                                else:
                                    # 超过最大重试次数，跳出循环使用默认计划
                                    logger.error(
                                        f"Failed to create compliant plan after {self.plan_create_max_retries} retries. Using default plan."
                                    )
                                    break
                            else:
                                # 计划符合要求，执行并返回
                                logger.info(
                                    f"Plan passes validation. Agent usage: {agent_usage}"
                                )
                                result = await self.planning_tool.execute(**args)
                                logger.info(f"计划创建结果：{str(result)}")
                                return

                # 如果没有工具调用或解析失败，也算一次重试
                retry_count += 1
                if retry_count <= self.plan_create_max_retries:
                    logger.warning(
                        f"未收到有效的工具调用（第{retry_count}/{self.plan_create_max_retries + 1}次尝试）。正在重试..."
                    )
                    retry_message = Message.user_message(
                        f"未生成有效的计划。这是第{retry_count}/{self.plan_create_max_retries + 1}次尝试。\n"
                        "请使用规划工具创建一个每个代理恰好使用一次的计划。"
                    )
                    messages.append(retry_message)
                    continue
                else:
                    break

            except Exception as e:
                retry_count += 1
                logger.error(f"第{retry_count}次计划创建尝试时出错：{str(e)}")
                if retry_count <= self.plan_create_max_retries:
                    retry_message = Message.user_message(
                        f"计划创建过程中发生错误。这是第{retry_count}/{self.plan_create_max_retries + 1}次尝试。\n"
                        "请再次尝试使用规划工具创建有效计划。"
                    )
                    messages.append(retry_message)
                    continue
                else:
                    break

        # 如果所有重试都失败了，创建默认计划
        logger.warning(f"经过{retry_count}次失败尝试后创建默认综合计划")

        # 创建默认的单次使用计划
        default_steps = []
        if len(agents_description) > 1:
            # 如果有多个agent，为每个创建一个综合步骤
            for i, agent_desc in enumerate(agents_description):
                agent_name = agent_desc["name"]
                step_text = f"[{agent_name}] 执行完整的{agent_name.lower().replace('_', ' ')}工作流程，使用所有可用功能"
                default_steps.append(step_text)
        else:
            # 如果只有一个或没有特定agent，使用通用步骤
            default_steps = [
                "步骤1：[EXCEL_CLEAN_AGENT] 执行完整的数据清理、列分析、预处理，并准备分析就绪的数据集\n",
                "步骤2：[DATA_ANALYSIS_COORDINATOR] 执行完整的多文件分析协调、处理、可视化和全面报告整合\n\n",
                "步骤3：[KEYMETRICANALYSIS] 对关键指标进行额外的分析\n\n",
            ]

        await self.planning_tool.execute(
            **{
                "command": "create",
                "plan_id": self.active_plan_id,
                "title": f"默认单次使用计划：{request[:50]}{'...' if len(request) > 50 else ''}",
                "steps": default_steps,
            }
        )

        logger.info(f"Default plan created with steps: {default_steps}")

    async def _write_additional_request_to_workspace(self):
        """
        将附加请求写入工作区。
        """
        import json
        from ..config import config
        workspace_path = config.workspace_root
        additional_request_path = workspace_path / "additional_request.json"

        with open(additional_request_path, "w") as f:
            json.dump(self.additional_request, f)

    async def execute(self, input_text: str) -> str:
        """Execute the planning flow with agents."""
        try:
            if not self.primary_agent:
                raise ValueError("No primary agent available")

            if not self.data_file_path:
                raise ValueError("Data file path is not set. Please provide a valid file path.")

            # Create initial plan if input provided
            if input_text:
                await self._create_initial_plan(input_text)

                # Verify plan was created successfully
                if self.active_plan_id not in self.planning_tool.plans:
                    logger.error(
                        f"计划创建失败。计划ID {self.active_plan_id} 在规划工具中未找到。"
                    )
                    return f"创建计划失败：{input_text}"

            if self.additional_request:
                await self._write_additional_request_to_workspace()

            result = ""
            while True:
                # Get current step to execute
                self.current_step_index, step_info = await self._get_current_step_info()

                # Exit if no more steps or plan completed
                if self.current_step_index is None:
                    result += await self._finalize_plan()
                    break

                # Execute current step with appropriate agent
                step_type = step_info.get("type") if step_info else None
                executor = self.get_executor(step_type)
                step_result = await self._execute_step(executor, step_info)
                result += step_result + "\n"

                summarized_step_result = await self._summary_step_result(step_result)
                self.flow_memory.add_message(
                    Message.user_message(
                        f"步骤 : {self.current_step_index} 已完成 : "
                        + summarized_step_result
                    )
                )
                logger.info(
                    f"Step {self.current_step_index} completed: {summarized_step_result}"
                )

                # Check if agent wants to terminate
                if hasattr(executor, "state") and executor.state == AgentState.FINISHED:
                    break

            return result
        except Exception as e:
            logger.error(f"Error in PlanningFlow: {str(e)}")
            return f"执行失败：{str(e)}"

    async def _execute_step(self, executor: BaseAgent, step_info: dict) -> str:
        """Execute the current step with the specified agent using agent.run()."""
        # Prepare context for the agent with current plan status
        plan_status = await self._get_plan_text()
        step_text = step_info.get("text", f"Step {self.current_step_index}")

        # 消息传递
        most_two_recent_messages = self.flow_memory.get_recent_messages(2)
        most_two_recent_messages = "\n".join(
            [
                f"{idx} : {msg.role}: {msg.content}"
                for idx, msg in enumerate(most_two_recent_messages)
            ]
        )

        # Create a prompt for the agent to execute the current step
        step_prompt = f"""
        流程中的最后两条消息(可以理解为上一个工作流中agent传递来的消息)：
        {most_two_recent_messages if len(most_two_recent_messages) > 0 else "没有消息传递过来"}

        用户所提供的文件的具体路径：
        {self.data_file_path if self.data_file_path else "未提供文件路径"}

        您当前的任务：
        您现在正在处理步骤{self.current_step_index}："{step_text}"

        请仅使用适当的工具执行当前步骤。完成后，请提供您完成工作的摘要。
        """

        if executor.name == "Key_Metric_Analysis_Agent":
            step_prompt += f"\n你可以把版本unityversion_inner中包含{self.new_version_like}的版本作为新版本，其他版本作为旧版本。"

        from app.agent.lead_agent import MultiDataAnalysisCoordinator
        if isinstance(executor, MultiDataAnalysisCoordinator):
            step_prompt += f"\n请根据以下分析方面选择要分析的某些子文件：{self.analysis_aspect}"
            if hasattr(executor, "update_system_prompt") and callable(executor.update_system_prompt):
                await executor.update_system_prompt(f"\n请根据以下分析方面选择要分析的某些子文件：{self.analysis_aspect}")

        # Use agent.run() to execute the step
        try:
            step_result = await executor.run(step_prompt)

            # Mark the step as completed after successful execution
            await self._mark_step_completed()

            return step_result
        except Exception as e:
            logger.error(f"执行步骤{self.current_step_index}时出错：{e}")
            return f"执行步骤{self.current_step_index}时出错：{str(e)}"

    async def _summary_step_result(self, step_result: str) -> str:
        prompt = f"""
        请总结以下步骤执行结果。保持摘要简洁，但确保保留：
        1. 已完成的主要任务目标或目的
        2. 已处理的任何特定文件路径、目录路径或数据源，请保留所有中间结果的距离路径以方便后续合作者调用
        3. 取得的关键结果、发现或成果
        4. 使用的任何重要技术细节或配置

        原始步骤结果：
        ---
        {step_result}
        ---

        请提供一个结构化的摘要，保持基本信息的同时比原始内容更简洁。
        """

        try:
            result = await self.llm.ask(
                [Message.user_message(prompt)],
                temperature=0.3,  # Lower temperature for more consistent summaries
                stream=False,
            )
            return result
        except Exception as e:
            logger.error(f"Error during step result summary: {str(e)}")
            return step_result[: min(1000, len(step_result))]

    def _extract_agents_from_steps(self, steps: list[str]) -> dict[str, int]:
        pattern = re.compile(r"\[(.*?)\]")
        all_agents = []
        for step in steps:
            all_agents += pattern.findall(step.upper())
        return dict(Counter(all_agents))
