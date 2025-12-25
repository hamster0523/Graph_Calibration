from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from pydantic import Field

from app.agent.base import BaseAgent
from app.llm import LLM
from app.schema import AgentState, Memory, Message
from app.logger import logger

SUMMARIZE_PROMPT = """你是一个专业的AI代理工作总结助手。请基于以下代理的执行轨迹，生成一个简洁而全面的工作总结报告。

        **任务要求：**
        1. 分析代理的整个工作流程和执行步骤
        2. 识别代理使用的工具和执行的关键操作
        3. 总结代理完成的主要任务和取得的成果
        4. 指出工作过程中的重要决策点和解决方案
        5. 评估整体执行效果和结果质量

        **输出格式：**
        请按照以下结构生成总结报告：

        ## 🎯 任务概述
        - 代理执行的主要任务
        - 处理的数据类型和规模

        ## 🔧 执行流程
        - 关键步骤和操作序列
        - 使用的工具和方法
        - 重要的决策点

        ## 📊 主要成果
        - 完成的具体工作
        - 生成的文件和输出
        - 数据处理结果

        ## 💡 关键洞察
        - 工作过程中的重要发现
        - 解决的问题和挑战
        - 采用的策略和方法

        ## ✅ 执行总结
        - 整体完成状态
        - 结果质量评估
        - 后续建议（如有）

        **代理执行轨迹：**
        {memory_trace}

        请基于以上轨迹生成一份专业的工作总结报告。"""

class ReActAgent(BaseAgent, ABC):
    name: str
    description: Optional[str] = None

    system_prompt: Optional[str] = None
    next_step_prompt: Optional[str] = None

    llm: Optional[LLM] = Field(default_factory=LLM)
    memory: Memory = Field(default_factory=Memory)
    state: AgentState = AgentState.IDLE

    max_steps: int = 10
    current_step: int = 0

    @abstractmethod
    async def think(self) -> bool:
        """Process current state and decide next action"""

    @abstractmethod
    async def act(self) -> str:
        """Execute decided actions"""

    async def update_system_memory(self, *args, **kwargs):
        """Update the system memory with new information."""
        pass

    async def step(self) -> str:
        """Execute a single step: think and act."""
        should_act = await self.think()
        if not should_act:
            return "Thinking complete - no action needed"
        return await self.act()

    async def _summarize_agent_work(self) -> str:
        """Generate a comprehensive summary of the agent's work based on memory trace."""

        memory_dict: List[Dict[str, Any]] = self.memory.model_dump()['messages']
        memory_str = self._form_memory_dict(memory_dict)

        # Use the LLM to summarize the agent's work
        prompt = SUMMARIZE_PROMPT.format(memory_trace=memory_str)

        try:
            summary = await self.llm.ask(
                [Message.user_message(content=prompt)],
                stream=False,
                temperature=0.8)
            logger.info(f"Agent {self.name} work summary : {summary}")
            return summary
        except Exception as e:
            # Fallback to a simple summary if LLM fails
            logger.error(f"LLM summary failed with exception : {e}, using simple summary fallback.")
            return self._generate_simple_summary(memory_dict)

    def _generate_simple_summary(self, memory_dict: List[Dict[str, Any]]) -> str:
        """Generate a simple fallback summary when LLM is unavailable."""

        total_messages = len(memory_dict)
        tool_calls = []
        user_queries = []
        assistant_responses = []

        for memo in memory_dict:
            role = memo.get('role', 'unknown')
            content = memo.get('content', '')

            if role == 'user':
                user_queries.append(content[:100] + "..." if len(content) > 100 else content)
            elif role == 'assistant':
                assistant_responses.append(content[:100] + "..." if len(content) > 100 else content)

            # Extract tool calls
            tool_calls_in_msg = memo.get('tool_calls', [])
            for tool_call in tool_calls_in_msg:
                func = tool_call.get('function', {})
                func_name = func.get('name', 'unknown_function')
                tool_calls.append(func_name)

        unique_tools = list(set(tool_calls))

        summary = f"""## 🎯 代理工作总结

            **执行统计:**
            - 处理的消息总数: {total_messages}
            - 用户查询: {len(user_queries)}
            - 助手响应: {len(assistant_responses)}
            - 使用的工具: {len(unique_tools)}

            **执行的工具:**
            {chr(10).join(f"- {tool}" for tool in unique_tools)}

            **关键交互:**
            {chr(10).join(f"- 用户: {query}" for query in user_queries[:3])}

            **最近的响应:**
            {chr(10).join(f"- 助手: {response}" for response in assistant_responses[-2:])}

            **状态:** 执行完成，共进行了 {len(tool_calls)} 次工具调用，涉及 {len(unique_tools)} 个不同的工具。
            """

        return summary

    def _form_memory_dict(self, memory_dict: List[Dict[str, Any]]) -> str:
        """Format memory dictionary into a readable string for LLM processing."""

        final_str = ""
        for i, memo in enumerate(memory_dict, 1):
            role = memo.get('role', 'unknown_role')
            content = memo.get('content', '')

            final_str += f"--- Step {i} ---\n"
            final_str += f"Role: {role}\n"
            final_str += f"Content: {content}\n"

            tool_calls = memo.get('tool_calls', [])
            if tool_calls:
                final_str += "Tool Calls:\n"
                for j, tool in enumerate(tool_calls, 1):
                    func = tool.get('function', {})
                    func_name = func.get('name', 'unknown_function')
                    func_args = func.get('arguments', '{}')
                    final_str += f"  {j}. Function: {func_name}\n"
                    final_str += f"     Arguments: {func_args}\n"

            final_str += "\n"

        return final_str
