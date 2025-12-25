from enum import Enum
from typing import Dict, List, Union

from app.agent.base import BaseAgent
from app.flow.base import BaseFlow
from app.flow.data_analysis_flow import data_analysis_flow
from app.flow.planning import PlanningFlow


class FlowType(str, Enum):
    PLANNING = "planning"
    GAME_DATA_ANALYSIS = "game_data_analysis"
    DATA_ANALYSIS_FLOW = "data_analysis_flow"


class FlowFactory:
    """Factory for creating different types of flows with support for multiple agents"""

    @staticmethod
    def create_flow(
        flow_type: FlowType,
        agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]],
        **kwargs,
    ) -> BaseFlow:
        flows = {
            FlowType.PLANNING: PlanningFlow,
            FlowType.GAME_DATA_ANALYSIS: data_analysis_flow,
            FlowType.DATA_ANALYSIS_FLOW: data_analysis_flow,
        }

        flow_class = flows.get(flow_type)
        if not flow_class:
            raise ValueError(f"Unknown flow type: {flow_type}")

        return flow_class(agents, **kwargs)
