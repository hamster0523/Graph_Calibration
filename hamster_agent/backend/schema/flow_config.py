from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

class FlowConfiguration(BaseModel):
    """Flow configuration request"""
    flow_type: str = Field(..., description="Type of flow to configure")
    primary_agent: str = Field(..., description="Primary agent to use")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Additional flow parameters")

class FlowConfigResponse(BaseModel):
    """Flow configuration response"""
    success: bool = Field(..., description="Whether configuration was successful")
    message: str = Field(..., description="Status message")
    current_flow: Optional[str] = Field(default=None, description="Currently configured flow")
    current_agent: Optional[str] = Field(default=None, description="Currently configured agent")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Current flow parameters")