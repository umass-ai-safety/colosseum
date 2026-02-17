from typing import Any, Dict, Optional
from envs.dcops.hospital.hospital_env import HospitalEnvironment
from experiments.agent_misalignment.tools import MisalignmentHospitalTools

class MisalignmentHospitalEnvironment(HospitalEnvironment):
    """
    Custom Hospital Environment for Misalignment experiments.
    Uses MisalignmentHospitalTools to allow tool execution during planning.
    """
    def __init__(self, communication_protocol, config, tool_logger):
        super().__init__(communication_protocol, config, tool_logger)
        # Override the tool implementation with our permissive version
        # The communication_protocol acts as the blackboard_manager
        self.tools_impl = MisalignmentHospitalTools(self.communication_protocol)
        
    def get_tools(self, phase: str):
        return self.tools_impl.get_tools(phase)

    def handle_tool_call(
        self, 
        tool_name: str, 
        agent_name: str, 
        arguments: Dict[str, Any], 
        phase: Optional[str] = None, 
        iteration: Optional[int] = None
    ) -> Dict[str, Any]:
        return self.tools_impl.handle_tool_call(
            tool_name, 
            agent_name, 
            arguments, 
            phase, 
            iteration, 
            env_state=self.get_serializable_state()
        )
