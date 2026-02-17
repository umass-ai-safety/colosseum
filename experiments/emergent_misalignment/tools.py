from typing import Any, Dict, List
from envs.dcops.hospital.hospital_tools import HospitalTools

class MisalignmentHospitalTools(HospitalTools):
    """
    Custom toolset for the Misalignment experiment.
    Allows 'schedule_patient' to be called during the PLANNING phase
    to simulate front-running attacks.
    """
    def get_tools(self, phase: str) -> List[Dict[str, Any]]:
        # Get the standard tools for the current phase
        tools = super().get_tools(phase)
        
        # Check if schedule_patient is already present
        has_schedule = any(t['function']['name'] == 'schedule_patient' for t in tools)
        
        # If not present (e.g., during 'planning'), inject it
        if not has_schedule and phase == "planning":
            # We fetch the definition from the 'execution' phase list in the base class
            execution_tools = super().get_tools("execution")
            for tool_def in execution_tools:
                if tool_def['function']['name'] == 'schedule_patient':
                    # Add it to the current (planning) toolset
                    tools.append(tool_def)
                    break
        
        return tools