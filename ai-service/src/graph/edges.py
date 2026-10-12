from typing import List
from src.graph.state import AgentState

def route_to_specialists(state: AgentState) -> List[str]:
    """
    Reads the Planner Agent's output and determines which specialist nodes to execute in parallel.
    Returns a list of node names corresponding to the active medical domains.
    """
    selected_specialists = state.get("active_specialists", [])
    
    # Map the Pydantic output strings directly to the registered LangGraph node names
    routing_map = {
        "Cardiology": "cardiology_node",
        "Neurology": "neurology_node",
        "Pulmonology": "pulmonology_node",
        "Gastroenterology": "gastroenterology_node",
        "Endocrinology": "endocrinology_node",
        "Dermatology": "dermatology_node"
    }
    
    nodes_to_execute = []
    for specialty in selected_specialists:
        if specialty in routing_map:
            nodes_to_execute.append(routing_map[specialty])
            
    # Fallback: If the planner fails to select any specialist, route to a default general node or end
    if not nodes_to_execute:
        return ["debate_node"] # Or implement a fallback generalist node
        
    return nodes_to_execute