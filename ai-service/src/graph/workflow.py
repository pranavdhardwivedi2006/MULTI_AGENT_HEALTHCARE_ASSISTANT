from langgraph.graph import StateGraph, START, END
from src.graph.state import AgentState
from src.agents.analyzer import symptom_analyzer_node
from src.agents.planner import planner_node
from src.agents.specialists.cardiology import cardiology_node
from src.agents.specialists.neurology import neurology_node
from src.agents.specialists.pulmonology import pulmonology_node
from src.agents.specialists.gastroenterology import gastroenterology_node
from src.agents.specialists.endocrinology import endocrinology_node
from src.agents.specialists.dermatology import dermatology_node
from src.agents.debate import debate_node
from src.agents.consensus import consensus_node
from src.agents.safety import safety_node
from src.graph.edges import route_to_specialists

def build_workflow() -> StateGraph:
    """Compiles the multi-agent healthcare workflow into an executable graph."""
    
    # 1. Initialize Graph with shared state
    builder = StateGraph(AgentState)
    
    # 2. Add Nodes
    builder.add_node("symptom_analyzer_node", symptom_analyzer_node)
    builder.add_node("planner_node", planner_node)
    
    # Specialist Layer
    builder.add_node("cardiology_node", cardiology_node)
    builder.add_node("neurology_node", neurology_node)
    builder.add_node("pulmonology_node", pulmonology_node)
    builder.add_node("gastroenterology_node", gastroenterology_node)
    builder.add_node("endocrinology_node", endocrinology_node)
    builder.add_node("dermatology_node", dermatology_node)
    
    # Resolution Layer
    builder.add_node("debate_node", debate_node)
    builder.add_node("consensus_node", consensus_node)
    builder.add_node("safety_node", safety_node)

    # 3. Define Standard Edges (Linear Flow)
    builder.add_edge(START, "symptom_analyzer_node")
    builder.add_edge("symptom_analyzer_node", "planner_node")
    
    # 4. Define Conditional Edges (Dynamic Routing)
    # The planner node routes to the dynamically selected specialists
    builder.add_conditional_edges(
        "planner_node",
        route_to_specialists,
        # Map the returned list of node names to themselves
        {
            "cardiology_node": "cardiology_node",
            "neurology_node": "neurology_node",
            "pulmonology_node": "pulmonology_node",
            "gastroenterology_node": "gastroenterology_node",
            "endocrinology_node": "endocrinology_node",
            "dermatology_node": "dermatology_node",
            "debate_node": "debate_node" # Fallback mapping
        }
    )
    
    # 5. Define Fan-In Edges (Merging back to standard flow)
    # All specialists must map to the debate node to compare their output
    specialist_nodes = [
        "cardiology_node", "neurology_node", "pulmonology_node", 
        "gastroenterology_node", "endocrinology_node", "dermatology_node"
    ]
    for node in specialist_nodes:
        builder.add_edge(node, "debate_node")
        
    # 6. Complete the Resolution Layer
    builder.add_edge("debate_node", "consensus_node")
    builder.add_edge("consensus_node", "safety_node")
    builder.add_edge("safety_node", END)
    
    # 7. Compile Graph
    return builder.compile()