from langgraph.graph import StateGraph, START, END
from src.graph.state import AgentState

def build_base_graph() -> StateGraph:
    workflow = StateGraph(AgentState)
    return workflow 