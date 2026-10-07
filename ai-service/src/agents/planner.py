from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.graph.state import AgentState, PlannerRouting

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

def planner_node(state: AgentState) -> dict:
    planner_llm = llm.with_structured_output(PlannerRouting)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are the Planner Agent. Your job is to determine which specialist agents should participate "
            "in the analysis. Instead of activating every specialist, select only potentially relevant agents "
            "based on the symptoms and available information to reduce unnecessary processing. "
            "Available specialists: Cardiology, Neurology, Pulmonology, Gastroenterology, Endocrinology, Dermatology."
        )),
        ("human", "Extracted Symptoms Data: {extracted_entities}")
    ])
    
    chain = prompt | planner_llm
    result = chain.invoke({
        "extracted_entities": state.get("extracted_entities", {})
    })
    
    return {"active_specialists": result.selected_specialists}