from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.graph.state import AgentState, DebateDeliberation
from src.config import get_model

llm = get_model(temperature=0.1)

def debate_node(state: AgentState) -> dict:
    """Compares specialist outputs and identifies agreements, disagreements, and uncertainty."""
    debate_llm = llm_pro.with_structured_output(DebateDeliberation)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are the Debate Agent in a multi-agent healthcare system. "
            "Different specialist agents have analyzed the patient's symptoms. "
            "Your objective is to compare these outputs and identify agreements, disagreements, "
            "supporting evidence, missing information, and uncertainty. "
            "You are creating a collaborative reasoning stage before the final response."
        )),
        ("human", (
            "Raw Symptoms: {raw_symptoms}\n"
            "Patient History: {patient_history}\n"
            "Specialist Assessments: {specialist_assessments}"
        ))
    ])
    
    chain = prompt | debate_llm
    result = chain.invoke({
        "raw_symptoms": state.get("raw_symptoms", ""),
        "patient_history": state.get("patient_history", []),
        "specialist_assessments": state.get("specialist_assessments", {})
    })
    
    return {"debate_output": result.model_dump()}