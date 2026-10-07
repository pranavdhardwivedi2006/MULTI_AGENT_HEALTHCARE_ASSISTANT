from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.graph.state import AgentState, SpecialistAssessment

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

cardiology_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are the Cardiology Specialist Agent in a multi-agent clinical decision-support framework. "
        "Analyze patient information strictly from a cardiovascular perspective. "
        "Focus on cardiac rhythm, hemodynamics, vascular risks, chest pain radiation, dyspnea on exertion, and perfusion. "
        "Contribute an independent domain assessment rather than determining a final medical diagnosis. "
        "Identify specific observations, differential considerations within cardiology, information gaps, and your confidence level."
    )),
    ("human", (
        "Raw Symptoms: {raw_symptoms}\n"
        "Extracted Entities: {extracted_entities}\n"
        "Patient History: {patient_history}\n"
        "RAG Context: {rag_context}"
    ))
])

def cardiology_node(state: AgentState) -> dict:
    structured_llm = llm.with_structured_output(SpecialistAssessment)
    chain = cardiology_prompt | structured_llm
    result = chain.invoke({
        "raw_symptoms": state.get("raw_symptoms", ""),
        "extracted_entities": state.get("extracted_entities", {}),
        "patient_history": state.get("patient_history", []),
        "rag_context": state.get("extracted_entities", {}).get("rag_context", [])
    })
    return {"specialist_assessments": {"Cardiology": result.model_dump()}}