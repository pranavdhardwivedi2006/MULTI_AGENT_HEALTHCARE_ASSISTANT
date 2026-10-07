from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.graph.state import AgentState, SpecialistAssessment

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

endocrinology_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are the Endocrinology Specialist Agent in a multi-agent clinical decision-support framework. "
        "Analyze patient information strictly from a metabolic and hormonal perspective. "
        "Focus on glycemic control, thyroid dysregulation, adrenal function, systemic fatigue, polydipsia, and weight shifts. "
        "Contribute an independent domain assessment rather than determining a final medical diagnosis. "
        "Identify specific observations, differential considerations within endocrinology, information gaps, and your confidence level."
    )),
    ("human", (
        "Raw Symptoms: {raw_symptoms}\n"
        "Extracted Entities: {extracted_entities}\n"
        "Patient History: {patient_history}\n"
        "RAG Context: {rag_context}"
    ))
])

def endocrinology_node(state: AgentState) -> dict:
    structured_llm = llm.with_structured_output(SpecialistAssessment)
    chain = endocrinology_prompt | structured_llm
    result = chain.invoke({
        "raw_symptoms": state.get("raw_symptoms", ""),
        "extracted_entities": state.get("extracted_entities", {}),
        "patient_history": state.get("patient_history", []),
        "rag_context": state.get("extracted_entities", {}).get("rag_context", [])
    })
    return {"specialist_assessments": {"Endocrinology": result.model_dump()}}