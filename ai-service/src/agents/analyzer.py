from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.graph.state import AgentState, SymptomExtraction

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

def symptom_analyzer_node(state: AgentState) -> dict:
    analyzer_llm = llm.with_structured_output(SymptomExtraction)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are the Symptom Analyzer Agent. Your responsibility is to perform an initial analysis "
            "of the user's symptoms, identify important symptoms, extract relevant information, "
            "identify possible medical domains, and determine whether additional information may be required."
        )),
        ("human", "Raw symptoms: {raw_symptoms}\nPatient History: {patient_history}")
    ])
    
    chain = prompt | analyzer_llm
    result = chain.invoke({
        "raw_symptoms": state["raw_symptoms"],
        "patient_history": state.get("patient_history", [])
    })
    
    return {"extracted_entities": result.model_dump()}