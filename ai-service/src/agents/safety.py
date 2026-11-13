from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.graph.state import AgentState, SafetyVerdict
from src.config import get_model

llm = get_model(temperature=0)

def safety_node(state: AgentState) -> dict:
    """Verifies the consensus draft for medical safety, calculates confidence, and generates a final status."""
    safety_llm = llm.with_structured_output(SafetyVerdict)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are the Safety Agent in a multi-agent healthcare system. "
            "Your objective is to evaluate the Consensus Agent's drafted response before it is delivered to the user. "
            "You must check whether the situation requires urgent medical attention (red flags), "
            "whether the response contains excessive certainty, whether professional consultation should be recommended, "
            "and whether the available information is insufficient. "
            "Based on the completeness of the data and the consensus draft, generate an overall 'confidence_score' (e.g., 'Medium (60%)'). "
            "Output your safety verdict, confidence score, and format an appropriate disclaimer."
        )),
        ("human", (
            "Patient Symptoms: {raw_symptoms}\n"
            "Consensus Draft: {consensus_draft}"
        ))
    ])
    
    chain = prompt | safety_llm
    result = chain.invoke({
        "raw_symptoms": state.get("raw_symptoms", ""),
        "consensus_draft": state.get("consensus_draft", "")
    })
    
    # Inject the confidence score into the final user-facing response
    if result.status == "CRITICAL":
        final_response = (
            f"**URGENT MEDICAL WARNING:** {result.urgency_level}\n\n"
            f"{result.recommended_action}\n\n"
            f"*Original System Draft (System Confidence: {result.confidence_score}):*\n{state.get('consensus_draft', '')}"
        )
    else:
        final_response = (
            f"{state.get('consensus_draft', '')}\n\n"
            f"**System Confidence Score:** {result.confidence_score}\n"
            f"---\n*Disclaimer: {result.disclaimer}*"
        )
        
    return {
        "safety_verdict": result.model_dump(),
        "final_response": final_response
    }