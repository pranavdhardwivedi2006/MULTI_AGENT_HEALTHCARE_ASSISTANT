from typing import TypedDict, List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field

class SymptomExtraction(BaseModel):
    primary_symptoms: List[str] = Field(description="List of identified important symptoms")
    duration: Optional[str] = Field(default=None)
    severity: Optional[str] = Field(default=None)
    possible_domains: List[str] = Field(description="List of possible medical domains involved based on the symptoms")
    requires_more_info: bool = Field(description="Flag indicating whether additional information may be required")

class PlannerRouting(BaseModel):
    selected_specialists: List[str] = Field(description="List of selected specialist agents. Must strictly be a subset of: Cardiology, Neurology, Pulmonology, Gastroenterology, Endocrinology, Dermatology")
    reasoning: str = Field(description="Reasoning for selecting these specific specialists to participate")

class SpecialistAssessment(BaseModel):
    specialty: str
    clinical_observations: List[str]
    differential_considerations: List[str]
    information_gaps: List[str]
    confidence_level: str
    rag_sources: List[str] = Field(default_factory=list)

class DebateDeliberation(BaseModel):
    points_of_agreement: List[str]
    points_of_disagreement: List[str]
    unaddressed_symptoms: List[str]
    uncertainty_factors: List[str]

class SafetyVerdict(BaseModel):
    status: str
    urgency_level: str
    red_flags_identified: List[str]
    recommended_action: str
    disclaimer: str
    
# --- Reducer Function ---
def merge_assessments(left: Dict[str, Dict[str, Any]], right: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Merges dictionaries from parallel specialist nodes into one state object."""
    if not left:
        left = {}
    if not right:
        right = {}
    merged = left.copy()
    merged.update(right)
    return merged

# --- Graph State ---
class AgentState(TypedDict):
    patient_id: str
    raw_symptoms: str
    patient_history: List[str]
    extracted_entities: Optional[Dict[str, Any]]
    active_specialists: List[str]
    specialist_assessments: Annotated[Dict[str, Dict[str, Any]], merge_assessments]
    debate_output: Optional[Dict[str, Any]]
    consensus_draft: Optional[str]
    safety_verdict: Optional[Dict[str, Any]]
    final_response: Optional[str]