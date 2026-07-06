from pydantic import BaseModel, Field

class QueryClassificationSchema(BaseModel):
    category: str = Field(
        description="Phân loại: 'faq', 'medical', 'emergency', hoặc 'out-of-scope'"
    )

class WebSearchQuerySchema(BaseModel):
    search_query: str = Field(description="English search query for medical search")
    entities: list[str] = Field(default_factory=list, description="Extracted medical entities")

class EvidenceGradingSchema(BaseModel):
    relevant: bool = Field(description="Chunk có liên quan đến câu hỏi không")
    score: float = Field(ge=0.0, le=1.0, description="Điểm relevance")
    reason: str = Field(default="", description="Lý do ngắn gọn")

class EmergencyValidationSchema(BaseModel):
    is_emergency: bool = Field(
        description="True if the query describes a life-threatening or urgent medical emergency happening right now."
    )
    rationale: str = Field(
        description="Brief explanation of why this query represents or does not represent a crisis."
    )

class SafetyValidationSchema(BaseModel):
    contains_diagnosis: bool = Field(
        description="True if the answer contains unauthorized or direct diagnosis of a condition."
    )
    contains_dosage_prescription: bool = Field(
        description="True if the answer contains direct prescriptions of drugs or specific dosage instructions."
    )
    validation_issues: list[str] = Field(
        default_factory=list,
        description="List of specific safety violations found in the response."
    )

