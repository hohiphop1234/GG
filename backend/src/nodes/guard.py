from __future__ import annotations

import re
from typing import Any

from src.utils.helpers import normalize_for_match
from src.pipeline.registry import (
    EMERGENCY_TRIGGERS,
    RESPONSE_EMERGENCY,
    RESPONSE_OUT_OF_SCOPE,
    RESPONSE_INSUFFICIENT,
    DISCLAIMERS,
)

def get_disclaimer(risk_level: str) -> str:
    return DISCLAIMERS.get(risk_level, DISCLAIMERS["medium"])


class SafetyGuard:
    """Safety checks that run before and after retrieval."""

    def __init__(self):
        # Normalize patterns at startup to support both accented and unaccented matching
        normalized_patterns = [normalize_for_match(p) for p in EMERGENCY_TRIGGERS]
        self.triggers = [re.compile(rf"\b{p}\b", re.IGNORECASE) for p in normalized_patterns]

    def is_emergency(self, query: str) -> bool:
        q_norm = normalize_for_match(query)
        
        # Lọc bằng Regex
        if not any(t.search(q_norm) for t in self.triggers):
            return False
            
        return True

    def emergency_response(self, query: str) -> dict[str, Any]:
        return {
            "type": "emergency",
            "message": RESPONSE_EMERGENCY,
            "risk_level": "critical",
            "requires_human": True,
            "language": "vi",
        }

    def out_of_scope_response(self, query: str) -> dict[str, Any]:
        return {
            "type": "out_of_scope",
            "message": RESPONSE_OUT_OF_SCOPE,
            "risk_level": "none",
            "language": "vi",
        }

    def insufficient_evidence_response(self, query: str) -> dict[str, Any]:
        return {
            "type": "insufficient_evidence",
            "message": RESPONSE_INSUFFICIENT,
            "risk_level": "medium",
            "language": "vi",
        }
