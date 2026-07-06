"""Central registry for all externalized data: drugs, medical terms, keywords and prompts."""

import json
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _BACKEND_DIR / "data"
_PROMPTS_DIR = _BACKEND_DIR / "prompts"

def _load_json(name: str) -> dict:
    with open(_DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)

def _load_prompt(name: str) -> str:
    with open(_PROMPTS_DIR / name, encoding="utf-8") as f:
        return f.read().strip()

# Drug data
_drugs = _load_json("drug_registry.json")
DRUG_ALIASES: dict[str, str] = _drugs["aliases"]
KNOWN_DRUG_ENTITIES: set[str] = set(_drugs["known_entities"])
DRUG_EXPANSIONS: dict[str, list[str]] = _drugs["expansions"]

# Medical terms
_terms = _load_json("medical_terms.json")
INTERACTION_CUES: list[str] = _terms["interaction_cues"]
WARNING_CUES: list[str] = _terms["warning_cues"]
SIDE_EFFECT_PATTERNS: list[str] = _terms["side_effect_patterns"]
CONCRETE_SYMPTOMS_EN: list[str] = _terms["concrete_symptoms"]["en"]
CONCRETE_SYMPTOMS_VI: list[str] = _terms["concrete_symptoms"]["vi"]
CONTEXT_CUES: list[str] = _terms["context_cues"]
FALSE_POSITIVES: list[str] = _terms["false_positives"]

# Keywords
_kw = _load_json("keywords.json")
FAQ_TRIGGERS: list[str] = _kw["faq_triggers"]
EMERGENCY_TRIGGERS: list[str] = _kw["emergency_triggers"]
PROHIBITED_PATTERNS: list[str] = _kw["prohibited_patterns"]

# Responses
_resp = _load_json("responses.json")
RESPONSE_EMERGENCY: str = _resp["emergency"]
RESPONSE_OUT_OF_SCOPE: str = _resp["out_of_scope"]
RESPONSE_INSUFFICIENT: str = _resp["insufficient_evidence"]
DISCLAIMERS: dict[str, str] = _resp["disclaimers"]
