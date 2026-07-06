from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.utils.helpers import normalize_for_match, tokenize
from config import MIN_EVIDENCE_CHUNKS, EVIDENCE_THRESHOLD, HIGH_EVIDENCE_SCORE, MIN_WEAK_EVIDENCE

@dataclass
class GradingResult:
    relevant_chunks: list[dict[str, Any]]
    score: float
    needs_crawl: bool
    confidence: str


class EvidenceGrader:
    """Grade whether retrieved chunks are useful enough to answer.
    Uses a fast deterministic rule-based scoring method.
    """

    def __init__(self):
        pass

    def grade(self, question: str, chunks: list[dict[str, Any]]) -> GradingResult:
        relevant = []
        for chunk in chunks:
            rule_score = self._rule_score(question, chunk)
            if rule_score >= EVIDENCE_THRESHOLD:
                relevant.append({**chunk, "relevance_score": rule_score})

        # Sort by relevance score
        relevant.sort(key=lambda item: item.get("relevance_score", 0.0), reverse=True)

        if not relevant and chunks:
            # Fallback to the best weak evidence if nothing is found
            best = max(chunks, key=lambda item: item.get("score", item.get("fused_score", 0)))
            best_score = self._rule_score(question, best)
            if best_score >= MIN_WEAK_EVIDENCE:
                relevant.append({**best, "relevance_score": best_score})

        average = sum(item.get("relevance_score", 0.0) for item in relevant) / max(len(relevant), 1)
        
        # Check if we need to crawl more
        needs_crawl = len(relevant) < MIN_EVIDENCE_CHUNKS and not any(
            c.get("relevance_score", 0) >= HIGH_EVIDENCE_SCORE for c in relevant
        )
        
        confidence = "high" if average >= 0.75 else "medium" if average >= 0.45 else "low"
        
        return GradingResult(
            relevant_chunks=relevant,
            score=average,
            needs_crawl=needs_crawl,
            confidence=confidence,
        )

    def _rule_score(self, question: str, chunk: dict[str, Any]) -> float:
        q_tokens = set(tokenize(question))
        c_tokens = set(tokenize(chunk.get("content", "")))
        if not q_tokens or not c_tokens:
            return 0.0

        query_entities = self._query_entities(question)
        overlap = len(q_tokens & c_tokens) / max(len(q_tokens), 1)
        
        metadata = chunk.get("metadata", {})
        entity = normalize_for_match(str(metadata.get("entity") or chunk.get("entity", "")))
        question_normalized = normalize_for_match(question)
        content_normalized = normalize_for_match(chunk.get("content", ""))
        search_score = float(chunk.get("score", chunk.get("vector_score", chunk.get("fused_score", 0.0))) or 0.0)

        entity_matches_query = bool(entity and entity in question_normalized)
        content_has_query_entity = any(entity_name in content_normalized for entity_name in query_entities)
        
        # Entity mismatch check
        if query_entities and not entity_matches_query and not content_has_query_entity:
            return min(0.2, overlap * 0.4)

        entity_bonus = 0.3 if (entity_matches_query or content_has_query_entity) else 0.0
        search_bonus = min(max(search_score, 0.0), 1.0) * 0.2
        return min(1.0, overlap * 0.7 + entity_bonus + search_bonus)

    def _query_entities(self, question: str) -> set[str]:
        normalized = normalize_for_match(question)
        known = {
            "warfarin",
            "ibuprofen",
            "acetaminophen",
            "paracetamol",
            "metformin",
            "insulin",
            "aspirin",
            "omeprazole",
            "famotidine",
            "phenytoin",
        }
        return {entity for entity in known if entity in normalized}
