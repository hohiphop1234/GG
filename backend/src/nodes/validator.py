from __future__ import annotations

import re
from typing import Any

from src.nodes.guard import get_disclaimer
from src.utils.helpers import normalize_for_match
from src.pipeline.registry import PROHIBITED_PATTERNS


class ResponseValidator:
    """Post-generation safety checks using rule-based metrics and prohibited patterns."""

    def __init__(self):
        pass

    def validate(self, response: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
        answer = response.get("answer", "")
        issues: list[str] = []

        # 1. Kiểm tra nguồn (Citations count check)
        cited_numbers = [int(num) for num in re.findall(r"\[(\d+)\]", answer)]
        for number in cited_numbers:
            if number > len(chunks):
                issues.append(f"Citation [{number}] references a missing source")

        # 2. Fast check: Regex patterns check
        normalized_answer = normalize_for_match(answer)
        for pattern in PROHIBITED_PATTERNS:
            if re.search(pattern, normalized_answer, flags=re.IGNORECASE):
                issues.append(f"Prohibited pattern found: {pattern}")

        category = response.get("category")
        exit_type = response.get("exit_type")
        if len(answer) > 100 and not cited_numbers and category not in ("faq", "out_of_scope") and exit_type != "insufficient_evidence":
            issues.append("Answer contains medical claims but no citations")

        language = response.get("language", "vi")
        risk_level = response.get("risk_level", "medium")
        response["disclaimer"] = get_disclaimer(risk_level)
        response["validation_issues"] = list(set(issues)) # Deduplicate issues
        response["is_valid"] = len(issues) == 0

        if any("Prohibited" in issue for issue in issues):
            response["answer"] = (
                "Tôi không thể đưa ra chẩn đoán, đơn thuốc, hoặc liều dùng cá nhân. "
                "Vui lòng trao đổi với bác sĩ hoặc dược sĩ."
                if language == "vi"
                else "I cannot provide diagnosis, prescriptions, or personal dosage instructions. "
                "Please consult a doctor or pharmacist."
            )
        return response
