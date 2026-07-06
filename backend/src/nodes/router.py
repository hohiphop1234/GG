import os
import numpy as np
from dataclasses import dataclass
from src.pipeline.registry import FAQ_TRIGGERS

@dataclass
class QueryClassification:
    intent: str
    category: str
    entities: list[str]
    risk_level: str
    confidence: float
    requires_rag: bool

class QueryRouter:
    """Classify medical queries before retrieval using PhoBERT ONNX.
    Fails immediately if the ONNX model cannot be loaded (no LLM fallback for setup errors).
    """

    def __init__(self, model_path="models/phobert-intent-onnx"):
        # Resolve path
        if not os.path.isabs(model_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            src_dir = os.path.dirname(current_dir)
            backend_dir = os.path.dirname(src_dir)
            resolved_model_path = os.path.join(backend_dir, model_path)
        else:
            resolved_model_path = model_path

        if not os.path.exists(resolved_model_path):
            raise FileNotFoundError(f"[Query Router] ONNX Model path not found: {resolved_model_path}")

        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer
        
        print(f"Loading ONNX Model from {resolved_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        self.model = ORTModelForSequenceClassification.from_pretrained(resolved_model_path)
        self.id2label = self.model.config.id2label
        print("ONNX Model Loaded Successfully!")

    def classify(self, query: str) -> QueryClassification:
        # Rule-based bypass for common greetings and identity questions (FAQ)
        from src.utils.helpers import normalize_for_match
        normalized_query = normalize_for_match(query)
        normalized_faq_triggers = [normalize_for_match(kw) for kw in FAQ_TRIGGERS]
        if any(kw in normalized_query for kw in normalized_faq_triggers) or normalized_query in ["chao", "hi"]:
            return QueryClassification(
                intent="faq",
                category="faq",
                entities=[],
                risk_level="low",
                confidence=1.0,
                requires_rag=False
            )

        # ONNX inference. If it fails, let the exception propagate.
        inputs = self.tokenizer(
            query, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=256
        )
        outputs = self.model(**inputs)
        logits = outputs.logits.detach().numpy()
        
        # Compute softmax to get confidence
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        predicted_id = int(np.argmax(logits, axis=-1)[0])
        confidence = float(np.max(probs))
        
        intent_label = self.id2label[predicted_id].lower()
        
        # Standardized routes
        if intent_label == "medical":
            category = "medical"
            risk_level = "high"
        elif intent_label == "emergency":
            category = "emergency"
            risk_level = "critical"
        elif intent_label == "out-of-scope":
            category = "out-of-scope"
            risk_level = "low"
        elif intent_label == "faq":
            category = "faq"
            risk_level = "low"
        else:
            category = "medical"
            risk_level = "high"

        return QueryClassification(
            intent=category,
            category=category,
            entities=[],
            risk_level=risk_level,
            confidence=confidence,
            requires_rag=(category == "medical")
        )
