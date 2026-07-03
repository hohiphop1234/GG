import os
import json
import numpy as np
from dataclasses import dataclass

@dataclass
class QueryClassification:
    intent: str
    category: str
    entities: list[str]
    risk_level: str
    confidence: float
    requires_rag: bool

class QueryRouter:
    """Classify medical queries before retrieval using PhoBERT ONNX."""

    def __init__(self, llm_client=None, model_path="models/phobert-intent-onnx"):
        self.llm = llm_client
        self.use_onnx = False
        self.model = None
        self.tokenizer = None
        
        try:
            # Ensure model_path is absolute based on project root if it's relative
            if not os.path.isabs(model_path):
                current_dir = os.path.dirname(os.path.abspath(__file__))
                backend_dir = os.path.dirname(current_dir)
                project_root = os.path.dirname(backend_dir)
                resolved_model_path = os.path.join(project_root, model_path)
            else:
                resolved_model_path = model_path

            if os.path.exists(resolved_model_path):
                from optimum.onnxruntime import ORTModelForSequenceClassification
                from transformers import AutoTokenizer
                
                print(f"Loading ONNX Model from {resolved_model_path}...")
                self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
                self.model = ORTModelForSequenceClassification.from_pretrained(resolved_model_path)
                self.id2label = self.model.config.id2label
                self.use_onnx = True
                print("ONNX Model Loaded Successfully!")
            else:
                print(f"ONNX Model path not found: {resolved_model_path}. Falling back to LLM.")
        except Exception as e:
            print(f"Failed to load ONNX model: {e}. Falling back to LLM.")

    def classify(self, query: str) -> QueryClassification:
        if self.use_onnx:
            try:
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
            except Exception as e:
                print(f"ONNX inference failed: {e}. Falling back to LLM.")

        # Fallback to LLM implementation if ONNX is missing or fails
        if not self.llm:
            return QueryClassification(
                intent="medical",
                category="medical",
                entities=[],
                risk_level="high",
                confidence=0.9,
                requires_rag=True
            )

        prompt = """
Bạn là hệ thống định tuyến (router) cho một trợ lý y tế AI.
Hãy phân loại câu hỏi sau thành ĐÚNG MỘT TRONG BỐN nhóm:
1. "faq": Các câu hỏi chào hỏi (hello, xin chào), khả năng của bạn (bạn làm được gì, bạn là ai), hoặc lời cảm ơn.
2. "medical": Tất cả các câu hỏi liên quan đến sức khỏe, triệu chứng bệnh, điều trị, thông tin thuốc, y tế nói chung.
3. "emergency": Các tình huống cấp cứu nguy kịch cần xử lý khẩn cấp (chảy máu nhiều, khó thở, tai nạn, ngộ độc, v.v.).
4. "out-of-scope": Các câu hỏi ngoài luồng, KHÔNG liên quan đến y tế hoặc trợ lý y tế (ví dụ: thời tiết, nấu ăn, lịch sử).

Trả về kết quả dưới định dạng JSON hợp lệ với 1 key duy nhất là "category".
Ví dụ: {"category": "faq"} hoặc {"category": "medical"} hoặc {"category": "emergency"} hoặc {"category": "out-of-scope"}
"""
        try:
            response = self.llm.generate_answer(query, system_prompt=prompt, max_new_tokens=512)
            start_idx = response.find("{")
            end_idx = response.rfind("}")
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx+1]
                data = json.loads(json_str)
                category = data.get("category", "medical").lower()
            else:
                category = "medical"
                
            if category not in ["faq", "medical", "emergency", "out-of-scope"]:
                category = "medical"
        except Exception:
            category = "medical"

        risk_level = "low"
        if category == "medical":
            risk_level = "high"
        elif category == "emergency":
            risk_level = "critical"

        return QueryClassification(
            intent=category,
            category=category,
            entities=[],
            risk_level=risk_level,
            confidence=0.9,
            requires_rag=(category == "medical")
        )
