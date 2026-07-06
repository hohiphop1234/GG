from __future__ import annotations

import os
import logging
from src.llm import get_llm
from src.pipeline.registry import _load_prompt
from config import SHORT_QUERY_WORDS, MAX_REWRITE_LENGTH

logger = logging.getLogger(__name__)

class QueryRewriter:
    """Rewrite raw medical queries into optimized keywords for retrieval using ViT5 ONNX."""

    def __init__(self, use_llm: bool = True, model_path="models/vit5-rewrite-onnx"):
        self.use_llm = use_llm
        if self.use_llm:
            self.llm = get_llm()
        self.use_onnx = False
        self.model = None
        self.tokenizer = None
        self.system_prompt = _load_prompt("rewrite.txt")
        
        try:
            if not os.path.isabs(model_path):
                current_dir = os.path.dirname(os.path.abspath(__file__))
                src_dir = os.path.dirname(current_dir)
                backend_dir = os.path.dirname(src_dir)
                resolved_model_path = os.path.join(backend_dir, model_path)
            else:
                resolved_model_path = model_path

            if os.path.exists(resolved_model_path):
                from optimum.onnxruntime import ORTModelForSeq2SeqLM
                from transformers import AutoTokenizer
                
                logger.info(f"[Query Rewriter] Loading ONNX Model from {resolved_model_path}...")
                self.tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
                self.model = ORTModelForSeq2SeqLM.from_pretrained(
                    resolved_model_path,
                    use_cache=False,
                    encoder_file_name="encoder_model.onnx",
                    decoder_file_name="decoder_model.onnx"
                )
                self.use_onnx = True
                logger.info("[Query Rewriter] ONNX Model Loaded Successfully!")
            else:
                logger.warning(f"[Query Rewriter] ONNX Model path not found: {resolved_model_path}. Falling back to LLM.")
        except Exception as e:
            logger.error(f"[Query Rewriter] Failed to load ONNX model: {e}. Falling back to LLM.")

    def rewrite(self, question: str, entities: list[str] | None = None) -> str:
        """
        Rewrite the query using local ViT5 ONNX. 
        If it fails, fallback to LLM.
        If LLM fails or returns an invalid/empty result, fallback to the original query.
        """
        words = question.strip().split()
        if len(words) <= SHORT_QUERY_WORDS:
            logger.info(f"[Query Rewriter] Query is already short ({len(words)} words). Bypassing rewrite.")
            return question

        if self.use_onnx:
            try:
                task_prefix = "viết lại câu hỏi y tế: "
                input_text = task_prefix + question
                inputs = self.tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True)
                outputs = self.model.generate(
                    **inputs,
                    max_length=64,
                    num_beams=4,
                    early_stopping=True
                )
                rewritten = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                if rewritten:
                    logger.info(f"[Query Rewriter] (ONNX) Rewrote '{question}' -> '{rewritten}'")
                    return rewritten
            except Exception as e:
                logger.error(f"[Query Rewriter] ONNX inference failed: {e}. Falling back to LLM.")

        if not self.use_llm:
            return question

        prompt = f"User question: {question}\n"
        if entities:
            prompt += f"Detected medical entities: {', '.join(entities)}\n"
        prompt += "Optimized search query:"

        try:
            rewritten = self.llm.generate_answer(
                question=prompt,
                max_new_tokens=256,
                system_prompt=self.system_prompt
            )
            rewritten = rewritten.strip().strip('"').strip("'")
            
            if (rewritten and 
                not rewritten.startswith("Lỗi:") and 
                not rewritten.startswith("Xin lỗi") and
                len(rewritten) < MAX_REWRITE_LENGTH): 
                
                logger.info(f"[Query Rewriter] (LLM) Rewrote '{question}' -> '{rewritten}'")
                return rewritten
        except Exception as e:
            logger.error(f"[Query Rewriter] Error during LLM rewrite: {e}")
        
        logger.info(f"[Query Rewriter] Fallback to original query: '{question}'")
        return question
