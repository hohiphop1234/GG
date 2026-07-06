from __future__ import annotations

import re
from typing import Any

from config import LLM_CONTEXT_CHUNKS
from src.nodes.router import QueryClassification
from src.llm import get_llm
from src.pipeline.registry import _load_prompt

class ResponseGenerator:
    """Generate cited answers from graded context using direct llama-server calls."""

    def __init__(self):
        self.llm = get_llm()
        self.system_prompt = _load_prompt("rag_generation.txt")

    def generate(
        self,
        question: str,
        chunks: list[dict[str, Any]],
        classification: QueryClassification,
    ) -> dict[str, Any]:
        used_chunks = chunks
        try:
            answer = self._generate_with_llm(question, chunks)
            if answer.startswith("Lỗi:"):
                raise Exception(answer)
        except Exception as e:
            print(f"\n[RAG Pipeline] Local LLM Failed: {e}\n")
            answer, used_chunks = self._generate_extractive_fallback(question, chunks)

        return {
            "answer": answer,
            "sources": self._format_sources(used_chunks),
            "risk_level": classification.risk_level,
            "category": classification.category,
            "classification_confidence": classification.confidence,
        }

    def _generate_with_llm(self, question: str, chunks: list[dict[str, Any]]) -> str:
        top_chunks = chunks[:LLM_CONTEXT_CHUNKS]
        context = self._build_context(top_chunks)
        prompt = (
            f"Context documents:\n{context}\n\n"
            f"User question: {question}\n\n"
            "Answer using only the context above. Cite sources with [1], [2], etc. "
            "Ignore citation numbers that appear inside a context document; only use "
            "the source numbers assigned to the context documents."
        )
        return self.llm.generate_answer(
            question=prompt,
            max_new_tokens=2048,
            system_prompt=self.system_prompt
        )
        
    def generate_stream(
        self,
        question: str,
        chunks: list[dict[str, Any]],
        classification: QueryClassification,
    ):
        """Streaming version of answer generator"""
        top_chunks = chunks[:LLM_CONTEXT_CHUNKS]
        context = self._build_context(top_chunks)
        prompt = (
            f"Context documents:\n{context}\n\n"
            f"User question: {question}\n\n"
            "Answer using only the context above. Cite sources with [1], [2], etc. "
            "Ignore citation numbers that appear inside a context document; only use "
            "the source numbers assigned to the context documents."
        )
        
        try:
            yield from self.llm.stream_answer(
                question=prompt,
                max_new_tokens=2048,
                system_prompt=self.system_prompt
            )
        except Exception as e:
            print(f"\n[RAG Pipeline] Local LLM Stream Failed: {e}\n")
            ans, _ = self._generate_extractive_fallback(question, chunks)
            yield ans

    def _build_context(self, chunks: list[dict[str, Any]]) -> str:
        parts = []
        for index, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            source = metadata.get("source") or chunk.get("source", "Unknown")
            title = metadata.get("title") or chunk.get("title", "")
            section = metadata.get("section") or chunk.get("section", "")
            content = chunk['content']
            
            content = re.sub(r"Sức khỏe\s+Quay lại.*?(?=#|\n|$)", "", content)
            content = re.sub(r"Quay lại\s+\w+\s+Quay lại", "", content)
            content = re.sub(r"^\s*#\s*$", "", content, flags=re.MULTILINE)
            content = re.sub(r"\n{3,}", "\n\n", content).strip()
            parts.append(
                f"[{index}] Source: {source} | {title} | {section}\n{content}"
            )
        return "\n\n---\n\n".join(parts)

    def _generate_extractive_fallback(
        self,
        question: str,
        chunks: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """A simple generic extractive fallback when LLM fails."""
        if not chunks:
            return "Không đủ thông tin trong các nguồn được truy xuất để trả lời chính xác.", []
            
        intro = "Dựa trên các nguồn được truy xuất:\n"
        bullets = []
        used_chunks = []
        seen_sentences = set()
        
        for index, chunk in enumerate(chunks[:LLM_CONTEXT_CHUNKS], 1):
            content = chunk.get("content", "")
            sentences = re.split(r"(?<=[.!?])\s+|\s+\*\s+", content)
            for sentence in sentences:
                clean = re.sub(r"\s+", " ", sentence).strip()
                if len(clean.split()) < 8 or len(clean.split()) > 50:
                    continue
                if clean in seen_sentences:
                    continue
                seen_sentences.add(clean)
                bullets.append(f"- {clean} [{index}]")
                if chunk not in used_chunks:
                    used_chunks.append(chunk)
                if len(bullets) >= 3:
                    break
            if len(bullets) >= 3:
                break
                
        if not bullets:
            return "Không đủ thông tin trong các nguồn được truy xuất để trả lời chính xác.", []
            
        return intro + "\n" + "\n".join(bullets), used_chunks

    def _format_sources(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sources = []
        for index, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            sources.append(
                {
                    "index": index,
                    "title": metadata.get("title") or chunk.get("title", "Unknown"),
                    "source": metadata.get("source") or chunk.get("source", ""),
                    "url": metadata.get("url") or chunk.get("url", ""),
                    "section": metadata.get("section") or chunk.get("section", ""),
                    "score": chunk.get("fused_score", chunk.get("score", 0.0)),
                }
            )
        return sources
