from __future__ import annotations

from typing import Any
from config import EMBEDDING_MODEL_VI
from src.utils import stable_hash

class EmbeddingManager:
    """
    Embedding manager CHỈ SỬ DỤNG MÔ HÌNH CHÍNH (No Fallback).
    Đảm bảo tính đồng nhất tuyệt đối cho không gian vector RAG.
    """

    def __init__(self, allow_fallback: bool = False):
        # Giữ tham số allow_fallback = False để tương thích với các lời gọi cũ nhưng không sử dụng
        self._model: Any | None = None
        self._cache: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        cache_key = stable_hash(text, 32)
        if cache_key in self._cache:
            return self._cache[cache_key]
        embedding = self.embed_batch([text])[0]
        self._cache[cache_key] = embedding
        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        try:
            return model.encode(texts, show_progress_bar=len(texts) > 8).tolist()
        except Exception as e:
            raise RuntimeError(f"❌ Lỗi khi tạo vector embedding với model '{EMBEDDING_MODEL_VI}': {e}") from e

    def get_embedding_dim(self) -> int:
        model = self._load_model()
        try:
            return int(model.get_sentence_embedding_dimension())
        except Exception as e:
            raise RuntimeError(f"❌ Không thể lấy số chiều của model '{EMBEDDING_MODEL_VI}': {e}") from e

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "❌ Thiếu thư viện 'sentence-transformers'. Vui lòng chạy 'pip install sentence-transformers' để dùng model chính!"
            ) from e

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
            
        print(f"⚡ Đang load model embedding chính ({EMBEDDING_MODEL_VI}) trên thiết bị: {device.upper()}")

        try:
            # Ưu tiên load từ cache local trước
            self._model = SentenceTransformer(EMBEDDING_MODEL_VI, device=device, local_files_only=True, trust_remote_code=True)
        except Exception:
            try:
                # Nếu chưa có trong cache thì tải online từ HuggingFace
                print(f"🌐 Đang tải model '{EMBEDDING_MODEL_VI}' từ HuggingFace...")
                self._model = SentenceTransformer(EMBEDDING_MODEL_VI, device=device, trust_remote_code=True)
            except Exception as e:
                raise RuntimeError(
                    f"❌ KHÔNG THỂ TẢI MODEL EMBEDDING CHÍNH '{EMBEDDING_MODEL_VI}'.\n"
                    f"Hệ thống RAG yêu cầu bắt buộc phải có model này để đảm bảo đồng nhất dữ liệu.\n"
                    f"Chi tiết lỗi: {e}"
                ) from e

        return self._model