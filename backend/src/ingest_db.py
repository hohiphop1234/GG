import json
import logging
from pathlib import Path
from typing import Any

from config import (
    BM25_INDEX_PATH,
    CHROMA_PERSIST_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from src.store.bm25_store import BM25Store
from src.data_cleaner import DataCleaner
from src.store.embeddings import EmbeddingManager
from src.utils.helpers import ensure_dir, load_jsonl
from src.store.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_data() -> dict[str, int]:
    ensure_dir(PROCESSED_DATA_DIR)
    ensure_dir(Path(BM25_INDEX_PATH).parent)
    
    print("Bắt đầu giai đoạn 1: Làm sạch & chuẩn hóa dữ liệu thô...")
    cleaner = DataCleaner()
    cleaner.process_dataset(RAW_DATA_DIR, PROCESSED_DATA_DIR)
    
    vector_store = VectorStore(CHROMA_PERSIST_DIR)
    vector_store.reset()

    all_chunks: dict[str, list[dict[str, Any]]] = {}
    chunks_path = Path(PROCESSED_DATA_DIR) / "chunks_vi.jsonl"
    chunks = load_jsonl(chunks_path)
    all_chunks["vi"] = chunks
    texts = [chunk["content"] for chunk in chunks]
    
    print(f"\nBắt đầu giai đoạn 2: Tạo vector embedding cho {len(texts):,} chunks...")
    embedding_manager = EmbeddingManager()
    embeddings = embedding_manager.embed_batch(texts)
    
    print(f"\nBắt đầu giai đoạn 3: Nạp vào ChromaDB...")
    vector_store.add_documents(chunks, embeddings)
    
    print(f"\nBắt đầu giai đoạn 4: Xây dựng chỉ mục từ khóa BM25...")
    bm25_store = BM25Store()
    bm25_store.build_index(chunks)
    bm25_store.save(BM25_INDEX_PATH)
    
    stats = vector_store.get_stats()
    stats["bm25_vi_count"] = len(all_chunks.get("vi", []))
    return stats

if __name__ == "__main__":
    print("Ingesting data...")
    stats = ingest_data()
    print(f"Done: {stats}")
