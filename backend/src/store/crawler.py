from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlparse

from config import CHUNK_OVERLAP, CHUNK_SIZE, CRAWL_RATE_LIMIT, CRAWL_WHITELIST, ENGLISH_CRAWL_WHITELIST, TAVILY_API_KEY
from src.utils.helpers import chunk_words, stable_hash

try:
    import requests
except ImportError:  # pragma: no cover - optional dependency
    requests = None


class WebCrawler:
    """On-demand crawler limited to trusted medical domains."""

    def __init__(self):
        self.last_request = 0.0

    def search(self, query: str, entities: list[str], use_english: bool = False) -> list[dict[str, Any]]:
        if requests is None or not TAVILY_API_KEY:
            return []
        
        chunks: list[dict[str, Any]] = []
        main_entity = entities[0] if entities else query
        
        for result in self._search_results(query, use_english)[:3]:
            content = result.get("content", "")
            url = result.get("url", "")
            if content and url:
                chunks.extend(self._chunk_content(
                    content,
                    url=url, 
                    entity=main_entity, 
                    title=result.get("title", ""),
                    publish_date=result.get("publish_date", "")
                ))
        return chunks

    def _search_results(self, query: str, use_english: bool = False) -> list[dict[str, str]]:
        if requests is None or not TAVILY_API_KEY:
            return []

        try:
            domains = ENGLISH_CRAWL_WHITELIST if use_english else CRAWL_WHITELIST
            payload = {
                "api_key": TAVILY_API_KEY,
                "query": query,
                "include_domains": domains,
                "max_results": 5,
                "include_raw_content": True,
            }
            self._rate_limit()
            response = requests.post(
                "https://api.tavily.com/search",
                json=payload,
                timeout=12
            )
            if response.status_code != 200:
                return []
            data = response.json()
            parsed_results: list[dict[str, str]] = []
            for result in data.get("results", []):
                url = str(result.get("url", "")).strip()
                if not url or not self._allowed(url) or self._is_search_result_url(url):
                    continue
                content = str(result.get("raw_content") or result.get("content") or "").strip()
                if not content:
                    continue
                parsed_results.append(
                    {
                        "url": url,
                        "title": str(result.get("title", "")).strip(),
                        "content": content,
                        "publish_date": str(result.get("published_date", "")).strip(),
                    }
                )
            return parsed_results
        except Exception:
            return []

    def _chunk_content(self, content: str, url: str, entity: str, title: str = "", publish_date: str = "") -> list[dict[str, Any]]:
        chunks = []
        for index, text in enumerate(chunk_words(content, CHUNK_SIZE, CHUNK_OVERLAP), 1):
            if len(text.split()) < 50:
                continue
            chunks.append(
                {
                    "id": f"crawl-{stable_hash(url + str(index), 20)}",
                    "content": text,
                    "source": urlparse(url).netloc,
                    "entity": entity,
                    "url": url,
                    "is_crawled": True,
                    "metadata": {
                        "source": urlparse(url).netloc,
                        "entity": entity,
                        "url": url,
                        "title": title or "Crawled trusted source",
                        "publish_date": publish_date or "",
                        "section": "",
                        "risk_level": "medium",
                        "category": "crawled",
                    },
                }
            )
        return chunks

    def _allowed(self, url: str) -> bool:
        host = urlparse(url).netloc.lower()
        allowed_domains = CRAWL_WHITELIST + ENGLISH_CRAWL_WHITELIST
        return any(host == domain or host.endswith("." + domain) for domain in allowed_domains)

    def _is_search_result_url(self, url: str) -> bool:
        parsed = urlparse(url)
        path = parsed.path.lower()
        query = parsed.query.lower()
        return (
            "search.cfm" in path
            or "query-meta" in path
            or "search" in path and "query=" in query
        )

    def _rate_limit(self) -> None:
        min_interval = 1.0 / max(CRAWL_RATE_LIMIT, 0.1)
        elapsed = time.time() - self.last_request
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request = time.time()
