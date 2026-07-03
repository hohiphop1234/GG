from __future__ import annotations

import time
from typing import Any
from urllib.parse import quote_plus, urljoin, urlparse

from config import CHUNK_OVERLAP, CHUNK_SIZE, CRAWL_RATE_LIMIT, CRAWL_WHITELIST, ENGLISH_CRAWL_WHITELIST, TAVILY_API_KEY
from src.utils import chunk_words, stable_hash

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - optional dependency
    requests = None
    BeautifulSoup = None


class WebCrawler:
    """On-demand crawler limited to trusted medical domains."""

    SEARCH_URLS_VI = [
        "https://html.duckduckgo.com/html/?q={query}+site:tamanhhospital.vn+OR+site:vinmec.com+OR+site:hellobacsi.com+OR+site:moh.gov.vn",
    ]
    SEARCH_URLS_EN = [
        "https://html.duckduckgo.com/html/?q={query}+site:mayoclinic.org+OR+site:webmd.com+OR+site:nhs.uk+OR+site:drugs.com",
    ]

    def __init__(self):
        self.cache: dict[str, str] = {}
        self.last_request = 0.0

    def search(self, query: str, entities: list[str], use_english: bool = False) -> list[dict[str, Any]]:
        if requests is None or BeautifulSoup is None:
            return []
        
        chunks: list[dict[str, Any]] = []
        main_entity = entities[0] if entities else query
        
        for url in self._search_links(query, use_english)[:3]:
            page_data = self._fetch_page(url)
            if page_data:
                chunks.extend(self._chunk_content(
                    page_data["text"], 
                    url=url, 
                    entity=main_entity, 
                    title=page_data["title"], 
                    publish_date=page_data["publish_date"]
                ))
        return chunks

    def _search_links(self, query: str, use_english: bool = False) -> list[str]:
        if requests is None:
            return []

        if not TAVILY_API_KEY:
            # Fallback to DuckDuckGo search
            links: list[str] = []
            templates = self.SEARCH_URLS_EN if use_english else self.SEARCH_URLS_VI
            for template in templates:
                search_url = template.format(query=quote_plus(query))
                html = self._get(search_url)
                if not html:
                    continue
                soup = BeautifulSoup(html, "html.parser")
                for anchor in soup.find_all("a", href=True):
                    href = anchor["href"]
                    if "uddg=" in href:
                        from urllib.parse import parse_qs, unquote
                        parsed = urlparse(href)
                        qs = parse_qs(parsed.query)
                        if "uddg" in qs:
                            href = unquote(qs["uddg"][0])
                    href = urljoin(search_url, href)
                    if self._allowed(href) and not self._is_search_result_url(href) and href not in links:
                        links.append(href)
            return links

        # Tavily Search API Implementation
        try:
            domains = ENGLISH_CRAWL_WHITELIST if use_english else CRAWL_WHITELIST
            payload = {
                "api_key": TAVILY_API_KEY,
                "query": query,
                "include_domains": domains,
                "max_results": 5
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
            return [result["url"] for result in data.get("results", [])]
        except Exception:
            return []

    def _fetch_page(self, url: str) -> dict[str, Any] | None:
        if self._is_search_result_url(url):
            return None
        html = self._get(url)
        if not html:
            return None
        soup = BeautifulSoup(html, "html.parser")
        
        # Extract Title
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        if not title:
            h1 = soup.find("h1")
            if h1:
                title = h1.get_text().strip()
        if not title:
            title = "Crawled trusted source"

        # Extract Publish Date
        publish_date = ""
        meta_selectors = [
            {"property": "article:published_time"},
            {"name": "pubdate"},
            {"name": "publish-date"},
            {"property": "og:pubdate"},
            {"property": "og:published_time"},
            {"name": "release-date"},
            {"name": "datePublished"},
            {"itemprop": "datePublished"}
        ]
        for selector in meta_selectors:
            meta = soup.find("meta", attrs=selector)
            if meta and meta.get("content"):
                publish_date = meta.get("content").strip()
                break
                
        if not publish_date:
            time_tag = soup.find("time")
            if time_tag:
                publish_date = time_tag.get("datetime") or time_tag.get_text()
                if publish_date:
                    publish_date = publish_date.strip()
                    
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        main = soup.find("article") or soup.find("main") or soup.body
        if not main:
            return None
        text = main.get_text(separator=" ", strip=True)
        normalized = text.lower()
        if "search results for:" in normalized or "no drug package labels found" in normalized:
            return None
        return {
            "text": text,
            "title": title,
            "publish_date": publish_date
        }

    def _get(self, url: str) -> str | None:
        if not self._allowed(url):
            return None
        if url in self.cache:
            return self.cache[url]
        self._rate_limit()
        try:
            response = requests.get(
                url,
                timeout=12,
                headers={"User-Agent": "MedicalRAG/1.0 educational prototype"},
            )
            if response.status_code != 200:
                return None
            self.cache[url] = response.text
            return response.text
        except Exception:
            return None

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
