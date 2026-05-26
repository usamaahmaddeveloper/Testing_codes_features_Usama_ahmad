"""Enterprise-grade mini web-crawler + BERT retrieval chatbot.

Usage:
  python test.py --url https://example.com --max-pages 200 --max-depth 2

What it does:
  1) Fetches and stores robots.txt (interpreted as "robot.text") if present.
  2) If robots.txt is not found, it crawls internal links (same host only) and
     searches for robot-related markers in discovered pages.
  3) Builds a BERT embedding index over crawled text chunks.
  4) Runs an interactive chatbot that answers by retrieving the most relevant
     passages (extractive) from the crawled corpus.

Notes:
  - This is an offline retrieval chatbot; it does not fine-tune BERT.
  - Respect website load by using conservative limits/timeouts.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from html import unescape
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F


DEFAULT_UA = (
    "Mozilla/5.0 (compatible; BlackboxAI-Crawler/1.0; +https://example.com/bot)"
)


ROBOT_MARKERS = [
    # Common robots.txt structure tokens
    "user-agent",
    "disallow",
    "allow",
    "sitemap",
    "crawl-delay",
    "host:",
]


@dataclass(frozen=True)
class CrawlConfig:
    start_url: str
    base_scheme: str
    base_netloc: str
    base_root: str
    max_pages: int
    max_depth: int
    timeout_s: int
    sleep_s: float
    user_agent: str
    respect_same_domain: bool = True


def normalize_start_url(url: str) -> Tuple[str, str, str]:
    """Return (scheme, netloc, root_url_without_path)."""
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url

    p = urlparse(url)
    scheme = p.scheme.lower() if p.scheme else "https"
    netloc = p.netloc.lower()

    # Root = scheme://netloc
    root = urlunparse((scheme, netloc, "", "", "", ""))
    return scheme, netloc, root


def is_internal_url(cfg: CrawlConfig, candidate: str) -> bool:
    try:
        p = urlparse(candidate)
        if not p.scheme or not p.netloc:
            return True  # relative urls handled earlier, treat as internal
        return p.netloc.lower() == cfg.base_netloc
    except Exception:
        return False


def canonicalize_url(cfg: CrawlConfig, url: str) -> Optional[str]:
    try:
        if not re.match(r"^https?://", url, re.I):
            url = urljoin(cfg.base_root + "/", url)
        p = urlparse(url)
        if not p.scheme or not p.netloc:
            return None
        if cfg.respect_same_domain and p.netloc.lower() != cfg.base_netloc:
            return None

        # Drop fragment; keep path/query.
        p = p._replace(fragment="")
        return urlunparse(p)
    except Exception:
        return None


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Prefer main content-ish: paragraphs and headings
    parts: List[str] = []
    for el in soup.find_all(["p", "li", "h1", "h2", "h3", "article", "section"]):
        t = el.get_text(" ", strip=True)
        if t:
            parts.append(t)

    text = "\n".join(parts)
    text = unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def extract_internal_links(base_url: str, html: str, cfg: CrawlConfig) -> Set[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: Set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        if not href:
            continue
        cand = urljoin(base_url, href)
        if not is_internal_url(cfg, cand):
            continue
        canon = canonicalize_url(cfg, cand)
        if canon:
            links.add(canon)
    return links


def try_fetch_robots_text(cfg: CrawlConfig) -> Optional[str]:
    # Standard robots.txt at {scheme}://{netloc}/robots.txt
    robots_url = urljoin(cfg.base_root + "/", "robots.txt")
    try:
        r = requests.get(
            robots_url,
            headers={"User-Agent": cfg.user_agent},
            timeout=cfg.timeout_s,
        )
        if r.status_code == 200 and r.text and len(r.text.strip()) > 0:
            return r.text
    except Exception:
        return None
    return None


def find_robot_related_text(html_text: str) -> bool:
    t = html_text.lower()
    # Look for explicit markers
    return any(m in t for m in ROBOT_MARKERS)


def chunk_text(text: str, chunk_size: int = 256, overlap: int = 32) -> List[str]:
    """Chunk by whitespace for embedding.

    We avoid tokenizer-aware chunking for simplicity.
    """
    tokens = text.split()
    if not tokens:
        return []
    chunks = []
    i = 0
    while i < len(tokens):
        j = min(len(tokens), i + chunk_size)
        chunk_tokens = tokens[i:j]
        chunk = " ".join(chunk_tokens).strip()
        if chunk:
            chunks.append(chunk)
        if j == len(tokens):
            break
        i = max(0, j - overlap)
    return chunks


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


class BertEmbedder:
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @torch.no_grad()
    def embed_texts(self, texts: List[str], batch_size: int = 8) -> torch.Tensor:
        """Return normalized embeddings [n, hidden]."""
        all_embs: List[torch.Tensor] = []
        for k in range(0, len(texts), batch_size):
            batch = texts[k : k + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model(**inputs)
            # Mean pool over attention mask
            last_hidden = out.last_hidden_state  # [b, seq, h]
            mask = inputs["attention_mask"].unsqueeze(-1).type_as(last_hidden)
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            emb = summed / counts
            emb = F.normalize(emb, p=2, dim=1)
            all_embs.append(emb.cpu())
        return torch.cat(all_embs, dim=0)


def build_index(
    cfg: CrawlConfig,
    pages: List[Dict],
    embedder: BertEmbedder,
    out_dir: str,
) -> Dict:
    """Build chunk index + embeddings."""

    os.makedirs(out_dir, exist_ok=True)

    chunk_records: List[Dict] = []
    corpus_chunks: List[str] = []

    for pg in pages:
        text = pg.get("text", "") or ""
        url = pg.get("url", "")
        for idx, ch in enumerate(chunk_text(text)):
            if len(ch) < 30:
                continue
            chunk_records.append(
                {
                    "chunk_id": len(chunk_records),
                    "source_url": url,
                    "chunk_index_in_page": idx,
                    "text": ch,
                }
            )
            corpus_chunks.append(ch)

    if not corpus_chunks:
        raise RuntimeError("No text chunks extracted; cannot build chatbot index.")

    embeddings = embedder.embed_texts(corpus_chunks, batch_size=8)
    index_path = os.path.join(out_dir, "index.pt")
    meta_path = os.path.join(out_dir, "index_meta.json")

    torch.save({"embeddings": embeddings}, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunk_records}, f, ensure_ascii=False, indent=2)

    return {
        "index_path": index_path,
        "meta_path": meta_path,
        "num_pages": len(pages),
        "num_chunks": len(chunk_records),
    }


def load_index(out_dir: str) -> Tuple[torch.Tensor, List[Dict]]:
    index_path = os.path.join(out_dir, "index.pt")
    meta_path = os.path.join(out_dir, "index_meta.json")
    pack = torch.load(index_path, map_location="cpu")
    embeddings: torch.Tensor = pack["embeddings"]
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return embeddings, meta["chunks"]


@torch.no_grad()
def answer_query(
    query: str,
    embedder: BertEmbedder,
    index_embeddings: torch.Tensor,
    chunks: List[Dict],
    top_k: int = 5,
) -> Dict:
    q_emb = embedder.embed_texts([query], batch_size=1)  # [1, h]
    # cosine similarity since embeddings normalized
    sims = (index_embeddings @ q_emb[0]).cpu()  # [n]
    top = torch.topk(sims, k=min(top_k, sims.shape[0]))
    top_ids = top.indices.tolist()

    passages = []
    for cid in top_ids:
        c = chunks[cid]
        passages.append(
            {
                "source_url": c["source_url"],
                "chunk_id": c["chunk_id"],
                "score": float(sims[cid].item()),
                "text": c["text"],
            }
        )

    # Extractive answer: return concatenation of top passages (bounded)
    answer_text = "\n\n".join([p["text"][:800] for p in passages])
    return {
        "query": query,
        "top_passages": passages,
        "answer": answer_text,
    }


def crawl_site(cfg: CrawlConfig, out_pages_dir: str) -> Tuple[List[Dict], Optional[str]]:
    os.makedirs(out_pages_dir, exist_ok=True)

    # 1) Try robots.txt first (robot.text)
    robots_text = try_fetch_robots_text(cfg)
    pages: List[Dict] = []

    robots_path = os.path.join(out_pages_dir, "robots.txt")
    if robots_text is not None:
        with open(robots_path, "w", encoding="utf-8") as f:
            f.write(robots_text)
        pages.append({"url": urljoin(cfg.base_root + "/", "robots.txt"), "text": robots_text})
        # Even if robots exists, still crawl some pages for chatbot usefulness.
    else:
        # Mark for later detection; still crawl to find robot-related markers.
        robots_path = None

    # 2) Crawl internal pages
    to_visit = collections.deque([(cfg.start_url, 0)])
    visited: Set[str] = set()

    start_canon = canonicalize_url(cfg, cfg.start_url)
    if start_canon:
        to_visit = collections.deque([(start_canon, 0)])

    session = requests.Session()
    session.headers.update({"User-Agent": cfg.user_agent})

    found_robot_related_pages: List[str] = []

    page_count = 0
    while to_visit and page_count < cfg.max_pages:
        url, depth = to_visit.popleft()
        canon = canonicalize_url(cfg, url)
        if not canon or canon in visited:
            continue
        visited.add(canon)

        if depth > cfg.max_depth:
            continue

        try:
            r = session.get(canon, timeout=cfg.timeout_s, allow_redirects=True)
            ct = r.headers.get("content-type", "")
            if r.status_code != 200:
                continue
            if "text/html" not in ct and "application/xhtml+xml" not in ct:
                continue
            html = r.text

            text = extract_text_from_html(html)
            if not text or len(text) < 80:
                continue

            pages.append({"url": canon, "text": text})
            page_count += 1

            if robots_text is None and find_robot_related_text(text):
                found_robot_related_pages.append(canon)

            if depth < cfg.max_depth:
                links = extract_internal_links(canon, html, cfg)
                for link in links:
                    lcanon = canonicalize_url(cfg, link)
                    if not lcanon or lcanon in visited:
                        continue
                    to_visit.append((lcanon, depth + 1))

            if cfg.sleep_s > 0:
                time.sleep(cfg.sleep_s)

        except Exception:
            continue

    if robots_text is None:
        # Persist a note that robots.txt was not found but robot markers were detected.
        meta_path = os.path.join(out_pages_dir, "robot_detection.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"found_robot_related_pages": found_robot_related_pages}, f, ensure_ascii=False, indent=2)

    # Persist pages.jsonl
    pages_path = os.path.join(out_pages_dir, "pages.jsonl")
    with open(pages_path, "w", encoding="utf-8") as f:
        for pg in pages:
            f.write(json.dumps({"url": pg["url"], "text": pg["text"][:20000]}, ensure_ascii=False) + "\n")

    # Persist corpus.txt
    corpus_path = os.path.join(out_pages_dir, "corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for pg in pages:
            f.write(f"\n\n===== SOURCE: {pg['url']} =====\n")
            f.write(pg["text"])

    return pages, robots_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Web crawl + BERT retrieval chatbot")
    parser.add_argument("--url", type=str, default=None, help="Starting URL/domain")
    parser.add_argument("--max-pages", type=int, default=100, help="Max pages to crawl")
    parser.add_argument("--max-depth", type=int, default=2, help="Max crawl depth")
    parser.add_argument("--timeout-s", type=int, default=15, help="Request timeout")
    parser.add_argument("--sleep-s", type=float, default=0.2, help="Sleep between requests")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="HF BERT model")
    parser.add_argument("--top-k", type=int, default=5, help="Top passages to use in answers")
    parser.add_argument("--out", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    url_in = args.url
    if not url_in:
        url_in = input("Enter domain or URL to crawl: ").strip()

    scheme, netloc, root = normalize_start_url(url_in)

    out_dir = args.out
    if not out_dir:
        # create deterministic output folder
        out_dir = os.path.join("crawled_data", f"{netloc}_{md5(root)}")

    cfg = CrawlConfig(
        start_url=url_in if re.match(r"^https?://", url_in, re.I) else "https://" + url_in,
        base_scheme=scheme,
        base_netloc=netloc,
        base_root=root,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        timeout_s=args.timeout_s,
        sleep_s=args.sleep_s,
        user_agent=DEFAULT_UA,
    )

    print(f"[INFO] Base domain: {cfg.base_netloc}")
    print(f"[INFO] Output dir: {out_dir}")

    pages_dir = os.path.join(out_dir, "pages")
    print("[INFO] Crawling...")
    pages, robots_text = crawl_site(cfg, pages_dir)

    print(f"[INFO] Extracted {len(pages)} pages (including robots.txt if present).")

    # Build or load embed index
    index_dir = os.path.join(out_dir, "index")
    os.makedirs(index_dir, exist_ok=True)

    # Always rebuild for simplicity; deterministic enough for this tool.
    print("[INFO] Loading BERT embedder...")
    embedder = BertEmbedder(model_name=args.model)

    print("[INFO] Building embedding index (chunked retrieval)...")
    index_info = build_index(cfg, pages, embedder, index_dir)
    print(f"[INFO] Index built: {index_info['num_chunks']} chunks")

    print("[INFO] Chat ready. Type 'exit' to quit.")

    index_embeddings, chunks = load_index(index_dir)

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            break

        resp = answer_query(q, embedder, index_embeddings, chunks, top_k=args.top_k)
        print("\nAssistant (retrieval-based):")
        print(resp["answer"][:2500])
        print("\nSources:")
        for p in resp["top_passages"]:
            print(f"- {p['source_url']} (score={p['score']:.4f})")
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)

