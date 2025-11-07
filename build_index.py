# -*- coding: utf-8 -*-
"""
긴 텍스트에 최적화된 완전 로컬 인덱서 (Ollama 임베딩 + Chroma)
- 임베딩: langchain-ollama / mxbai-embed-large (기본)
- 긴 문서 대비: 큰 청크 + 상한 가드 + 자동 재분할
- Azure/클라우드 의존성 없음
사용법:
  1) .env(선택):
     CSV_DEFAULT=test.csv
     PERSIST_DIR=./chroma_creation
     OLLAMA_BASE_URL=http://localhost:11434
     OLLAMA_EMBED_MODEL=mxbai-embed-large   # 또는 nomic-embed-text
     CHUNK_CHARS=1800
     CHUNK_OVERLAP=200
     MAX_EMBED_CHARS=6000
     ENRICH=false
     FRESH=true
  2) 인덱싱:
     python build_index_ollama_long.py
"""

import os, re, time, hashlib, shutil, random
import pandas as pd
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===================== 기본 설정 =====================
load_dotenv()

CSV_DEFAULT   = os.getenv("CSV_DEFAULT", "test.csv")
PERSIST_DIR   = os.getenv("PERSIST_DIR", "./chroma_creation")

# ENRICH: url 본문을 추가로 긁어와서 content 보강 (선택)
ENRICH        = os.getenv("ENRICH", "false").lower() in ("1","true","yes","y")
FRESH         = os.getenv("FRESH",  "true").lower() in ("1","true","yes","y")

FORCE_FETCH_DOMAINS = {"creation.kr"}
MIN_CONTENT_LEN     = 200

# Ollama
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").replace("/v1","")
# 🔥 긴 텍스트에 적합한 모델 기본값
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# 긴 텍스트 대비 파라미터 (필요하면 .env로 조절)
CHUNK_CHARS        = int(os.getenv("CHUNK_CHARS", "1800"))    # 1800자 정도
CHUNK_OVERLAP      = int(os.getenv("CHUNK_OVERLAP", "200"))
MAX_EMBED_CHARS    = int(os.getenv("MAX_EMBED_CHARS", "6000"))  # 최종 상한 (문자 기준)

# ===================== 유틸 / 전처리 =====================
def df_fingerprint(df: pd.DataFrame) -> str:
    parts = [(row.get("title","") or "") + (row.get("content","") or "") for _, row in df.iterrows()]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()

def persist_path(persist_dir: str, fp: str):
    d = os.path.join(persist_dir, f"chroma_{fp[:12]}")
    return d, f"creation_{fp[:12]}"

def load_csv(csv_path: str):
    csv_path = csv_path if os.path.isabs(csv_path) else os.path.join(os.getcwd(), csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    need_cols = {"url","title","content","references","further_refs"}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV에 필요한 컬럼이 누락되었습니다: {missing}")
    return df

def need_force_fetch(url: str) -> bool:
    try:
        host = re.sub(r"^https?://", "", url).split("/")[0]
        return any(host.endswith(dom) for dom in FORCE_FETCH_DOMAINS)
    except Exception:
        return False

def _smart_select_main(soup: BeautifulSoup):
    for css in ["article",".fr-view",".rd-content",".board_view",".boardView",".content",
                "#content","#article","#view",".editor_content",".xe_content",".se-component"]:
        node = soup.select_one(css)
        if node and node.get_text(strip=True): return node
    return soup.body or soup

def fetch_url_text(url: str, timeout: int = 12, max_len: int = 25000, retries: int = 2) -> str:
    if not url or not re.match(r"^https?://", url): return ""
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers={"User-Agent":"Mozilla/5.0 (CreationKR/1.0)"}, timeout=timeout)
            r.raise_for_status()
            if not r.encoding or r.encoding.lower() in ("iso-8859-1","ascii"):
                r.encoding = r.apparent_encoding or "utf-8"
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script","style","nav","footer","header","aside","form"]): tag.decompose()
            main = _smart_select_main(soup)
            text = (main or soup).get_text("\n")
            lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
            text = "\n".join([ln for ln in lines if ln])[:max_len]
            if len(text) < 150 and attempt < retries:
                time.sleep(0.4); continue
            return text
        except Exception:
            time.sleep(0.3)
    return ""

def safe_truncate(s: str, max_chars: int) -> str:
    """임베딩 컨텍스트 초과를 막기 위한 안전 자르기(문자 기준)."""
    if len(s) <= max_chars:
        return s
    # 문장 경계 근처에서 자르기(간단한 휴리스틱)
    cut = s[:max_chars]
    last = max(cut.rfind("\n"), cut.rfind(". "), cut.rfind("。"), cut.rfind("! "), cut.rfind("? "))
    if last >= max_chars * 0.7:
        return cut[:last].rstrip()
    return cut.rstrip()

# ===================== 문서 청크 =====================
def docs_from_df(df: pd.DataFrame, do_network_enrich: bool = False):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_CHARS,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = []
    for ridx, row in df.iterrows():
        title = (row.get("title") or "").strip()
        content = (row.get("content") or "").strip()
        url = (row.get("url") or "").strip()
        references_raw = (row.get("references") or "").strip()
        further_refs_raw = (row.get("further_refs") or "").strip()

        if do_network_enrich and url and (len(content) < MIN_CONTENT_LEN or need_force_fetch(url)):
            fetched = fetch_url_text(url)
            if fetched:
                content = (content + "\n\n" + fetched).strip()

        if not title and not content:
            continue

        base_text = f"{title}\n\n{content}".strip()

        # 1차 분할
        chunks = splitter.split_text(base_text)

        # 최종 상한 가드 + 과대청크 자동 재분할
        final_chunks = []
        for ch in chunks:
            if len(ch) <= MAX_EMBED_CHARS:
                final_chunks.append(ch)
            else:
                # 너무 크면 상한에 맞춰 여러 조각으로 추가 분할
                # 간단한 고정폭 슬라이스 (필요시 재귀적 split 사용 가능)
                start = 0
                while start < len(ch):
                    piece = safe_truncate(ch[start:start + MAX_EMBED_CHARS + 500], MAX_EMBED_CHARS)
                    if not piece:
                        break
                    final_chunks.append(piece)
                    start += len(piece)

        for cidx, chunk in enumerate(final_chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "title": title, "url": url,
                    "references_raw": references_raw, "further_refs_raw": further_refs_raw,
                    "row_id": str(ridx), "chunk_id": f"{ridx}-{cidx}",
                }
            ))
    return docs

# ===================== 인덱싱 루프 =====================
def add_with_backoff(store: Chroma, docs, batch_size=32, max_retries=8):
    n = len(docs); i = 0
    while i < n:
        j = min(i + batch_size, n)
        batch = docs[i:j]
        attempt = 0
        while True:
            try:
                store.add_documents(batch)
                break
            except Exception as e:
                wait = min(2 ** attempt, 60) + random.uniform(0, 0.5)
                print(f"[index] add_documents error: {e} (sleep {wait:.1f}s)")
                time.sleep(wait)
                attempt += 1
                if attempt >= max_retries:
                    raise
        print(f"[index] added {j}/{n}")
        i = j

# ===================== 메인 =====================
def main(csv_path=CSV_DEFAULT, persist_dir=PERSIST_DIR, enrich=False, fresh=True):
    print(f"[index] CSV: {csv_path}")
    df = load_csv(csv_path)
    fp = df_fingerprint(df)
    d, cname = persist_path(persist_dir, fp)
    print(f"[index] fingerprint   : {fp[:12]}")
    print(f"[index] persist dir   : {d}")
    print(f"[index] collection    : {cname}")
    print(f"[index] enrich via net: {enrich}")
    print(f"[index] fresh build   : {fresh}")
    print(f"[index] embed model   : {OLLAMA_EMBED_MODEL}")
    print(f"[index] chunk/overlap : {CHUNK_CHARS}/{CHUNK_OVERLAP}, max_embed_chars={MAX_EMBED_CHARS}")

    if fresh and os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

    docs = docs_from_df(df, do_network_enrich=enrich)
    print(f"[index] total chunks  : {len(docs)}")

    # Ollama 임베딩
    emb = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    # 모델 헬스체크 & 404/500 등 친절 로그
    try:
        _ = emb.embed_query("health check")
    except Exception as e:
        hint = ""
        msg = str(e)
        if "model" in msg.lower() and "not found" in msg.lower():
            hint = f"\n💡 해결: `ollama pull {OLLAMA_EMBED_MODEL}` 먼저 실행하세요."
        print(f"[index] ❌ 임베딩 모델 호출 실패: {e}{hint}")
        raise

    store = Chroma(persist_directory=d, collection_name=cname, embedding_function=emb)
    add_with_backoff(store, docs, batch_size=32)
    print("[index] ✅ done.")

if __name__ == "__main__":
    enrich_flag = os.getenv("ENRICH", "false").lower() in ("1","true","yes","y")
    fresh_flag  = os.getenv("FRESH",  "true").lower() in ("1","true","yes","y")
    main(CSV_DEFAULT, PERSIST_DIR, enrich=enrich_flag, fresh=fresh_flag)
