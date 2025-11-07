# -*- coding: utf-8 -*-
"""
Streamlit RAG 챗봇 (완전 로컬) — 고속·고정확 버전
- 임베딩: Ollama (mxbai-embed-large)
- 벡터DB: Chroma (디스크 영속)
- 생성모델: Ollama (llama3.1:8b-instruct-q4_0)

실행:
  streamlit run streamlit_rag_ollama_app.py

핵심 개선(정확도↑ 속도↑):
1) 하이브리드 검색 재랭킹: RRF(Reciprocal Rank Fusion) + 키워드 보너스
2) MMR 다양화: 중복 문서 억제 → 컨텍스트 품질↑
3) 멀티쿼리: HyDE + 2개 패러프레이즈(빠른 경로 캐시)
4) 초경량 검증: 답변 직후 핵심사실 3~6개 추출 → 후보 문서 entailment 미니체크(토글)
5) 컨텍스트 압축: 문단·문장 단위 스코어링으로 3,500자 내 고정보 유지
6) 캐시/리소스 재사용 최적화: BM25/멀티쿼리/엔테일먼트 캐시

주의: 외부 네트워크 호출 없음. ollama pull 등은 사전 준비 필요.
"""

import os, re, time, hashlib, random, shutil, math
from typing import List, Dict, Any, Tuple
import concurrent.futures as _f

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
import numpy as np

# Streamlit 페이지 설정(최상단)
st.set_page_config(page_title="로컬 RAG 챗봇 (Ollama + Chroma)", page_icon="🧠", layout="wide")

# LangChain & Chroma/Ollama
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 옵션: BM25 있으면 사용
try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except Exception:
    _BM25_AVAILABLE = False

# ===================== 환경 로드 =====================
load_dotenv()

EMB_BACKEND        = os.getenv("EMB_BACKEND", "ollama").lower()
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").replace("/v1","")
OLLAMA_GEN_MODEL   = os.getenv("OLLAMA_GEN_MODEL", "llama3.1:8b-instruct-q4_0")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")

CSV_DEFAULT        = os.getenv("CSV_DEFAULT", "test.csv")
PERSIST_DIR        = os.getenv("PERSIST_DIR", "./chroma_creation")
COLLECTION_NAME    = os.getenv("COLLECTION_NAME", "")

CHUNK_CHARS        = int(os.getenv("CHUNK_CHARS", "1600"))
CHUNK_OVERLAP      = int(os.getenv("CHUNK_OVERLAP", "150"))
MAX_EMBED_CHARS    = int(os.getenv("MAX_EMBED_CHARS", "4000"))
FRESH              = os.getenv("FRESH", "true").lower() in ("1","true","yes","y")

FORCE_FETCH_DOMAINS = {"creation.kr"}
MIN_CONTENT_LEN      = 200

# ===================== Fast 옵션 플래그 기본값 =====================
FAST_SOURCE_SELECT_DEFAULT = True  # 출처선정에서 LLM 엔테일먼트 생략(키워드 적합도 기반)

# ===================== 공용 캐시/리소스 =====================
@st.cache_resource(show_spinner=False)
def _get_emb() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)

@st.cache_resource(show_spinner=False)
def _get_llm(temp: float = 0.2, num_predict: int = 200) -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_GEN_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temp,
        model_kwargs={
            "num_predict": num_predict,
            "keep_alive": "10m",
            "num_thread": 0,
        },
    )

@st.cache_resource(show_spinner=False)
def _get_llm_zero() -> ChatOllama:
    return ChatOllama(model=OLLAMA_GEN_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0,
                      model_kwargs={"keep_alive": "10m", "num_thread": 0})

@st.cache_resource(show_spinner=False)
def _warm_llm_once() -> bool:
    try:
        _ = _get_llm_zero().invoke([{"role":"user","content":"ping"}])
    except Exception:
        pass
    return True

# ===================== 유틸 =====================
@st.cache_data(show_spinner=False)
def _cached_read_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    return df

def df_fingerprint(df: pd.DataFrame) -> str:
    parts = [(row.get("title","") or "") + (row.get("content","") or "") for _, row in df.iterrows()]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()

def persist_path(persist_dir: str, fp: str) -> Tuple[str, str]:
    d = os.path.join(persist_dir, f"chroma_{fp[:12]}")
    return d, f"creation_{fp[:12]}"

def load_csv(csv_path: str) -> pd.DataFrame:
    csv_path = csv_path if os.path.isabs(csv_path) else os.path.join(os.getcwd(), csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
    df = _cached_read_csv(csv_path)
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
    for css in [
        "article",".fr-view",".rd-content",".board_view",".boardView",".content",
        "#content","#article","#view",".editor_content",".xe_content",".se-component",
    ]:
        node = soup.select_one(css)
        if node and node.get_text(strip=True):
            return node
    return soup.body or soup

@st.cache_data(show_spinner=False)
def fetch_url_text(url: str, timeout: int = 12, max_len: int = 25000, retries: int = 2) -> str:
    if not url or not re.match(r"^https?://", url):
        return ""
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers={"User-Agent":"Mozilla/5.0 (CreationKR/1.0)"}, timeout=timeout)
            r.raise_for_status()
            if not r.encoding or r.encoding.lower() in ("iso-8859-1","ascii"):
                r.encoding = r.apparent_encoding or "utf-8"
            soup = BeautifulSoup(r.text, "htmlparser" if "htmlparser" in str(BeautifulSoup).lower() else "html.parser")
            for tag in soup(["script","style","nav","footer","header","aside","form"]):
                tag.decompose()
            main = _smart_select_main(soup)
            text = (main or soup).get_text("\n")
            lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
            text = "\n".join([ln for ln in lines if ln])[:max_len]
            if len(text) < 150 and attempt < retries:
                time.sleep(0.2)
                continue
            return text
        except Exception:
            time.sleep(0.2)
    return ""

def safe_truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars]
    last = max(cut.rfind("\n"), cut.rfind(". "), cut.rfind("。"), cut.rfind("! "), cut.rfind("? "))
    if last >= max_chars * 0.7:
        return cut[:last].rstrip()
    return cut.rstrip()

@st.cache_data(show_spinner=False)
def _split_base_text(base_text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_CHARS,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(base_text)

def docs_from_df(df: pd.DataFrame, do_network_enrich: bool = False) -> List[Document]:
    docs: List[Document] = []
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
        chunks = _split_base_text(base_text)

        final_chunks = []
        for ch in chunks:
            if len(ch) <= MAX_EMBED_CHARS:
                final_chunks.append(ch)
            else:
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
                    "title": title,
                    "url": url,
                    "references_raw": references_raw,
                    "further_refs_raw": further_refs_raw,
                    "row_id": str(ridx),
                    "chunk_id": f"{ridx}-{cidx}",
                }
            ))
    return docs

def _collection_is_empty(store: Chroma) -> bool:
    try:
        cnt = store._collection.count()  # type: ignore[attr-defined]
        return (cnt or 0) == 0
    except Exception:
        try:
            got = store._collection.get(limit=1)  # type: ignore[attr-defined]
            return not bool(got.get("ids"))
        except Exception:
            try:
                _ = store.similarity_search("ping", k=1)
                return False
            except Exception:
                return True

@st.cache_resource(show_spinner=False)
def _open_store(persist_directory: str, collection_name: str, _emb: OllamaEmbeddings) -> Chroma:
    # Streamlit 캐시 해시 오류 회피: 언더스코어 접두 파라미터는 해시에서 제외됨
    return Chroma(persist_directory=persist_directory, collection_name=collection_name, embedding_function=_emb)

@st.cache_data(show_spinner=False)
def _store_id(persist_dir: str, coll: str) -> str:
    return hashlib.sha1(f"{persist_dir}::{coll}".encode("utf-8")).hexdigest()

def build_or_load_store(csv_path: str, persist_dir: str, collection_name_env: str, fresh: bool) -> Tuple[Chroma, str, str]:
    df = load_csv(csv_path)
    fp = df_fingerprint(df)
    d_auto, cname_auto = persist_path(persist_dir, fp)

    d = d_auto
    cname = (collection_name_env or "").strip() or cname_auto

    if fresh and os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

    emb = _get_emb()

    # 임베딩 모델 헬스체크
    try:
        _ = emb.embed_query("health check")
    except Exception as e:
        msg = str(e)
        hint = ""
        if "model" in msg.lower() and "not found" in msg.lower():
            hint = f"\n💡 해결: `ollama pull {OLLAMA_EMBED_MODEL}` 먼저 실행하세요."
        raise RuntimeError(f"임베딩 모델 호출 실패: {e}{hint}")

    store = _open_store(d, cname, emb)

    need_index = fresh or _collection_is_empty(store)
    if need_index:
        with st.spinner("인덱스를 생성/갱신 중..."):
            docs = docs_from_df(df, do_network_enrich=False)
            add_with_backoff(store, docs, batch_size=32)

    return store, d, cname

def add_with_backoff(store: Chroma, docs: List[Document], batch_size=32, max_retries=8):
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
                wait = min(2 ** attempt, 10) + random.uniform(0, 0.2)
                st.warning(f"[index] add_documents error: {e} (재시도 {wait:.1f}s)")
                time.sleep(wait)
                attempt += 1
                if attempt >= max_retries:
                    raise
        i = j

# ===================== 프롬프트 & RAG 체인 =====================
SYSTEM_PROMPT = (
    "당신은 신뢰할 수 있는 한국어 연구 보조자입니다. "
    "항상 제공된 컨텍스트 범위에서만 답하고, 모르면 모른다고 말하세요. "
    "본문에는 출처를 쓰지 말고, 마지막에 단 1개만 표기합니다."
)

USER_Q_TEMPLATE = (
    "질문: {question}\n\n"
    "[컨텍스트]\n{context}\n\n"
    "요구사항:\n"
    "- 위 컨텍스트만 근거로 '정답'만을 작성하세요.\n"
    "- 정답은 정확하게 체크해서 써야 합니다.\n"
    "- 본문에는 링크/출처를 쓰지 마세요.\n"
    "- 잘못된 내용을 지어내지 마세요.\n"
    "- 맞춤법은 지키세요마세요.\n"    
)

# ===================== 컨텍스트 압축 =====================
def _sent_tokenize(text: str) -> List[str]:
    # 간단 문장 분할(영/한 혼합용)
    parts = re.split(r'(?<=[\.\?\!。])\s+|\n+', text)
    return [p.strip() for p in parts if p and len(p.strip()) >= 5]

def _keyword_score(sent: str, terms: List[str]) -> float:
    s = sent.lower()
    return sum(1 for t in terms if t in s) / max(1, len(terms) or 1)

def _compress_context(docs_serialized: List[Tuple[str, Dict[str, Any]]], question: str, max_chars: int = 3500) -> str:
    terms = [t for t in re.split(r"[\W_]+", question.lower()) if len(t) >= 2]
    scored_sents = []
    for page_content, meta in docs_serialized:
        title = meta.get("title") or ""
        title_boost = 0.2 * _keyword_score(title, terms)
        for s in _sent_tokenize(page_content):
            scored_sents.append((0.8 * _keyword_score(s, terms) + title_boost, title, s))
    scored_sents.sort(key=lambda x: x[0], reverse=True)
    out = []
    used = 0
    for _, title, s in scored_sents:
        block = f"### {title}\n{s}"
        if used + len(block) + 2 > max_chars:
            break
        out.append(block); used += len(block) + 2
        if used > max_chars * 0.9:
            break
    if not out:
        # fallback: 원본 앞부분 압축 없이
        parts = []
        for page_content, meta in docs_serialized:
            head = meta.get("title") or "(제목 없음)"
            parts.append(f"### {head}\n{page_content}")
        return ("\n\n".join(parts))[:max_chars]
    return "\n\n".join(out)

# -------------------- 일반화 키워드/스코어 유틸 --------------------
_STOPWORDS = set("""
은 는 이 가 을 를 에 의 와 과 도 로 으로 에서 한 하고 이나 나 또는 혹은 그리고 그러나 그래서
the a an and or of to in on for with from by as at is are was were be been being this that those these it its if then
""".split())

def extract_keywords(text: str, min_len: int = 2) -> List[str]:
    toks = [t.lower() for t in re.split(r"[\W_]+", text) if t]
    toks = [t for t in toks if len(t) >= min_len and t not in _STOPWORDS]
    if not toks:
        return []
    vals, cnts = np.unique(toks, return_counts=True)
    pairs = sorted(zip(vals, cnts), key=lambda x: x[1], reverse=True)
    return [w for w, _ in pairs[:20]]

def _normalize_q(q: str) -> str:
    q = q.strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q

# ===================== 검색 & 재랭킹(정확도 핵심) =====================
@st.cache_data(show_spinner=False)
def _load_all_chunks_cached(persist_dir_used: str, collection_used: str) -> Tuple[List[List[str]], List[str], List[Dict[str, Any]]]:
    emb = _get_emb()
    store = _open_store(persist_dir_used, collection_used, emb)
    docs = store._collection.get(include=["documents", "metadatas"])  # type: ignore[attr-defined]
    texts = docs.get("documents", []) or []
    metas = docs.get("metadatas", []) or []
    tokenized = [t.split() for t in texts]
    return tokenized, texts, metas

def _current_store_id() -> str:
    persist = st.session_state.get("persist_dir_used", PERSIST_DIR)
    coll = st.session_state.get("collection_used", COLLECTION_NAME or "(auto)")
    return hashlib.sha1(f"{persist}::{coll}".encode("utf-8")).hexdigest()

def _ensure_bm25_index(store: Chroma):
    if not _BM25_AVAILABLE:
        return None
    sid = _current_store_id()
    if "bm25" in st.session_state and st.session_state.get("bm25_store_id") == sid:
        return st.session_state.bm25
    tokenized, texts, metas = _load_all_chunks_cached(
        st.session_state.get("persist_dir_used", PERSIST_DIR),
        st.session_state.get("collection_used", COLLECTION_NAME or "(auto)")
    )
    if not texts:
        return None
    bm25 = BM25Okapi(tokenized)
    st.session_state.bm25 = bm25
    st.session_state.bm25_texts = texts
    st.session_state.bm25_metas = metas
    st.session_state.bm25_store_id = sid
    return bm25

def _dense_search_with_scores(store: Chroma, query: str, k: int) -> List[Tuple[Document, float]]:
    try:
        pairs = store.similarity_search_with_relevance_scores(query, k=k)
        return [(doc, float(score if score is not None else 0.0)) for doc, score in pairs]
    except Exception:
        docs = store.similarity_search(query, k=k)
        return [(d, 0.0) for d in docs]

@st.cache_data(show_spinner=False)
def _hyde_query_cached(q: str) -> str:
    qn = _normalize_q(q)
    llm_tmp = _get_llm_zero()
    prompt = "질문에 대한 간결한 가설적 답변을 한 단락으로 작성하세요.\n질문: " + qn
    resp = llm_tmp.invoke([{"role":"user","content":prompt}])
    return (getattr(resp, "content", str(resp)) or "").strip()

@st.cache_data(show_spinner=False)
def _paraphrases_cached(q: str, n: int = 2) -> List[str]:
    qn = _normalize_q(q)
    llm0 = _get_llm_zero()
    prompt = (
        "아래 질문을 서로 다른 관점의 질의 2개로 짧게 패러프레이즈하세요.\n"
        "출력은 줄바꿈으로 구분된 2줄, 불릿 금지.\n질문: " + qn
    )
    rsp = llm0.invoke([{"role":"user","content":prompt}])
    raw = (getattr(rsp, "content", str(rsp)) or "").strip()
    out = [ln.strip("-• ").strip() for ln in raw.splitlines() if ln.strip()]
    return out[:n]

def _keyword_bonus(text: str, terms: List[str]) -> float:
    s = text.lower()
    return sum(1 for t in terms if t in s) / max(1, len(terms) or 1)

def _rrf_fusion(rank_lists: List[List[Tuple[str, Tuple[Document,float]]]], k: int, K: int = 60) -> List[Tuple[Document, float]]:
    # rank_lists: [ [(key,(doc,score)), ...], ... ]
    agg: Dict[str, Tuple[Document, float]] = {}
    for ranks in rank_lists:
        for r, (key, (doc, _)) in enumerate(ranks, start=1):
            prev = agg.get(key, (doc, 0.0))
            agg[key] = (doc, prev[1] + 1.0 / (K + r))
    items = list(agg.items())
    items.sort(key=lambda x: x[1][1], reverse=True)
    return [(doc, score) for _, (doc, score) in items[:k]]

def _rankify(pairs: List[Tuple[Document, float]], terms: List[str]) -> List[Tuple[str, Tuple[Document,float]]]:
    # 키워드 보너스를 살짝 더해 순위 형성
    scored = []
    for doc, s in pairs:
        meta = doc.metadata or {}
        t = (meta.get("title","") + " " + doc.page_content)
        scored.append((doc, s + 0.1*_keyword_bonus(t, terms)))
    scored.sort(key=lambda x: x[1], reverse=True)
    out = []
    for i, (doc, sc) in enumerate(scored, start=1):
        key = f"{meta.get('row_id','?')}-{meta.get('chunk_id','?')}-{i}"
        out.append((key, (doc, sc)))
    return out

def _mmr_diversify(docs: List[Document], top_k: int, lamb: float = 0.7) -> List[Document]:
    # 토큰 교집합 근사 유사도(Jaccard-like)
    def _tokset(text: str):
        return set([t for t in re.split(r"[\W_]+", text.lower()) if len(t) >= 3])
    cand = [(doc, _tokset(doc.page_content)) for doc in docs]
    selected: List[Tuple[Document, set]] = []
    while cand and len(selected) < top_k:
        best_i, best_score = 0, -1e9
        for i, (doc_i, set_i) in enumerate(cand):
            rel = 1.0  # 이미 재랭크된 리스트라 가중치 비슷하게 취급
            div = 0.0
            if selected:
                div = max(len(set_i & sset)/max(1,len(set_i|sset)) for _, sset in selected)
            score = lamb*rel - (1-lamb)*div
            if score > best_score:
                best_score, best_i = score, i
        selected.append(cand.pop(best_i))
    return [d for d,_ in selected]

def _neighbor_docs_serialized(chunk_id: str, texts: List[str], metas: List[Dict[str, Any]], window: int = 1) -> List[Tuple[str, Dict[str, Any]]]:
    try:
        ridx_str, cidx_str = chunk_id.split("-")
        ridx = int(ridx_str); cidx = int(cidx_str)
    except Exception:
        return []
    out: List[Tuple[str, Dict[str, Any]]] = []
    for meta, text in zip(metas, texts):
        if str(meta.get("row_id")) != str(ridx):
            continue
        try:
            cid = meta.get("chunk_id", "0-0")
            _, c = cid.split("-")
            if abs(int(c) - cidx) <= window and int(c) != cidx:
                out.append((text, meta))
        except Exception:
            continue
    return out

def hybrid_retrieve_with_scores(store: Chroma, question: str, k: int, mode: str,
                               dense_weight: float = 0.6, window: int = 1,
                               use_multiquery: bool = True) -> List[Document]:
    terms = extract_keywords(question)
    queries = [question]

    if mode in ("HyDE+Hybrid", "Hybrid") and use_multiquery:
        try:
            hyp = _hyde_query_cached(question)
            queries.append(f"{question}\n{hyp}")
        except Exception:
            pass
        try:
            for pq in _paraphrases_cached(question, n=2):
                if pq and pq not in queries:
                    queries.append(pq)
        except Exception:
            pass

    # Dense/BM25 각각 상위 후보들을 RRF로 결합
    dense_ranklists = []
    bm25_ranklists  = []
    for q in queries:
        dense_pairs = _dense_search_with_scores(store, q, k=max(k*2, k))
        dense_ranklists.append(_rankify(dense_pairs, terms))
        if _BM25_AVAILABLE and mode in ("Lexical", "Hybrid", "HyDE+Hybrid"):
            bm25 = _ensure_bm25_index(store)
            if bm25 is not None:
                tokenized_q = q.split()
                scores = bm25.get_scores(tokenized_q)
                idx_sorted = np.argsort(scores)[::-1][:max(k*2, k)]
                texts = st.session_state.bm25_texts
                metas = st.session_state.bm25_metas
                bm25_pairs = []
                for i in idx_sorted:
                    meta = metas[int(i)]
                    text = texts[int(i)]
                    bm25_pairs.append((Document(page_content=text, metadata=meta), float(scores[int(i)])))
                bm25_ranklists.append(_rankify(bm25_pairs, terms))

    fused_docs: List[Document] = []
    # RRF 결합
    ranklists = []
    if dense_ranklists: ranklists += dense_ranklists
    if bm25_ranklists:  ranklists += bm25_ranklists
    if ranklists:
        fused = _rrf_fusion(ranklists, k=max(k*4, k))
        fused_docs = [doc for doc, _ in fused]
    else:
        # Dense fallback
        fused_docs = [d for d,_ in _dense_search_with_scores(store, question, k=max(k*3, k))]

    # MMR 다양화
    diversified = _mmr_diversify(fused_docs, top_k=k)

    # 이웃 확장
    texts = st.session_state.get("bm25_texts", [])
    metas = st.session_state.get("bm25_metas", [])
    out: List[Document] = []
    seen = set()
    for doc in diversified:
        cid = doc.metadata.get("chunk_id","?")
        if cid not in seen:
            out.append(doc); seen.add(cid)
        if texts and metas and window > 0:
            for page_text, meta in _neighbor_docs_serialized(cid, texts, metas, window=window):
                ncid = meta.get("chunk_id","?")
                if ncid not in seen:
                    out.append(Document(page_content=page_text, metadata=meta)); seen.add(ncid)
    return out[:max(k+window*2, k)]

# -------------------- 출처 선택 & 미니 검증 --------------------
@st.cache_data(show_spinner=False)
def _extract_facts_from_answer_cached(answer_text: str) -> List[str]:
    llm0 = _get_llm_zero()
    prompt = (
        "다음 답변에서 검증이 필요한 핵심 사실만 3~6개 bullet로 매우 짧게 추출하세요.\n"
        "형식: 각 줄 하나의 사실. 불필요한 수식어 제거. 고유명사/수량/관계를 살리되 80자 이내.\n\n"
        f"{answer_text}\n"
    )
    rsp = llm0.invoke([{"role":"user","content":prompt}])
    raw = (getattr(rsp, "content", str(rsp)) or "")
    facts = []
    for line in raw.splitlines():
        line = line.strip("-• \t").strip()
        if line:
            facts.append(line)
    return facts[:6]

@st.cache_data(show_spinner=False)
def _entailment_score_cached(fact: str, candidate_text: str) -> float:
    llm0 = _get_llm_zero()
    judge = (
        "문서가 주어진 사실을 뒷받침하는지 평가하세요.\n"
        "- 출력은 0.0~1.0 사이 소수점 하나만(설명 금지)\n"
        "- 1.0=강하게 뒷받침, 0.5=애매, 0.0=부정/관련없음\n\n"
        f"[사실]\n{fact}\n\n"
        f"[문서]\n{candidate_text[:2800]}\n"
    )
    r = llm0.invoke([{"role":"user","content":judge}])
    s = (getattr(r, "content", str(r)) or "").strip()
    try:
        v = float(re.findall(r"[01](?:\.\d+)?", s)[0])
        return max(0.0, min(1.0, v))
    except Exception:
        return 0.0

def _select_primary_source_fast(question: str, candidates_serialized: List[Tuple[str, Dict[str, Any]]], top_n: int = 6) -> Dict[str, Any]:
    terms = [t for t in re.split(r"[\W_]+", question.lower()) if len(t) >= 2]
    best, best_s = None, -1.0
    for (page_content, meta) in candidates_serialized[:max(1, top_n)]:
        t = (meta.get("title") or "") + " " + page_content
        sc = _keyword_bonus(t, terms)
        if sc > best_s:
            best_s, best = sc, meta
    return best or (candidates_serialized[0][1] if candidates_serialized else {})

def _primary_source_line(meta: Dict[str, Any]) -> str:
    if not meta:
        return "- (출처 없음)"
    title = (meta.get("title") or "").strip() or "(제목 없음)"
    url = (meta.get("url") or "").strip()
    if url:
        return f"{title} | {url}"
    return f"{title}"

def _dynamic_num_predict(question: str, ctx_chars: int) -> int:
    base = 160 if len(question) > 60 or ctx_chars > 2500 else 120
    return max(96, min(200, base))

def rag_answer(store: Chroma, question: str, k: int = 5,
               mode: str = "Hybrid", dense_weight: float = 0.6, neighbor_window: int = 1,
               fast_mode: bool = True, use_multiquery: bool = True, strict_verify: bool = False
               ) -> Tuple[str, List[Document]]:
    # 1) 검색 + 재랭크 + 다양화
    fetched_docs = hybrid_retrieve_with_scores(
        store, question, k=k, mode=mode, dense_weight=dense_weight, window=neighbor_window,
        use_multiquery=use_multiquery and not fast_mode  # Fast일 때 멀티쿼리 off로 지연 최소화
    )

    total_chars = sum(len(d.page_content) for d in fetched_docs)
    if len(fetched_docs) == 0 or total_chars < 400:
        return (
            "죄송해요. 제공된 문서들만으로는 질문에 답하기에 충분한 근거를 찾지 못했어요. "
            "질문을 더 구체화하거나 CSV에 관련 자료를 추가해 주세요.",
            fetched_docs,
        )

    # 2) 컨텍스트 압축(정확도 유지 + 토큰 절약)
    docs_serialized: List[Tuple[str, Dict[str, Any]]] = [
        (d.page_content, dict(d.metadata)) for d in fetched_docs
    ]
    ctx = _compress_context(docs_serialized, question, max_chars=3500)

    # 3) 생성
    llm = _get_llm(temp=0.2, num_predict=_dynamic_num_predict(question, len(ctx)) if fast_mode else 200)
    prompt = USER_Q_TEMPLATE.format(question=question, context=ctx)
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    resp = llm.invoke(msgs)
    answer_text = (getattr(resp, "content", str(resp)) or "").strip()

    # 4) 출처 1개 선택(빠른 키워드 적합도 기반)
    primary_meta = _select_primary_source_fast(question, docs_serialized, top_n=max(3, k))
    source_line = _primary_source_line(primary_meta)

    # 5) (선택) 초경량 검증 — 사실 평균 0.6 미만이면 주의 문구
    if strict_verify and not fast_mode:
        facts = _extract_facts_from_answer_cached(answer_text)
        if facts:
            to_eval_text = (primary_meta.get("title","") + "\n" + fetched_docs[0].page_content) if fetched_docs else ""
            with _f.ThreadPoolExecutor(max_workers=min(4, len(facts))) as ex:
                vals = list(ex.map(lambda f: _entailment_score_cached(f, to_eval_text), facts))
            avg_ent = sum(vals) / max(1, len(vals))
            if avg_ent < 0.6:
                answer_text += "\n\n(참고: 근거 일치도가 낮습니다. 질문을 더 구체화하거나 추가 문서를 제공해 주세요.)"

    final = f"{answer_text}\n\n출처:\n{source_line}"
    return final, fetched_docs

# ===================== UI =====================
st.title("🤖 창조 과학 챗봇 (고속·고정확)")

# LLM 워밍(첫 질의 딜레이 완화)
_ = _warm_llm_once()

with st.sidebar:
    st.subheader("⚙️ 설정")
    st.markdown("**임베딩/생성 모델은 .env 로 제어됩니다.**")

    st.text_input("CSV 경로", key="csv_path", value=CSV_DEFAULT)
    st.text_input("PERSIST_DIR", key="persist_dir", value=PERSIST_DIR)
    st.text_input("COLLECTION_NAME(선택)", key="collection_name", value=COLLECTION_NAME)

    st.markdown("---")
    top_k = st.slider("Top-K 문서", min_value=2, max_value=12, value=5, step=1)
    mode = st.selectbox("Retrieval Mode", ["Hybrid", "HyDE+Hybrid", "Lexical", "Dense"], index=0)
    dense_w = st.slider("Dense 가중치(하이브리드)", 0.0, 1.0, 0.6, 0.05)
    nb_win = st.slider("이웃 청크 범위", 0, 2, 1, 1)

    st.markdown("---")
    do_rebuild = st.button("🔁 인덱스 재빌드(FRESH)")

    st.markdown("---")
    fast_mode = st.checkbox("🚀 Fast Mode (최소 토큰·즉시 출처선정·HyDE 약화)", value=True)
    strict_verify = st.checkbox("🛡️ Strict Verify (사실-문서 미니 엔테일먼트)", value=False,
                                help="정확도↑(약간 느려짐). Fast Mode가 꺼져 있을 때 효과적")
    use_multiquery = st.checkbox("🔎 MultiQuery (HyDE+패러프레이즈)", value=True,
                                 help="정확도↑(검색↑). Fast Mode가 켜져 있으면 자동 약화")

    st.session_state.fast_mode = fast_mode
    st.session_state.strict_verify = strict_verify
    st.session_state.use_multiquery = use_multiquery

    # 진단 패널
    st.markdown("### 🔍 현재 설정 진단")
    st.code(f"CSV={st.session_state.get('csv_path', 'N/A')}\n"
            f"PERSIST_DIR={st.session_state.get('persist_dir', 'N/A')}\n"
            f"COLLECTION_NAME={st.session_state.get('collection_name', 'N/A')}\n"
            f"OLLAMA_BASE_URL={os.getenv('OLLAMA_BASE_URL')}\n"
            f"EMBED_MODEL={os.getenv('OLLAMA_EMBED_MODEL')}\n", language="bash")
    if st.button("CSV 존재/컬럼 점검"):
        try:
            _df = pd.read_csv(st.session_state.get('csv_path', ''), dtype=str).fillna("")
            st.success(f"CSV 로드 OK, shape={_df.shape}")
            miss = {'url','title','content','references','further_refs'} - set(_df.columns)
            st.write("누락 컬럼:", miss if miss else "없음 ✅")
        except Exception as e:
            st.error(f"CSV 로드 실패: {e}")

# 세션 상태
if "history" not in st.session_state:
    st.session_state.history = []

# 스토어 준비
if "store" not in st.session_state or do_rebuild:
    try:
        store, d_used, cname_used = build_or_load_store(
            csv_path=st.session_state.csv_path,
            persist_dir=st.session_state.persist_dir,
            collection_name_env=st.session_state.collection_name,
            fresh=True if do_rebuild else False,
        )
        st.session_state.store = store
        st.session_state.persist_dir_used = d_used
        st.session_state.collection_used = cname_used
    except FileNotFoundError as e:
        st.error(f"CSV 경로 문제: {e}")
    except ValueError as e:
        st.error(f"CSV 컬럼 문제: {e}")
    except RuntimeError as e:
        st.error(f"임베딩 모델 문제: {e}")
    except Exception as e:
        st.exception(e)

# 채팅 영역
st.markdown("### 💬 대화")
for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

user_msg = st.chat_input("질문을 입력하세요… (로컬 문서 기반으로 답합니다)")
if user_msg:
    st.session_state.history.append(("user", user_msg))
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        if "store" not in st.session_state:
            st.error("벡터 스토어가 준비되지 않았습니다. 좌측에서 설정 확인 후 재빌드하세요.")
            answer = "스토어 미준비"
        else:
            with st.spinner("검색 및 생성 중…"):
                try:
                    answer, used_docs = rag_answer(
                        st.session_state.store,
                        question=user_msg,
                        k=top_k,
                        mode=mode,
                        dense_weight=dense_w,
                        neighbor_window=nb_win,
                        fast_mode=st.session_state.get("fast_mode", True),
                        use_multiquery=st.session_state.get("use_multiquery", True),
                        strict_verify=st.session_state.get("strict_verify", False),
                    )
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"응답 생성 실패: {e}")
                    answer = f"오류: {e}"

    st.session_state.history.append(("assistant", answer))
