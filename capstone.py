# -*- coding: utf-8 -*-
import os
import re
import glob
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# === RAG용: Chroma + Ollama 임베딩 ===
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# === Ollama (공식 SDK) for generation ===
import ollama  # OpenAI SDK 미사용

# === 호환 임포트 (splitter / Document) ===
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # fallback

try:
    from langchain_core.documents import Document
except ModuleNotFoundError:
    from langchain.docstore.document import Document  # fallback


# ========= 환경설정 =========
load_dotenv()

CSV_DEFAULT = os.getenv("CSV_DEFAULT", "test.csv")  # 옵션
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_GEN_MODEL = os.getenv("OLLAMA_GEN_MODEL", "llama3.1:8b-instruct-q4_0")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")

PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_creation")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", None)  # 미지정 시 최신 폴더명에서 유추

# Ollama 클라이언트
client = ollama.Client(host=OLLAMA_BASE)


# ========= 유틸 =========
def normalize_ko(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^\uac00-\ud7a3a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_ko(s: str):
    return re.findall(r"[가-힣a-z0-9]{2,}", normalize_ko(s))

def df_fingerprint(df: pd.DataFrame) -> str:
    parts = []
    cols = set(df.columns)
    title_col = "title" if "title" in cols else None
    content_col = "content" if "content" in cols else None
    for _, row in df.iterrows():
        t = (row.get(title_col, "") or "") if title_col else ""
        c = (row.get(content_col, "") or "") if content_col else ""
        parts.append(t + c)
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()

def load_csv(csv_path: str):
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(os.getcwd(), csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    need_cols = {"url", "title", "content", "references", "further_refs"}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV에 필요한 컬럼이 누락되었습니다: {missing}")
    return df

def docs_for_bm25(df: pd.DataFrame):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=200, length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = []
    for ridx, row in df.iterrows():
        title = (row.get("title") or "").strip()
        content = (row.get("content") or "").strip()
        base_text = f"{title}\n\n{content}".strip()
        for cidx, chunk in enumerate(splitter.split_text(base_text)):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "title": title,
                    "row_id": str(ridx),
                    "chunk_id": f"{ridx}-{cidx}",
                    "url": (row.get("url") or "").strip()
                }
            ))
    return docs

# 문자열 잘라주는 함수
def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n...[일부 생략]"


# ========= Chroma 연결 =========
def _pick_latest_collection_dir(base_dir: str) -> str:
    cands = sorted(
        glob.glob(os.path.join(base_dir, "chroma_*")),
        key=os.path.getmtime,
        reverse=True
    )
    if not cands:
        raise RuntimeError(f"Chroma 컬렉션 폴더를 찾을 수 없음: {base_dir}")
    return cands[0]

def get_store() -> Chroma:
    emb = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE)
    persist_dir = _pick_latest_collection_dir(PERSIST_DIR)
    col_name = COLLECTION_NAME or os.path.basename(persist_dir).replace("chroma_", "creation_")
    store = Chroma(
        persist_directory=persist_dir,
        collection_name=col_name,
        embedding_function=emb
    )
    return store

def warn_if_embedding_mismatch(store: Chroma):
    try:
        raw_coll = getattr(store, "_collection", None)
        name = getattr(raw_coll, "name", None) or getattr(store, "collection_name", None)
        client_inner = getattr(store, "_client", None)
        if client_inner and hasattr(client_inner, "get_collection"):
            coll = client_inner.get_collection(name)
            meta = getattr(coll, "metadata", None) or {}
        else:
            meta = {}
        idx_model = meta.get("embedding_model")
        if idx_model and idx_model != OLLAMA_EMBED_MODEL:
            st.warning(
                f"⚠️ 이 컬렉션은 인덱싱 시 '{idx_model}'로 임베딩되었고, "
                f"현재 검색 임베딩은 '{OLLAMA_EMBED_MODEL}' 입니다. "
                f"가능하면 동일 모델로 맞추세요."
            )
    except Exception:
        pass


# ========= 검색 & 컨텍스트 구성 (MMR + 하이브리드) =========
def keyword_overlap_score(query: str, text: str) -> float:
    q_toks = set(tokenize_ko(query))
    t_toks = set(tokenize_ko(text))
    if not q_toks:
        return 0.0
    inter = len(q_toks & t_toks)
    return inter / max(3, len(q_toks))

def retrieve_context(
    store: Chroma,
    query: str,
    k: int = 5,
    fetch_k: int = 24,
    ctx_char_limit: int = 4500,
    alpha: float = 0.65,
    use_mmr: bool = True,
    mmr_lambda: float = 0.55
):
    """
    MMR + 임베딩/키워드 하이브리드 재랭킹.
    FAQ 제목(의문문) 과다 노출 완화.
    """
    if use_mmr and hasattr(store, "max_marginal_relevance_search_with_score"):
        raw = store.max_marginal_relevance_search_with_score(
            query, k=fetch_k, fetch_k=max(fetch_k * 2, 40), lambda_mult=mmr_lambda
        )
    else:
        raw = store.similarity_search_with_score(query, k=fetch_k)

    candidates, seen = [], set()
    for doc, dist in raw:
        key = (doc.metadata.get("chunk_id"), doc.page_content[:120])
        if key in seen:
            continue
        seen.add(key)
        sim = 1.0 / (1.0 + float(dist))
        text_for_kw = f"{doc.metadata.get('title','')}\n{doc.page_content}"
        ko = keyword_overlap_score(query, text_for_kw)
        candidates.append((doc, float(dist), sim, ko))

    def hybrid_score(item):
        _, _, sim, ko = item
        return alpha * sim + (1 - alpha) * ko

    candidates.sort(key=hybrid_score, reverse=True)
    results = candidates[:max(k, 5)]

    ctx_blocks, sources, running_len = [], [], 0
    for doc, dist, sim, ko in results:
        title = (doc.metadata.get("title") or "").strip()
        url = (doc.metadata.get("url") or "").strip()

        block = f"### {title}\n{doc.page_content}"
        if url:
            block += f"\n\n[원문]({url})"
        block += f"\n\n(유사도≈ {sim:.4f} / kw≈ {ko:.4f} / distance={dist:.4f})"

        blk = _truncate(block, max_chars=1400)
        if running_len + len(blk) > ctx_char_limit:
            continue
        ctx_blocks.append(blk)
        running_len += len(blk)
        sources.append({"title": title, "url": url, "score": sim, "kw": ko})

    return "\n\n---\n\n".join(ctx_blocks), sources


# ========= 증거(문장) 추출 & 안전 요약 =========
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+|[。！？」]\s*")

def build_sentence_pool(context_md: str, max_pool: int = 140):
    """
    의문문/메뉴/헤더 제거 + 중복 제거.
    """
    text = re.sub(r"^#+\s.*$", "", context_md, flags=re.MULTILINE)
    sents_raw = [s.strip() for s in SENT_SPLIT.split(text)]
    sents = []
    for s in sents_raw:
        if len(s) < 10 or len(s) > 400:
            continue
        if "원문](" in s or "유사도≈" in s or "kw≈" in s:
            continue
        if s.endswith("?") or "?" in s:
            continue
        if re.match(r"^[\-•\*]\s", s):
            continue
        sents.append(s)

    uniq, seen = [], set()
    for s in sents:
        key = s[:96]
        if key not in seen:
            uniq.append(s)
            seen.add(key)
        if len(uniq) >= max_pool:
            break
    return uniq

def embed_vecs(texts, emb: OllamaEmbeddings):
    if not texts:
        return []
    return emb.embed_documents(texts)

def cosine(a, b):
    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

_FACT_HINT = re.compile(r"(이다|였습니다|였다|입니다|으로|으로서|으로써|라고|명|세|살|년|월|일|장|절|족보|아들|딸|왕|개)")
_NUM = re.compile(r"\d")

def rerank_sentences(query: str, sents: list[str], emb: OllamaEmbeddings, top_n: int = 6, beta: float = 0.40):
    """
    코사인(임베딩) + 키워드 겹침(β) + 사실/숫자 boost, 의문문 penalty
    """
    if not sents:
        return []
    qv = emb.embed_query(query)
    dvs = embed_vecs(sents, emb)

    scored = []
    for s, dv in zip(sents, dvs):
        sim = cosine(qv, dv)
        ko = keyword_overlap_score(query, s)
        boost = 1.0
        if _FACT_HINT.search(s):
            boost += 0.08
        if _NUM.search(s):
            boost += 0.05
        if "?" in s:
            boost -= 0.30
        score = (beta * ko + (1 - beta) * sim) * boost
        scored.append((score, s, sim, ko))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_n]

def allowed_token_set(evidence_text: str):
    return set(tokenize_ko(evidence_text))

KOREAN_STOPWORDS = {
    "그리고","그러나","하지만","또한","또","혹은","또는","때문","때문에","이는","이런","이러한","그","그런","그러한",
    "것","사실","문제","해석","설명","동일","인물","이름","여러","기록","번역","차이","경우","본문","성경",
    "의미","사용","예","같다","있다","없다","로","은","는","이","가","을","를","에","에서","과","와","으로",
    "에게","보다","까지","부터","처럼","이다","였다","아니다","된다","되며","되어","되었다","등","따라서","그러므로","즉",
    "전통적으로","정확히","명시","근거","요약"
}

def is_safe_summary(answer: str, evidence_text: str, coverage_threshold: float = 0.60) -> bool:
    """
    증거 어휘 포함률(coverage)로 요약의 '근거 충실성'을 확인.
    coverage_threshold를 낮출수록 패러프레이즈 허용 폭이 커짐.
    """
    toks = tokenize_ko(answer)
    if not toks:
        return False
    allow = allowed_token_set(evidence_text)
    ok = [t for t in toks if (t in allow or t in KOREAN_STOPWORDS)]
    coverage = len(ok) / len(toks)
    if coverage < coverage_threshold:
        return False
    # 숫자/길이 3+ 토큰은 증거 외 생성 금지
    for t in toks:
        if (re.search(r"\d", t) or len(t) >= 3) and (t not in allow and t not in KOREAN_STOPWORDS):
            return False
    if answer.strip().endswith("?"):
        return False
    return True

def _gen_with_ollama(system: str, user: str, temperature: float, num_predict: int):
    res = client.chat(
        model=OLLAMA_GEN_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        options={"temperature": temperature, "num_predict": num_predict, "repeat_penalty": 1.15},
    )
    return (res.get("message", {}).get("content") or "").strip()

def safe_summarize(query: str, evidence_text: str, num_predict: int = 320, temperature: float = 0.15) -> str:
    """
    안전 요약(표현은 자연스럽게, 사실은 증거 안에서만).
    """
    system = (
        "너는 한국어 RAG 비서다. 아래 [증거] 범위를 절대 벗어나지 말라.\n"
        "- 숫자, 고유명사, 전문용어는 [증거]에 있는 것만 사용한다.\n"
        "- 표현은 자연스럽게 바꿔 말해도 된다(패러프레이즈 허용).\n"
        "- 질문/반문, 감탄, 추측, 사족 금지. 한 단락의 서술형 답만 출력."
    )
    prompt = (
        f"[증거]\n{evidence_text}\n\n"
        f"[질문]\n{query}\n\n"
        "형식: 한 단락, 서술형. 증거 밖 사실 금지. 표현은 자연스럽게."
    )
    return _gen_with_ollama(system, prompt, temperature=temperature, num_predict=num_predict)

def gentle_summarize(query: str, evidence_text: str, num_predict: int = 280, temperature: float = 0.22) -> str:
    """
    가드에 걸렸을 때 재시도용: 동일 제약이되 톤을 더 부드럽게.
    """
    system = (
        "너는 한국어 RAG 비서다. [증거]만 근거로 삼아 간결하고 자연스럽게 요약한다.\n"
        "- 사실/숫자/명칭은 증거에서만 가져온다.\n"
        "- 부드러운 연결어를 허용하되 과장/추측은 금지.\n"
        "- 한 단락으로 출력."
    )
    prompt = f"[증거]\n{evidence_text}\n\n[질문]\n{query}\n\n형식: 한 단락 요약."
    return _gen_with_ollama(system, prompt, temperature=temperature, num_predict=num_predict)

def answer_generic(query: str, context_md: str, emb_for_sent: OllamaEmbeddings,
                   summary_mode: str = "Safe Summary", num_predict: int = 320,
                   coverage_threshold: float = 0.55, base_temperature: float = 0.15) -> str:
    sents = build_sentence_pool(context_md, max_pool=160)
    top = rerank_sentences(query, sents, emb_for_sent, top_n=6, beta=0.40)
    if not top:
        return "컨텍스트에 없음"

    evidence_sents = [s for _, s, _, _ in top]
    evidence_text = " ".join(evidence_sents)

    if summary_mode == "Quotes only":
        return " ".join([s for s in evidence_sents if "?" not in s][:3])

    # 1차: 안전 요약
    ans = safe_summarize(query, evidence_text, num_predict=num_predict, temperature=base_temperature)
    if not ans.strip().endswith("?") and is_safe_summary(ans, evidence_text, coverage_threshold=coverage_threshold):
        return ans

    # 2차: 부드러운 재요약(패러프레이즈폭 살짝↑)
    ans2 = gentle_summarize(query, evidence_text, num_predict=max(160, num_predict-40),
                            temperature=min(0.30, base_temperature + 0.05))
    if not ans2.strip().endswith("?") and is_safe_summary(ans2, evidence_text, coverage_threshold=max(0.40, coverage_threshold-0.05)):
        return ans2

    # 3차: 인용 폴백
    quotes = [s for s in evidence_sents if "?" not in s][:3]
    return " ".join(quotes) if quotes else "컨텍스트에 없음"


# ========= Streamlit UI =========
st.set_page_config(page_title="Creation.kr Q&A (Chroma+Ollama RAG)", page_icon="🧭", layout="centered")
st.title("🤖 창조과학 Q&A 챗봇 — Chroma DB 기반 RAG (일반형·복붙 최소화 튜닝)")

with st.sidebar:
    st.subheader("검색/생성 설정")
    mode = st.radio(
        "Answer Mode",
        options=["Safe Summary (권장)", "Quotes only", "Strict Substring"],
        index=0,
        help="Safe Summary: 증거 기반 안전 요약(자연스러운 표현) / Quotes only: 증거 문장 인용 / Strict: 짧은 구 발췌"
    )
    fast_mode = st.toggle("⚡ Fast Mode (더 빠른 응답)", value=True, help="num_predict를 낮추고 컨텍스트 길이 제한")
    k = st.slider("Top-k", 2, 15, 5, 1)
    fetch_k = st.slider("Fetch-k (넓게 긁기)", 8, 64, 32, 4)
    ctx_limit = st.slider("Context 길이 제한(문자 수)", 2000, 9000, 4500, 500)
    alpha = st.slider("하이브리드 가중치 α (임베딩 비중)", 0.0, 1.0, 0.65, 0.05)
    mmr_lambda = st.slider("MMR λ (다양성 가중)", 0.1, 0.9, 0.55, 0.05)

    # 🔧 추가: 요약 자유도와 자연스러움 컨트롤
    paraphrase_level = st.select_slider(
        "요약 자유도(자연스러움)",
        options=["보수적", "보통", "자유"],
        value="보통",
        help="높일수록 표현이 자연스러워지지만, 증거 어휘 일치 비율은 낮아질 수 있음"
    )
    if paraphrase_level == "보수적":
        coverage_th = 0.60
        base_temp = 0.12
    elif paraphrase_level == "자유":
        coverage_th = 0.45
        base_temp = 0.22
    else:  # 보통
        coverage_th = 0.55
        base_temp = 0.15

    gen_tokens = 192 if fast_mode else 320
    st.caption("유사도/키워드 점수는 아래 '컨텍스트/출처'에서 확인할 수 있습니다.")

# 벡터 스토어 연결
try:
    store = get_store()
    st.success("Chroma 연결 성공 ✅ (기존 벡터 DB 사용)")
    warn_if_embedding_mismatch(store)
except Exception as e:
    st.error(f"Chroma 연결 실패: {e}")
    st.stop()

# (옵션) CSV 로드 (정합 확인 용도)
try:
    if CSV_DEFAULT:
        df = load_csv(CSV_DEFAULT)
        _ = docs_for_bm25(df)
except Exception:
    pass  # CSV 없어도 동작

# 세션 상태 (채팅 기록)
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 입력 & 응답
user_q = st.chat_input("질문을 입력하세요. (컨텍스트 기반, 일반형)")
if user_q:
    st.session_state["messages"].append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("검색 및 답변 생성 중…"):
            context_md, sources = retrieve_context(
                store, user_q, k=k, fetch_k=fetch_k, ctx_char_limit=ctx_limit, alpha=alpha,
                use_mmr=True, mmr_lambda=mmr_lambda
            )

            if not context_md.strip():
                st.warning("검색 결과가 비어 있습니다. 컬렉션/임베딩 모델 일치 여부를 확인하세요.")
                answer_md = "컨텍스트에 없음"
                sources = []
            else:
                emb_for_sent = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE)

                if mode == "Strict Substring":
                    STRICT_SYSTEM = (
                        "너는 한국어 RAG 비서다. 반드시 [컨텍스트]에 들어있는 텍스트만 사용한다. "
                        "정답은 [컨텍스트]에 문자 그대로 존재하는 짧은 구(2~40자)만 출력. "
                        "역질문/사족 금지. 없으면 '컨텍스트에 없음'만 출력."
                    )
                    prompt = f"[컨텍스트]\n{context_md}\n\n[질문]\n{user_q}\n\n형식: 정답 구(2~40자)만 출력."
                    res = client.chat(
                        model=OLLAMA_GEN_MODEL,
                        messages=[{"role": "system", "content": STRICT_SYSTEM},
                                  {"role": "user", "content": prompt}],
                        options={"temperature": 0.0, "top_p": 0.1, "repeat_penalty": 1.1, "num_predict": 128},
                    )
                    a = (res.get("message", {}).get("content") or "").strip()
                    if normalize_ko(a) and normalize_ko(a) in normalize_ko(context_md) and 2 <= len(a) <= 40:
                        answer_md = a
                    else:
                        answer_md = "컨텍스트에 없음"

                elif mode == "Quotes only":
                    # 인용만 출력
                    sents = build_sentence_pool(context_md, max_pool=160)
                    top = rerank_sentences(user_q, sents, emb_for_sent, top_n=6, beta=0.40)
                    quotes = [s for _, s, _, _ in top if "?" not in s][:3]
                    answer_md = " ".join(quotes) if quotes else "컨텍스트에 없음"

                else:  # Safe Summary (권장)
                    answer_md = answer_generic(
                        user_q, context_md, emb_for_sent,
                        summary_mode="Safe Summary", num_predict=gen_tokens,
                        coverage_threshold=coverage_th, base_temperature=base_temp
                    )

        # 1) 본문 답변
        st.markdown(answer_md)

        # 2) 원문 URL 목록
        if sources:
            st.markdown("\n\n**원문 링크**")
            seen = set()
            for s in sources:
                url = s.get("url", "")
                title = s.get("title", "") or url
                if url and url not in seen:
                    st.markdown(f"- [{title}]({url})")
                    seen.add(url)

        # 3) 컨텍스트/유사도는 expander로
        with st.expander("🔎 사용한 컨텍스트 / 출처 (유사도·키워드 포함)"):
            st.markdown(context_md if context_md else "_(비어 있음)_")

    st.session_state["messages"].append({"role": "assistant", "content": answer_md})
