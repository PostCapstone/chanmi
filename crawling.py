# -*- coding: utf-8 -*-
import re
import csv
import os
import io
import builtins
from contextlib import contextmanager
import requests
from collections import OrderedDict
from bs4 import BeautifulSoup

URL = "https://creation.kr/animals/?idx=5088632&bmode=view"
OUT_CSV = "test.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
}

REFERENCES_HEADER_RE = re.compile(r"^(References?|Bibliography)\s*:?\s*$", re.I)
STAR_CHAMJO_RE      = re.compile(r"^\*?\s*참조[:：]?\s*$")

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def get_lines_for_p(node) -> list[str]:
    raw = node.get_text("\n", strip=True)
    raw = re.sub(r"[ \t]*\n[ \t]*", "\n", raw)
    return [l for l in (x.strip() for x in raw.split("\n")) if l]

def get_line_for_li(node) -> str:
    t = " ".join(node.stripped_strings)
    t = re.sub(r"\bCO\s*2\b", "CO2", t)
    return norm_space(t)

def is_meta_tail_line(txt: str) -> bool:
    t = norm_space(txt)
    if t.startswith("|"):
        return True
    # '출처 :', '주소 :', '번역 :' 라인 전부 제외
    return bool(re.match(r"^(출처|주소|번역)\s*[:：]", t))

def looks_like_author_or_cite(line: str) -> bool:
    t = norm_space(line)
    if not t:
        return False
    if t.lower().startswith("cite this article"):
        return True
    if re.search(r"\bby\s+[A-Z][A-Za-z.\- ]+$", t):
        return True
    if re.search(r"\b(is|was)\s+.*\b(Research|Professor|Coordinator|Author)\b", t, flags=re.I):
        return True
    return False

def strip_english_title_parenthesis(line: str) -> str:
    if re.fullmatch(r"\(\s*[^)]+\s*\)", line.strip()):
        return ""
    return line

def norm_url(u: str) -> str:
    u = (u or "").strip()
    if u.endswith("/"):
        u = u[:-1]
    return u

class LinkCollector:
    def __init__(self):
        self.url_items = OrderedDict()  # url_norm -> label or ""
        self.order = []

    def add(self, href: str, label: str = ""):
        u = norm_url(href)
        if not u:
            return
        if u not in self.url_items:
            self.url_items[u] = norm_space(label)
            self.order.append(u)
        else:
            if not self.url_items[u] and label:
                self.url_items[u] = norm_space(label)

    def add_from_line_if_any(self, line: str):
        m = re.search(r"(https?://\S+)", line)
        if not m:
            return False
        url = m.group(1).rstrip(").,;」』]")
        label = norm_space(line.replace(url, "").strip(" -–—:"))
        self.add(url, label)
        return True

    def to_lines(self):
        out = []
        for u in self.order:
            lbl = self.url_items[u]
            out.append(f"{lbl}|{u}" if lbl else u)
        return out

def scrape():
    r = requests.get(URL, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # 컨테이너
    container = None
    for div in soup.select("div"):
        cls = " ".join(div.get("class", []))
        if "_comment_body_" in cls:
            container = div
            break
    if container is None:
        container = soup

    blocks = list(container.select("p, li"))

    title = ""
    content_lines = []
    biblio_refs = []          # References(서지 텍스트 전용)
    further_links = LinkCollector()  # *참조(링크 전용)

    mode = "content"  # content | refs | chamjo
    seen_title = False
    stop_all = False  # 메타블록 시작 시 전체 중단

    for node in blocks:
        if stop_all:
            break

        # p / li별로 줄 뽑기
        if node.name == "p":
            lines = get_lines_for_p(node)
        else:  # li
            one = get_line_for_li(node)
            lines = [one] if one else []

        if not lines:
            continue

        # 문서 첫 줄을 제목으로
        if not seen_title:
            if node.name == "p":
                title = lines[0]
                rest = lines[1:]
                seen_title = True
                for ln in rest:
                    if is_meta_tail_line(ln):
                        stop_all = True
                        break
                    ln = strip_english_title_parenthesis(ln)
                    if ln and not looks_like_author_or_cite(ln):
                        content_lines.append(ln)
                if stop_all:
                    break
                continue
            else:
                # 첫 블록이 li이면 패스
                continue

        # 메타블록 만나면 거기서 중단 (어떤 모드든)
        if any(is_meta_tail_line(ln) for ln in lines):
            stop_all = True
            break

        # 섹션 헤더 감지 (p에서만)
        if node.name == "p":
            switched = False
            new_lines = []
            for ln in lines:
                if REFERENCES_HEADER_RE.fullmatch(ln):
                    mode = "refs"
                    switched = True
                    continue
                if STAR_CHAMJO_RE.fullmatch(ln):
                    mode = "chamjo"
                    switched = True
                    continue
                new_lines.append(ln)
            lines = new_lines

            if switched:
                if mode == "refs":
                    for ln in lines:
                        if not is_meta_tail_line(ln) and not looks_like_author_or_cite(ln):
                            biblio_refs.append(norm_space(ln))
                elif mode == "chamjo":
                    for ln in lines:
                        if not is_meta_tail_line(ln):
                            further_links.add_from_line_if_any(ln)
                continue

        # 모드별 수집
        if mode == "content":
            for ln in lines:
                if is_meta_tail_line(ln):
                    stop_all = True
                    break
                ln = strip_english_title_parenthesis(ln)
                if ln and not looks_like_author_or_cite(ln):
                    content_lines.append(ln)
            if stop_all:
                break

        elif mode == "refs":
            if node.name == "li":
                # References는 li 텍스트 한 줄만 (링크 무시)
                ln = lines[0]
                if ln and not looks_like_author_or_cite(ln):
                    biblio_refs.append(ln)
            else:
                for ln in lines:
                    if not is_meta_tail_line(ln) and not looks_like_author_or_cite(ln):
                        biblio_refs.append(norm_space(ln))

        elif mode == "chamjo":
            if node.name == "li":
                anchors = node.select("a[href]")
                if anchors:
                    for a in anchors:
                        href = a.get("href", "").strip()
                        label = norm_space(a.get_text(" ", strip=True))
                        if href:
                            further_links.add(href, label)
                else:
                    further_links.add_from_line_if_any(lines[0])
            else:
                for ln in lines:
                    if not further_links.add_from_line_if_any(ln):
                        # URL 없는 텍스트는 스킵(링크 중심)
                        pass

    # 중복 제거(순서 보존)
    def dedupe(seq):
        seen = set()
        out = []
        for s in seq:
            t = s.strip()
            if not t or t in seen:
                continue
            seen.add(t)
            out.append(t)
        return out

    row = {
        "url": URL,
        "title": norm_space(title),
        "content": "\n".join(dedupe(content_lines)),
        "references": "\n".join(dedupe(biblio_refs)),
        "further_refs": "\n".join(further_links.to_lines()),
    }

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

    print(f"Saved -> {OUT_CSV}")

# =========================
# 배치 래퍼 (임시 파일 생성 없이 메모리 캡처 후 한 번에 test.csv 작성)
# =========================

class _NoCloseStringIO(io.StringIO):
    # close()되어도 내용을 유지하려고 덮어씀
    def close(self):
        pass

@contextmanager
def _capture_scrape_writes():
    """
    scrape() 내부의 'with open(..., "w", ...)' 호출을 메모리 버퍼로 가로채고,
    종료 후 [(filepath, buffer), ...] 목록을 돌려준다.
    """
    original_open = builtins.open
    buffers = []

    def open_interceptor(file, mode='r', *args, **kwargs):
        if isinstance(mode, str) and 'w' in mode:
            buf = _NoCloseStringIO()
            buffers.append((file, buf))
            return buf
        return original_open(file, mode, *args, **kwargs)

    builtins.open = open_interceptor
    try:
        yield buffers
    finally:
        builtins.open = original_open

def run_batch_to_single(csv_path: str, start_row: int = 822, end_row: int = 1182, url_col: int = 0, output_csv: str = "test.csv"):
    """
    Creation.csv의 '물리적 행 번호'(1부터) start_row~end_row(포함)를 순회.
    각 URL마다 scrape()을 그대로 호출하되, 파일 출력은 메모리로 가로채고,
    마지막에 한 번만 output_csv(test.csv)에 합쳐서 기록.
    """
    global URL, OUT_CSV

    # 입력 CSV 읽기
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        all_rows = list(csv.reader(f))
    if not all_rows:
        raise ValueError(f"CSV 비어있음: {csv_path}")

    print(f"[BATCH] file={csv_path}, total_rows={len(all_rows)}, pick_rows={start_row}~{end_row}")

    collected = []   # list[dict] : 헤더 -> 값
    header = None

    # 선택한 물리적 행만 순회
    for phys_idx, row in enumerate(all_rows, start=1):
        if phys_idx < start_row or phys_idx > end_row:
            continue
        if not row or len(row) <= url_col:
            print(f"[SKIP {phys_idx}] 첫 열 비어있음")
            continue

        url = (row[url_col] or "").strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            print(f"[SKIP {phys_idx}] URL 아님: {url}")
            continue

        URL = url
        OUT_CSV = "test.csv"  # scrape()이 쓰는 경로(실제로는 메모리로 캡처됨)
        print(f"[{phys_idx}] Fetch -> {URL}")

        try:
            with _capture_scrape_writes() as bufs:
                scrape()  # *** 파싱/저장 로직은 그대로 실행되지만, 파일은 메모리로 흡수됨 ***

            # scrape()에서 쓴 CSV 텍스트 파싱(항상 1행 헤더 + 1행 데이터)
            if not bufs:
                print(f"[WARN {phys_idx}] 캡처된 쓰기 없음")
                continue

            # 마지막 write 타겟(보통 하나)
            _, mem = bufs[-1]
            csv_text = mem.getvalue()
            if not csv_text.strip():
                print(f"[WARN {phys_idx}] 비어있는 결과")
                continue

            rdr = csv.reader(io.StringIO(csv_text))
            rows_list = list(rdr)
            if len(rows_list) < 2:
                print(f"[WARN {phys_idx}] 데이터 행 없음")
                continue

            cur_header = rows_list[0]
            data = rows_list[1]

            if header is None:
                header = cur_header
            else:
                # 헤더 불일치 시 공통열 기준 맞추기 (일반적으로 동일)
                if header != cur_header:
                    print(f"[INFO] 헤더 불일치 감지. 기존 기준에 맞춰 정렬.")
                    # 누락 컬럼 채우기
                    mapping = {name: i for i, name in enumerate(cur_header)}
                    normalized = [data[mapping.get(h, -1)] if mapping.get(h, -1) != -1 and mapping.get(h, -1) < len(data) else "" for h in header]
                    data = normalized

            if header == cur_header:
                row_dict = {header[j]: (data[j] if j < len(data) else "") for j in range(len(header))}
            else:
                row_dict = {header[j]: (data[j] if j < len(data) else "") for j in range(len(header))}

            collected.append(row_dict)

        except Exception as e:
            print(f"[ERROR {phys_idx}] {e}")

    if not collected:
        raise RuntimeError("수집된 결과가 없습니다. (모든 행이 스킵/에러)")

    # 한 번만 최종 test.csv 작성
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=header)
        writer.writeheader()
        for rd in collected:
            writer.writerow(rd)

    print(f"[DONE] {len(collected)} rows -> {output_csv}")

if __name__ == "__main__":
    csv_path = "Creation.csv"  # 같은 폴더의 Creation.csv 사용
    if os.path.exists(csv_path):
        run_batch_to_single(csv_path, start_row=822, end_row=1182, url_col=0, output_csv="test.csv")
    else:
        # 없으면 단일 모드로 1회 실행(원래 동작 유지)
        scrape()
