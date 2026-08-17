#!/usr/bin/env python3
"""
레인 A — 외부 자산 링크 체커
『손으로 배우는 인공지능』 실습 코드 점검 체계

assets.csv에 적힌 외부 자산이 아직 살아 있는지 확인한다.
표준 라이브러리만 사용하므로 GitHub Actions에서 pip install 없이 바로 돌아간다.

사용법:
    python check_assets.py                     # 전체 점검
    python check_assets.py --list              # 점검 없이 목록만 출력
    python check_assets.py --only entry,drive  # 특정 분류만
    python check_assets.py --id page-making-ai # 특정 항목만
    python check_assets.py --include-manual    # 수동 확인 항목까지 표에 표시
    python check_assets.py --strict            # WARN도 실패로 간주
    python check_assets.py --json report.json --markdown summary.md

종료 코드:
    0  이상 없음 (또는 medium 등급만 실패)
    1  critical/high 등급 항목 실패  ← 워크플로가 이 값을 보고 알림을 띄운다
    2  설정 오류 (CSV 없음/형식 오류)

환경변수:
    GDRIVE_API_KEY   설정하면 드라이브 폴더를 HTML 추정 대신 Drive API로 확인한다.
                     (공개 저장소에 키를 넣지 말 것. Settings → Secrets에 등록해 참조)
    HF_TOKEN         설정하면 게이트된 HuggingFace 모델도 조회한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "assets.csv"

USER_AGENT = (
    "Mozilla/5.0 (compatible; book-asset-checker/1.0; "
    "+https://mlfundamentals.github.io/making-ai/)"
)

VALID_CHECKS = {"http", "hf", "gsheet", "drive", "manual"}
VALID_SEVERITY = {"critical", "high", "medium"}
BLOCKING_SEVERITY = {"critical", "high"}

KST = timezone(timedelta(hours=9))

STATUS_MARK = {"OK": "✅", "WARN": "⚠️ ", "FAIL": "❌", "SKIP": "⏭️ "}


# ─────────────────────────────────────────────────────────────
# 자료 구조
# ─────────────────────────────────────────────────────────────

@dataclass
class Asset:
    id: str
    category: str
    name: str
    target: str
    check: str
    severity: str
    section: str = ""
    notes: str = ""

    @property
    def probe_url(self) -> str:
        """실제로 두드릴 주소."""
        if self.check == "hf":
            return f"https://huggingface.co/api/models/{self.target}"
        if self.check == "gsheet":
            return (
                f"https://docs.google.com/spreadsheets/d/{self.target}"
                "/gviz/tq?tqx=out:csv&sheet=Sheet1"
            )
        if self.check == "drive":
            if os.getenv("GDRIVE_API_KEY"):
                key = urllib.parse.quote(os.environ["GDRIVE_API_KEY"])
                return (
                    f"https://www.googleapis.com/drive/v3/files/{self.target}"
                    f"?fields=id,name,trashed,mimeType&key={key}"
                )
            return f"https://drive.google.com/drive/folders/{self.target}"
        return self.target

    @property
    def display_url(self) -> str:
        """보고서에 찍을 주소 (API 키는 가린다)."""
        url = self.probe_url
        if "key=" in url:
            url = url.split("&key=")[0] + "&key=***"
        return url


@dataclass
class Result:
    asset: Asset
    status: str = "SKIP"        # OK / WARN / FAIL / SKIP
    detail: str = ""
    elapsed_ms: int = 0
    http_status: int | None = None
    extra: dict = field(default_factory=dict)

    @property
    def blocking(self) -> bool:
        return self.status == "FAIL" and self.asset.severity in BLOCKING_SEVERITY


# ─────────────────────────────────────────────────────────────
# HTTP 하부 계층
# ─────────────────────────────────────────────────────────────

class Response:
    def __init__(self, status, final_url, headers, body: bytes):
        self.status = status
        self.final_url = final_url
        self.headers = headers
        self.body = body

    @property
    def content_type(self) -> str:
        return (self.headers.get("Content-Type") or "").lower()

    def text(self, limit: int = 4096) -> str:
        return self.body[:limit].decode("utf-8", errors="replace")


def request(
    url: str,
    method: str = "GET",
    timeout: int = 20,
    read_bytes: int = 65536,
    headers: dict | None = None,
) -> Response:
    """예외를 삼키지 않고 그대로 올린다. HTTPError는 응답으로 변환한다."""
    hdrs = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    if headers:
        hdrs.update(headers)

    req = urllib.request.Request(url, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = b"" if method == "HEAD" else resp.read(read_bytes)
            return Response(resp.status, resp.geturl(), dict(resp.headers), body)
    except urllib.error.HTTPError as e:
        try:
            body = e.read(read_bytes)
        except Exception:
            body = b""
        return Response(e.code, e.geturl(), dict(e.headers or {}), body)


def request_with_retry(url: str, retries: int = 2, backoff: float = 1.5, **kwargs) -> Response:
    """네트워크 오류와 5xx는 재시도한다. 일시적 장애를 실패로 오인하지 않기 위함."""
    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = request(url, **kwargs)
            if resp.status >= 500 and attempt < retries:
                time.sleep(backoff * (attempt + 1))
                continue
            return resp
        except Exception as e:                      # URLError, timeout, SSL 등
            last_exc = e
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
    raise last_exc  # type: ignore[misc]


def looks_like_login_page(resp: Response) -> bool:
    if "text/html" not in resp.content_type:
        return False
    body = resp.text().lower()
    markers = (
        "accounts.google.com", "signinchooser", "로그인이 필요",
        "sign in", "액세스 권한이 필요", "request access", "you need access",
    )
    return any(m in body for m in markers)


# ─────────────────────────────────────────────────────────────
# 체커
# ─────────────────────────────────────────────────────────────

def check_http(asset: Asset, timeout: int, retries: int) -> tuple[str, str, dict]:
    url = asset.probe_url
    # HEAD를 거부하는 서버가 많아 405/403/501이면 GET으로 되묻는다.
    resp = request_with_retry(url, retries=retries, method="HEAD", timeout=timeout)
    if resp.status in (403, 405, 501) or resp.status >= 500:
        resp = request_with_retry(
            url, retries=retries, method="GET", timeout=timeout,
            headers={"Range": "bytes=0-2047"},
        )

    extra = {"http_status": resp.status, "final_url": resp.final_url}
    size = resp.headers.get("Content-Length")
    if size:
        extra["content_length"] = size

    if resp.status in (200, 206):
        detail = f"HTTP {resp.status}"
        if size:
            detail += f" · {int(size):,} bytes"
        if resp.final_url.rstrip("/") != url.rstrip("/"):
            return "WARN", f"{detail} · 리다이렉트 → {resp.final_url}", extra
        return "OK", detail, extra

    if resp.status in (301, 302, 307, 308):
        return "WARN", f"HTTP {resp.status} 리다이렉트 미해결", extra
    return "FAIL", f"HTTP {resp.status}", extra


def check_hf(asset: Asset, timeout: int, retries: int) -> tuple[str, str, dict]:
    headers = {}
    if os.getenv("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"

    resp = request_with_retry(
        asset.probe_url, retries=retries, method="GET",
        timeout=timeout, headers=headers, read_bytes=200000,
    )
    extra = {"http_status": resp.status}

    if resp.status == 401:
        return "FAIL", "인증 필요 — 비공개 전환 의심", extra
    if resp.status == 403:
        body = resp.text().lower()
        if "gated" in body or "access" in body:
            return "WARN", "접근 제한(게이트) — 독자가 약관 동의해야 다운로드 가능", extra
        return "WARN", "HTTP 403 — 게이트 여부 또는 네트워크 차단 확인 필요", extra
    if resp.status == 404:
        return "FAIL", "모델 없음 — 삭제 또는 이름 변경", extra
    if resp.status != 200:
        return "FAIL", f"HTTP {resp.status}", extra

    try:
        meta = json.loads(resp.body.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return "WARN", "JSON 응답 아님 — 수동 확인 필요", extra

    extra["model_id"] = meta.get("id", "")
    extra["downloads"] = meta.get("downloads")
    extra["last_modified"] = meta.get("lastModified", "")

    if meta.get("gated"):
        extra["gated"] = meta["gated"]
        return "WARN", f"게이트 모델(gated={meta['gated']}) — 독자 동의 절차 안내 필요", extra
    if meta.get("private"):
        return "FAIL", "비공개 전환됨", extra
    if meta.get("disabled"):
        return "FAIL", "비활성화됨", extra

    modified = (extra["last_modified"] or "")[:10]
    return "OK", f"공개 · 최종수정 {modified or '?'}", extra


def check_gsheet(asset: Asset, timeout: int, retries: int) -> tuple[str, str, dict]:
    """gviz CSV 엔드포인트. 인증 없이 응답·헤더·행 수까지 확인 가능(책에서 소개한 방식)."""
    resp = request_with_retry(
        asset.probe_url, retries=retries, method="GET",
        timeout=timeout, read_bytes=4_000_000,
    )
    extra = {"http_status": resp.status}

    if resp.status == 404:
        return "FAIL", "시트 없음 — 삭제 또는 ID 변경", extra
    if resp.status != 200:
        return "FAIL", f"HTTP {resp.status}", extra
    if looks_like_login_page(resp) or "text/html" in resp.content_type:
        return "FAIL", "비공개 전환 의심 — 로그인 페이지 반환", extra

    text = resp.body.decode("utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "FAIL", "빈 응답", extra

    try:
        rows = list(csv.reader(lines))
    except csv.Error:
        return "WARN", "CSV 파싱 실패 — 수동 확인 필요", extra

    header = rows[0]
    data_rows = len(rows) - 1
    extra["columns"] = len(header)
    extra["data_rows"] = data_rows
    extra["header"] = header[:12]

    if data_rows < 1:
        return "FAIL", "데이터 행 없음", extra
    return "OK", f"{data_rows:,}행 × {len(header)}열 · 헤더: {', '.join(header[:3])}", extra


def check_drive(asset: Asset, timeout: int, retries: int) -> tuple[str, str, dict]:
    use_api = bool(os.getenv("GDRIVE_API_KEY"))
    resp = request_with_retry(
        asset.probe_url, retries=retries, method="GET",
        timeout=timeout, read_bytes=200000,
    )
    extra = {"http_status": resp.status, "mode": "api" if use_api else "html"}

    if use_api:
        if resp.status == 404:
            return "FAIL", "폴더 없음 — 삭제 또는 ID 변경", extra
        if resp.status in (401, 403):
            return "FAIL", "접근 거부 — 공개 설정 해제 의심", extra
        if resp.status != 200:
            return "FAIL", f"Drive API HTTP {resp.status}", extra
        try:
            meta = json.loads(resp.body.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            return "WARN", "Drive API 응답 파싱 실패", extra
        extra["drive_name"] = meta.get("name", "")
        if meta.get("trashed"):
            return "FAIL", "휴지통으로 이동됨", extra
        return "OK", f"공개 폴더 '{meta.get('name', '?')}'", extra

    # HTML 추정 경로 — 구글 페이지 구조가 바뀌면 흔들리므로 판정을 보수적으로 둔다.
    if resp.status == 404:
        return "FAIL", "폴더 없음(404)", extra
    if resp.status != 200:
        return "FAIL", f"HTTP {resp.status}", extra
    if "accounts.google.com" in resp.final_url:
        return "FAIL", "로그인 요구 — 공개 설정 해제 의심", extra
    if looks_like_login_page(resp):
        return "WARN", "로그인/권한 안내 문구 감지 — 브라우저 시크릿 창으로 확인할 것", extra
    return "OK", "공개 접근 가능(HTML 추정) · GDRIVE_API_KEY 설정 시 정확도 향상", extra


CHECKERS = {
    "http": check_http,
    "hf": check_hf,
    "gsheet": check_gsheet,
    "drive": check_drive,
}


def run_check(asset: Asset, timeout: int, retries: int) -> Result:
    if asset.check == "manual":
        return Result(
            asset, "SKIP",
            "수동 확인 대상 — 드라이브 폴더 목록 대조(레인 C)로만 확인",
        )

    started = time.monotonic()
    try:
        status, detail, extra = CHECKERS[asset.check](asset, timeout, retries)
    except Exception as e:
        status, detail, extra = "FAIL", f"{type(e).__name__}: {e}", {}

    return Result(
        asset=asset,
        status=status,
        detail=detail,
        elapsed_ms=int((time.monotonic() - started) * 1000),
        http_status=extra.pop("http_status", None),
        extra=extra,
    )


# ─────────────────────────────────────────────────────────────
# CSV 로딩
# ─────────────────────────────────────────────────────────────

def load_assets(path: Path) -> list[Asset]:
    if not path.exists():
        raise SystemExit(f"[설정 오류] 인벤토리 파일이 없다: {path}")

    required = {"id", "category", "name", "target", "check", "severity"}
    assets: list[Asset] = []
    seen: set[str] = set()

    # 인코딩 사고를 사람이 읽을 수 있는 오류로 바꾼다.
    # (기업용 문서보안(DRM) 프로그램이 CSV를 암호화해 올린 사례가 실제로 있었다.)
    raw = path.read_bytes()
    if b"DRM" in raw[:64] or b"encrypted" in raw[:200].lower():
        raise SystemExit(
            f"[설정 오류] {path} 가 문서보안(DRM) 프로그램에 암호화된 상태다.\n"
            "  → GitHub 웹 편집기(연필 아이콘)에서 내용을 직접 붙여넣어 다시 커밋할 것.\n"
            "     로컬에 파일을 저장하면 DRM이 다시 걸린다."
        )
    try:
        raw.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise SystemExit(
            f"[설정 오류] {path} 를 UTF-8로 읽을 수 없다 (위치 {e.start}, 바이트 0x{raw[e.start]:02x}).\n"
            "  → 파일이 UTF-8이 아니거나 손상되었다. GitHub 웹 편집기에서 다시 붙여넣을 것."
        )

    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"[설정 오류] 필수 열 누락: {', '.join(sorted(missing))}")

        for lineno, row in enumerate(reader, start=2):
            if not (row.get("id") or "").strip():
                continue
            asset = Asset(
                id=row["id"].strip(),
                category=(row.get("category") or "").strip(),
                name=(row.get("name") or "").strip(),
                target=(row.get("target") or "").strip(),
                check=(row.get("check") or "").strip(),
                severity=(row.get("severity") or "").strip(),
                section=(row.get("section") or "").strip(),
                notes=(row.get("notes") or "").strip(),
            )
            if asset.id in seen:
                raise SystemExit(f"[설정 오류] {lineno}행: id 중복 '{asset.id}'")
            if asset.check not in VALID_CHECKS:
                raise SystemExit(
                    f"[설정 오류] {lineno}행 '{asset.id}': check='{asset.check}' "
                    f"(가능: {', '.join(sorted(VALID_CHECKS))})"
                )
            if asset.severity not in VALID_SEVERITY:
                raise SystemExit(
                    f"[설정 오류] {lineno}행 '{asset.id}': severity='{asset.severity}' "
                    f"(가능: {', '.join(sorted(VALID_SEVERITY))})"
                )
            if not asset.target:
                raise SystemExit(f"[설정 오류] {lineno}행 '{asset.id}': target 비어 있음")
            seen.add(asset.id)
            assets.append(asset)

    if not assets:
        raise SystemExit(f"[설정 오류] 점검 대상이 없다: {path}")
    return assets


# ─────────────────────────────────────────────────────────────
# 출력
# ─────────────────────────────────────────────────────────────

def _width(s: str) -> int:
    """한글을 2칸으로 계산한다."""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s: str, n: int) -> str:
    return s + " " * max(0, n - _width(s))


def print_report(results: list[Result], include_manual: bool, elapsed: float) -> None:
    shown = [r for r in results if include_manual or r.status != "SKIP"]

    name_w = max([_width(r.asset.name) for r in shown] + [12])
    id_w = max([_width(r.asset.id) for r in shown] + [8])

    print()
    print("─" * 100)
    print("레인 A — 외부 자산 점검")
    print("─" * 100)

    current_cat = None
    for r in shown:
        if r.asset.category != current_cat:
            current_cat = r.asset.category
            print(f"\n[{current_cat}]")
        mark = STATUS_MARK.get(r.status, "  ")
        print(
            f"  {mark} {_pad(r.asset.name, name_w)}  "
            f"{_pad(r.asset.id, id_w)}  {r.detail}"
        )

    counts = {s: sum(1 for r in results if r.status == s) for s in ("OK", "WARN", "FAIL", "SKIP")}
    print()
    print("─" * 100)
    print(
        f"정상 {counts['OK']} · 주의 {counts['WARN']} · 실패 {counts['FAIL']} · "
        f"수동 {counts['SKIP']}   (소요 {elapsed:.1f}초)"
    )

    problems = [r for r in results if r.status in ("FAIL", "WARN")]
    if problems:
        print("\n조치가 필요한 항목")
        for r in sorted(problems, key=lambda x: (x.status != "FAIL", x.asset.severity)):
            loc = f"본문 {r.asset.section}" if r.asset.section not in ("", "-") else "본문 위치 없음"
            print(f"  {STATUS_MARK[r.status]} [{r.asset.severity}] {r.asset.name} ({loc})")
            print(f"       {r.detail}")
            print(f"       {r.asset.display_url}")
    print("─" * 100)


def summarize(results: list[Result]) -> tuple[str, str]:
    """site/index.html의 check-result에 넣을 한 줄."""
    fails = [r for r in results if r.status == "FAIL"]
    warns = [r for r in results if r.status == "WARN"]
    if fails:
        return "fail", f"점검 실패 {len(fails)}건 — 확인 중"
    if warns:
        return "warn", f"주의 {len(warns)}건 — 확인 중"
    return "ok", "전체 정상"


def write_json(results: list[Result], path: Path, elapsed: float) -> None:
    state, message = summarize(results)
    now = datetime.now(KST)
    payload = {
        "checked_at": now.strftime("%Y-%m-%d"),
        "checked_at_iso": now.isoformat(timespec="seconds"),
        "state": state,
        "check_result": message,          # ← site/index.html 갱신 스크립트가 그대로 사용
        "elapsed_sec": round(elapsed, 1),
        "counts": {
            s: sum(1 for r in results if r.status == s)
            for s in ("OK", "WARN", "FAIL", "SKIP")
        },
        "results": [
            {
                "id": r.asset.id,
                "category": r.asset.category,
                "name": r.asset.name,
                "section": r.asset.section,
                "severity": r.asset.severity,
                "check": r.asset.check,
                "url": r.asset.display_url,
                "status": r.status,
                "http_status": r.http_status,
                "detail": r.detail,
                "elapsed_ms": r.elapsed_ms,
                **({"extra": r.extra} if r.extra else {}),
            }
            for r in results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(results: list[Result], path: Path) -> None:
    """GitHub Actions 요약(job summary)용."""
    _, message = summarize(results)
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
    lines = [
        "## 레인 A — 외부 자산 점검",
        "",
        f"**{message}** · {now}",
        "",
        "| | 항목 | 본문 | 등급 | 결과 |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        if r.status == "SKIP":
            continue
        lines.append(
            f"| {STATUS_MARK.get(r.status, '').strip()} | {r.asset.name} | "
            f"{r.asset.section} | {r.asset.severity} | {r.detail} |"
        )
    manual = [r for r in results if r.status == "SKIP"]
    if manual:
        lines += ["", f"<sub>수동 확인 대상 {len(manual)}건은 레인 C에서 폴더 목록 대조.</sub>"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description="레인 A — 외부 자산 링크 체커",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="인벤토리 CSV 경로")
    p.add_argument("--only", default="", help="분류 필터 (쉼표 구분: entry,drive,gsheet,dataset,hf_model)")
    p.add_argument("--id", default="", help="특정 id만 점검 (쉼표 구분)")
    p.add_argument("--include-manual", action="store_true", help="수동 확인 항목도 표에 표시")
    p.add_argument("--timeout", type=int, default=20, help="요청 제한시간(초)")
    p.add_argument("--retries", type=int, default=2, help="재시도 횟수")
    p.add_argument("--workers", type=int, default=8, help="동시 요청 수")
    p.add_argument("--json", type=Path, default=SCRIPT_DIR / "report.json", help="JSON 보고서 경로")
    p.add_argument("--markdown", type=Path, default=None, help="마크다운 요약 경로")
    p.add_argument("--strict", action="store_true", help="WARN과 medium 실패도 종료코드 1")
    p.add_argument("--list", action="store_true", help="점검 없이 목록만 출력")
    args = p.parse_args()

    try:
        assets = load_assets(args.csv)
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 2

    if args.only:
        wanted = {c.strip() for c in args.only.split(",") if c.strip()}
        assets = [a for a in assets if a.category in wanted]
    if args.id:
        wanted = {i.strip() for i in args.id.split(",") if i.strip()}
        assets = [a for a in assets if a.id in wanted]
    if not assets:
        print("[설정 오류] 필터 결과가 비었다.", file=sys.stderr)
        return 2

    if args.list:
        print(f"점검 대상 {len(assets)}건 ({args.csv})\n")
        for a in assets:
            print(f"  [{a.severity:8}] {a.check:7} {a.id}")
            print(f"             {a.display_url}")
        return 0

    print(f"인벤토리 {args.csv} — {len(assets)}건 점검 시작")
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(
            lambda a: run_check(a, args.timeout, args.retries), assets
        ))
    elapsed = time.monotonic() - started

    # 보고서를 먼저 남긴다. 화면 출력에서 예외가 나도 결과가 사라지지 않도록.
    if args.json:
        write_json(results, args.json, elapsed)
    if args.markdown:
        write_markdown(results, args.markdown)

    print_report(results, args.include_manual, elapsed)

    if args.json:
        print(f"\nJSON 보고서: {args.json}")
    if args.markdown:
        print(f"마크다운 요약: {args.markdown}")

    if args.strict:
        bad = any(r.status in ("FAIL", "WARN") for r in results)
    else:
        bad = any(r.blocking for r in results)
    return 1 if bad else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:
        # 자산 문제(1)와 스크립트 문제(2)를 구분해서 알린다.
        print(f"\n[스크립트 오류] {type(e).__name__}: {e}", file=sys.stderr)
        print("자산이 죽은 것이 아니라 점검 도구 자체의 문제일 수 있다.", file=sys.stderr)
        sys.exit(2)
