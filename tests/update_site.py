#!/usr/bin/env python3
"""
점검 결과 → 페이지 반영
『손으로 배우는 인공지능』 실습 코드 점검 체계

check_assets.py가 만든 report.json을 읽어, index.html의 세 곳만 갱신한다.
  1) id="checked-at"   → 점검 날짜
  2) id="check-result" → 상태 문구
  3) id="status-dot"   → 점 색깔 (초록 / 주황 / 빨강)

그 외의 내용은 한 글자도 건드리지 않는다.

사용법:
    python update_site.py                    # 기본 경로로 갱신
    python update_site.py --dry-run          # 무엇이 바뀌는지만 출력
    python update_site.py --html path.html --report path.json

종료 코드:
    0  갱신 완료 (또는 바뀔 내용이 없음)
    2  파일 없음 / 앵커를 찾지 못함 / 보고서 형식 오류
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_REPORT = SCRIPT_DIR / "report.json"
DEFAULT_HTML = REPO_ROOT / "index.html"

# report.json의 state → 점 색깔 클래스
DOT_CLASS = {"ok": "dot", "warn": "dot warn", "fail": "dot fail"}


def replace_span_text(html: str, anchor_id: str, new_text: str) -> tuple[str, str]:
    """id="anchor_id"인 span의 안쪽 텍스트만 교체. (새 html, 이전 텍스트) 반환."""
    pattern = re.compile(
        r'(<span[^>]*\bid="' + re.escape(anchor_id) + r'"[^>]*>)(.*?)(</span>)',
        re.DOTALL,
    )
    m = pattern.search(html)
    if not m:
        raise SystemExit(f'[오류] id="{anchor_id}" 앵커를 찾지 못했다.')
    old = m.group(2)
    return pattern.sub(lambda x: x.group(1) + new_text + x.group(3), html, count=1), old


def replace_dot_class(html: str, new_class: str) -> tuple[str, str]:
    """id="status-dot"인 요소의 class 속성만 교체."""
    pattern = re.compile(r'(<span\s+class=")([^"]*)("\s+id="status-dot")')
    m = pattern.search(html)
    if not m:
        # class와 id의 순서가 뒤바뀐 경우도 시도
        pattern = re.compile(r'(<span\s+id="status-dot"\s+class=")([^"]*)(")')
        m = pattern.search(html)
    if not m:
        raise SystemExit('[오류] id="status-dot" 앵커를 찾지 못했다.')
    old = m.group(2)
    return pattern.sub(lambda x: x.group(1) + new_class + x.group(3), html, count=1), old


def main() -> int:
    p = argparse.ArgumentParser(description="점검 결과를 페이지에 반영")
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--html", type=Path, default=DEFAULT_HTML)
    p.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않고 변경 내용만 출력")
    args = p.parse_args()

    if not args.report.exists():
        print(f"[오류] 보고서가 없다: {args.report}", file=sys.stderr)
        return 2
    if not args.html.exists():
        print(f"[오류] 페이지가 없다: {args.html}", file=sys.stderr)
        return 2

    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        checked_at = report["checked_at"]
        check_result = report["check_result"]
        state = report["state"]
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[오류] 보고서 형식이 맞지 않는다: {e}", file=sys.stderr)
        return 2

    if state not in DOT_CLASS:
        print(f"[오류] 알 수 없는 상태값: {state}", file=sys.stderr)
        return 2

    html = original = args.html.read_text(encoding="utf-8")

    try:
        html, old_date = replace_span_text(html, "checked-at", checked_at)
        html, old_result = replace_span_text(html, "check-result", check_result)
        html, old_dot = replace_dot_class(html, DOT_CLASS[state])
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 2

    changes = [
        ("최종 점검일", old_date, checked_at),
        ("상태 문구", old_result, check_result),
        ("점 색깔", old_dot, DOT_CLASS[state]),
    ]
    changed = [c for c in changes if c[1] != c[2]]

    if not changed:
        print("바뀔 내용이 없다. 페이지는 이미 최신이다.")
        return 0

    for label, old, new in changed:
        print(f"  {label}: {old}  →  {new}")

    if args.dry_run:
        print("\n--dry-run이므로 파일을 쓰지 않았다.")
        return 0

    if html == original:
        print("변경 사항 없음.")
        return 0

    args.html.write_text(html, encoding="utf-8")
    print(f"\n갱신 완료: {args.html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
