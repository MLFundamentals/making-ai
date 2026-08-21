#!/usr/bin/env python3
"""
점검 결과 → 페이지 반영
『손으로 배우는 인공지능』 실습 코드 점검 체계

check_assets.py가 만든 report.json을 읽어, index.html의 정해진 자리만 갱신한다.
  1) id="checked-at"     → 점검 날짜
  2) id="check-result"   → 상태 문구
  3) id="status-dot"     → 점 색깔 (초록 / 주황 / 빨강)
  4) id="asset-count"    → 자산 건수 (report.json 의 결과 수를 센다)
  5) id="notebook-count" → 노트북 편수 (notebooks/ 를 실제로 센다)
  6) id="lane-c-at"      → 레인 C 마지막 실행일 (tests/lane_c_last_run.txt)

그 외의 내용은 한 글자도 건드리지 않는다.

갱신과 별개로 **매주 잔소리하는 항목**이 둘 있다. 손으로 박아 두고 잊는 것이
이 프로젝트의 사고 유형이라, 잊을 수 없게 만들어 둔다.
  - index.html 에 제보 이메일 자리표가 남아 있는가
  - 레인 C 를 마지막으로 돈 지 100일이 넘었는가
둘 다 경고만 낸다. 종료 코드는 바꾸지 않는다 — 페이지 갱신을 막을 일은 아니다.

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

NOTEBOOK_DIR = REPO_ROOT / "notebooks"
LANE_C_FILE = SCRIPT_DIR / "lane_c_last_run.txt"

# 레인 C 를 이만큼 안 돌면 잔소리한다. 분기(90일)에 여유를 조금 얹은 값.
LANE_C_STALE_DAYS = 100

# 계약이 정해지면 채울 자리표. 채워질 때까지 매주 경고가 나간다.
EMAIL_PLACEHOLDER = "이메일 주소"


def count_notebooks() -> str:
    """notebooks/ 의 .ipynb 를 실제로 센다.

    손으로 '53' 이라 적어 두면 노트북이 늘거나 줄어도 페이지는 53 이라고
    계속 말한다. 이 프로젝트에서 거짓말을 한 것은 늘 결과가 아니라 라벨이었다.
    """
    if not NOTEBOOK_DIR.is_dir():
        raise SystemExit(f"[오류] 노트북 폴더가 없다: {NOTEBOOK_DIR}")
    n = len(list(NOTEBOOK_DIR.rglob("*.ipynb")))
    if n == 0:
        raise SystemExit(f"[오류] 노트북을 하나도 찾지 못했다: {NOTEBOOK_DIR}")
    return str(n)


def read_lane_c() -> tuple[str, int | None]:
    """레인 C 마지막 실행일과 경과일. (표시할 문자열, 경과일) 반환.

    경과일이 None 이면 날짜를 읽지 못한 것이다.
    """
    import datetime

    if not LANE_C_FILE.exists():
        return "(기록 없음)", None
    for line in LANE_C_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            d = datetime.date.fromisoformat(line)
        except ValueError:
            return "(형식 오류)", None
        return d.isoformat(), (datetime.date.today() - d).days
    return "(기록 없음)", None


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


def warn_leftovers(html: str, lane_c_days: int | None) -> None:
    """매주 잔소리해야 하는 것들. 경고만 내고 종료 코드는 건드리지 않는다.

    ⚠ 이 함수는 '바뀔 내용이 없다'로 일찍 돌아가기 **전에** 불려야 한다.
       페이지가 그대로라고 해서 잔소리까지 쉬면, 잔소리가 필요한 바로 그
       상황(아무도 아무것도 안 하는 동안)에 조용해진다.
    """
    if EMAIL_PLACEHOLDER in html:
        print(f'::warning::index.html 의 제보 이메일이 아직 자리표("{EMAIL_PLACEHOLDER}")다.')
        print("  책에 인쇄되는 것은 이 페이지 주소 하나뿐이고, 제보 창구는 여기 얹혀 있다.")
        print("  자리표인 채로 인쇄되면 독자가 연락할 곳이 없다.")

    if lane_c_days is None:
        print("::warning::레인 C 마지막 실행일을 읽지 못했다 "
              "(tests/lane_c_last_run.txt).")
    elif lane_c_days > LANE_C_STALE_DAYS:
        print(f"::warning::레인 C(Colab 실전 점검)를 돈 지 {lane_c_days}일 지났다 "
              f"(기준 {LANE_C_STALE_DAYS}일).")
        print("  레인 A·B 가 초록불이어도 설치 줄·대형 모델·그림은 확인되지 않는다.")
        print("  tests/lane_c_checklist.md 를 따라 Colab 에서 돌 것.")


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
        asset_count = str(len(report["results"]))      # ← 추가
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[오류] 보고서 형식이 맞지 않는다: {e}", file=sys.stderr)
        return 2

    if state not in DOT_CLASS:
        print(f"[오류] 알 수 없는 상태값: {state}", file=sys.stderr)
        return 2

    html = original = args.html.read_text(encoding="utf-8")

    notebook_count = count_notebooks()
    lane_c_text, lane_c_days = read_lane_c()

    try:
        html, old_date = replace_span_text(html, "checked-at", checked_at)
        html, old_result = replace_span_text(html, "check-result", check_result)
        html, old_count = replace_span_text(html, "asset-count", asset_count)
        html, old_nb = replace_span_text(html, "notebook-count", notebook_count)
        html, old_lane_c = replace_span_text(html, "lane-c-at", lane_c_text)
        html, old_dot = replace_dot_class(html, DOT_CLASS[state])
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 2

    changes = [
        ("최종 점검일", old_date, checked_at),
        ("상태 문구", old_result, check_result),
        ("자산 건수", old_count, asset_count),
        ("노트북 편수", old_nb, notebook_count),
        ("실환경 점검일", old_lane_c, lane_c_text),
        ("점 색깔", old_dot, DOT_CLASS[state]),
    ]
    # 페이지가 그대로여도 잔소리는 한다 — 아래 이른 반환보다 먼저 부른다.
    warn_leftovers(html, lane_c_days)

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
