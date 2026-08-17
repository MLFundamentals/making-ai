"""
tests/probe_gviz.py — gviz CSV가 노트북이 기대하는 모양인지 한 번 확인한다
=========================================================================

레인 B의 대역 gspread는 구글시트를 공개 gviz 주소로 읽는다.
행·열 수는 레인 A가 이미 확인했지만, **값의 표기 형식**은 확인한 적이 없다.

문제가 되는 경우:
  시트에 백분율 서식이 걸려 있으면 gviz는 0.72 대신 "72.00%"를 준다.
  → 노트북의 df['Hypertension_risk'].astype(float) 가 그 자리에서 깨진다.

이 스크립트는 네 시트의 앞부분을 그대로 찍어보고, 숫자로 바꿀 수 없는 칸이
있으면 지목한다. 컨테이너에서는 구글 접속이 막혀 있으므로 **Actions에서 한 번**
돌린다. 한 번 통과하면 다시 돌릴 일은 없다.

    python tests/probe_gviz.py

표준 라이브러리만 사용한다.
"""

from __future__ import annotations

import csv
import io
import sys
import urllib.request

GVIZ = "https://docs.google.com/spreadsheets/d/{id}/gviz/tq?tqx=out:csv"

# (시트 ID, 절, 기대 행수(헤더 제외), 기대 열수, 숫자여야 하는 열)
SHEETS = [
    ("12RmvY2KI-6UcMEHXX4E65R_oso4965jNpgAbKeIgI9c", "III-3", 10, 4,
     ["BMI", "Exercise_per_week", "Hypertension_risk"]),
    ("1eC4YnMRr5a6v2EmxcZBUb-mGv6k8Qz-8UcNTSoeM2qU", "III-4", 100, 9,
     ["Age", "BMI", "Exercise_per_week", "Alcohol_intake",
      "Smoking", "Family_history", "Hypertension_risk"]),
    ("1MI-HWfwR5-Dp-A8dJ4DfOVGGau-9QdysfvsQHvZ5VY0", "III-5", 2000, 7, "ALL"),
    ("1SyBg64IX_uX290hsIwGH3zZwe_Aeb1SBLx146zSGIro", "IX-3", 1200, 26, None),
]

BAD_MARKS = ("%", "₩", "$", ",")   # 서식이 섞여 들어온 흔적


def fetch(sheet_id: str) -> list[list[str]]:
    with urllib.request.urlopen(GVIZ.format(id=sheet_id), timeout=30) as r:
        text = r.read().decode("utf-8", errors="replace")
    if text.lstrip()[:1] == "<":
        raise RuntimeError("CSV가 아니라 HTML이 왔다 — 비공개 전환 의심")
    return [row for row in csv.reader(io.StringIO(text)) if any(c.strip() for c in row)]


def main() -> int:
    problems = 0

    for sheet_id, section, exp_rows, exp_cols, numeric in SHEETS:
        print("=" * 72)
        print(f"{section}  {sheet_id}")
        try:
            rows = fetch(sheet_id)
        except Exception as exc:
            print(f"  ❌ 읽기 실패: {type(exc).__name__}: {exc}")
            problems += 1
            continue

        header, data = rows[0], rows[1:]
        print(f"  행 {len(data)} (기대 {exp_rows}) · 열 {len(header)} (기대 {exp_cols})")
        if len(data) != exp_rows or len(header) != exp_cols:
            print("  ⚠ 크기 불일치 — assets.csv 및 원고 기재값 재확인 필요")
            problems += 1

        print(f"  헤더: {header}")
        for row in data[:2]:
            print(f"  샘플: {row}")

        # 숫자여야 하는 칸이 정말 숫자로 바뀌는지
        if numeric is None:
            print("  (숫자 검사 생략 — 노트북이 gspread로 읽지 않는 시트)")
            continue
        cols = list(range(len(header))) if numeric == "ALL" else [
            header.index(c) for c in numeric if c in header
        ]
        missing = [] if numeric == "ALL" else [c for c in numeric if c not in header]
        if missing:
            print(f"  ❌ 헤더에 없는 열: {missing}")
            problems += 1

        bad = []
        for r_i, row in enumerate(data):
            for c_i in cols:
                if c_i >= len(row):
                    continue
                cell = row[c_i].strip()
                if cell == "":
                    continue
                try:
                    float(cell)
                except ValueError:
                    mark = [m for m in BAD_MARKS if m in cell]
                    bad.append((r_i + 2, header[c_i], cell, mark))
        if bad:
            problems += 1
            print(f"  ❌ 숫자로 못 바꾸는 칸 {len(bad)}개 — astype(float)에서 깨진다")
            for r, c, v, m in bad[:5]:
                hint = f"  ← 서식 흔적 {m}" if m else ""
                print(f"      {r}행 [{c}] = {v!r}{hint}")
        else:
            print("  ✅ 숫자 열 전부 float 변환 가능")

    print("=" * 72)
    if problems:
        print(f"문제 {problems}건 — 대역 gspread를 그대로 쓰면 안 된다.")
        return 1
    print("전부 정상 — 대역 gspread가 노트북에 넘길 값이 원래와 같은 모양이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
