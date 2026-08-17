"""
tests/run_chapter3.py — III장 노트북 4개를 대역으로 실행해 본다 (1회용 점검)
==========================================================================

레인 B 본체(run_notebooks.py)의 축소판이다. 하는 일은 세 가지뿐이다.

  1. 저장소의 노트북 파일을 **읽기만** 한다 (수정·저장 없음)
  2. prelude.install()로 대역을 세운다
  3. 코드를 실행하고, 책에 인쇄된 값과 대조한다

컨테이너에서는 구글 접속이 막혀 가짜 시트로만 확인했다.
이 스크립트는 **진짜 시트로 같은 일을 하는지**를 Actions에서 확인하기 위한 것이다.

    python tests/run_chapter3.py

필요한 것: numpy, pandas, scikit-learn (워크플로가 설치한다)
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NOTEBOOKS = os.path.join(ROOT, "notebooks")
sys.path.insert(0, HERE)

import prelude  # noqa: E402

# (파일명, 절, 완주 판정 문구, 정확도 하한, 책 인쇄값)
#   정확도 하한이 None이면 '완주 여부'만 본다.
#   3paras·8paras는 가중치 초기화와 SGD에 난수가 섞여 실행마다 값이 달라지므로
#   숫자로 판정하지 않는다. (지침서 6-4의 '혼동 조합 5개'와 같은 이유)
CASES = [
    ("Multiple Linear Regression_Hypertension_3paras.ipynb", "III-3",
     "%입니다.", None, None),
    ("Multiple Linear Regression_Hypertension_8paras.ipynb", "III-4",
     "예측된 고혈압 발병 확률", None, None),
    ("Binary Classification.ipynb", "III-5",
     "학습된 모델의 예측 정확도", 0.85, 0.8735),
    ("Multiclass Classification_Iris.ipynb", "III-6",
     "모델 정확도", 0.95, 0.9867),
]

ACCURACY_RE = re.compile(r"정확도[:：]\s*([0-9]*\.?[0-9]+)")


def load_source(filename: str) -> str:
    """노트북의 코드 셀을 순서대로 이어 붙인다 (파일은 건드리지 않는다)."""
    with open(os.path.join(NOTEBOOKS, filename), encoding="utf-8") as f:
        nb = json.load(f)
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    return "\n\n".join("".join(c["source"]) for c in cells)


def run_one(filename: str, section: str, marker: str,
            floor: float | None, printed: float | None) -> dict:
    result = {"notebook": filename, "section": section, "status": "FAIL", "detail": ""}
    src = load_source(filename)

    # 노트북마다 깨끗한 상태에서 시작한다
    for mod in [m for m in list(sys.modules) if m.startswith(("google", "gspread"))]:
        del sys.modules[mod]

    buf = io.StringIO()
    try:
        prelude.install(filename, verbose=False)
        with contextlib.redirect_stdout(buf):
            exec(compile(src, filename, "exec"), {"__name__": "__main__"})
    except prelude.PreludeError as exc:
        result["detail"] = f"대역 오류 (노트북 코드 문제 아님): {exc}"
        result["tail"] = buf.getvalue()[-800:]
        return result
    except Exception:
        result["detail"] = "노트북 실행 중 예외 — 아래 역추적 참조"
        result["traceback"] = traceback.format_exc(limit=4)
        result["tail"] = buf.getvalue()[-800:]
        return result

    out = buf.getvalue()

    leftover = prelude.unused_inputs()
    if leftover > 0:
        result["detail"] = (
            f"대역 입력 {leftover}개가 남았다 — 노트북의 input() 개수가 줄었을 수 있다. "
            f"prelude.py의 NOTEBOOK_INPUTS 확인 필요."
        )
        return result

    if marker not in out:
        result["detail"] = f"완주 문구를 찾지 못했다: {marker!r}"
        result["tail"] = out[-800:]
        return result

    if floor is not None:
        found = ACCURACY_RE.search(out)
        if not found:
            result["detail"] = "정확도 출력을 찾지 못했다 — 출력 형식이 바뀌었는지 확인할 것."
            result["tail"] = out[-800:]
            return result
        value = float(found.group(1))
        result["accuracy"] = value
        result["printed"] = printed
        if value <= floor:
            result["detail"] = f"정확도 {value:.4f} ≤ 하한 {floor}"
            return result
        gap = abs(value - printed) if printed is not None else 0.0
        result["status"] = "OK"
        result["detail"] = (
            f"정확도 {value:.4f} (책 {printed} · 차이 {gap:.4f})"
            if printed is not None else f"정확도 {value:.4f}"
        )
        if printed is not None and gap > 0.02:
            result["status"] = "WARN"
            result["detail"] += "  ⚠ 책 인쇄값과 벌어졌다 — 데이터나 라이브러리 변화 의심"
        return result

    result["status"] = "OK"
    result["detail"] = "완주"
    return result


def main() -> int:
    print(f"III장 노트북 {len(CASES)}개 — 대역 실행 점검")
    print(f"노트북 경로: {NOTEBOOKS}\n")

    results = [run_one(*c) for c in CASES]

    print("=" * 74)
    for r in results:
        icon = {"OK": "✅", "WARN": "⚠️", "FAIL": "❌"}[r["status"]]
        print(f"{icon} {r['section']:6s} {r['notebook']}")
        print(f"     {r['detail']}")
        if "traceback" in r:
            print("     --- 역추적 ---")
            for line in r["traceback"].strip().split("\n"):
                print(f"     {line}")
        if r["status"] != "OK" and r.get("tail"):
            print("     --- 마지막 출력 ---")
            for line in r["tail"].strip().split("\n")[-8:]:
                print(f"     {line[:100]}")
        print()

    ok = sum(1 for r in results if r["status"] == "OK")
    warn = sum(1 for r in results if r["status"] == "WARN")
    fail = sum(1 for r in results if r["status"] == "FAIL")
    print("=" * 74)
    print(f"정상 {ok} · 주의 {warn} · 실패 {fail}")

    # Actions 요약 탭에도 같은 표를 남긴다
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("\n## III장 노트북 대역 실행\n\n")
            f.write("| | 절 | 노트북 | 결과 |\n|---|---|---|---|\n")
            for r in results:
                icon = {"OK": "✅", "WARN": "⚠️", "FAIL": "❌"}[r["status"]]
                f.write(f"| {icon} | {r['section']} | `{r['notebook']}` | {r['detail']} |\n")
            f.write(f"\n**정상 {ok} · 주의 {warn} · 실패 {fail}**\n")

    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
