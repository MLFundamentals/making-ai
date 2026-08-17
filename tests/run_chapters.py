"""
tests/run_chapters.py — III·IV·V장 노트북을 대역으로 실행해 본다
================================================================

run_chapter3.py를 III장 밖으로 넓힌 것이다. 하는 일은 같다.

  1. 저장소의 노트북 파일을 **읽기만** 한다 (수정·저장 없음)
  2. prelude.install()로 대역을 세운다
  3. 코드를 실행하고, 책에 인쇄된 값과 대조한다

    python tests/run_chapters.py                # III·IV·V장 전부
    python tests/run_chapters.py --chapters 4 5 # IV·V장만

필요한 것: numpy, pandas, scikit-learn, tensorflow(또는 tensorflow-cpu),
           gymnasium, seaborn, ipython  (워크플로가 설치한다)

판정 방식은 세 가지다.
  - complete : 완주 문구가 나왔는가만 본다 (난수 때문에 값이 흔들리는 노트북)
  - floor    : 지표가 하한을 넘는가
  - measure  : 값을 재서 보고만 한다. 실패시키지 않는다
               (에포크를 줄였을 때 정확도가 얼마나 되는지 아직 모르는 노트북)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import time
import traceback

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # TensorFlow 잡음 줄이기

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NOTEBOOKS = os.path.join(ROOT, "notebooks")
sys.path.insert(0, HERE)

import prelude  # noqa: E402

# 지표를 뽑는 정규식
ACC_FRACTION = (re.compile(r"정확도[:：]\s*([0-9]*\.?[0-9]+)\s*$"), 1.0)      # 0.9867
ACC_PERCENT = (re.compile(r"정확도[:：]\s*([0-9.]+)\s*%"), 100.0)             # 97.47 %
SUCCESS_RATE = (re.compile(r"성공률[:：]\s*([0-9.]+)\s*%"), 100.0)            # 76.90%

# ---------------------------------------------------------------------------
# 점검 대상
#
# printed = 책 또는 저장 출력의 값. 비교용이며 판정에 쓰이지 않는다.
# ---------------------------------------------------------------------------
CASES = [
    # ── III장 ────────────────────────────────────────────────────────────────
    dict(chapter=3, section="III-3", file="Multiple Linear Regression_Hypertension_3paras.ipynb",
         marker="%입니다.", mode="complete",
         note="가중치 초기화가 난수라 값이 흔들린다"),
    dict(chapter=3, section="III-4", file="Multiple Linear Regression_Hypertension_8paras.ipynb",
         marker="예측된 고혈압 발병 확률", mode="complete",
         note="SGD에 난수가 섞인다"),
    dict(chapter=3, section="III-5", file="Binary Classification.ipynb",
         marker="정확도", mode="floor", metric=ACC_FRACTION, floor=0.85, printed=0.8735,
         note="가중치를 0에서 시작해 결정적이다"),
    dict(chapter=3, section="III-6", file="Multiclass Classification_Iris.ipynb",
         marker="모델 정확도", mode="floor", metric=ACC_FRACTION, floor=0.95, printed=0.9867,
         note="외부 데이터를 쓰지 않는다"),

    # ── IV장 ─────────────────────────────────────────────────────────────────
    dict(chapter=4, section="IV-1", file="Perceptron_AND.ipynb",
         marker="학습 완료된 AND", mode="complete",
         note="input() 반복. 대역이 '1 0' 뒤에 'q'를 넣어 빠져나온다"),
    dict(chapter=4, section="IV-3", file="Multilayer Perceptron_XOR.ipynb",
         marker="학습 완료된 XOR", mode="complete",
         note="씨앗이 3개 다 걸려 있어 완전히 결정적"),
    dict(chapter=4, section="IV-4", file="MNIST_DNN.ipynb",
         marker="모델 정확도", mode="floor", metric=ACC_PERCENT, floor=0.95, printed=0.9747,
         note="6에포크로 상한. 3에포크 실측 0.9644 → 상향 후 0.97 안팎 예상"),
    dict(chapter=4, section="IV-5", file="MNIST_CNN.ipynb",
         marker="모델 정확도", mode="floor", metric=ACC_PERCENT, floor=0.97, printed=0.9894,
         note="4에포크로 상한. 2에포크 실측 0.9828 → 상향 후 0.987 안팎 예상"),
    dict(chapter=4, section="IV-6", file="RNN_hello world.ipynb",
         marker="예측된 다음 글자", mode="complete",
         note="input() 반복. 대역이 'exit'로 빠져나온다"),

    # ── V장 ──────────────────────────────────────────────────────────────────
    dict(chapter=5, section="V-1", file="Reinforcement Learning_Level 1.ipynb",
         marker=None, expect_var="trajectory", mode="complete",
         note="print가 하나도 없고 애니메이션만 만든다 → 변수 trajectory로 완주를 확인"),
    dict(chapter=5, section="V-2", file="Reinforcement Learning_Level 2.ipynb",
         marker="행동-결과", mode="complete"),
    dict(chapter=5, section="V-3", file="Reinforcement Learning_Level 3-1.ipynb",
         marker="성공률", mode="floor", metric=SUCCESS_RATE, floor=0.20, printed=0.3370,
         note="실측 범위 28.5~37.1% · 하한은 최저값보다 8%p 아래로 둔다"),
    dict(chapter=5, section="V-3", file="Reinforcement Learning_Level 3-2.ipynb",
         marker="성공률", mode="floor", metric=SUCCESS_RATE, floor=0.40, printed=0.6140,
         note="실측 범위 57.0~70.9% · 하한은 최저값보다 17%p 아래로 둔다"),
    dict(chapter=5, section="V-4", file="Reinforcement Learning_Level 4-1.ipynb",
         marker="성공률", mode="floor", metric=SUCCESS_RATE, floor=0.40, printed=0.7390,
         note="실측 범위 56.9~78.1%. 하한을 넉넉히 둔 이유"),
    dict(chapter=5, section="V-4", file="Reinforcement Learning_Level 4-2.ipynb",
         marker="성공률", mode="measure", metric=SUCCESS_RATE, printed=0.5981,
         note="실측 9.8~65.2%. 학습이 통째로 실패하는 실행이 있어 값으로 판정하지 않는다"),
]


def load_source(filename: str) -> str:
    """노트북의 코드 셀을 순서대로 이어 붙인다 (파일은 건드리지 않는다)."""
    with open(os.path.join(NOTEBOOKS, filename), encoding="utf-8") as f:
        nb = json.load(f)
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    return "\n\n".join("".join(c["source"]) for c in cells)


def run_one(case: dict) -> dict:
    name = case["file"]
    out = {"case": case, "status": "FAIL", "detail": "", "seconds": 0.0}

    try:
        src = load_source(name)
    except FileNotFoundError:
        out["detail"] = "노트북 파일을 찾을 수 없다 — 파일명이 바뀌었는지 확인할 것."
        return out

    # 노트북마다 깨끗한 상태에서 시작한다 (google/gspread 대역이 겹치지 않도록)
    for mod in [m for m in list(sys.modules) if m.startswith(("google.colab", "gspread"))]:
        del sys.modules[mod]

    buf = io.StringIO()
    started = time.time()
    try:
        prelude.install(name, verbose=False)
        namespace = {"__name__": "__main__"}
        with contextlib.redirect_stdout(buf):
            exec(compile(src, name, "exec"), namespace)
    except prelude.PreludeError as exc:
        out["seconds"] = time.time() - started
        out["detail"] = f"대역 오류 (노트북 코드 문제가 아니다): {exc}"
        out["tail"] = buf.getvalue()[-800:]
        return out
    except Exception:
        out["seconds"] = time.time() - started
        out["detail"] = "노트북 실행 중 예외 — 아래 역추적 참조"
        out["traceback"] = traceback.format_exc(limit=4)
        out["tail"] = buf.getvalue()[-800:]
        return out

    out["seconds"] = time.time() - started
    text = buf.getvalue()
    out["epochs_capped"] = prelude.epoch_cap_applied()

    leftover = prelude.unused_inputs()
    if leftover > 0:
        out["detail"] = (
            f"대역 입력 {leftover}개가 남았다 — 노트북의 input() 개수가 달라졌을 수 있다. "
            "prelude.py의 NOTEBOOK_INPUTS 확인 필요."
        )
        return out

    # 화면에 아무것도 찍지 않는 노트북은 변수의 존재로 완주를 확인한다
    var = case.get("expect_var")
    if var is not None:
        if var not in namespace:
            out["detail"] = f"완주 확인용 변수를 찾을 수 없다: {var!r}"
            out["tail"] = text[-800:]
            return out
        if not len(namespace[var]):
            out["detail"] = f"변수 {var!r}가 비어 있다 — 학습이 진행되지 않았다."
            return out

    if case["marker"] is not None and case["marker"] not in text:
        out["detail"] = f"완주 문구를 찾지 못했다: {case['marker']!r}"
        out["tail"] = text[-800:]
        return out

    mode = case["mode"]
    if mode == "complete":
        out["status"] = "OK"
        out["detail"] = "완주"
        return out

    pattern, scale = case["metric"]
    found = None
    for line in text.split("\n"):
        m = pattern.search(line.strip())
        if m:
            found = m                      # 마지막으로 찍힌 값을 쓴다
    if not found:
        out["detail"] = "지표 출력을 찾지 못했다 — 출력 형식이 바뀌었는지 확인할 것."
        out["tail"] = text[-800:]
        return out

    value = float(found.group(1)) / scale
    out["value"] = value
    printed = case.get("printed")
    shown = f"{value:.4f}"
    if printed is not None:
        shown += f" (책/저장값 {printed:.4f})"

    if mode == "measure":
        out["status"] = "MEASURE"
        out["detail"] = shown
        return out

    if value <= case["floor"]:
        out["detail"] = f"{shown} ≤ 하한 {case['floor']}"
        return out

    out["status"] = "OK"
    out["detail"] = f"{shown} · 하한 {case['floor']}"
    return out


ICON = {"OK": "✅", "MEASURE": "📏", "WARN": "⚠️", "FAIL": "❌"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapters", nargs="+", type=int, default=[3, 4, 5],
                    help="점검할 장 번호 (기본: 3 4 5)")
    args = ap.parse_args()

    cases = [c for c in CASES if c["chapter"] in args.chapters]
    print(f"점검 대상: {len(cases)}개 (장 {args.chapters})")
    print(f"노트북 경로: {NOTEBOOKS}\n")

    results = []
    for case in cases:
        print(f"▶ {case['section']:6s} {case['file']}", flush=True)
        r = run_one(case)
        results.append(r)
        print(f"   {ICON[r['status']]} {r['detail']}  ({r['seconds']:.1f}초)\n", flush=True)

    print("=" * 78)
    for r in results:
        c = r["case"]
        print(f"{ICON[r['status']]} {c['section']:6s} {c['file']}")
        print(f"     {r['detail']}  ({r['seconds']:.1f}초)")
        if r.get("epochs_capped"):
            for asked, capped in r["epochs_capped"]:
                print(f"     에포크 {asked} → {capped} 로 낮춰 실행")
        if c.get("note"):
            print(f"     ※ {c['note']}")
        if "traceback" in r:
            for line in r["traceback"].strip().split("\n"):
                print(f"     {line}")
        if r["status"] == "FAIL" and r.get("tail"):
            print("     --- 마지막 출력 ---")
            for line in r["tail"].strip().split("\n")[-8:]:
                print(f"     {line[:100]}")
        print()

    ok = sum(1 for r in results if r["status"] == "OK")
    meas = sum(1 for r in results if r["status"] == "MEASURE")
    fail = sum(1 for r in results if r["status"] == "FAIL")
    total_time = sum(r["seconds"] for r in results)
    print("=" * 78)
    print(f"정상 {ok} · 측정 {meas} · 실패 {fail} · 합계 {total_time/60:.1f}분")

    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(f"\n## 노트북 대역 실행 (장 {args.chapters})\n\n")
            f.write("| | 절 | 노트북 | 결과 | 소요 |\n|---|---|---|---|---|\n")
            for r in results:
                c = r["case"]
                f.write(f"| {ICON[r['status']]} | {c['section']} | `{c['file']}` | "
                        f"{r['detail']} | {r['seconds']:.0f}초 |\n")
            f.write(f"\n**정상 {ok} · 측정 {meas} · 실패 {fail} · 합계 {total_time/60:.1f}분**\n")
            if meas:
                f.write("\n📏 는 값을 재기만 한 것이다. 이 값을 보고 임계값을 정한 뒤 "
                        "`run_chapters.py`의 mode를 floor로 바꾼다.\n")

    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
