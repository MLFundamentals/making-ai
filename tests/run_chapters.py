"""
tests/run_chapters.py — 노트북을 대역으로 실행해 본다 (I·II·III·IV·V장)
================================================================

하는 일은 셋뿐이다.

  1. 저장소의 노트북 파일을 **읽기만** 한다 (수정·저장 없음)
  2. prelude.install()로 대역을 세운다
  3. 코드를 실행하고, 책에 인쇄된 값과 대조한다

    python tests/run_chapters.py                # 등록된 장 전부
    python tests/run_chapters.py --chapters 1 2 # 파이썬 기초만
    python tests/run_chapters.py --chapters 4 5 # IV·V장만

장 번호는 **책의 장**을 따른다. 파이썬 기초 7개는 I-4·I-5절이라 1장,
Array_Dimension은 II-3절이라 2장이다. 폴더 이름(python-basics)이 아니다.

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
import tempfile
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
APPLE_PRICE = (re.compile(r"사과 10개일 때 예상 가격[:：]\s*([0-9.]+)"), 1.0)  # 9.9798

# ---------------------------------------------------------------------------
# 점검 대상
#
# printed     = 책 또는 저장 출력의 값. 비교용이며 판정에 쓰이지 않는다.
# marker      = 이 문구가 화면에 나와야 완주로 본다. None이면 문구를 보지 않는다.
# expect_text = marker 말고 **추가로** 확인할 문구들 (선택).
#               파이썬 기초처럼 셀마다 예시가 따로 노는 노트북에서 쓴다.
#               맨 끝 문구 하나만 보면 중간 셀이 조용히 달라져도 초록불이 켜지기 때문이다.
#               ⚠ 라이브러리 판형에 따라 흔들리는 문구는 넣지 말 것
#                  (numpy 배열 출력의 칸 맞춤 등). 무해한 차이로 빨간불이 뜬다.
# ---------------------------------------------------------------------------
CASES = [
    # ── I장 · 파이썬 기초 ────────────────────────────────────────────────────
    # 외부 자산도 무거운 모델도 없다. 전부 몇 초 안에 끝난다.
    dict(chapter=1, section="I-4-2", file="python-basics/01_Variables_Expressions_Outputs_Inputs.ipynb",
         marker="MAP ≈", mode="complete",
         expect_text=["Your BMI is 22.86"],
         note="input() 2번. 대역이 '175'·'70'을 넣는다 → BMI 22.86 이 나와야 대역이 제대로 흐른 것"),
    dict(chapter=1, section="I-4-3", file="python-basics/02_Basic Data Types.ipynb",
         marker="고혈압 대상자: ['Lee']", mode="complete",
         note="난수·외부 자산 없음. 완전히 결정적"),
    dict(chapter=1, section="I-4-4", file="python-basics/03_If Statement_Loop.ipynb",
         marker="첫번째 발견:", mode="complete",
         expect_text=["Lee → 고혈압 + 40세 이상"],
         note="맨 끝 셀이 range(3) 출력이라 완주 문구로 쓸 수 없다 → 그 앞 셀을 기준으로 삼았다"),
    dict(chapter=1, section="I-4-5", file="python-basics/04_Function.ipynb",
         marker="Help on function get_pass_scores", mode="complete",
         expect_text=["합격자 점수: [72, 88, 91, 79]"],
         note="맨 끝이 help(). 도움말 출력이 화면으로 나오는지까지 확인한다"),
    dict(chapter=1, section="I-5-2", file="python-basics/05_Numpy.ipynb",
         marker="과체중 BMI:", mode="complete",
         expect_text=["평균 나이: 38.75"],
         note="배열 자체의 출력 모양은 확인하지 않는다 — numpy 판형에 따라 칸 맞춤이 달라진다"),
    dict(chapter=1, section="I-5-3", file="python-basics/06_Pandas.ipynb",
         marker="나이 표준편차:", mode="complete",
         expect_text=["최대 혈압: 140"],
         note="checkup.csv 를 만든다 → 실행기가 임시 폴더에서 돌리므로 저장소는 더러워지지 않는다"),
    dict(chapter=1, section="I-5-4", file="python-basics/07_Matplotlib.ipynb",
         marker=None, expect_var="df", mode="complete",
         note="print가 하나도 없고 그림만 그린다 → V-1과 같은 방식으로 변수 df 로 완주를 확인"),

    # ── II장 ─────────────────────────────────────────────────────────────────
    dict(chapter=2, section="II-3", file="python-basics/Array_Dimension.ipynb",
         marker="차원 수: 4", mode="complete",
         expect_text=["==== 3차원: 텐서 ===="],
         note="단일 셀. 0~4차원을 차례로 찍는다"),

    # ── III장 ────────────────────────────────────────────────────────────────
    dict(chapter=3, section="III-2", file="Linear Regression_Apple Price.ipynb",
         marker="사과 10개일 때 예상 가격", mode="floor",
         metric=APPLE_PRICE, floor=9.5, printed=9.9798,
         expect_text=["Epoch 2000:"],
         note="시작점이 난수지만 2000번이면 늘 10 근처로 모인다(300회 실측 9.9684~9.9985). "
              "하한 9.5는 값의 흔들림이 아니라 학습이 발산해 버린 경우를 잡기 위한 것이다"),
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
    dict(chapter=3, section="III-7", file="MNIST_Sample Image Viewer_1.ipynb",
         marker=None, expect_var="sample_img", mode="complete",
         note="그림만 그리고 print가 없다 → 변수 sample_img 로 완주를 확인"),
    dict(chapter=3, section="III-7", file="MNIST_Sample Image Viewer_2.ipynb",
         marker=None, expect_var="sample_img", mode="complete",
         note="위와 같다. 격자를 덧그리는 판형"),
    dict(chapter=3, section="III-7", file="MNIST_Sample Image Viewer_3.ipynb",
         marker=None, expect_var="sample_img", mode="complete",
         note="위와 같다. seaborn 히트맵을 쓰는 유일한 III장 노트북"),
    dict(chapter=3, section="III-8", file="MNIST_Multiclass Logistic Regression.ipynb",
         marker="모델이 예측한 레이블", mode="floor",
         metric=ACC_PERCENT, floor=0.90, printed=0.9272,
         note="printed는 **저장 출력**이다(2026-08-18 재실행). 책 인쇄값은 92.39%. "
              "씨앗이 없어 실행마다 0.3%p 안팎으로 흔들린다 — 원고는 그대로 둔다"),
    dict(chapter=3, section="III-10", file="Evaluation Metrics.ipynb",
         marker="[ 가장 많이 혼동한 조합 5개 ]", mode="floor",
         metric=ACC_PERCENT, floor=0.90, printed=0.9268,
         expect_text=["정확도 : 0.95", "정밀도 : 0.00", "재현율 : 0.00",
                      "정확도 : 0.93", "정밀도 : 0.40", "재현율 : 0.80", "F1 점수: 0.53"],
         note="예제 1(모델 A·B)은 난수가 전혀 없어 책 인쇄값과 한 자리까지 같아야 한다 → expect_text로 못박았다. "
              "'혼동 조합 5개'는 4·5위가 동률이라 실행마다 순서가 바뀌므로 문구만 확인하고 값은 보지 않는다"),

    # ── IV장 ─────────────────────────────────────────────────────────────────
    dict(chapter=4, section="IV-1", file="Perceptron_AND.ipynb",
         marker="학습 완료된 AND", mode="complete",
         note="input() 반복. 대역이 '1 0' 뒤에 'q'를 넣어 빠져나온다"),
    dict(chapter=4, section="IV-3", file="Multilayer Perceptron_XOR.ipynb",
         marker="학습 완료된 XOR", mode="complete",
         note="씨앗이 3개 다 걸려 있어 완전히 결정적"),
    dict(chapter=4, section="IV-4", file="MNIST_DNN.ipynb",
         marker="모델 정확도", mode="floor", metric=ACC_PERCENT, floor=0.95, printed=0.9747,
         note="상한 없음(10에포크). 6에포크 실측 0.9665 → 책 값 0.9747 부근 예상"),
    dict(chapter=4, section="IV-5", file="MNIST_CNN.ipynb",
         marker="모델 정확도", mode="floor", metric=ACC_PERCENT, floor=0.97, printed=0.9912,
         note="상한 없음(10에포크). 저장 출력을 99.12%로 갱신함(2026-08-18)"),
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

    # ── VI장 ─────────────────────────────────────────────────────────────────
    # 착수 순서: 모델 없는 것 → 의심 API → Gradio + 실제 모델.
    # 한 번에 다 넣지 않는다. 문제가 생겼을 때 원인이 어디인지 바로 보이게 하기 위해서다.
    dict(chapter=6, section="VI-1", file="NLP_Preprocessing.ipynb",
         marker="④ 인덱스 시퀀스로 변환된 문장:", mode="complete",
         expect_text=["['나는', '머신러닝을', '정말', '좋아한다']",
                      "'파이썬으로': 9", "[9, 3, 8, 10]"],
         note="순수 파이썬(re 모듈만). 난수도 모델도 없어 **완전히 결정적**이다 — "
              "책 인쇄값과 한 자리까지 같아야 하므로 결과를 못박았다"),
    dict(chapter=6, section="VI-3", file="NLP_Word Embedding.ipynb",
         marker="✅ 학습된 워드 임베딩", mode="complete",
         expect_text=["amazing", "waiting", "was"],
         note="지침서가 의심 후보로 지목한 keras Tokenizer·pad_sequences 를 쓴다. "
              "임베딩 값 자체는 씨앗이 없어 매번 다르므로, 단어 목록만 확인한다"),
    # VI-9 — 2026-08-18 복구 완료.
    # transformers v5가 pipeline("question-answering")·("summarization")·("translation")을
    # 제거해 통째로 고장났던 노트북이다. 세 셀 모두 모델을 직접 부르는 방식으로 고쳤고,
    # 파일 이름도 NLP_Transformers-pipeline.ipynb → NLP_Transformers.ipynb 로 바뀌었다
    # (더 이상 pipeline 을 쓰지 않으므로).
    dict(chapter=6, section="VI-9", file="NLP_Transformers.ipynb",
         cells=[0, 1],
         marker="[대역] 화면을 띄우는 대신", mode="complete",
         note="셀 3개가 서로 독립된 프로그램이다(각자 import·모델·Interface를 갖고, "
              "변수도 qa_/sum_/trans_ 로 나뉘어 있다). "
              "**셀 2는 NLLB-200-600M(약 2.4GB)이라 레인 B 제외 대상**이므로 0·1만 돌린다. "
              "셀 0 질의응답(xlm-roberta) · 셀 1 요약(KoBART). 셀 2는 레인 C에서 확인할 것. "
              "노트북은 판번호를 못박아 두었지만 여기서는 !pip 줄을 지우고 최신판으로 돌린다 — "
              "**빨간불은 '독자가 지금 못 쓴다'가 아니라 '고정을 풀면 깨진다'는 뜻이다**"),

    # VI-9 — 같은 절의 두 번째 노트북. 위의 NLP_Transformers 와는 별개의 파일이다.
    # 2026-08-20 복구 완료: transformers v5 는 기본 어텐션이 sdpa 라서
    # output_attentions=True 만으로는 어텐션 행렬을 아예 만들지 않는다(빈 튜플이 온다).
    # 네 셀 모두 attn_implementation="eager" 를 명시해 되살렸다.
    dict(chapter=6, section="VI-9", file="NLP_Transformer with BertViz.ipynb",
         marker="['[CLS]', 'the', 'cat', 'sat'",
         mode="complete", expect_var="attn",
         note="코드 셀 5개 전부 실행한다 — 0 준비 · 1 BERT head_view · 2 GPT-2 head_view · "
              "3 Marian model_view · 4 Neuron View(bertviz 자체 BERT 복사본). "
              "그림 자체는 브라우저가 그리므로 러너가 볼 수 없지만, **bertviz 의 세 함수는 "
              "모두 호출 즉시 어텐션을 소비한다** (head_view·model_view → format_attention(), "
              "neuron_view.show() → get_attention()). 어텐션이 비면 torch.stack 에서 "
              "RuntimeError, 차원이 4가 아니면 ValueError 로 터진다 — 이번 고장도 그렇게 잡혔다. "
              "따라서 '조용히 통과'는 일어나지 않는다. marker 는 셀 1의 print(tokens) 출력이고 "
              "난수가 없어 완전히 결정적이다. expect_var='attn' 은 셀 2(GPT-2)의 어텐션 "
              "튜플이 비어 있지 않은지 보는 덤이다(셀 3·4가 덮어쓰지 않는다). "
              "⚠ 셀 4의 model·tokenizer 는 len() 이 되지 않으므로 expect_var 로 쓰면 안 된다. "
              "⚠ 워크플로에 bertviz 설치가 필요하다 — 레인 B 는 !pip 줄을 지운다"),

    dict(chapter=6, section="VI-10", file="NLP_BERT_pipeline.ipynb",
         marker="[대역] 화면을 띄우는 대신", mode="complete",
         note="셀 3개 모두 작은 모델이라 통째로 돌린다 — bert-base-uncased(마스크 채우기) · "
              "jhgan/ko-sbert-nli(문장 유사도) · distilbert-sst2(감정 분석). "
              "sentence-transformers 설치가 필요하다. "
              "출력 길이 실측 247·16·22자 — **상한은 걸지 않는다.** 이 셀들에는 길이를 정하는 "
              "인자 자체가 없어서, VI-11 같은 '인자가 조용히 무시되는' 고장이 일어날 수 없다"),
    dict(chapter=6, section="VI-11", file="NLP_GPT_pipeline.ipynb",
         marker="[대역] 화면을 띄우는 대신", mode="complete",
         gradio_max_chars=300,
         note="Gradio 대역의 첫 실전 시험. KoGPT2를 실제로 내려받아 문장을 생성한다. "
              "생성 결과는 매번 다르므로 '대역이 fn을 실제로 불렀는가'만 본다. "
              "출력 상한 300자는 실측 근거가 있다 — 정상 85자 vs max_length가 무시됐을 때 721자"),
]


# contextlib.chdir 은 Python 3.11부터 있다. 워크플로는 3.12를 쓰지만,
# 저자가 손에 있는 오래된 파이썬으로 돌려볼 수도 있으므로 대체품을 둔다.
if hasattr(contextlib, "chdir"):
    _chdir = contextlib.chdir
else:                                                   # pragma: no cover
    @contextlib.contextmanager
    def _chdir(path):
        before = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(before)


# 노트북에만 있는 문법. 파이썬은 이 줄들을 이해하지 못하고 SyntaxError 로 죽는다.
#   !pip install ...  → 지운다. 워크플로가 미리 설치하므로 다시 깔 필요가 없고,
#                       빠뜨린 라이브러리가 있으면 import 에서 요란하게 드러난다.
#   %매직            → 지운다 (%matplotlib inline 등, 화면 표시용이라 서버에선 무의미)
#   그 밖의 !명령     → 실제로 실행한다. !wget·!unzip 처럼 자료를 내려받는 줄은
#                       진짜로 돌아야 노트북이 이어지기 때문이다.
SHELL_LINE = re.compile(r"^(\s*)([!%])(.+?)\s*$")
PIP_LINE = re.compile(r"^(pip3?|%pip)\b|^\S*python\S*\s+-m\s+pip\b")


def rewrite_notebook_only_lines(src: str) -> tuple[str, list[str]]:
    """셸·매직 줄을 파이썬이 실행할 수 있는 형태로 바꾼다. (바뀐 코드, 건너뛴 설치줄)"""
    out, skipped = [], []
    for line in src.split("\n"):
        m = SHELL_LINE.match(line)
        if not m:
            out.append(line)
            continue
        indent, sigil, body = m.groups()
        if sigil == "%" or PIP_LINE.match(body.strip()):
            skipped.append(body.strip())
            out.append(f"{indent}# [실행기가 건너뜀] {sigil}{body}")
        else:
            out.append(f"{indent}__prelude_shell__({body.strip()!r})")
    return "\n".join(out), skipped


def load_source(filename: str, cells: list[int] | None = None) -> tuple[str, list[str]]:
    """노트북의 코드 셀을 순서대로 이어 붙인다 (파일은 건드리지 않는다).

    cells 를 주면 **그 번호의 코드 셀만** 골라 잇는다.
    한 노트북 안에 서로 독립된 프로그램이 여러 개 들어 있고, 그중 하나가
    레인 B 제외 대상(대형 모델)일 때 쓴다. 번호는 코드 셀만 센 것이다.

    ⚠ 이 기능은 위험하다. 셀을 빼면 그만큼 점검되지 않는데 결과는 초록불이다.
      `skip-test` 태그를 버린 이유와 같은 함정이므로, 반드시
      ① 빠진 셀이 남은 셀과 독립인지 확인하고 ② note 에 이유를 적고
      ③ 실행기가 표에 '셀 N개 중 M개만' 이라고 드러내게 한다.
    """
    with open(os.path.join(NOTEBOOKS, filename), encoding="utf-8") as f:
        nb = json.load(f)
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    if cells is not None:
        picked = []
        for i in cells:
            if i >= len(code_cells):
                raise IndexError(f"{filename}: 코드 셀이 {len(code_cells)}개뿐인데 {i}번을 지정했다")
            picked.append(code_cells[i])
        code_cells = picked
    return rewrite_notebook_only_lines("\n\n".join("".join(c["source"]) for c in code_cells))


def run_one(case: dict) -> dict:
    name = case["file"]
    out = {"case": case, "status": "FAIL", "detail": "", "seconds": 0.0}

    try:
        src, skipped_installs = load_source(name, case.get("cells"))
    except FileNotFoundError:
        out["detail"] = "노트북 파일을 찾을 수 없다 — 파일명이 바뀌었는지 확인할 것."
        return out

    # 노트북마다 깨끗한 상태에서 시작한다 (google/gspread 대역이 겹치지 않도록)
    for mod in [m for m in list(sys.modules) if m.startswith(("google.colab", "gspread"))]:
        del sys.modules[mod]

    # 앞 노트북이 열어 둔 그림을 닫는다. plt.show()는 화면 없는 서버에서 그림을 닫지
    # 않기 때문에, 그냥 두면 20장을 넘긴 순간부터 무해한 경고가 로그를 채운다.
    try:
        import matplotlib.pyplot as _plt
        _plt.close("all")
    except ImportError:
        pass

    buf = io.StringIO()
    started = time.time()
    try:
        prelude.install(name, verbose=False)
        namespace = {"__name__": "__main__", "__prelude_shell__": prelude.shell}
        # 노트북을 빈 임시 폴더에서 돌린다. Colab의 /content 자리에 해당한다.
        # I-5-3(Pandas)이 checkup.csv 를 만들기 때문에, 저장소 안에서 돌리면
        # 점검을 한 번 할 때마다 없던 파일이 하나씩 생긴다.
        # 폴더는 실행이 끝나면 통째로 사라진다.
        with tempfile.TemporaryDirectory() as scratch, _chdir(scratch), \
                contextlib.redirect_stdout(buf):
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
    out["skipped_installs"] = skipped_installs

    # Gradio 대역이 함수를 부르지 못한 채 넘어갔다면 실패로 본다.
    # 모델만 불러오고 번역·요약·생성은 한 번도 안 해 본 초록불은 초록불이 아니다.
    blocked = prelude.gradio_blocked()
    if blocked:
        out["detail"] = "Gradio 대역이 실행할 함수를 찾지 못했다: " + " / ".join(blocked)
        out["tail"] = buf.getvalue()[-800:]
        return out
    out["gradio_runs"] = prelude.gradio_runs()

    # Gradio 출력 길이 상한.
    # VI-11에서 배운 것이다: max_length 인자가 조용히 무시되자 출력이 85자에서
    # 721자로 뛰었다. 터지지도 값이 어긋나지도 않아 점검은 초록불이었다.
    # 인자가 무시되면 길이가 껑충 뛰므로, 길이에 선을 그어 두면 같은 종류의
    # '조용한 고장'을 자동으로 잡을 수 있다.
    # ⚠ 상한은 반드시 실측 후에 정한다. 값을 재 보지 않은 노트북에는 걸지 않는다.
    ceiling = case.get("gradio_max_chars")
    if ceiling:
        for g in out["gradio_runs"]:
            if g["chars"] > ceiling:
                out["detail"] = (f"Gradio 출력이 {g['chars']}자로 상한 {ceiling}자를 넘었다 "
                                 f"— 인자가 무시되고 있을 수 있다 ({g['title']})")
                out["tail"] = buf.getvalue()[-800:]
                return out

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

    # 중간 셀의 결과도 확인한다 (선택). 셀마다 예시가 따로 노는 노트북용.
    for wanted in case.get("expect_text") or []:
        if wanted not in text:
            out["detail"] = f"확인 문구를 찾지 못했다: {wanted!r}"
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
    ap.add_argument("--chapters", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                    help="점검할 장 번호 (기본: 1 2 3 4 5)")
    ap.add_argument("--retry", type=int, default=0, metavar="N",
                    help="실패한 노트북만 N번까지 다시 돌려 본다 (기본: 0 = 다시 돌리지 않음)")
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

    # ── 실패한 것만 다시 돌려 본다 ────────────────────────────────────────────
    # MNIST 내려받기(구글 storage)나 구글시트 접속이 잠깐 흔들려 실패하는 일이 있다.
    # 그런 실패로 알림 메일이 오면, 사람은 곧 알림 자체를 무시하게 된다.
    # 그래서 한 번 더 돌려 보고, 그때도 안 되면 그제야 실패로 본다.
    # 다시 돌려서 통과한 것은 표에 '재시도 후 통과'로 남긴다 — 조용히 넘어가지 않는다.
    if args.retry:
        targets = [i for i, r in enumerate(results) if r["status"] == "FAIL"]
        if targets:
            print("=" * 78)
            print(f"실패한 {len(targets)}개를 다시 돌려 본다 (최대 {args.retry}회).\n")
            for i in targets:
                case = results[i]["case"]
                first = results[i]["detail"]
                for attempt in range(1, args.retry + 1):
                    print(f"▶ 재시도 {attempt}/{args.retry}  {case['section']:6s} {case['file']}",
                          flush=True)
                    again = run_one(case)
                    print(f"   {ICON[again['status']]} {again['detail']}\n", flush=True)
                    if again["status"] != "FAIL":
                        again["detail"] += f" · 재시도 {attempt}회 만에 통과"
                        again["retried_after"] = first
                        results[i] = again
                        break
            print()

    print("=" * 78)
    for r in results:
        c = r["case"]
        print(f"{ICON[r['status']]} {c['section']:6s} {c['file']}")
        print(f"     {r['detail']}  ({r['seconds']:.1f}초)")
        if r.get("retried_after"):
            print(f"     ⚠ 첫 실행에서는 실패했다: {r['retried_after']}")
        for g in r.get("gradio_runs") or []:
            print(f"     Gradio 화면 대신 직접 실행: {g['title']} → 출력 {g['chars']}자")
        if r["case"].get("cells") is not None:
            print(f"     ⚠ 코드 셀 일부만 실행했다: {r['case']['cells']}번")
        if r.get("skipped_installs"):
            print(f"     건너뛴 설치 줄: {' / '.join(r['skipped_installs'])}")
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
                detail = r["detail"]
                # 에포크를 낮춰 돌렸다면 표에서 바로 보이게 한다.
                # 로그를 뒤져야만 알 수 있으면 실측값의 근거를 놓치게 된다.
                for asked, capped in r.get("epochs_capped") or []:
                    detail += f" · 에포크 {asked}→{capped}"
                # Gradio 출력 길이도 표에 남긴다. 인자가 조용히 무시되면
                # 이 숫자가 껑충 뛰므로, 상한을 정하는 근거이자 조기 신호가 된다.
                for g in r.get("gradio_runs") or []:
                    detail += f" · Gradio 출력 {g['chars']}자"
                if c.get("cells") is not None:
                    detail += f" · **셀 {c['cells']}번만 실행**"
                f.write(f"| {ICON[r['status']]} | {c['section']} | `{c['file']}` | "
                        f"{detail} | {r['seconds']:.0f}초 |\n")
            f.write(f"\n**정상 {ok} · 측정 {meas} · 실패 {fail} · 합계 {total_time/60:.1f}분**\n")
            retried = [r for r in results if r.get("retried_after")]
            if retried:
                f.write(f"\n⚠ {len(retried)}개는 첫 실행에서 실패했다가 다시 돌려 통과했다. "
                        "대개 네트워크가 잠깐 흔들린 것이지만, **다음 달에도 같은 노트북이 "
                        "여기에 뜨면 진짜 문제**다.\n")
                for r in retried:
                    f.write(f"- `{r['case']['file']}` — 첫 실행: {r['retried_after']}\n")
            if meas:
                f.write("\n📏 는 값을 재기만 한 것이다. 이 값을 보고 임계값을 정한 뒤 "
                        "`run_chapters.py`의 mode를 floor로 바꾼다.\n")

    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
