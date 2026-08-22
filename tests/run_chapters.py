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
         marker="성공률", mode="floor", metric=SUCCESS_RATE, floor=0.08, printed=0.3370,
         note="실측(2026-08-22 · 600회): 중앙 34.2% · 5%ile 28.1% · 최저 17.1% · 최고 41.1%. "
              "🔴 하한이 0.20 이던 때 600회 중 3회(0.5%)가 걸렸다 — 월 1회 실행이면 연 6% 확률의 "
              "거짓 빨간불이었다. 이전 note 의 '실측 28.5~37.1%'는 몇 번 돌려 본 값이라 "
              "가운데만 보고 있었다. "
              "하한 0.08 의 근거: 관측 최저 17.1%보다 9.2%p 아래이면서, 학습이 전혀 듣지 않는 "
              "상태(ε을 1.0에 고정한 순수 랜덤, 40회 실측 0.8~2.2%)보다 3.6배 위다. "
              "성공률을 결정하는 것은 첫 성공이 언제 오느냐다(상관 -0.761). 최초 성공 "
              "에피소드는 중앙 53인데 꼬리가 470까지 간다"),
    dict(chapter=5, section="V-3", file="Reinforcement Learning_Level 3-2.ipynb",
         marker="성공률", mode="floor", metric=SUCCESS_RATE, floor=0.30, printed=0.6140,
         note="실측(2026-08-22 · 600회): 중앙 62.9% · 5%ile 54.7% · 최저 41.2% · 최고 77.8%. "
              "🔴 하한이 0.40 이던 때 여유가 1.2%p 뿐이었다 — 600회에서 걸리지 않은 것은 운이다. "
              "이전 note 의 '실측 57.0~70.9%'도 가운데만 본 값이었다. "
              "하한 0.30 의 근거: 관측 최저 41.2%보다 11.2%p 아래이면서 "
              "학습 실패 상태(0.8~2.2%)와는 열 배 넘게 떨어져 있다. "
              "첫 성공 시점과의 상관은 -0.453 으로 3-1·4-1 보다 약하다 — 감쇠가 후반을 "
              "이용 쪽으로 끌어당겨 초반 운의 영향을 줄이기 때문이다. "
              "✅ **종결된 의문 — 다시 조사하지 말 것.** 3-1 → 3-2 는 코드가 두 곳 다르다"
              "(35번 줄 조건 · 68~69번 줄 감쇠). 그래서 '상승이 감쇠 덕인지 알 수 없다'는 의심이 "
              "나오는데, **35번 줄은 동작을 바꾸지 않는다.** 3-2 가 더한 "
              "`or np.sum(q_table[state]) == 0` 이 없어도, Q행이 전부 0이면 3-1 의 else 가지에서 "
              "`flatnonzero(q==max(q))` 가 [0,1,2,3] 을 돌려주어 네 방향 균등 추첨이 되고 "
              "이는 `action_space.sample()` 과 같다(각 20만 회 실측 0.249~0.251). "
              "두 방식을 번갈아 220쌍 돌린 결과 p=0.412 로 구분되지 않았다. "
              "따라서 34%→63% 상승은 전부 감쇠 덕이고, 원고의 「한 줄 추가」는 실질적으로 참이다. "
              "⚠ 처음에는 따로 몰아 돌려 p=0.023·0.008 이 나왔는데 순서 효과였다 — "
              "이런 비교는 반드시 번갈아 돌릴 것"),
    dict(chapter=5, section="V-4", file="Reinforcement Learning_Level 4-1.ipynb",
         marker="성공률", mode="floor", metric=SUCCESS_RATE, floor=0.40, printed=0.7390,
         note="실측(2026-08-22 · 600회): 중앙 76.0% · 5%ile 69.1% · 최저 53.8% · 최고 80.1%. "
              "**하한을 바꾸지 않는다** — 관측 최저보다 13.8%p 아래라 이미 넉넉하다. "
              "같은 날 3-1·3-2 는 여유가 없어 낮췄다. "
              "⚠ 원고는 이 절의 성공률을 '70~80%에 달하고'라고 단정하는데 실측 최저는 53.8% 다. "
              "그리고 3-2(41.2~77.8%)와 범위가 겹쳐 γ 를 넣었는데 성공률이 떨어지는 실행이 나올 수 "
              "있다. γ 의 증거는 성공률이 아니라 최종 경로 길이와 Q값의 등급화다. "
              "✅ **종결된 의문 — 다시 조사하지 말 것.** 3-2 → 4-1 은 γ 추가 말고 "
              "ε 시작값도 0.1 → 1.0 으로 바뀐다. 상승의 몫을 쪼개 재면(다른 코드는 고정) "
              "γ 만 추가 → 중앙 87.6%(+24.7%p) · ε₀ 만 변경 → 중앙 38.0%(-24.9%p) · "
              "둘 다 → 76.0%. 즉 **ε₀=1.0 은 성공률을 11.6%p 깎지만 의도된 설계다** — "
              "초반을 충분히 탐험시키는 표준 감쇠 설정이고 4-2 가 같은 값으로 이어받는다. "
              "**'성공률을 올리려면 ε₀ 를 0.1 로 낮추자'고 제안하지 말 것.** "
              "원고가 상승을 γ 덕으로 돌린 것은 옳고, 오히려 보수적이다(γ 단독 효과가 더 크다)"),
    dict(chapter=5, section="V-4", file="Reinforcement Learning_Level 4-2.ipynb",
         marker="성공률", mode="measure", metric=SUCCESS_RATE, printed=0.5981,
         note="실측 9.8~65.2%. 학습이 통째로 실패하는 실행이 있어 값으로 판정하지 않는다. "
              "2026-08-22 에 68회를 더 돌렸으나 47~64% 로 9.8% 는 재현되지 않았다 — "
              "한 회에 8.6초가 들어(에피소드 10,000회 · is_slippery=True) 표본을 더 쌓지 못했다. "
              "9.8% 가 러너 환경에서만 나오는 것인지 드문 꼬리인지는 아직 모른다. "
              "**레인 C 2회차 확인 항목**이다. 어느 쪽이든 하한은 걸지 않는다"),

    # ── VI장 ─────────────────────────────────────────────────────────────────
    # 착수 순서: 모델 없는 것 → 의심 API → Gradio + 실제 모델.
    # 한 번에 다 넣지 않는다. 문제가 생겼을 때 원인이 어디인지 바로 보이게 하기 위해서다.
    dict(chapter=6, section="VI-1", file="NLP_Preprocessing.ipynb",
         marker="④ 인덱스 시퀀스로 변환된 문장:", mode="complete",
         expect_text=["['나는', '머신러닝을', '정말', '좋아한다']",
                      "'파이썬으로': 9", "[9, 3, 8, 10]"],
         note="순수 파이썬(re 모듈만). 난수도 모델도 없어 **완전히 결정적**이다 — "
              "책 인쇄값과 한 자리까지 같아야 하므로 결과를 못박았다"),
    dict(chapter=6, section="VI-2", file="NLP_Sentence Classification.ipynb",
         marker="→ 예측:", mode="floor", metric=ACC_FRACTION, floor=0.70, printed=0.7784,
         expect_text=['▶ 예시 문장: "완전 재미있었어요!"',
                      '▶ 예시 문장: "그저 그랬어요."'],
         note="NSMC 20만건을 내려받아(레인 A 감시 중: data-nsmc-train/test) TF-IDF + "
              "LogisticRegression. **0.7784 가 네 번 일치했다** — Colab 저장 출력 1회 + "
              "러너 3회(sklearn 1.9.0), 소수점 네 자리까지 같다. 난수 인자가 없다는 추정이 "
              "실측으로 확인되어 2026-08-20 measure → floor 로 올렸다. "
              "하한 0.70 은 값의 흔들림이 아니라 **학습이 통째로 망가진 경우**를 잡기 위한 것이다 "
              "— 이 노트북의 진짜 위험은 정확도 하락이 아니라 NSMC(개인 저장소)가 사라지는 것이고, "
              "그건 하한이 아니라 예외로 터진다. 예측 결과(긍정/부정·확률)는 확인하지 않는다"),

    dict(chapter=6, section="VI-3", file="NLP_Word Embedding.ipynb",
         marker="✅ 학습된 워드 임베딩", mode="complete",
         expect_text=["amazing", "waiting", "was"],
         note="지침서가 의심 후보로 지목한 keras Tokenizer·pad_sequences 를 쓴다. "
              "임베딩 값 자체는 씨앗이 없어 매번 다르므로, 단어 목록만 확인한다"),

    dict(chapter=6, section="VI-6", file="NLP_Language Model_01.ipynb",
         marker="다음에 나올 확률이 높은 단어: 'processing'", mode="complete",
         expect_text=["('natural', 'language')"],
         note="순수 파이썬(re · collections)뿐이다. 모델도 난수도 외부 자산도 없어 "
              "**완전히 결정적**이므로 책 인쇄값과 한 자리까지 같아야 한다 → 결과를 통째로 못박았다. "
              "⚠ 다만 'processing'과 'models'는 빈도가 1로 동률이고, Counter.most_common 이 "
              "먼저 들어온 것을 고르기 때문에 'processing'이 나온다(원고에도 그렇게 적혀 있다). "
              "파이썬이 이 동작을 바꾸면 여기서 빨간불이 뜬다 — 그때는 노트북이 아니라 "
              "이 marker 를 손볼 것"),
    dict(chapter=6, section="VI-7", file="NLP_Encoder-Decoder Model.ipynb",
         marker="▶ 입력: have a nice day", mode="complete",
         expect_text=["▶ 입력: nice to meet you", "▶ 입력: thank you"],
         note="Keras seq2seq 300에포크(문장쌍 10개라 몇 초면 끝난다). 씨앗이 없어 "
              "**번역 결과는 확인하지 않는다** — 저장 출력의 '반가워'·'고마워'·'좋은 하루 보내'를 "
              "expect_text 로 못박으면 무해한 차이로 빨간불이 뜬다. 확인하는 것은 "
              "translate() 세 번이 모두 예외 없이 끝났다는 것뿐이다. "
              "⚠ 학습이 통째로 실패해 엉뚱한 번역이 나와도 초록불이다 — 번역 품질은 레인 C 몫"),

    dict(chapter=6, section="VI-6", file="NLP_Language Model_02.ipynb",
         marker="▶ Input: 'RNN models are'", mode="complete",
         expect_text=["Top 3 predicted words:"],
         note="Keras LSTM 200에포크(학습 데이터가 문장 3개라 배치가 1개뿐이다). "
              "씨앗이 없어 **예측 단어와 확률은 확인하지 않는다** — 저장 출력 0.6581 vs "
              "원고 인쇄값 0.8001 로 이미 실측이 두 번 갈렸다. 확인하는 것은 입력 문구와 "
              "Top-3 출력 형식이 살아 있는가뿐이다. "
              "⚠ 예측이 'used'가 아닌 다른 단어로 바뀌어도 초록불이다 — 원고의 설명이 "
              "여전히 맞는지는 레인 C에서 사람이 볼 것"),

    dict(chapter=6, section="VI-8", file="NLP_Attention_arrows.ipynb",
         marker=None, expect_var="attention_scores", mode="complete",
         note="print 가 하나도 없고 ipywidgets 드롭다운으로 그림만 그린다 → 변수로 완주를 확인. "
              "어텐션 값은 학습한 것이 아니라 손으로 적어 넣은 시뮬레이션이라 완전히 결정적이다. "
              "맨 끝에서 plot_attention_arrows(0) 을 실제로 부르므로 그리기 코드가 깨지면 예외로 터진다. "
              "⚠ 그 호출이 `with output:`(ipywidgets.Output) 안에 있다. ipywidgets 8 의 __exit__ 은 "
              "**커널이 있을 때만** 예외를 삼키므로 러너에서는 예외가 그대로 올라온다(2026-08-20 실측). "
              "ipywidgets 7 은 무조건 삼켰다 — 판번호가 내려가면 이 노트북은 조용히 통과하게 된다. "
              "⚠ 노트북에 !pip 줄이 없다. 워크플로가 ipywidgets 를 깔아 주어야 한다"),

    dict(chapter=6, section="VI-8", file="NLP_Attention_VS_No-Attention_str.ipynb",
         marker="=== REVERSE TASK ===", mode="complete",
         expect_text=["=== COPY TASK ===", "입력(길이 30): abcdefghijklmnopqrstuvwxyzabcd"],
         note="코드 240여 줄에 seq2seq 모델 4개(copy·reverse × 기본·어텐션)를 15에포크씩 학습한다. "
              "**이 장에서 유일하게 씨앗이 걸린 노트북이다**(random·numpy·tf 셋). "
              "노트북 첫머리에 PYTHONHASHSEED 도 넣지만 그것은 인터프리터가 시작할 때 읽는 값이라 "
              "실행 중에 넣어도 효과가 없다. 실효 씨앗은 셋이다. "
              "그런데도 생성 결과는 확인하지 않는다 — 씨앗은 같은 TF 판번호 안에서만 재현을 보장하고, "
              "러너는 일부러 최신 TF로 돌리기 때문이다. 확인하는 것은 두 task 가 모두 끝났는가와 "
              "길이 30 입력 문자열뿐이다. "
              "🔴 그 예측이 2026-08-22 에 실증됐다. 씨앗을 그대로 둔 채 Colab 에서 다시 돌렸더니 "
              "(2025-08-15 TF 2.19 무렵 → 2026-08-22 TF 2.20.0) COPY 길이 5 의 기본 모델 출력이 "
              "abcde 에서 abced 로 갈렸다. 출력을 못박았다면 이날 빨간불이 됐다. "
              "⚠ '어텐션 모델은 장문을 재현하고 기본 모델은 실패한다'는 이 절의 핵심 주장은 "
              "점검되지 않는다 — 값으로 못박으면 TF 판번호가 오를 때마다 빨간불이 뜬다. 레인 C 몫. "
              "다만 길이 30 에 대해서는 세 번의 실행이 모두 '어텐션 완벽 · 기본 붕괴'로 일치했다"
              "(2025-08-15 · 2026-08-22 두 번). 흔들리는 것은 길이 5 쪽이다. "
              "※ 2026-08-22 에 셀이 둘에서 하나로 줄었다. 사라진 셀 0 은 TF 판번호를 찍고 "
              "os 를 import 하던 '(Optional)' 셀인데, 건너뛰면 셀 1 이 죽는 구조였다. "
              "TF 판번호는 워크플로의 mods 목록이 매달 따로 찍으므로 잃은 정보가 없다"),

    dict(chapter=6, section="VI-10", file="NLP_BERT.ipynb",
         marker="종료합니다.", mode="complete",
         expect_text=["모델 준비 완료!", "예측 결과:"],
         note="mBERT(bert-base-multilingual-cased, 약 700MB)를 특징추출기로만 쓰고 "
              "[CLS] 벡터를 sklearn LogisticRegression 에 넣는다. "
              "input() 반복인데 **종료 조건이 종료어가 아니라 빈 문자열**이라 "
              "prelude 의 마지막 칸이 \"\" 다 — IV-1('q')·IV-6('exit')과 다르니 주의. "
              "marker '종료합니다.'가 나왔다는 것은 준비된 답 4개가 모두 쓰였다는 뜻이고, "
              "모자라면 러너의 unused_inputs 검사가 따로 잡는다. "
              "예측 결과와 확률값은 확인하지 않는다 — torch 판번호에 따라 끝자리가 흔들린다"),

    dict(chapter=6, section="VI-11", file="NLP_BERT_GPT_output.ipynb",
         marker="🟠 [KoGPT2] 다음 단어 예측 (Top-3):", mode="complete",
         expect_text=["🔵 [KLUE BERT] 문맥 임베딩 출력:", "##를", "[SEP]"],
         note="klue/bert-base 와 skt/kogpt2-base-v2 를 함께 내려받는다(합쳐 약 1GB). "
              "input() 도 난수도 없고 두 모델 다 eval() 이라 결정적이지만, "
              "**임베딩 평균값과 확률은 확인하지 않는다** — torch 판번호에 따라 끝자리가 달라진다. "
              "대신 토큰 쪼개기 결과('##를'·'[SEP]')를 못박았다. 토크나이저는 모델에 딸린 "
              "고정 자산이라 안정적이고, 이것이 바뀌면 원고의 '##' 설명 자체가 틀려진다"),

    # VI장 미등록 3건은 **의도적 제외**이며 아직 등록하지 않은 것이 아니다.
    #   VI-4 NLP_Transfer Learning_01 — cats_vs_dogs 25,000장 × 모델 2개.
    #                                   원고가 "Colab GPU로도 약 10분"이라 적고 있다.
    #   VI-5 NLP_Transfer Learning_02 — GloVe 822MB 내려받기 + 양방향 LSTM 2개.
    #                                   원고가 "20분 이상"이라 적고 있다.
    #   VI-9 NLP_Transformer-based translation model — mBART-large-50(약 2.4GB).
    # 셋 다 러너(GPU 없는 2코어)에서 60분 제한을 넘길 것이 거의 확실하다.
    # 앞의 둘은 tensorflow_datasets 도 필요한데 워크플로에 없다.
    # → 레인 C에서 사람이 Colab으로 확인할 것.

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
              "출력 상한 300자는 실측 근거가 있다 — 정상 100자 안팎 vs "
              "max_length 가 무시됐을 때 721자. "
              "여섯 번의 실측이 83~104자에 흩어졌고, **잴 때마다 범위가 넓어졌다** "
              "(85 → 84~93 → 84~97 → 83~97 → 83~104). do_sample 이 걸려 있으므로 "
              "앞으로도 계속 넓어진다. **그래서 개별 값을 좇지 않기로 했다** — "
              "범위를 따라 적는 일은 라벨 관리 비용만 늘리고 판정에는 쓰이지 않는다. "
              "판정 인자는 gradio_max_chars 하나이고, 상한 300자는 '정상'과 "
              "'인자가 무시된 상태'를 가르는 여유를 이미 충분히 갖고 있다"),

    # ── VII장 · 생성형 AI ────────────────────────────────────────────────────
    # 원고 대조로 확정: 노트북 4개 / 절 3개 (VII-3 에 두 개 — VI-6·VI-8 과 같은 형태).
    # 그중 등록 대상은 2개다. 나머지 둘은 **아직 안 한 것이 아니라 안 하기로 한 것**이다.
    #
    #   VII-2  Generative AI_Diffusion
    #          노트북 메타데이터에 accelerator: "GPU" · gpuType: "T4" 가 저장돼 있고,
    #          원고도 "Colab의 T4 GPU를 사용하도록 런타임 설정을 해두었다"고 명시한다.
    #          러너는 GPU 없는 2코어다. 여기에 sd-turbo 라이선스 게이트까지 겹친다.
    #   VII-3① Generative AI_Music
    #          모델 2.36GB — 저장 출력에 그대로 찍혀 있다
    #          ("model.safetensors: reconstructing file: 0.00B / 2.36GB").
    #          원고도 "최초 실행 시에는 모델 설치 시간이 몇 분 걸릴 수 있다"고 적고 있다.
    #
    # 두 노트북의 고정 판번호는 레인 A 가 PyPI 로 감시한다(pypi-diffusers-0-39-0 등).
    # 실행 자체는 레인 C 에서 사람이 Colab 으로 확인한다.
    dict(chapter=7, section="VII-3", file="Generative AI_Voice.ipynb",
         marker="▶ 재생 (1.3x speed)", mode="complete", expect_var="y_fast",
         note="VII-3 의 두 번째 노트북(② 음성 생성 AI). VII장에서 유일하게 **모델을 "
              "하나도 내려받지 않는다** — gTTS 가 구글 서버에 실시간 요청을 보내 mp3 를 받고, "
              "librosa 로 1.3배속 처리만 한다. "
              "marker 는 저장 출력에서 그대로 가져왔다. time_stretch 다음 줄의 print 이므로 "
              "속도 변환이 터지면 marker 가 안 나온다. expect_var='y_fast' 는 그 위 — "
              "**터지지 않고 빈 배열이 돌아오는 경우**를 잡는 덤이다. "
              "⚠ 레인 B 에서 유일하게 **외부 API 응답에 실행이 걸리는 노트북**이다. "
              "구글 TTS 가 흔들리면 빨간불이 뜬다. --retry 1 이 한 번은 흡수해 주지만, "
              "두 달 연속 재시도에 걸리면 일시 장애가 아니라 진짜 문제다. "
              "접속이 막혔을 때의 모습은 이렇다 — "
              "gtts.tts.gTTSError: 403 (Forbidden) from TTS API, 8번째 줄의 .save() 에서. "
              "**이 서명이 보이면 노트북도 라이브러리도 아니고 구글 쪽이다.** "
              "구글이 데이터센터 IP 를 막을 가능성이 있으므로, 이 항목만 반복해서 빨간불이면 "
              "레인 C 로 옮기는 것을 검토할 것 — 책의 코드가 아니라 러너의 위치가 원인이기 때문이다. "
              "다만 **2026-08-20 러너 실행 두 번 모두 통과했다**(3초 ~ 7초) — "
              "현재 Actions 에서는 막히지 않는다. "
              "⚠ 워크플로에 gTTS·librosa 설치가 필요하다 — 레인 B 는 !pip 줄을 지운다. "
              "그리고 그 둘은 반드시 **다른 라이브러리와 같은 pip 명령 안**에 두어야 한다. "
              "따로 깔면 gTTS 가 click 을 끌어내려 huggingface-hub 가 조용히 망가지고, "
              "피해는 VII장이 아니라 **VI장 14개**에 간다. "
              "**2026-08-20 1~7장 전체 실행에서 VI장 14개가 모두 통과해 이 조치가 검증됐다.** "
              "mp3 디코딩에는 추가 설치가 필요 없다 — soundfile 이 품고 오는 "
              "libsndfile 1.2.2 가 MP3 를 지원한다(2026-08-20 실측). "
              "librosa 는 주 판번호가 1.0.0 으로 올라간 최신판에서도 "
              "time_stretch(y, rate=) 시그니처가 그대로다(실측: 44100 → 33923, 비율 1.3). "
              "노트북 고정값 0.11.0 과 Colab 기본값도 0.11.0 이다"),

    dict(chapter=7, section="VII-1", file="Generative AI_Text Generator.ipynb",
         marker="[대역] 사람이 누르는 대신", mode="complete",
         button_max_chars=600,
         note="버튼 대역의 첫 실전 시험. 이 노트북은 끝이 button.on_click(...) + display(...) "
              "라서, 그냥 돌리면 위젯을 만들어 얹는 시늉만 하고 끝난다 — **생성 로직은 한 번도 "
              "실행되지 않는다.** 저장 출력에도 위젯 여섯 개만 있고 생성 결과가 없다"
              "(원고는 결과를 별도 이미지 파일로 싣고 있다). 그대로 등록했다면 KoGPT2 를 "
              "내려받은 것까지만 확인하고 초록불이 켜졌을 것이다. "
              "대역이 on_click 등록 시점에 콜백을 한 번 불러 준다. 등록 시점에 generator· "
              "dropdown·temp·length·output 이 모두 준비돼 있어 안전하다. "
              "marker 는 대역이 찍는 문구다 — 이 노트북에는 print 가 하나도 없다. "
              "실측 세 번은 166자·165자·171자다(2026-08-20). 세 표본이 165~171 에 모여 "
              "**상한 600자를 걸었다** — 정상값의 약 3.5배로, VI-11 이 85자를 재고 300자를 "
              "정한 것과 같은 비율이다. 상한은 생성 품질을 보는 것이 아니라 "
              "**max_new_tokens=60 이 조용히 무시되는 것**을 잡기 위한 것이다. "
              "길이를 정하는 인자가 있는 노트북에만 거는 원칙에 따랐다. "
              "⚠ 이 166자는 **생성된 문장의 길이가 아니다.** 노트북이 마침표마다 '<br>' 를 "
              "붙이고(.replace) 머리말과 프롬프트가 더해진 값이다. 생성이 길어지면 마침표도 "
              "늘어 부풀림이 함께 커지므로 상한을 넉넉히 잡았다. "
              "그리고 VI-11 과 위험의 성격이 다르다 — VI-11 은 max_length 라는 "
              "**틀린 인자**를 써서 무시당했는데, 이 노트북은 max_new_tokens 로 맞는 인자를 쓴다. "
              "상한의 값어치가 VI-11 만큼 크지는 않다. "
              "⚠ **!pip 줄이 아예 없어 transformers 가 고정돼 있지 않다.** 같은 장의 "
              "VII-2·VII-3① 은 transformers==5.15.0 으로 못박혀 있는데 이것만 무방비다. "
              "판번호 고정 정책의 「설치 줄 11개」 목록은 **설치 줄이 없는 노트북을 "
              "구조적으로 볼 수 없었다** — VI-8 arrows 도 같은 처지다. "
              "다행히 VI-11 이 같은 pipeline('text-generation') 을 쓰며 매달 도니 "
              "조기 경보는 있다. 원고·노트북에 설치 줄을 넣을지는 미결"),

    # ── VIII장 · 멀티모달 AI ─────────────────────────────────────────────────
    # **등록 대상이 0개다.** 노트북 두 개가 모두 의도적 제외이며 레인 C 에서 사람이 본다.
    # 절은 셋인데 VIII-1(멀티모달 AI란)은 코딩 없는 Gemini 체험이라 노트북이 없다.
    #
    #   VIII-2 Multimodal AI_Image Captioning — BLIP · gr.Image 에 예시값이 없다
    #   VIII-3 Multimodal AI_Video Subtitle   — Whisper · gr.Video 에 예시값이 없다
    #
    # 용량도 문제지만 **더 근본적인 이유는 입력이 독자가 올리는 파일**이라는 것이다.
    # Gradio 대역은 value= 에서 값을 꺼내는데 여기엔 꺼낼 값이 없고, 지어내서도 안 된다
    # (지어낸 입력으로 통과시키면 그것 역시 아무것도 검증하지 않은 초록불이다).
    # VIII-3 은 여기에 ffmpeg 바이너리까지 필요하다 — ffmpeg-python 은 래퍼일 뿐이다.
    #
    # 다만 길이 아주 없지는 않다. 레인 A 가 샘플 파일 다섯 건을 이미 감시하고 있으므로
    # (file-video-teded · file-video-cosmoswag 등), 레인 C 에서 그 파일을 받아
    # fn 에 직접 넣어 주면 대역 없이도 추론을 확인할 수 있다.

    # ── IX장 · 기술, 그리고 사람 ─────────────────────────────────────────────
    # 절은 다섯인데 실습은 IX-3 하나뿐이다(나머지는 본문과 그림만 있다).
    dict(chapter=9, section="IX-3", file="Biased_Cheering.ipynb",
         marker="[학습 데이터 요약]", mode="complete",
         expect_text=["- 전체 경기 수: 1,200",
                      "- 청팀 승(1): 810건 (67.5%)",
                      "- 홍팀 승(0): 390건 (32.5%)"],
         note="버튼 대역의 두 번째 실전이자, **버튼이 셋인 첫 노트북**이다. "
              "한 줄에 셋이 등록되므로(btn_one·btn_many·btn_clear) 대역이 "
              "**단일 테스트 → 일괄 100회 → 초기화** 순으로 다 누른다. "
              "셋 다 display(HTML(...)) 출력이 있어 무출력 검사에 걸리지 않는다. "
              "⚠ **대역은 '초기화' 같은 파괴적 버튼도 누른다.** 여기서는 메모리 상태만 "
              "지우므로 무해하고, 길이 측정은 지우기 전에 이미 기록된다. 그러나 파일을 "
              "지우는 버튼이 있는 노트북을 붙일 때는 이 성질을 먼저 볼 것. "
              "모델을 하나도 받지 않고 !pip 줄도 없다 — numpy·pandas·sklearn·ipywidgets 만 "
              "쓰는데 ipywidgets 는 무거운 설치 조건(6|7|8|9)에 9가 들어 있어 이미 깔린다. "
              "expect_text 세 줄은 **저장 출력에서 그대로 가져왔고 완전히 결정적이다** "
              "— 시트를 세기만 하는 값이다. 그리고 이 세 줄에 값이 있다는 것이 중요하다: "
              "**레인 A 는 시트가 살아 있는지만 보지 내용이 바뀌었는지는 못 본다.** "
              "이 셋을 못박아 두면 시트가 조용히 바뀌는 것을 레인 B 가 잡는다 "
              "(III-10 에서 값 7개를 못박은 것과 같은 발상이고, 여기서는 그 값이 "
              "원고에도 인쇄돼 있다 — '810회(67.5%), 홍팀 승이 390회(32.5%)'). "
              "⚠ **이 노트북의 핵심 주장은 레인 B 가 못 잡는다.** '예측이 8:2로 편향된다'가 "
              "IX-3 의 논지인데 rng = np.random.default_rng() 에 씨앗이 없어 매번 다르다. "
              "VI-8 _str 의 어텐션 주장과 같은 부류다 → 레인 C. "
              "실측 2026-08-20: 829~831자 · 790자 · 681자 (단일·일괄·초기화). "
              "두 번의 실행에서 일괄·초기화가 자릿수까지 같았다 — 요약 텍스트라 거의 결정적이다. "
              "⚠ **일괄 100회가 단일 1회보다 짧다.** 오류가 아니다 — 색깔 네모 보드는 "
              "board.value 에 직접 꽂혀 display() 를 거치지 않으므로 대역이 재지 못한다. "
              "이 숫자는 요약 텍스트의 길이일 뿐이다. "
              "**button_max_chars 를 걸지 않았다** — 이 노트북에는 '무시되면 길이가 껑충 뛰는 "
              "인자'가 없다. 길이를 정하는 인자가 있는 노트북에만 건다는 원칙에 따랐다 "
              "(VI-10 에 상한을 걸지 않은 것과 같은 이유). "
              "⚠ **실행 시간이 4초 ~ 241초로 갈렸다** (9장 단독 4초 vs 1~9장 전체의 끝 241초, "
              "2026-08-20 같은 날). 전체 실행의 시간 증가분이 거의 전부 이 한 편이었다 — "
              "러너가 전반적으로 느려진 것이 아니다. **원인 미상.** 짐작으로는 구글 시트 "
              "응답 지연 · 45개를 돌린 뒤의 메모리 압박 · 러너 개별 변동인데, 바로 앞의 VII-1 이 "
              "KoGPT2 를 받고도 2초였으므로 메모리 설명은 약하다. **표본이 둘이라 아직 판단하지 "
              "않는다.** 다음 달 예약 실행 값을 볼 것"),
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

    # 버튼 대역도 같은 잣대로 본다.
    # 버튼을 눌러 보지도 못했거나, 눌렀는데 아무것도 안 나왔다면 그 초록불은
    # '모델을 불러왔다'까지만 뜻한다. 그건 이 노트북이 보여 주려는 것이 아니다.
    blocked = prelude.button_blocked()
    if blocked:
        out["detail"] = "버튼 대역이 확인에 실패했다: " + " / ".join(blocked)
        out["tail"] = buf.getvalue()[-800:]
        return out
    out["button_clicks"] = prelude.button_clicks()

    # 버튼 출력 길이 상한. gradio_max_chars 와 같은 목적이다
    # (VI-11: 인자가 조용히 무시되면 길이가 껑충 뛴다).
    # ⚠ 반드시 실측 후에 정한다. 재 보지 않은 노트북에는 걸지 않는다.
    ceiling = case.get("button_max_chars")
    if ceiling:
        for b in out["button_clicks"]:
            if b["chars"] > ceiling:
                out["detail"] = (f"버튼 출력이 {b['chars']}자로 상한 {ceiling}자를 넘었다 "
                                 f"— 인자가 무시되고 있을 수 있다 ({b['button']})")
                out["tail"] = buf.getvalue()[-800:]
                return out

    # Gradio 출력 길이 상한.
    # VI-11에서 배운 것이다: max_length 인자가 조용히 무시되자 출력이 83~97자에서
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
    # 기본 장 목록은 CASES 에서 뽑는다. 여기에 숫자를 옮겨 적으면 장이 늘어도
    # 따라오지 않아, 손으로 돌릴 때만 조용히 일부 장을 건너뛰게 된다.
    # (워크플로는 --chapters 를 항상 명시하므로 예약 실행은 이 값을 쓰지 않는다)
    all_chapters = sorted({c["chapter"] for c in CASES})
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapters", nargs="+", type=int, default=all_chapters,
                    help="점검할 장 번호 (기본: 등록된 전체 — "
                         + " ".join(map(str, all_chapters)) + ")")
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
                for b in r.get("button_clicks") or []:
                    detail += f" · 버튼 '{b['button']}' 출력 {b['chars']}자"
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
