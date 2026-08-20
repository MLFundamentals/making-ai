"""
tests/prelude.py — 레인 B(월간 노트북 실행) 전용 '대역' 모음
================================================================

이 파일은 노트북을 **고치지 않고** 자동 실행하기 위한 것이다.
실행기가 노트북을 메모리로 읽어들인 뒤, 아래 install()을 호출하는 셀 하나를
맨 앞에 임시로 끼워 넣는다. 그 셀은 저장되지 않는다.

대역을 세우는 곳은 세 군데뿐이다.

  1. input()              — 사람 대신 미리 정해둔 답을 순서대로 내어준다
  2. google.colab         — auth 는 인증 팝업 대신 "성공했다"고만 대답하고,
                            output 은 위젯 스위치를 켜는 시늉만 한다
  3. gspread              — 시트를 읽는 척하면서 공개 gviz CSV에서 같은 값을 가져온다

⚠ 대역이 가리는 것(레인 C에서 사람이 확인해야 하는 것)
   - 진짜 구글 인증 경로가 살아 있는지
   - Colab에서 gspread가 실제로 동작하는지
   이 두 가지는 여기서 통과해도 검증되지 않는다.

Python 표준 라이브러리만 사용한다.
"""

from __future__ import annotations

import builtins
import csv
import importlib.machinery
import io
import os
import sys
import types
import urllib.request

GVIZ_BASE = os.getenv(
    "BOOK_GVIZ_BASE",
    "https://docs.google.com/spreadsheets/d/{id}/gviz/tq?tqx=out:csv",
)

# ---------------------------------------------------------------------------
# 노트북별 대역 입력값
#
# 값은 노트북에 저장된 출력에서 그대로 가져왔다. 즉 저자가 실제로 넣었던 값이며,
# 책에 인쇄된 결과와 한 자리까지 대조할 수 있다. 바꾸지 말 것.
# ---------------------------------------------------------------------------
NOTEBOOK_INPUTS: dict[str, list[str]] = {
    # III-4 · 나이, 성별, BMI, 식습관, 주간운동, 음주량, 흡연량, 가족력 → 35.04%
    "Multiple Linear Regression_Hypertension_8paras.ipynb":
        ["20", "0", "18", "2", "5", "0", "0", "0"],
    # III-5 · 나이, 성별, 페이지수, 체류시간, 이전구매, 광고클릭 → 시그모이드 0.90
    "Binary Classification.ipynb":
        ["30", "1", "10", "20", "5", "1"],
    # III-6 · 꽃받침 길이/너비, 꽃잎 길이/너비 → Virginica
    "Multiclass Classification_Iris.ipynb":
        ["5.7", "2.8", "4.9", "1.7"],
    # III-3 은 input() 없음 (78행에 테스트값이 하드코딩되어 있다)
    "Multiple Linear Regression_Hypertension_3paras.ipynb": [],

    # I-4-2 · 키(cm), 몸무게(kg) → BMI 22.86
    #   반복문이 없어 정확히 2번만 물어본다. 종료어가 필요 없다.
    "01_Variables_Expressions_Outputs_Inputs.ipynb": ["175", "70"],

    # IV-1 · while True 반복. 마지막은 반드시 종료어('q')여야 한다.
    "Perceptron_AND.ipynb": ["1 0", "q"],
    # IV-6 · while True 반복. 마지막은 반드시 종료어('exit')여야 한다.
    "RNN_hello world.ipynb": ["hello wo", "exit"],

    # VI-10 · while True 반복. 종료 조건이 종료어가 아니라 **빈 문자열**이다
    #   (user_text == "" 이면 break). 그래서 마지막 칸은 반드시 "" 여야 한다.
    #   앞의 세 문장은 저장 출력에 찍힌 그대로다 → 예측 결과도 책과 대조할 수 있다.
    "NLP_BERT.ipynb": ["정말 재미있고 감동적이었어요",
                       "전개가 답답하고 몰입이 안 되네요",
                       "음악과 영상은 인상적이다",
                       ""],
}

# ---------------------------------------------------------------------------
# 학습 시간 상한 (에포크 수)
#
# 노트북에 하드코딩된 epochs를 실행 시점에만 낮춘다. 파일은 바뀌지 않는다.
# 값이 없으면 노트북에 적힌 그대로 돈다.
#
# ⚠ 에포크를 줄이면 정확도가 떨어진다. 반드시 임계값과 짝지어 정할 것.
#    (지침서 6-4의 Evaluation Metrics 사례: 1에포크로 줄이면 0.9147까지 내려가
#     하한 0.90과의 여유가 1.5%p밖에 남지 않는다)
# ---------------------------------------------------------------------------
MAX_EPOCHS: dict[str, int] = {
    # MNIST 두 건은 상한을 걷어냈다 (2026-08-18 판단).
    #   3에포크 DNN 0.9644 / 6에포크 0.9665  — 두 배로 늘려 얻은 것이 0.2%p뿐
    #   2에포크 CNN 0.9828 / 4에포크 0.9848  — 같은 양상
    # 반면 러너가 빨라(6에포크 15초, 4에포크 29초) 10에포크를 다 돌려도 1분 30초쯤이다.
    # 상한을 없애면 점검이 독자와 똑같은 코드를 돌리게 되고,
    # 실측값을 책 인쇄값과 그대로 비교할 수 있다. 설명해야 할 차이가 사라진다.
    #
    # 상한 장치 자체는 남겨 둔다. 나중에 무거운 노트북에서 필요해진다.
    # "MNIST_DNN.ipynb": 6,
    # "MNIST_CNN.ipynb": 4,
    # III-10 도 같은 이유로 상한을 걷어냈다 (2026-08-18).
    # 준비 셀은 III-8과 똑같은 모델이라 러너에서 30초 안쪽이다. 10에포크를 다 돌리면
    # 저장 출력 92.68% 와 그대로 대조된다. 3에포크로 줄이면 하한 0.90 과의 여유를
    # 다시 재야 하고, 독자가 보는 값과도 어긋난다.
    # "Evaluation Metrics.ipynb": 3,
    # Perceptron_AND(200) · XOR(300) · RNN(100)은 몇 초면 끝나므로 손대지 않는다
}


class PreludeError(RuntimeError):
    """대역 설치·동작 중의 오류. 노트북 코드의 오류와 구분하기 위해 따로 둔다."""


# ---------------------------------------------------------------------------
# 1. input() 대역
# ---------------------------------------------------------------------------
def _install_input(answers: list[str]) -> None:
    queue = list(answers)
    real_input = builtins.input

    def fake_input(prompt: str = "") -> str:
        if not queue:
            raise PreludeError(
                f"input() 호출이 준비된 답보다 많다. 마지막 프롬프트: {prompt!r}\n"
                f"→ prelude.py의 NOTEBOOK_INPUTS에 값을 추가할 것."
            )
        value = queue.pop(0)
        print(f"{prompt}{value}")          # Colab의 화면 모양을 그대로 흉내낸다
        return value

    fake_input._is_prelude_stub = True     # 검증용 표식
    fake_input._remaining = lambda: len(queue)
    builtins.input = fake_input
    builtins._prelude_real_input = real_input


def unused_inputs() -> int:
    """실행이 끝난 뒤 남은 대역 답의 수. 0이 아니면 준비값과 코드가 어긋난 것이다."""
    fn = builtins.input
    return fn._remaining() if getattr(fn, "_is_prelude_stub", False) else -1


# ---------------------------------------------------------------------------
# 2. google.colab / google.auth 대역
# ---------------------------------------------------------------------------
class _FakeCredentials:
    """gspread.authorize()에 넘겨질 자리표. 대역 gspread는 내용을 보지 않는다."""
    def __init__(self) -> None:
        self.token = "prelude-stub-token"
        self.valid = True

    def refresh(self, request=None) -> None:
        return None


def _fake_module(name: str, is_package: bool = False) -> types.ModuleType:
    """가짜 모듈을 만든다. **`__spec__`을 반드시 채운다.**

    파이썬 모듈에는 '이 모듈을 어디서 어떻게 불러왔는지' 적힌 명세(`__spec__`)가 붙어 있다.
    `types.ModuleType()`으로 손수 만든 모듈은 이 칸이 비어 있는데(`None`),
    평소에는 아무도 들여다보지 않으므로 문제가 드러나지 않는다.

    그런데 `importlib.util.find_spec()`으로 '이 라이브러리가 깔려 있나?'를 확인하는
    코드가 이 칸을 읽는다. 비어 있으면 "없다"가 아니라 **ValueError로 터진다.**
    transformers가 정확히 이 방식으로 google.colab을 확인하기 때문에,
    명세를 채우지 않으면 `import transformers` 한 줄에서 VI장 전체가 죽는다.
    """
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=is_package)
    if is_package:
        mod.__path__ = []
        mod.__spec__.submodule_search_locations = []
    return mod


def _install_google_auth() -> None:
    # ⚠ 여기가 함정이다.
    # 'google'은 여러 패키지가 나눠 쓰는 네임스페이스다. google.protobuf(TensorFlow가
    # 쓴다), google.auth, google.colab이 모두 이 이름 아래 들어온다.
    # 가짜 google 모듈을 만들어 덮어쓰면 __path__가 비어버려서, 그 뒤의
    # `import google.protobuf`가 "모듈 없음"으로 실패한다 → TensorFlow 전체가 죽는다.
    # 그래서 진짜 google을 먼저 불러오고, 없을 때만 새로 만든다.
    try:
        import google                            # 진짜 네임스페이스를 그대로 쓴다
    except ImportError:
        google = _fake_module("google", is_package=True)
        sys.modules["google"] = google

    # --- google.colab.auth ---------------------------------------------------
    # google.colab은 Colab 밖에 존재하지 않으므로 항상 가짜를 넣는다.
    colab = _fake_module("google.colab", is_package=True)
    auth = _fake_module("google.colab.auth")
    auth.authenticate_user = lambda *a, **k: None
    colab.auth = auth
    google.colab = colab
    sys.modules["google.colab"] = colab
    sys.modules["google.colab.auth"] = auth

    # --- google.colab.output -------------------------------------------------
    # 위젯을 쓰는 노트북의 머리에 늘 붙는 두 줄이다.
    #     from google.colab import output
    #     output.enable_custom_widget_manager()
    # Colab 화면에서 자바스크립트 위젯(BertViz의 어텐션 그림 등)을 허용하는 스위치다.
    # 러너에는 화면이 없으므로 스위치를 켜는 시늉만 한다.
    #
    # ⚠ 이것이 없으면 `from google.colab import output` 한 줄에서 노트북이 죽는다.
    #   위의 가짜 google.colab 은 __path__ 가 빈 리스트라서, 파이썬이 하위 모듈
    #   'output' 을 찾아 나설 때 찾아볼 곳이 하나도 없어 ModuleNotFoundError 가 된다.
    #   __path__ 를 비워 둔 것은 진짜 google.protobuf 를 지키기 위한 조치이므로
    #   (위의 주석 참조) 바꾸지 말고, 필요한 하위 모듈을 여기에 하나씩 넣는다.
    colab_output = _fake_module("google.colab.output")
    colab_output.enable_custom_widget_manager = lambda *a, **k: None
    colab_output.disable_custom_widget_manager = lambda *a, **k: None
    colab.output = colab_output
    sys.modules["google.colab.output"] = colab_output

    # --- google.auth.default -------------------------------------------------
    # google-auth가 실제로 깔려 있으면 default()만 갈아끼운다.
    try:
        import google.auth as real_auth          # noqa: F401
        real_auth.default = lambda *a, **k: (_FakeCredentials(), "prelude-project")
    except Exception:
        ga = _fake_module("google.auth", is_package=True)
        ga.default = lambda *a, **k: (_FakeCredentials(), "prelude-project")
        google.auth = ga
        sys.modules["google.auth"] = ga


# ---------------------------------------------------------------------------
# 3. gspread 대역 — 공개 gviz CSV에서 읽는다
# ---------------------------------------------------------------------------
def _fetch_gviz(sheet_id: str, timeout: int = 30) -> list[list[str]]:
    """gviz CSV를 '문자열 2차원 리스트'로 돌려준다.

    gspread의 get_all_values()가 돌려주는 모양과 같아야 한다.
    - 헤더 행을 포함한다
    - 모든 칸이 문자열이다
    """
    url = GVIZ_BASE.format(id=sheet_id)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read()
            ctype = resp.headers.get("Content-Type", "")
    except Exception as exc:
        raise PreludeError(f"gviz 읽기 실패: {url}\n  {type(exc).__name__}: {exc}") from exc

    text = raw.decode("utf-8", errors="replace")

    # 시트가 비공개로 바뀌면 CSV 대신 로그인 HTML이 온다. 조용히 넘기면
    # 엉뚱한 파싱 오류로 둔갑하므로 여기서 잡는다.
    if "text/html" in ctype or text.lstrip()[:1] == "<":
        raise PreludeError(
            f"gviz가 CSV 대신 HTML을 반환했다 — 시트 {sheet_id} 가 비공개로 바뀌었을 수 있다."
        )

    rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
    if not rows:
        raise PreludeError(f"gviz 응답이 비어 있다 — 시트 {sheet_id}")
    return rows


class _StubWorksheet:
    def __init__(self, sheet_id: str, title: str = "Sheet1") -> None:
        self._sheet_id = sheet_id
        self.title = title
        self._values: list[list[str]] | None = None

    def _load(self) -> list[list[str]]:
        if self._values is None:
            self._values = _fetch_gviz(self._sheet_id)
        return self._values

    # 노트북이 실제로 쓰는 메서드
    def get_all_values(self) -> list[list[str]]:
        return [row[:] for row in self._load()]

    # 쓰지는 않지만 gspread 사용자가 흔히 부르는 것들 — 넓게 받아둔다
    def get_all_records(self) -> list[dict]:
        rows = self._load()
        return [dict(zip(rows[0], r)) for r in rows[1:]]

    @property
    def row_count(self) -> int:
        return len(self._load())

    @property
    def col_count(self) -> int:
        return len(self._load()[0])

    def __getattr__(self, name):
        raise PreludeError(
            f"대역 gspread가 지원하지 않는 메서드: Worksheet.{name}()\n"
            f"→ prelude.py의 _StubWorksheet에 추가할 것."
        )


class _StubSpreadsheet:
    def __init__(self, sheet_id: str) -> None:
        self._sheet_id = sheet_id
        self._ws = _StubWorksheet(sheet_id)

    @property
    def sheet1(self) -> _StubWorksheet:
        return self._ws

    def get_worksheet(self, index: int) -> _StubWorksheet:
        if index != 0:
            raise PreludeError(
                "대역 gspread는 첫 번째 탭만 읽는다 (gviz에 sheet 파라미터를 주지 않기 때문). "
                f"요청된 인덱스: {index}"
            )
        return self._ws

    def worksheet(self, title: str) -> _StubWorksheet:
        return self._ws

    def __getattr__(self, name):
        raise PreludeError(
            f"대역 gspread가 지원하지 않는 메서드: Spreadsheet.{name}()\n"
            f"→ prelude.py의 _StubSpreadsheet에 추가할 것."
        )


class _StubClient:
    def open_by_key(self, sheet_id: str) -> _StubSpreadsheet:
        return _StubSpreadsheet(sheet_id)

    def open_by_url(self, url: str) -> _StubSpreadsheet:
        # .../spreadsheets/d/{ID}/... 에서 ID를 뽑는다
        parts = url.split("/d/")
        if len(parts) < 2:
            raise PreludeError(f"시트 ID를 찾을 수 없는 주소: {url}")
        return _StubSpreadsheet(parts[1].split("/")[0])

    def __getattr__(self, name):
        raise PreludeError(
            f"대역 gspread가 지원하지 않는 메서드: Client.{name}()\n"
            f"→ prelude.py의 _StubClient에 추가할 것."
        )


def _install_gspread() -> None:
    try:
        import gspread                            # 진짜가 깔려 있으면 authorize만 갈아끼운다
        gspread.authorize = lambda creds=None, *a, **k: _StubClient()
    except Exception:
        mod = _fake_module("gspread")
        mod.authorize = lambda creds=None, *a, **k: _StubClient()
        mod.Client = _StubClient
        mod.Spreadsheet = _StubSpreadsheet
        mod.Worksheet = _StubWorksheet
        sys.modules["gspread"] = mod


# ---------------------------------------------------------------------------
# 4. 학습 시간 상한 — model.fit()의 epochs를 실행 시점에만 낮춘다
# ---------------------------------------------------------------------------
_epoch_cap_log: list[tuple[int, int]] = []
_epoch_limit: list[int | None] = [None]        # 현재 걸린 상한. None이면 상한 없음


def _install_epoch_cap(limit: int | None) -> None:
    """Keras의 Model.fit을 감싸 epochs 상한을 씌운다.

    노트북 코드는 그대로 epochs=10을 넘기고, 중간에서 우리가 낮춰 받는다.
    한 프로세스에서 노트북을 여러 개 돌리므로, 상한 값은 매번 갈아끼운다.
    (DNN 3 → CNN 2처럼 노트북마다 다르기 때문이다)
    """
    _epoch_limit[0] = limit
    if limit is None:
        return

    try:
        import keras
    except ImportError:
        raise PreludeError(
            "에포크 상한을 걸려 했으나 keras를 불러올 수 없다. "
            "tensorflow(또는 tensorflow-cpu)가 설치되어 있는지 확인할 것."
        )

    if getattr(keras.Model.fit, "_is_prelude_capped", False):
        return                                   # 감싸기는 한 번이면 된다

    original = keras.Model.fit

    def capped_fit(self, *args, **kwargs):
        cap = _epoch_limit[0]
        if cap is None:
            return original(self, *args, **kwargs)
        asked = kwargs.get("epochs")
        if asked is None and len(args) >= 4:
            # epochs가 위치 인자로 넘어온 경우. 이 책의 노트북에는 없지만,
            # 조용히 통과시키면 상한이 걸리지 않은 채 오래 돌게 된다.
            raise PreludeError(
                "epochs가 위치 인자로 전달되어 상한을 걸 수 없다. "
                "prelude.py의 capped_fit을 손볼 것."
            )
        if asked is not None and asked > cap:
            _epoch_cap_log.append((asked, cap))
            kwargs["epochs"] = cap
        return original(self, *args, **kwargs)

    capped_fit._is_prelude_capped = True
    keras.Model.fit = capped_fit


def epoch_cap_applied() -> list[tuple[int, int]]:
    """(원래 에포크, 낮춘 에포크) 목록. 비어 있으면 상한이 걸리지 않은 것이다."""
    return list(_epoch_cap_log)


# ---------------------------------------------------------------------------
# 5. 그림 그리기 — 화면 없는 서버에서 창을 띄우려다 멈추는 것을 막는다
# ---------------------------------------------------------------------------
def _install_headless_matplotlib() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)         # 파일로만 그린다. 창을 열지 않는다
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# 6. Gradio — 화면을 띄우는 대신, 화면에 미리 채워져 있던 기본값으로 한 번 돌려 본다
#
# `.launch()`는 웹 화면을 띄우고 사람이 닫을 때까지 기다린다. 자동 점검에서는
# 영원히 끝나지 않는다는 뜻이다.
#
# 그렇다고 `.launch()`를 '아무것도 하지 않는 함수'로 바꾸면 안 된다.
# 이 노트북들의 알맹이는 gr.Interface(fn=...) 의 fn 안에 들어 있고, fn 은
# 사람이 화면에서 버튼을 눌러야 비로소 실행된다. 그냥 넘기면 모델을 불러오는
# 데까지만 확인하고 **번역·요약·생성이 실제로 되는지는 하나도 보지 않은 채**
# 초록불이 켜진다. 태그 방식을 버린 것과 똑같은 종류의 실패다.
#
# 다행히 저자가 모든 입력 칸에 value= 로 예시값을 넣어 두었다. 독자가 화면을
# 열면 이미 채워져 있는 그 값이다. 대역은 그 값을 그대로 꺼내 fn 을 한 번
# 호출한다. 즉 **독자가 화면을 열고 버튼을 한 번 누른 것과 같은 일**을 한다.
# ---------------------------------------------------------------------------
_gradio_runs: list[dict] = []
_gradio_blocked: list[str] = []


def _install_gradio() -> None:
    try:
        import gradio as gr
    except ImportError:
        return                                    # 설치되지 않았으면 할 일이 없다

    if getattr(gr.Blocks.launch, "_prelude", False):
        return                                    # 한 프로세스에서 두 번 걸지 않는다

    def launch(self, *args, **kwargs):            # noqa: ANN001
        fn = getattr(self, "fn", None)
        comps = getattr(self, "input_components", None)

        if fn is None or comps is None:
            # gr.Interface 가 아니라 gr.Blocks 를 직접 쓴 경우. 부를 함수가 없다.
            _gradio_blocked.append("gr.Interface 가 아니어서 호출할 함수를 찾지 못했다")
            return None, "", ""

        values, missing = [], []
        for c in comps:
            v = getattr(c, "value", None)
            if v is None:
                missing.append(type(c).__name__)   # Image·Video 처럼 예시값이 없는 칸
            values.append(v)

        if missing:
            # 예시값이 없으면 지어내지 않는다. 가짜 입력으로 통과시키면
            # 그것 역시 아무것도 검증하지 않은 초록불이 된다.
            _gradio_blocked.append(
                f"입력 칸에 예시값(value=)이 없어 호출하지 못했다: {', '.join(missing)}")
            return None, "", ""

        label = getattr(self, "title", None) or fn.__name__
        print(f"\n[대역] 화면을 띄우는 대신 기본값으로 한 번 실행한다 — {label}")
        for c, v in zip(comps, values):
            print(f"   입력 {getattr(c, 'label', '?')}: {str(v)[:60]}")

        result = fn(*values)                       # 여기서 터지면 그대로 실패로 잡힌다

        text = result if isinstance(result, str) else repr(result)
        _gradio_runs.append(dict(title=label, chars=len(text)))
        print(f"   출력({len(text)}자): {text[:300]}")
        if len(text) > 300:
            print("   ...")
        return None, "", ""

    launch._prelude = True
    gr.Blocks.launch = launch


def gradio_runs() -> list[dict]:
    """대역이 실제로 불러 본 Gradio 함수 목록."""
    return list(_gradio_runs)


def gradio_blocked() -> list[str]:
    """`.launch()`가 있었는데 함수를 부르지 못한 사유. 비어 있어야 정상이다."""
    return list(_gradio_blocked)


# ---------------------------------------------------------------------------
# ipywidgets 버튼 대역
# ---------------------------------------------------------------------------
# Gradio 대역과 목적이 같다. VII-1 'Generative AI_Text Generator' 의 알맹이는
# `on_click()` 안에 있고, 그 함수는 **사람이 버튼을 눌러야** 실행된다.
# 그대로 두면 러너는 위젯을 만들어 화면에 얹는 시늉만 하고 끝나, 모델 로딩까지만
# 확인한 채 초록불이 켜진다. 태그 방식을 버린 것과 똑같은 실패다.
#
# 그래서 `Button.on_click` 을 가로채, 콜백을 등록하는 그 자리에서 한 번 불러 준다.
# **독자가 화면을 열고 버튼을 한 번 누른 것과 같은 일**이다.
#
# ⚠ 등록 시점에 부르므로, 콜백이 **나중에 만들어지는 변수**를 참조하는 노트북에서는
#   NameError 가 난다. VII-1 은 on_click 등록 시점에 generator·dropdown·temp·
#   length·output 이 모두 준비돼 있어 안전하다. 새 노트북을 붙일 때 이 조건을 볼 것.
#
# ⚠ ipywidgets 8.x 의 `Output.__exit__` 는 IPython 커널이 없으면 예외를 삼키지 않는다
#   (8.1.9 실측). 그래서 `with output:` 안에서 터진 것이 러너까지 올라온다.
#   **7.x 는 무조건 삼켰다** — 판번호가 내려가면 이 대역이 통째로 무력해진다.
#   워크플로의 판번호 기록에 ipywidgets 가 들어 있는 이유다.
_button_clicks: list[dict] = []
_button_blocked: list[str] = []
_display_depth: list[int] = [0]          # 클릭이 도는 동안에만 출력을 기록한다
_display_seen: list[dict] = []


def _install_display_probe() -> None:
    """`display()` 로 넘어온 것의 길이를 잰다 (클릭이 도는 동안만).

    버튼만 눌러서는 '터지지 않았다'까지만 알 수 있다. KoGPT2 가 빈 문자열을
    돌려줘도 초록불이 된다. VI-11 에서 `max_length` 가 조용히 무시됐을 때와 같은
    자리다 — 터지지도 값이 어긋나지도 않는 고장.

    Markdown·HTML 객체는 원문을 `.data` 에 들고 있어 길이를 잴 수 있다.
    노트북이 `from IPython.display import display` 하기 **전에** 갈아끼워야
    노트북이 이 대역을 가져간다. prelude 는 노트북보다 먼저 도니 조건이 맞는다.
    클릭 중이 아니면 원본에 그대로 넘기므로 다른 노트북에는 영향이 없다.

    ⚠ **한계: `display()` 를 통과하는 것만 잰다.**
    위젯 속성에 직접 쓰는 코드는 지나간다. IX-3 의 색깔 네모 보드가 그렇다 —
    `board.value = ...` 로 꽂으므로 100회를 돌려 네모가 100개 그려져도 측정값은
    늘지 않는다(실측: 단일 1회 829자 > 일괄 100회 790자).
    **즉 이 숫자는 '요약 텍스트의 길이'이지 '독자가 보는 것의 양'이 아니다.**
    `.value` 대입까지 가로채려면 위젯 클래스에 손을 대야 하는데, 잡히는 것에 비해
    대역이 복잡해져 하지 않았다. 상한을 정할 때 이 성질을 기억할 것.
    """
    try:
        import IPython.display as ipd
    except ImportError:
        return

    if getattr(ipd.display, "_prelude", False):
        return

    original = ipd.display

    def display(*objs, **kwargs):         # noqa: ANN001
        if _display_depth[0] > 0:
            for o in objs:
                data = getattr(o, "data", None)
                if isinstance(data, str):
                    _display_seen.append(dict(kind=type(o).__name__,
                                              chars=len(data), text=data))
        return original(*objs, **kwargs)

    display._prelude = True
    ipd.display = display


def _install_button() -> None:
    try:
        import ipywidgets as widgets
    except ImportError:
        return                                    # 설치되지 않았으면 할 일이 없다

    if getattr(widgets.Button.on_click, "_prelude", False):
        return                                    # 한 프로세스에서 두 번 걸지 않는다

    original = widgets.Button.on_click

    def on_click(self, callback, remove=False):   # noqa: ANN001
        original(self, callback, remove=remove)
        if remove:
            return                                # 등록 해제는 그냥 넘긴다

        label = getattr(self, "description", None) or "이름 없는 버튼"
        if not callable(callback):
            _button_blocked.append(f"'{label}' 에 걸린 것이 함수가 아니다")
            return

        print(f"\n[대역] 사람이 누르는 대신 '{label}' 을 한 번 누른다")

        before = len(_display_seen)
        _display_depth[0] += 1
        try:
            callback(self)                        # 여기서 터지면 그대로 실패로 잡힌다
        finally:
            _display_depth[0] -= 1

        produced = _display_seen[before:]
        total = sum(d["chars"] for d in produced)
        _button_clicks.append(dict(button=label,
                                   callback=getattr(callback, "__name__", "?"),
                                   chars=total, items=len(produced)))
        if produced:
            head = produced[0]["text"].replace("\n", " ")
            print(f"   출력({total}자): {head[:300]}")
            if total > 300:
                print("   ...")
        else:
            # 터지지는 않았는데 아무것도 내놓지 않았다. 실패시키지는 않되
            # run_chapters 가 볼 수 있게 남긴다 — 조용한 성공을 만들지 않기 위해서다.
            _button_blocked.append(f"'{label}' 을 눌렀으나 display() 로 나온 것이 없다")
            print("   출력 없음 ⚠")

    on_click._prelude = True
    widgets.Button.on_click = on_click


def button_clicks() -> list[dict]:
    """대역이 실제로 눌러 본 버튼 목록."""
    return list(_button_clicks)


def button_blocked() -> list[str]:
    """버튼을 눌렀으나 아무것도 확인하지 못한 사유. 비어 있어야 정상이다."""
    return list(_button_blocked)


# ---------------------------------------------------------------------------
# 7. 셸 명령(`!wget` 같은 줄) — 노트북에만 있는 문법이라 파이썬이 이해하지 못한다
#
# `!pip install` 은 실행기가 아예 지워 버린다(워크플로가 미리 설치하므로).
# 그 밖의 `!` 줄은 이 함수를 거쳐 진짜로 실행된다. 자료를 내려받는 줄
# (`!wget`, `!unzip`)은 실제로 돌아야 노트북이 이어지기 때문이다.
# ---------------------------------------------------------------------------
def shell(command: str) -> None:
    import subprocess
    print(f"[대역] 셸 실행: {command}")
    done = subprocess.run(command, shell=True, capture_output=True, text=True)
    if done.stdout.strip():
        print(done.stdout.strip()[:500])
    if done.returncode != 0:
        raise PreludeError(
            f"셸 명령이 실패했다 (종료 코드 {done.returncode}): {command}\n"
            f"{done.stderr.strip()[:500]}")


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------
def install(notebook: str, inputs: list[str] | None = None, verbose: bool = True) -> None:
    """노트북 실행 직전에 대역을 세운다.

    notebook : 노트북 파일 이름 (경로 없이). NOTEBOOK_INPUTS 조회에 쓴다.
    inputs   : 표를 무시하고 직접 넘길 때만 사용.
    """
    name = os.path.basename(notebook)
    answers = inputs if inputs is not None else NOTEBOOK_INPUTS.get(name, [])

    _epoch_cap_log.clear()
    _gradio_runs.clear()
    _gradio_blocked.clear()
    _button_clicks.clear()
    _button_blocked.clear()
    _display_seen.clear()
    _display_depth[0] = 0
    _install_input(answers)
    _install_google_auth()
    _install_gspread()
    _install_headless_matplotlib()
    _install_gradio()
    _install_display_probe()
    _install_button()

    limit = MAX_EPOCHS.get(name)
    if limit is not None or _epoch_limit[0] is not None:
        # 상한이 있으면 걸고, 앞 노트북에 걸려 있던 상한은 여기서 푼다
        _install_epoch_cap(limit)

    if verbose:
        note = f" · 에포크 상한 {limit}" if limit else ""
        print(f"[prelude] {name} · 대역 설치 완료 (input {len(answers)}개 준비){note}")
