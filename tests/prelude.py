"""
tests/prelude.py — 레인 B(월간 노트북 실행) 전용 '대역' 모음
================================================================

이 파일은 노트북을 **고치지 않고** 자동 실행하기 위한 것이다.
실행기가 노트북을 메모리로 읽어들인 뒤, 아래 install()을 호출하는 셀 하나를
맨 앞에 임시로 끼워 넣는다. 그 셀은 저장되지 않는다.

대역을 세우는 곳은 세 군데뿐이다.

  1. input()              — 사람 대신 미리 정해둔 답을 순서대로 내어준다
  2. google.colab.auth    — 인증 팝업 대신 "성공했다"고만 대답한다
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


def _install_google_auth() -> None:
    # google 네임스페이스 패키지를 손상시키지 않도록, 이미 있으면 속성만 바꾼다.
    google = sys.modules.get("google")
    if google is None:
        google = types.ModuleType("google")
        google.__path__ = []                     # 패키지로 인식시킨다
        sys.modules["google"] = google

    # --- google.colab.auth ---------------------------------------------------
    colab = types.ModuleType("google.colab")
    colab.__path__ = []
    auth = types.ModuleType("google.colab.auth")
    auth.authenticate_user = lambda *a, **k: None
    colab.auth = auth
    google.colab = colab
    sys.modules["google.colab"] = colab
    sys.modules["google.colab.auth"] = auth

    # --- google.auth.default -------------------------------------------------
    # google-auth가 실제로 깔려 있으면 default()만 갈아끼운다.
    try:
        import google.auth as real_auth          # noqa: F401
        real_auth.default = lambda *a, **k: (_FakeCredentials(), "prelude-project")
    except Exception:
        ga = types.ModuleType("google.auth")
        ga.__path__ = []
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
        mod = types.ModuleType("gspread")
        mod.authorize = lambda creds=None, *a, **k: _StubClient()
        mod.Client = _StubClient
        mod.Spreadsheet = _StubSpreadsheet
        mod.Worksheet = _StubWorksheet
        sys.modules["gspread"] = mod


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

    _install_input(answers)
    _install_google_auth()
    _install_gspread()

    if verbose:
        print(f"[prelude] {name} · 대역 설치 완료 (input {len(answers)}개 준비)")
