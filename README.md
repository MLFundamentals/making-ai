# making-ai

『손으로 배우는 인공지능』 실습 코드 저장소.
실습자료실: https://mlfundamentals.github.io/making-ai/

## 구성
- `notebooks/` — 실습 노트북 53개 (Google Colab 전용)
- `tests/` — 외부 자산 점검 스크립트
- `index.html` — 실습자료실 페이지 (GitHub Pages)

## 자산 점검
```bash
python tests/check_assets.py          # 전체 점검
python tests/check_assets.py --list   # 대상 목록만
```
표준 라이브러리만 사용합니다. 결과는 `tests/report.json`에 기록됩니다.
