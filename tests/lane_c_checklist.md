레인 C 는 **독자의 자리에 앉아 Colab 에서 직접 돌려 보는 점검**이다.
레인 A(주간·자산)와 레인 B(월간·노트북)가 구조적으로 볼 수 없는 것을 맡는다.

> 이 목록은 `tests/lane_c_checklist.md` 를 그대로 옮긴 것이다.
> 고칠 것이 생기면 **워크플로가 아니라 그 파일**을 고친다.

---

## C-0 · 설치줄 점검 — CPU · 10분 안팎

레인 B 는 노트북의 `!pip` 줄을 **지우고** 최신판으로 돌린다. 그래서 설치 줄이
무엇이든 초록불이 뜬다. 실제로 `sentencepiece` 와 `accelerate` 가 빠져 있었는데도
몇 달 동안 초록불이었다.

- [ ] Colab 에서 `tests/lane_c0_install_check.ipynb` 를 연다 (런타임 유형은 그대로 — GPU 불필요)
- [ ] 위에서부터 셀 7개를 차례로 실행한다 (셀 6 이 10~20분)
- [ ] 셀 7 이 찍은 표 두 개를 지침서에 붙여넣는다
- [ ] 「저자가 판단할 것」 다섯 칸에 답을 적는다

**이번에 특히 볼 것**

- [ ] 사각지대 넷(`gymnasium`·`gspread`·`ipywidgets`·`tensorflow-datasets`)이 여전히 Colab 기본에 있는가
- [ ] 고정값 10개가 여전히 Colab 기본과 같은가
      → **달라졌다면 설치 줄이 「아무것도 안 하는 줄」에서 「내려받아 강등하는 줄」로 바뀐 것이다.**
        설치 시간이 몇 초에서 몇 분으로 뛰고 세션 재시작이 끼어들 수 있다.
        고정값 재조사가 필요하다는 신호다.

---

## C-1 · 제외 7건 실행 — GPU T4 · 2시간 안팎

레인 B 가 용량·GPU 때문에 등록하지 않은 노트북들. **한 번도 자동으로 돈 적이 없다.**
총 내려받기 약 11GB.

> ⚠ `VII-2 Diffusion` 만 T4 + fp16 이 진짜로 필요하다. 무료 등급에서 GPU 가 귀하면
> **이것부터** 돌린다.

- [ ] `VI-4` NLP_Transfer Learning_01 — cats_vs_dogs 25,000장 × 모델 2개
- [ ] `VI-5` NLP_Transfer Learning_02 — GloVe 822MB + 양방향 LSTM 2개
- [ ] `VI-9` NLP_Transformer-based translation model — mBART-large-50 약 2.4GB
- [ ] `VII-2` Generative AI_Diffusion — 라이선스 게이트 확인 · **GPU T4 필수**
- [ ] `VII-3①` Generative AI_Music — MusicGen 2.36GB
- [ ] `VIII-2` Multimodal AI_Image Captioning — `gr.Image` 에 예시값이 없어 대역이 못 돈다
- [ ] `VIII-3` Multimodal AI_Video Subtitle — `gr.Video` 에 예시값이 없어 대역이 못 돈다

**VIII장 둘은 Gradio 화면을 띄우지 말고 `fn` 을 직접 부른다.**
레인 A 가 감시 중인 실제 파일이 드라이브에 있다.

```python
generate_caption(Image.open("gift box.png"))     # VIII-2 : fn=generate_caption, 입력 PIL
transcribe("[TED-Ed] Sample Video_30s.mp4")      # VIII-3 : fn=transcribe,      입력 경로
```

**`input()` 이 있는 둘은 넣을 문장을 미리 정해 둔다** — 즉석에서 타이핑하면 시간 측정이 오염된다.

- [ ] 셀별 소요 시간을 남긴다
- [ ] **원고의 시간 문구와 대조한다**: VI-4 「Colab GPU로도 약 10분」 · VI-5 「20분 이상」

---

## C-2 · 드라이브 파일 11건 눈으로 대조

레인 A 는 이 11건을 **`manual` 로 표시하고 건너뛴다.** 드라이브 안의 개별 파일은
공개 폴더 URL 만으로 존재 여부를 확인할 수 없기 때문이다. 주간 보고서 맨 끝의
「수동 확인 대상 11건은 레인 C에서 폴더 목록 대조」가 이 절을 가리킨다.

드라이브의 **`실습데이터/` 폴더를 열고** 아래 이름이 그대로 있는지 본다.
**이름이 한 글자만 달라도 노트북은 파일을 못 찾는다.**

- [ ] `Patient Registration Data` (II-2-3)
- [ ] `Patient Clinical Data` (II-2-3)
- [ ] `Patient Health Monitoring Data` (II-2-3)
- [ ] `Patient Registration Data_Processed` (II-2-4)
- [ ] `Patient Health Monitoring Data_Processed` (II-2-4)
- [ ] `Data_Population Status_Kor_20251227` (II-1)
- [ ] `[TED-Ed] Sample Video_30s.mp4` (VIII-3)
- [ ] `[cosmoswag_kr] Sample Video_30s.mp4` (VIII-3)
- [ ] `[Sora] smiling woman.mp4` (VIII-1)
- [ ] `[DALL·E] gift box.png` (VIII-1) — **`·` 는 가운뎃점이다. 마침표가 아니다**
- [ ] `[Pixabay] crowd-cheers.mp3` (VIII-1)

- [ ] 폴더 안에 **목록에 없는 파일**이 있는가
      → 있다면 `assets.csv` 에 빠진 것이거나 원고에서만 쓰이는 것이다. 어느 쪽인지 확인한다

---

## C-3 · 눈으로 볼 것 — 레인 B 가 초록불인데 값으로 확인되지 않는 것들

**예외가 안 났다는 것은 성공이 아니다.** 아래는 코드가 판정할 수 없어 사람이 봐야 한다.

- [ ] `VI-9` **bertviz 그림** — 자바스크립트로 그려진다. 예외가 없어도 아무것도 안 보일 수 있다
- [ ] `VI-8` `_str` — 「어텐션 모델은 장문을 재현하고 기본 모델은 실패한다」가 실제로 그러한가
      (씨앗은 같은 TF 판번호 안에서만 재현을 보장하는데 러너는 최신 TF 로 돈다)
      → **길이 30 은 점이 셋이다.** 2025-08-15(TF 2.19 무렵) · 2026-08-22 두 번(TF 2.20.0)
        모두 「어텐션 완벽 · 기본 붕괴」. copy·reverse 양쪽에서 같았다. **이번 회차가 넷째 점이다**
      → ⚠ **흔들리는 것은 길이 5 쪽이다.** 2026-08-22 의 첫 실행에서 기본 모델이
        `abcde` 를 `abced` 로 냈다(두 번째 실행은 정상). **「짧은 문장은 두 모델 다 성공한다」는
        매 실행 참이 아니다.** 독자 중 일부는 다른 것을 본다 → 원고 문구를 볼 것
- [ ] `VI-7` 인코더-디코더의 **번역 품질** — 학습이 통째로 실패해도 초록불이다
- [ ] `VI-6` `_02` 예측 단어 — 저장 출력 0.6581 vs 원고 0.8001. 실측이 이미 두 번 갈렸다
- [ ] `IX-3` 「예측이 8:2로 편향된다」 — `default_rng()` 에 씨앗이 없다
- [ ] `VI-8` `_arrows` · `VII-1` · `IX-3` 의 **ipywidgets 7 동작**
      — Colab 은 7.7.1, 러너는 8.x 다. `with output:` 안에서 오류가 났을 때
        **독자에게 무엇이 보이는지** 확인된 적이 없다

---

## 마무리 — 잊으면 다음 분기가 조용해진다

- [ ] `tests/lane_c_last_run.txt` 의 날짜를 **오늘 날짜로** 고치고 커밋한다
- [ ] 이 이슈를 닫는다
- [ ] 지침서(작업 지침서)의 「현재 상태」와 「변경 이력」을 갱신한다
