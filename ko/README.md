# ko/ — 한국어 말장난 (K 계열)

영어 E5의 세 단계 생성과 청자 모델 고르기를 한국어로 옮긴다. 설계: [DESIGN.md](DESIGN.md).

- 1단계 범위: 한 단어(명사) 말장난. A 갈래(철자 다른 소리 쌍)와 B 갈래(같은 철자 동음이의어).
- 지금 단계: K0 도구 확인, K1 쌍 목록, K2 사람 말장난 모으기([K2_GUIDE.md](K2_GUIDE.md), 양식 `k2/k2_template.csv`).
- `k2/local/`은 깃에 올리지 않는다.

## 자료 출처

- `k1/nouns.csv`, `k1/pairs_*.csv`: [wordfreq](https://github.com/rspeer/wordfreq) 한국어 빈도 목록(CC BY-SA 4.0)에서 만들었다. 발음은 g2pkk 규칙 + Kiwi.
- `k1/homograph_candidates.csv`: 국립국어원 「한국어 학습용 어휘 목록」(2003, 공공누리 제1유형: 출처 표시)에서 낱말·동형어 번호·원어·등급만 옮겼다.
  원본 파일은 연구자가 내려받아 `k1/local/`에 둔다(올리지 않음).
- 표준국어대사전 오픈 API 결과(뜻풀이)는 `k1/local/`에 캐시만 하고 올리지 않는다.
