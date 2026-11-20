"""표준국어대사전 오픈 API (https://stdict.korean.go.kr/openapi). 결과는 ko/k1/local/stdict_cache.jsonl에 쌓는다(깃에 올리지 않음).

인증키는 환경 변수 STDICT_KEY에서만 읽는다(Windows에서 이미 떠 있는 프로세스가 시스템 변수를 못 받았으면 레지스트리에서 읽는다).
키는 출력하거나 파일에 쓰지 않는다.
"""
import json
import os
import time
from pathlib import Path

import requests

API = "https://stdict.korean.go.kr/api/search.do"
CACHE = Path(__file__).resolve().parents[1] / "k1" / "local" / "stdict_cache.jsonl"


def _key() -> str:
    k = os.environ.get("STDICT_KEY")
    if k:
        return k
    try:
        import winreg
        for hive, path in ((winreg.HKEY_CURRENT_USER, "Environment"),
                           (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")):
            try:
                with winreg.OpenKey(hive, path) as h:
                    return winreg.QueryValueEx(h, "STDICT_KEY")[0]
            except OSError:
                continue
    except ImportError:
        pass
    raise RuntimeError("STDICT_KEY 환경 변수가 없다")


class Stdict:
    def __init__(self, sleep: float = 0.05):
        self.key, self.sleep = _key(), sleep
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        if CACHE.exists():
            for line in CACHE.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.cache[r["q"]] = r["items"]

    def lookup(self, word: str) -> list[dict]:
        """정확히 일치하는 표제어(명사 등 모든 품사)의 항목들: word, sup_no(어깨번호), pos, definition."""
        if word in self.cache:
            return self.cache[word]
        params = dict(key=self.key, q=word, req_type="json", advanced="y", method="exact", num=100)
        for attempt in range(3):
            try:
                r = requests.get(API, params=params, timeout=20)
                break
            except requests.RequestException:
                time.sleep(2 + 3 * attempt)
        else:
            raise RuntimeError(f"요청 실패: {word}")
        items = []
        txt = r.text.strip()
        if txt:
            try:
                data = r.json()
            except ValueError:
                raise RuntimeError(f"JSON이 아닌 응답 (상태 {r.status_code}): {txt[:200]}")
            for it in (data.get("channel", {}) or {}).get("item", []) or []:
                sense = it.get("sense") or {}
                if isinstance(sense, list):
                    sense = sense[0] if sense else {}
                items.append(dict(word=it.get("word"), sup_no=it.get("sup_no"), pos=it.get("pos"),
                                  definition=sense.get("definition"), target_code=it.get("target_code")))
        self.cache[word] = items
        with CACHE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(q=word, items=items), ensure_ascii=False) + "\n")
        time.sleep(self.sleep)
        return items
