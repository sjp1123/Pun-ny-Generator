"""E5 생성기 백엔드. 같은 프롬프트(system + user 한 턴)를 두 방식으로 돌린다.

- hf: transformers (GPU 주 실행). 예: allenai/OLMo-2-1124-7B-Instruct, bf16
- ollama: 로컬 Ollama 서버 (CPU 시험용, 양자화 모델). 예: olmo2:7b, ministral-3:8b

호출마다 시드를 정해 넘기므로 같은 백엔드·모델·시드면 같은 출력이 나온다(GPU 비결정성 범위 안에서).
온도 0이면 탐욕 해독(판정용).
"""
import re

from e5 import SYSTEM


def tag_of(model: str) -> str:
    """결과 폴더 이름: allenai/OLMo-2-1124-7B-Instruct → olmo-2-1124-7b-instruct, olmo2:7b → olmo2-7b"""
    return re.sub(r"[^a-z0-9.]+", "-", model.split("/")[-1].lower()).strip("-")


class Generator:
    def __init__(self, backend: str, model: str, device: str | None = None, dtype: str = "bfloat16",
                 host: str = "http://localhost:11434", think: bool | None = None):
        self.backend, self.model_name, self.host, self.think = backend, model, host, think
        self.info = dict(backend=backend, model=model)
        if backend == "hf":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.torch = torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.tok = AutoTokenizer.from_pretrained(model)
            self.model = AutoModelForCausalLM.from_pretrained(model, dtype=getattr(torch, dtype)).to(self.device).eval()
            self.info.update(dtype=dtype, device=self.device,
                             revision=getattr(self.model.config, "_commit_hash", None),
                             transformers=__import__("transformers").__version__, torch=torch.__version__)
        elif backend == "ollama":
            import requests
            self.http = requests
            r = requests.post(f"{host}/api/show", json=dict(model=model), timeout=60).json()
            self.info.update(digest=r.get("digest") or r.get("model_info", {}).get("general.file_type"),
                             details=r.get("details"),
                             ollama=requests.get(f"{host}/api/version", timeout=10).json().get("version"))
            tags = requests.get(f"{host}/api/tags", timeout=10).json().get("models", [])
            for m in tags:
                if m.get("name") == model or m.get("model") == model:
                    self.info["digest"] = m.get("digest")
        else:
            raise ValueError(backend)

    def chat(self, user: str, seed: int, temperature: float, max_new_tokens: int = 1500, system: str = SYSTEM) -> str:
        msgs = [dict(role="system", content=system), dict(role="user", content=user)]
        if self.backend == "hf":
            torch = self.torch
            enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
            ids = enc["input_ids"].to(self.device)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            kw = dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0) if temperature > 0 else dict(do_sample=False)
            with torch.inference_mode():
                out = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                          pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id, **kw)
            return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        body = dict(model=self.model_name, messages=msgs, stream=False,
                    options=dict(temperature=temperature, seed=seed, num_predict=max_new_tokens, top_p=1.0, top_k=0))
        if self.think is not None:
            body["think"] = self.think
        r = self.http.post(f"{self.host}/api/chat", json=body, timeout=1800)
        r.raise_for_status()
        return r.json()["message"]["content"]
