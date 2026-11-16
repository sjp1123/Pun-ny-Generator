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


def hf_revision(name: str, model=None) -> str | None:
    """Hugging Face 가중치 커밋 해시. transformers 5에서는 config._commit_hash가 비어 있어서 내려받은 스냅샷 폴더 이름으로 찾는다."""
    rev = getattr(getattr(model, "config", None), "_commit_hash", None)
    if rev:
        return rev
    try:
        from pathlib import Path
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(name, local_files_only=True)).name
    except Exception:
        return None


class Generator:
    def __init__(self, backend: str, model: str, device: str | None = None, dtype: str = "bfloat16",
                 host: str = "http://localhost:11434", think: bool | None = None, load_4bit: bool = False):
        self.backend, self.model_name, self.host, self.think = backend, model, host, think
        self.info = dict(backend=backend, model=model)
        if backend == "hf":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.torch = torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.tok = AutoTokenizer.from_pretrained(model)
            if load_4bit:  # bitsandbytes NF4, 계산은 dtype(bf16). 32B를 24~48GB GPU에 올릴 때
                from transformers import BitsAndBytesConfig
                q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=getattr(torch, dtype))
                self.model = AutoModelForCausalLM.from_pretrained(model, quantization_config=q,
                                                                  device_map={"": self.device}).eval()
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model, dtype=getattr(torch, dtype)).to(self.device).eval()
            self.info.update(dtype=dtype, device=self.device, revision=hf_revision(model, self.model),
                             quantization="bitsandbytes nf4 (4bit)" if load_4bit else None,
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

    def chat(self, user: str, seed: int, temperature: float, max_new_tokens: int = 1500, system: str | None = SYSTEM,
             top_p: float = 1.0) -> str:
        """system=None이면 시스템 메시지 없이 사용자 한 턴만 보낸다(v2 단계 프롬프트)."""
        msgs = ([dict(role="system", content=system)] if system else []) + [dict(role="user", content=user)]
        if self.backend == "hf":
            torch = self.torch
            enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
            ids = enc["input_ids"].to(self.device)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            kw = dict(do_sample=True, temperature=temperature, top_p=top_p, top_k=0) if temperature > 0 else dict(do_sample=False)
            with torch.inference_mode():
                out = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                          pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id, **kw)
            return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        body = dict(model=self.model_name, messages=msgs, stream=False,
                    options=dict(temperature=temperature, seed=seed, num_predict=max_new_tokens, top_p=top_p, top_k=0))
        if self.think is not None:
            body["think"] = self.think
        r = self.http.post(f"{self.host}/api/chat", json=body, timeout=1800)
        r.raise_for_status()
        return r.json()["message"]["content"]
