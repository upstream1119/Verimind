import os
import re
import time
from dataclasses import dataclass
from typing import Optional, Protocol

from dotenv import load_dotenv
from llama_index.llms.dashscope import DashScope
from openai import OpenAI
from zhipuai import ZhipuAI

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

REFUSAL_FLAGS = [
    "information not found",
    "not found",
    "cannot find",
    "unable to determine",
    "rejected",
    "无证据",
    "未找到",
    "未提及",
]


@dataclass
class ClientConfig:
    provider: str
    model_name: str
    api_key_env: str
    base_url: Optional[str] = None
    disable_thinking: bool = False
    max_retries: int = 3
    retry_base_sec: float = 3.0


class LLMClient(Protocol):
    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.01) -> str:
        ...

    def score_faithfulness(self, answer: str, context: str) -> int:
        ...


def _require_env(env_name: str) -> str:
    value = os.environ.get(env_name, "").strip()
    if not value:
        raise ValueError(f"Missing required env var: {env_name}")
    return value


def _parse_score(raw_text: str, default_score: int = 6) -> int:
    m = re.search(r"\b(10|[0-9])\b", raw_text or "")
    if not m:
        return default_score
    score = int(m.group(1))
    return max(0, min(10, score))


def _contains_refusal(answer: str) -> bool:
    low = (answer or "").lower()
    return any(flag in low for flag in REFUSAL_FLAGS)


def _format_call_error(provider: str, err: Exception) -> str:
    return f"{provider} call failed: {type(err).__name__}: {err}"


def _make_faith_prompt(answer: str, context: str) -> str:
    return (
        "Evaluate if the [Generated Answer] is strictly derived from the [Retrieved Context].\n"
        "Output ONLY an integer score from 0 to 10.\n"
        "10 = completely faithful; 0 = completely hallucinated.\n\n"
        f"[Context]: {(context or '')[:3000]}\n"
        f"[Answer]: {answer}\n"
        "Score:"
    )


class _BaseClient:
    def __init__(self, config: ClientConfig) -> None:
        self.config = config

    def _complete_once(self, prompt: str, max_tokens: int, temperature: float) -> str:
        raise NotImplementedError

    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.01) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                return self._complete_once(prompt, max_tokens, temperature).strip()
            except Exception as err:
                last_err = err
                if attempt < self.config.max_retries - 1:
                    sleep_sec = self.config.retry_base_sec * (2 ** attempt)
                    time.sleep(sleep_sec)
        raise RuntimeError(_format_call_error(self.config.provider, last_err or RuntimeError("unknown")))

    def score_faithfulness(self, answer: str, context: str) -> int:
        if _contains_refusal(answer):
            return 10
        prompt = _make_faith_prompt(answer, context)
        raw = self.complete(prompt, max_tokens=16, temperature=0.01)
        return _parse_score(raw)


class QwenClient(_BaseClient):
    def __init__(self, config: ClientConfig) -> None:
        super().__init__(config)
        self._api_key = _require_env(config.api_key_env)

    def _complete_once(self, prompt: str, max_tokens: int, temperature: float) -> str:
        llm = DashScope(
            model_name=self.config.model_name,
            api_key=self._api_key,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return llm.complete(prompt).text


class GLMClient(_BaseClient):
    def __init__(self, config: ClientConfig) -> None:
        super().__init__(config)
        self._client = ZhipuAI(api_key=_require_env(config.api_key_env))

    def _complete_once(self, prompt: str, max_tokens: int, temperature: float) -> str:
        kwargs = {}
        if self.config.disable_thinking:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        resp = self._client.chat.completions.create(
            model=self.config.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return resp.choices[0].message.content or ""


class DeepSeekClient(_BaseClient):
    def __init__(self, config: ClientConfig) -> None:
        super().__init__(config)
        self._client = OpenAI(
            api_key=_require_env(config.api_key_env),
            base_url=config.base_url or "https://api.deepseek.com",
        )

    def _complete_once(self, prompt: str, max_tokens: int, temperature: float) -> str:
        resp = self._client.chat.completions.create(
            model=self.config.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


def build_client(
    provider: str,
    max_retries: int = 3,
    retry_base_sec: float = 3.0,
    model_name_override: Optional[str] = None,
) -> LLMClient:
    p = (provider or "").strip().lower()
    if p == "qwen":
        model_name = model_name_override or "qwen-plus"
        return QwenClient(
            ClientConfig(
                provider="qwen",
                model_name=model_name,
                api_key_env="DASHSCOPE_API_KEY",
                max_retries=max_retries,
                retry_base_sec=retry_base_sec,
            )
        )
    if p == "glm":
        model_name = model_name_override or "glm-4.5-air"
        return GLMClient(
            ClientConfig(
                provider="glm",
                model_name=model_name,
                api_key_env="ZHIPU_API_KEY",
                disable_thinking=True,
                max_retries=max_retries,
                retry_base_sec=retry_base_sec,
            )
        )
    if p == "deepseek":
        model_name = model_name_override or "deepseek-chat"
        return DeepSeekClient(
            ClientConfig(
                provider="deepseek",
                model_name=model_name,
                api_key_env="DEEPSEEK_API_KEY",
                base_url="https://api.deepseek.com",
                max_retries=max_retries,
                retry_base_sec=retry_base_sec,
            )
        )
    raise ValueError(f"Unsupported provider: {provider}. Use qwen|glm|deepseek.")
