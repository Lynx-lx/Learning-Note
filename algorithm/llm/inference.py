from __future__ import annotations

from abc import ABC, abstractmethod

from algorithm.llm.prompt import Message


class LLMClient(ABC):
    """大模型推理接口。对接 OpenAI / 本地 vLLM / Ollama 时继承此类。"""

    @abstractmethod
    def chat(self, messages: list[Message], **kwargs) -> str:
        raise NotImplementedError


class EchoLLM(LLMClient):
    """离线占位实现，便于本地跑通流水线。"""

    def chat(self, messages: list[Message], **kwargs) -> str:
        last = messages[-1].content if messages else ""
        return f"[echo] {last}"
