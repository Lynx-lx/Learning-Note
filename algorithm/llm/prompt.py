from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Message:
    role: str
    content: str


class PromptTemplate:
    """占位符模板，例如：'根据图像描述回答：{question}'。"""

    def __init__(self, template: str) -> None:
        self.template = template

    def format(self, **kwargs: str) -> str:
        return self.template.format(**kwargs)

    def as_messages(self, system: str | None = None, **kwargs: str) -> list[Message]:
        messages: list[Message] = []
        if system:
            messages.append(Message("system", system))
        messages.append(Message("user", self.format(**kwargs)))
        return messages
