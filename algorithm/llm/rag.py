from __future__ import annotations

import numpy as np

from algorithm.llm.inference import LLMClient
from algorithm.llm.prompt import Message, PromptTemplate


class NumpyVectorStore:
    """基于余弦相似度的最小向量库。"""

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.vectors: np.ndarray | None = None

    def add(self, texts: list[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=float)
        self.texts.extend(texts)
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])

    def search(self, query: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
        if self.vectors is None or not self.texts:
            return []
        q = np.asarray(query, dtype=float).ravel()
        q_norm = np.linalg.norm(q) or 1.0
        v_norm = np.linalg.norm(self.vectors, axis=1)
        v_norm[v_norm == 0] = 1.0
        scores = (self.vectors @ q) / (v_norm * q_norm)
        k = min(top_k, len(self.texts))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.texts[i], float(scores[i])) for i in idx]


class SimpleRAG:
    def __init__(self, llm: LLMClient, store: NumpyVectorStore, embed_fn) -> None:
        self.llm = llm
        self.store = store
        self.embed_fn = embed_fn
        self.template = PromptTemplate("参考资料：\n{context}\n\n问题：{question}")

    def add_documents(self, texts: list[str]) -> None:
        vectors = np.stack([self.embed_fn(t) for t in texts])
        self.store.add(texts, vectors)

    def ask(self, question: str, top_k: int = 3) -> str:
        hits = self.store.search(self.embed_fn(question), top_k=top_k)
        context = "\n".join(f"- {text}" for text, _ in hits) or "（无检索结果）"
        prompt = self.template.format(context=context, question=question)
        return self.llm.chat([Message("user", prompt)])
