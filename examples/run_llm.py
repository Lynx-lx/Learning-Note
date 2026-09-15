from __future__ import annotations

import numpy as np

from algorithm.llm import EchoLLM, NumpyVectorStore, PromptTemplate, SimpleRAG


def bag_of_chars(text: str, dim: int = 32) -> np.ndarray:
    vec = np.zeros(dim, dtype=float)
    for ch in text:
        vec[ord(ch) % dim] += 1.0
    return vec


def main() -> None:
    prompt = PromptTemplate("用一句话解释：{topic}")
    llm = EchoLLM()
    print("提示词:", prompt.format(topic="图像分类"))
    print("推理:", llm.chat(prompt.as_messages(topic="图像分类")))

    rag = SimpleRAG(llm=llm, store=NumpyVectorStore(), embed_fn=bag_of_chars)
    rag.add_documents(
        [
            "卷积神经网络适合图像分类与检测。",
            "检索增强生成把外部知识注入大模型回答。",
            "逻辑回归是经典二分类机器学习模型。",
        ]
    )
    print("RAG:", rag.ask("什么是检索增强生成？", top_k=2))


if __name__ == "__main__":
    main()
