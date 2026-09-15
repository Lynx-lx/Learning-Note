from __future__ import annotations

import numpy as np

from algorithm.common.metrics import classification_report
from algorithm.common.pipeline import Pipeline
from algorithm.ml import LogisticRegression
from algorithm.nlp import CountVectorizer


def main() -> None:
    texts = [
        "图像 分类 卷积 网络",
        "目标 检测 图像 框",
        "提示词 大模型 生成 文本",
        "检索 增强 大模型 问答",
        "图像 分割 像素 分类",
        "聊天 大模型 对齐",
    ]
    labels = np.array([0, 0, 1, 1, 0, 1])
    pipe = Pipeline(steps=[CountVectorizer()], estimator=LogisticRegression(epochs=400))
    pipe.fit(texts, labels)
    pred = pipe.predict(texts)
    print("文本分类:", classification_report(labels, pred))


if __name__ == "__main__":
    main()
