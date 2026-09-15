# Algorithm-Learning-Note

个人学习笔记。

## 算法基础框架

问卷中的算法方向均落到同一套接口：`Estimator` / `Transform` / `Pipeline`。

| 问卷项 | 模块 | 最小实现 |
| --- | --- | --- |
| 机器学习 | `algorithm/ml` | 线性回归、逻辑回归、KNN、KMeans |
| 大模型算法 | `algorithm/llm` | Prompt、LLM 接口、最小 RAG |
| 图像算法 | `algorithm/vision` | 预处理、Softmax 分类 |
| 规控算法 | `algorithm/control` | 栅格 A*、一维 PID |
| 自然语言处理 | `algorithm/nlp` | 词袋 + 逻辑回归 |
| 推荐算法 | `algorithm/rec` | 评分矩阵分解 |
| SLAM 算法 | `algorithm/slam` | 二维 ICP |
| 语音算法 | `algorithm/speech` | 波形特征 + 分类 |
| 多模态算法 | `algorithm/multimodal` | 特征拼接早期融合 |

公共层在 `algorithm/common`：统一估计器、预处理流水线、指标与配置。

### 目录

```text
Learning-Note/
├── algorithm/
│   ├── common/          # Pipeline / Estimator / 指标 / 配置
│   ├── ml/
│   ├── llm/
│   ├── vision/
│   ├── control/
│   ├── nlp/
│   ├── rec/
│   ├── slam/
│   ├── speech/
│   └── multimodal/
├── configs/default.yaml
├── examples/
├── requirements.txt
└── README.md
```

### 环境

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

将仓库根目录加入 `PYTHONPATH` 后运行示例：

```powershell
$env:PYTHONPATH = "."
python examples/run_ml.py
python examples/run_llm.py
python examples/run_vision.py
python examples/run_control.py
python examples/run_nlp.py
python examples/run_rec.py
python examples/run_slam.py
python examples/run_speech.py
python examples/run_multimodal.py
```

### 模块说明

#### 机器学习

- `LinearRegression`：正规方程最小二乘
- `LogisticRegression`：二分类梯度下降
- `KNNClassifier`：近邻分类
- `KMeans`：无监督聚类
- `StandardScaler`：标准化，可接入 `Pipeline`

#### 大模型算法

- `PromptTemplate`：提示词模板与 chat messages
- `LLMClient` / `EchoLLM`：推理抽象；`EchoLLM` 用于离线跑通
- `SimpleRAG` + `NumpyVectorStore`：文档入库、余弦检索、拼上下文再调用 LLM

接入真实模型时，继承 `LLMClient` 实现 `chat()`，替换 `embed_fn` 为句子向量模型即可。

#### 图像算法

- `Resize` / `ImageNormalize` / `load_image`：读图与标准化
- `SoftmaxClassifier`：将图像展平后做线性分类

#### 规控算法

- `GridAStar`：占用栅格上的四邻域 A*
- `PIDController`：一维 PID，示例中用于把质点跟踪到目标位置

后续可换成轨迹优化、MPC、纯跟踪等，只要仍实现 `fit` / `predict`。

#### 自然语言处理

- `CountVectorizer`：空格分词词袋（`Transform`）
- 示例中与 `LogisticRegression` 组成文本分类流水线

#### 推荐算法

- `MatrixFactorization`：用户/物品隐向量 + 偏置，输入 `[user_id, item_id]` 预测评分

#### SLAM 算法

- `ICP2D`：最近邻匹配 + SVD 刚体对齐，示例用已知旋转平移的点云验证 RMSE

#### 语音算法

- `WaveformNormalize` / `FrameFeature`：去直流，提取能量、过零率、标准差
- 示例：有声 / 无声二分类

#### 多模态算法

- `ConcatFusion`：图像特征与文本特征在特征维拼接（早期融合）
- 示例：融合后再走逻辑回归

### 统一流水线

```python
from algorithm.common.pipeline import Pipeline
from algorithm.ml import LinearRegression, StandardScaler

pipe = Pipeline(steps=[StandardScaler()], estimator=LinearRegression())
pipe.fit(x_train, y_train)
y_hat = pipe.predict(x_test)
```

NLP / 语音 / 多模态同样把预处理做成 `Transform`，估计器复用 `ml` 或 `vision` 中的分类器。

### 设计约定

- 核心实现以 NumPy 为主，依赖少
- 训练与推理走同一套 `fit` / `predict`
- 大模型只定义接口，不绑定具体云厂商 SDK

