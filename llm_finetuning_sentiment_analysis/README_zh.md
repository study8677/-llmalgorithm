# 大语言模型情感分析微调项目

## 项目概览
本项目演示了如何使用IMDB电影评论数据集微调预训练的语言模型（DistilBERT）以进行情感分析。目标是将评论分为正面或负面。

## 项目结构

```
llm_finetuning_sentiment_analysis/
├── data/                     # 存储处理后的数据集 (训练集、验证集、测试集)
├── models/                   # 存储微调后的模型、日志和结果
│   ├── fine_tuned_sentiment_model/ # 保存的微调模型和分词器
│   ├── logs/                   # 训练日志 (例如 TensorBoard)
│   └── results/                # 训练结果和检查点
├── src/                      # 项目源代码
│   ├── prepare_data.py       # 下载、预处理和标记数据的脚本
│   ├── fine_tune_model.py    # 微调语言模型的脚本
│   └── evaluate_model.py     # 评估微调模型的脚本
├── requirements.txt          # Python 依赖项
└── README.md                 # 英文版说明文件
└── README_zh.md              # 本文件 (中文版说明)
```

## 环境设置

1.  **克隆仓库** (占位符)
    ```bash
    # git clone <repository-url>
    # cd llm_finetuning_sentiment_analysis
    ```

2.  **创建并激活虚拟环境**
    ```bash
    python -m venv venv
    # Windows 系统
    # venv\Scripts\activate
    # macOS/Linux 系统
    # source venv/bin/activate
    ```

3.  **安装依赖项**
    ```bash
    pip install -r requirements.txt
    ```

## 运行项目

请按顺序执行以下步骤来准备数据、微调模型和评估其性能。

### 第一步：准备数据

此脚本下载IMDB数据集，对文本进行预处理（分词、填充、截断），并将其分为训练集、验证集和测试集。处理后的数据集将保存到 `llm_finetuning_sentiment_analysis/data/` 目录中。

```bash
python src/prepare_data.py
```

### 第二步：微调模型

此脚本从 `data/` 目录加载处理后的数据，并微调预训练的DistilBERT模型。在验证集上表现最佳的模型（基于F1分数）将保存到 `llm_finetuning_sentiment_analysis/models/fine_tuned_sentiment_model/`。训练日志和中间结果（检查点）分别保存在 `llm_finetuning_sentiment_analysis/models/logs/` 和 `llm_finetuning_sentiment_analysis/models/results/` 中。

```bash
python src/fine_tune_model.py
```
*注意：微调过程可能需要大量时间和计算资源，具体取决于您的硬件。*

### 第三步：评估模型

此脚本从 `llm_finetuning_sentiment_analysis/models/fine_tuned_sentiment_model/` 加载微调后的模型，并从 `llm_finetuning_sentiment_analysis/data/` 加载处理后的测试集。然后，它评估模型在测试集上的性能，并打印准确率、精确率、召回率和F1分数等指标。

```bash
python src/evaluate_model.py
```

## 使用示例 (推理)

模型微调并保存后，您可以将其用于对新句子进行情感分析。以下是一个演示如何操作的Python代码片段：

```python
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch # 可选：用于检查设备

# 加载微调后的模型和分词器
model_path = "./llm_finetuning_sentiment_analysis/models/fine_tuned_sentiment_model" # 如果从不同目录运行，请调整路径
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 确定设备 (如果可用则为GPU，否则为CPU)
device = 0 if torch.cuda.is_available() else -1 

# 创建情感分析管道
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)

# 示例文本
positive_text = "这部电影绝对精彩！强烈推荐。"
negative_text = "我真的很不喜欢这部电影。它既无聊又冗长。"
another_example = "演技令人难以置信，故事情节也让我全神贯注。"

# 获取预测结果
print(f"'{positive_text}' -> {sentiment_analyzer(positive_text)}")
print(f"'{negative_text}' -> {sentiment_analyzer(negative_text)}")
print(f"'{another_example}' -> {sentiment_analyzer(another_example)}")

# 多句子示例:
# results = sentiment_analyzer([positive_text, negative_text, another_example])
# for text, result in zip([positive_text, negative_text, another_example], results):
# print(f"'{text}' -> {result}")
```

## 依赖项

本项目依赖于以下关键Python库：

*   **transformers**: 用于访问预训练模型和训练基础设施。
*   **datasets**: 用于轻松下载和处理像IMDB这样的数据集。
*   **torch**: transformers 使用的深度学习框架。
*   **scikit-learn**: 用于计算评估指标。
*   **tensorboard**: 用于记录和可视化训练进度 (可选)。

请参阅 `requirements.txt` 获取完整的依赖项列表。
