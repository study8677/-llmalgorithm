# FLHF 通过大型模型 API 实现个性化内容生成

## 1. 概述
本项目探索通过联邦学习与人类反馈 (FLHF) 实现个性化内容生成，模拟客户端通过 API 与一个强大的中央大型语言模型 (LLM)（例如类 GPT-3/4 模型）交互以生成内容的场景。FLHF 机制用于在客户端训练或微调一个*较小的辅助模型*或*提示策略*。这些辅助组件随后在服务器端聚合（如果以个性化为核心，则可保留在本地），以帮助个性化、引导或调整主 LLM 的输出，使其适应特定用户需求或上下文，而无需直接微调 LLM 本身。

## 2. 项目结构
项目组织结构如下：

```
flhf_content_generation/
├── data/               # 数据集占位目录 (例如：提示、反馈数据)
├── notebooks/          # Jupyter Notebooks 用于实验和概念验证
│   └── poc_flhf_summarization.ipynb # 演示 FLHF 流程
├── src/                # FLHF 框架的源代码
│   ├── federated_learning/ # 核心联邦学习组件
│   │   ├── __init__.py
│   │   ├── model.py        # 定义辅助模型/提示策略 (例如：AuxiliaryPromptStrategyModel)
│   │   ├── client.py       # 定义客户端逻辑：使用辅助模型、查询 LLM API、训练辅助模型
│   │   └── server.py       # 定义服务器逻辑：聚合辅助模型/策略的更新
│   ├── feedback/         # 人类反馈模拟组件
│   │   ├── __init__.py
│   │   └── feedback_simulator.py # 模拟人类反馈 (评分、偏好)
│   ├── __init__.py       # 使 'src' 成为一个包
│   ├── data_utils.py     # 用于辅助模型训练的数据加载和预处理工具
│   ├── llm_api_simulator.py # 模拟来自强大的 LLM API 的响应
│   └── flhf_process.py   # 协调 FLHF 模拟与 LLM API 交互的主脚本
├── tests/              # 各组件的单元测试
│   ├── __init__.py
│   ├── test_model.py
│   ├── test_client.py
│   ├── test_server.py
│   ├── test_feedback_simulator.py
│   └── test_data_utils.py
├── README.md           # 英文版 README 文件
├── README_zh.md        # 本文件 (中文版 README)
└── requirements.txt    # Python 依赖项 (例如：torch)
```

## 3. 核心组件

*   **`src/federated_learning/model.py`**: 定义 `AuxiliaryPromptStrategyModel` (示例名称)，代表客户端的较小模型或提示工程策略。该组件通过 FLHF 过程进行更新。
*   **`src/federated_learning/client.py`**: 定义 `Client` 类。客户端使用其本地辅助模型/策略来制定有效的提示，查询中央（模拟的）LLM API 以生成内容，接收关于生成内容的人类反馈，然后基于此反馈训练/更新其辅助模型/策略。
*   **`src/federated_learning/server.py`**: 定义 `Server` 类，现在负责聚合来自客户端的分布式辅助模型或提示策略的更新。
*   **`src/feedback/feedback_simulator.py`**: 定义 `FeedbackSimulator` 来模拟人类对 LLM 生成内容的反馈（评分、偏好）。
*   **`src/data_utils.py`**: 提供数据处理的工具函数，主要用于加载与训练辅助模型/提示策略相关的数据（例如，提示、反馈数据、上下文）。
*   **`src/llm_api_simulator.py`**: 模拟一个强大的通用 LLM API 的行为和响应。这允许在没有实际 API 成本或依赖的情况下进行开发和测试。
*   **`src/flhf_process.py`**: 包含 `run_flhf_simulation`，这是协调整个 FLHF 过程的主脚本。现在包括客户端制定提示、查询模拟的 LLM API、接收反馈、更新其辅助模型/策略以及服务器端聚合。

## 4. 设置与安装

1.  **先决条件**:
    *   Python 3.x (例如：Python 3.7+)

2.  **克隆仓库 (如果适用)**:
    ```bash
    # git clone <repository_url>
    # cd flhf_content_generation
    ```

3.  **安装依赖**:
    创建虚拟环境 (推荐):
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows 系统: venv\Scripts\activate
    ```
    安装所需的包:
    ```bash
    pip install -r requirements.txt
    ```
    目前, `requirements.txt` 内容极少，可能主要包含 `torch`。

## 5. 运行概念验证 (PoC)

概念验证 (PoC) 使用虚拟数据、模拟的 LLM API 和占位符辅助模型逻辑来演示基于 API 的 FLHF 流程。

1.  导航到 `notebooks` 目录:
    ```bash
    cd flhf_content_generation/notebooks
    ```
2.  启动 Jupyter Notebook 或 JupyterLab:
    ```bash
    jupyter notebook poc_flhf_summarization.ipynb
    # 或者
    # jupyter lab poc_flhf_summarization.ipynb
    ```
3.  打开 `poc_flhf_summarization.ipynb` 并按顺序运行所有单元格。该 Notebook 会处理必要的导入并使用 `run_flhf_simulation` 函数（已针对新的基于 API 的流程进行调整）。

    *注意*: Notebook 中的导入路径配置假定其从 `notebooks` 目录运行，或者项目根目录已正确添加到 `sys.path`。

## 6. 运行测试

项目提供了单元测试来验证各个组件的功能。

1.  导航到项目的根目录 (`flhf_content_generation`):
    ```bash
    cd path/to/flhf_content_generation
    ```
2.  使用 Python 的 `unittest` 模块运行测试:
    ```bash
    python -m unittest discover tests
    ```
    此命令将自动发现并运行 `tests` 目录中所有名为 `test_*.py` 的测试文件。（测试将需要更新以反映组件交互的变化）。

## 7. 未来工作 / 当前状态

本项目已重新聚焦于模拟与强大的 LLM 通过 API 交互的 FLHF 过程，训练辅助模型/提示策略。此模拟的核心架构已基本就绪。

未来发展的关键领域包括:
*   **实现复杂的辅助模型**: 开发更高级的辅助模型（例如，用于提示参数化的小型神经网络、基于规则的系统或可学习的提示嵌入）。
*   **开发多样化的提示工程策略**: 探索并实现各种可通过 FLHF 优化的可学习提示工程策略。
*   **优化 LLM API 模拟器**: 增强 `llm_api_simulator.py` 以实现更真实的 LLM 行为，包括模拟不同的响应风格、延迟和潜在的 API 错误。
*   **实现客户端辅助模型训练**: 在 `Client` 内部开发实际的训练循环，以根据反馈和 LLM 响应更新辅助模型/策略。
*   **高级反馈集成**: 探索更细致的方式将人类反馈整合到辅助模型/提示策略的训练中。
*   **真实数据集集成**: 使用真实的提示数据集、用户偏好和上下文信息来驱动模拟。
*   **与实际 LLM API 集成 (可选/实验性)**: 如果可行且资源允许，探索与实际 LLM API 集成以进行验证。
*   **评估指标**: 定义并实现指标，以评估辅助模型/提示策略在改善 LLM 输出个性化和质量方面的有效性。
*   **配置管理**: 引入更强大的配置系统。
*   **日志记录和实验跟踪**: 集成全面的日志记录和实验跟踪。
```
