# LLM Fine-tuning for Sentiment Analysis

## Overview
This project demonstrates how to fine-tune a pre-trained language model (DistilBERT) for sentiment analysis using the IMDB movie review dataset. The goal is to classify reviews as either positive or negative.

## Project Structure

```
llm_finetuning_sentiment_analysis/
├── data/                     # Stores processed datasets (train, validation, test)
├── models/                   # Stores fine-tuned model, logs, and results
│   ├── fine_tuned_sentiment_model/ # Saved fine-tuned model and tokenizer
│   ├── logs/                   # Training logs (e.g., TensorBoard)
│   └── results/                # Training results and checkpoints
├── src/                      # Source code for the project
│   ├── prepare_data.py       # Script to download, preprocess, and tokenize data
│   ├── fine_tune_model.py    # Script to fine-tune the language model
│   └── evaluate_model.py     # Script to evaluate the fine-tuned model
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Setup

1.  **Clone the Repository** (Placeholder)
    ```bash
    # git clone <repository-url>
    # cd llm_finetuning_sentiment_analysis
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    python -m venv venv
    # On Windows
    # venv\Scripts\activate
    # On macOS/Linux
    # source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project

Follow these steps in order to prepare the data, fine-tune the model, and evaluate its performance.

### Step 1: Prepare Data

This script downloads the IMDB dataset, preprocesses the text (tokenization, padding, truncation), and splits it into training, validation, and test sets. The processed datasets are then saved to the `llm_finetuning_sentiment_analysis/data/` directory.

```bash
python src/prepare_data.py
```

### Step 2: Fine-tune Model

This script loads the processed data from the `data/` directory and fine-tunes a pre-trained DistilBERT model. The best performing model (based on F1-score on the validation set) is saved to `llm_finetuning_sentiment_analysis/models/fine_tuned_sentiment_model/`. Training logs and intermediate results (checkpoints) are saved in `llm_finetuning_sentiment_analysis/models/logs/` and `llm_finetuning_sentiment_analysis/models/results/` respectively.

```bash
python src/fine_tune_model.py
```
*Note: Fine-tuning can take a significant amount of time and computational resources depending on your hardware.*

### Step 3: Evaluate Model

This script loads the fine-tuned model from `llm_finetuning_sentiment_analysis/models/fine_tuned_sentiment_model/` and the processed test set from `llm_finetuning_sentiment_analysis/data/`. It then evaluates the model's performance on the test set and prints metrics such as accuracy, precision, recall, and F1-score.

```bash
python src/evaluate_model.py
```

## Example Usage (Inference)

Once the model is fine-tuned and saved, you can use it for sentiment analysis on new sentences. Here's a Python code snippet demonstrating how:

```python
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch # Optional: for checking device

# Load the fine-tuned model and tokenizer
model_path = "./llm_finetuning_sentiment_analysis/models/fine_tuned_sentiment_model" # Adjust path if running from a different directory
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Determine device (GPU if available, otherwise CPU)
device = 0 if torch.cuda.is_available() else -1 

# Create a sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)

# Example sentences
positive_text = "This movie was absolutely fantastic! Highly recommended."
negative_text = "I really disliked this film. It was boring and too long."
another_example = "The acting was incredible and the storyline kept me engaged."

# Get predictions
print(f"'{positive_text}' -> {sentiment_analyzer(positive_text)}")
print(f"'{negative_text}' -> {sentiment_analyzer(negative_text)}")
print(f"'{another_example}' -> {sentiment_analyzer(another_example)}")

# Example with multiple sentences:
# results = sentiment_analyzer([positive_text, negative_text, another_example])
# for text, result in zip([positive_text, negative_text, another_example], results):
# print(f"'{text}' -> {result}")
```

## Dependencies

This project relies on the following key Python libraries:

*   **transformers**: For accessing pre-trained models and the training infrastructure.
*   **datasets**: For easily downloading and processing datasets like IMDB.
*   **torch**: The deep learning framework used by transformers.
*   **scikit-learn**: For calculating evaluation metrics.
*   **tensorboard**: For logging and visualizing training progress (optional).

Please refer to `requirements.txt` for a full list of dependencies.
