import datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, EvalPrediction
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os

def compute_metrics(eval_pred: EvalPrediction):
    """Computes accuracy, precision, recall, and F1-score."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def fine_tune():
    # Define paths
    data_base_path = 'llm_finetuning_sentiment_analysis/data'
    train_dataset_path = os.path.join(data_base_path, 'train_dataset')
    validation_dataset_path = os.path.join(data_base_path, 'validation_dataset')
    
    output_dir_results = './llm_finetuning_sentiment_analysis/models/results'
    logging_dir_logs = './llm_finetuning_sentiment_analysis/models/logs'
    save_model_path = './llm_finetuning_sentiment_analysis/models/fine_tuned_sentiment_model'

    # Create directories if they don't exist
    if not os.path.exists(output_dir_results):
        os.makedirs(output_dir_results)
    if not os.path.exists(logging_dir_logs):
        os.makedirs(logging_dir_logs)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    # Load datasets
    if not os.path.exists(train_dataset_path) or not os.path.exists(validation_dataset_path):
        print(f"Error: Processed data not found at {train_dataset_path} or {validation_dataset_path}")
        print("Please run the data preparation script first.")
        return

    train_dataset = datasets.load_from_disk(train_dataset_path)
    validation_dataset = datasets.load_from_disk(validation_dataset_path)

    # Load pre-trained model and tokenizer
    model_name = 'distilbert-base-uncased'
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir_results,
        num_train_epochs=1,  # For demo purposes
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=logging_dir_logs,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1", # Optional: specify metric to monitor for best model
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer, # Pass tokenizer to handle padding correctly if not already done
    )

    # Start fine-tuning
    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning finished.")

    # Save the best model and tokenizer
    print(f"Saving model to {save_model_path}")
    trainer.save_model(save_model_path)
    tokenizer.save_pretrained(save_model_path)
    print("Model and tokenizer saved.")

if __name__ == '__main__':
    fine_tune()
