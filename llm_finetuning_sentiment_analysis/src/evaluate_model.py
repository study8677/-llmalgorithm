import datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import torch
import os

def compute_metrics_eval(predictions, labels):
    """Computes and prints accuracy, precision, recall, and F1-score."""
    # Ensure predictions are in the correct format (0 or 1)
    # The pipeline might return 'LABEL_0' or 'LABEL_1'
    # This was handled in the prediction loop, ensuring `preds_int` are 0 or 1.
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', pos_label=1)
    acc = accuracy_score(labels, predictions)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def evaluate_model_performance():
    # Define paths
    test_dataset_path = 'llm_finetuning_sentiment_analysis/data/test_dataset'
    model_path = 'llm_finetuning_sentiment_analysis/models/fine_tuned_sentiment_model'

    # Check if model and data exist
    if not os.path.exists(model_path):
        print(f"Error: Fine-tuned model not found at {model_path}")
        print("Please run the fine-tuning script first.")
        return
    if not os.path.exists(test_dataset_path):
        print(f"Error: Test dataset not found at {test_dataset_path}")
        print("Please run the data preparation script first.")
        return

    # Load the processed test dataset
    print(f"Loading test dataset from {test_dataset_path}...")
    try:
        test_dataset = datasets.load_from_disk(test_dataset_path)
    except Exception as e:
        print(f"Failed to load test dataset: {e}")
        return
    print("Test dataset loaded.")

    # Load the fine-tuned model and tokenizer
    print(f"Loading model and tokenizer from {model_path}...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Failed to load model or tokenizer: {e}")
        return
    print("Model and tokenizer loaded.")

    # Create a sentiment analysis pipeline
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'cuda:0' if device == 0 else 'cpu'}")
    nlp_pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device=device)

    # Extract true labels and texts
    # The test_dataset is already tokenized, but the pipeline needs raw text.
    # We need to reload the original text for evaluation or pass tokenized inputs.
    # For simplicity with pipeline, let's assume we need to pass raw text.
    # However, the current `test_dataset` saved by `prepare_data.py` does not contain raw 'text'.
    # This is a mismatch. For now, let's try to load the original IMDB test split for its text.
    # A better long-term solution would be to save 'text' in the processed dataset if needed by evaluation.
    
    print("Loading original IMDB dataset to get raw text for evaluation...")
    try:
        # This is a workaround. Ideally, prepare_data.py should save text or evaluation should use tokenized inputs.
        original_imdb_dataset = datasets.load_dataset('imdb', split='test')
        # The test set was split 50/50 for validation and test. We need the text from the actual test portion.
        # This requires knowing the indices or having a unique ID.
        # Given the current structure, this is complicated.
        # Let's assume for now that the 'test_dataset' saved *does* have a 'text' column or
        # that the pipeline can somehow work with the tokenized inputs if the model is correctly configured.

        # The `pipeline` expects raw text. The `test_dataset` contains `input_ids`, `attention_mask`, `label`.
        # Let's try decoding `input_ids` back to text for the pipeline.
        # This is inefficient but works around the missing raw text.
        
        if 'text' not in test_dataset.column_names:
            print("Raw 'text' column not found in the processed test dataset. Attempting to decode from input_ids.")
            texts_for_pipeline = [tokenizer.decode(ids, skip_special_tokens=True) for ids in test_dataset['input_ids']]
            if not texts_for_pipeline:
                 print("Could not decode texts from input_ids. Aborting evaluation.")
                 return
        else:
            texts_for_pipeline = test_dataset['text']

    except Exception as e:
        print(f"Could not load or process texts for pipeline: {e}")
        return

    true_labels = test_dataset['label'].tolist()
    print(f"Extracted {len(true_labels)} true labels.")

    # Get predictions
    print("Getting predictions from the pipeline...")
    try:
        # Batching for potentially large datasets
        predictions_output = []
        for i in range(0, len(texts_for_pipeline), 32): # Process in batches of 32
             batch_texts = texts_for_pipeline[i:i+32]
             predictions_output.extend(nlp_pipeline(batch_texts))
        print(f"Received {len(predictions_output)} predictions from pipeline.")
    except Exception as e:
        print(f"Error during pipeline prediction: {e}")
        return

    # Convert string labels (e.g., 'LABEL_1') to integer format (0 or 1)
    # The fine-tuning script maps positive to 1 and negative to 0.
    # Default pipeline behavior: LABEL_0 (negative), LABEL_1 (positive)
    # If model config has id2label: use that. Otherwise, assume LABEL_1 is positive.
    # Let's check model's config for label mapping
    id2label = model.config.id2label
    positive_label_str = 'LABEL_1' # Default assumption
    if id2label:
        for key, val in id2label.items():
            if val.upper() == 'POSITIVE' or key == 1 : # if label for id 1 is 'POSITIVE' or if id 1 exists
                positive_label_str = id2label.get(1, 'LABEL_1') # Get the string for label 1
                break
    
    print(f"Interpreting '{positive_label_str}' as positive sentiment (1).")

    preds_int = []
    for p in predictions_output:
        if p['label'] == positive_label_str:
            preds_int.append(1)
        else:
            preds_int.append(0)

    if len(preds_int) != len(true_labels):
        print(f"Mismatch in number of predictions ({len(preds_int)}) and true labels ({len(true_labels)}). Aborting.")
        return
        
    # Compute and print metrics
    print("Computing metrics...")
    compute_metrics_eval(preds_int, true_labels)

if __name__ == '__main__':
    evaluate_model_performance()
