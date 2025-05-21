import datasets
from transformers import BertTokenizerFast
import os

def preprocess_data():
    # Load the IMDB dataset
    dataset = datasets.load_dataset('imdb')

    # Initialize tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Define tokenization function
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    # Apply tokenization
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Remove 'unsupervised' split if it exists
    if 'unsupervised' in tokenized_datasets:
        tokenized_datasets.pop('unsupervised')

    # Split the 'test' set into 'validation' and 'test'
    if 'test' in tokenized_datasets:
        test_valid_split = tokenized_datasets['test'].train_test_split(test_size=0.5, seed=42)
        tokenized_datasets['test'] = test_valid_split['test']
        tokenized_datasets['validation'] = test_valid_split['train'] # train_test_split names the first part 'train'

    # Select relevant columns and set format to 'torch'
    for split in tokenized_datasets.keys():
        tokenized_datasets[split] = tokenized_datasets[split].select_columns(['input_ids', 'attention_mask', 'label'])
        tokenized_datasets[split].set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # Define save paths
    base_save_path = 'llm_finetuning_sentiment_analysis/data'
    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path) # Should have been created already, but good to have

    train_path = os.path.join(base_save_path, 'train_dataset')
    validation_path = os.path.join(base_save_path, 'validation_dataset')
    test_path = os.path.join(base_save_path, 'test_dataset')

    # Save the processed datasets
    if 'train' in tokenized_datasets:
        tokenized_datasets['train'].save_to_disk(train_path)
        print(f"Processed train dataset saved to {train_path}")
    if 'validation' in tokenized_datasets:
        tokenized_datasets['validation'].save_to_disk(validation_path)
        print(f"Processed validation dataset saved to {validation_path}")
    if 'test' in tokenized_datasets:
        tokenized_datasets['test'].save_to_disk(test_path)
        print(f"Processed test dataset saved to {test_path}")

if __name__ == '__main__':
    preprocess_data()
