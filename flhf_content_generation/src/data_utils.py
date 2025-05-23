import torch
from torch.utils.data import Dataset, DataLoader

# Define a fixed padding length for simplicity
MAX_SEQ_LEN = 20 # Max number of tokens in a sequence
PAD_TOKEN_ID = 0 # Assuming 0 is the padding token ID

class TextDataset(Dataset):
    """
    A PyTorch Dataset for text data, specifically for text-summary pairs.

    This dataset takes lists of texts and summaries, tokenizes them using a
    provided tokenizer function and vocabulary, and returns tensor pairs.

    Attributes:
        texts (list of str): The raw input texts.
        summaries (list of str): The raw target summaries.
        tokenizer_func (callable): The function used for tokenizing texts and summaries.
        vocab (dict): The vocabulary mapping tokens to integer IDs.
        tokenized_texts (list of list of int): Tokenized and padded input texts.
        tokenized_summaries (list of list of int): Tokenized and padded target summaries.
    """
    def __init__(self, texts, summaries, tokenizer_func, vocab):
        """
        Initializes the TextDataset.

        Args:
            texts (list of str): List of input texts.
            summaries (list of str): List of corresponding target summaries.
            tokenizer_func (callable): A function that takes a string, a vocabulary,
                                       max length, and pad token ID, and returns a
                                       list of token IDs (padded/truncated).
            vocab (dict): A vocabulary mapping words (tokens) to integer IDs.
        """
        self.texts = texts
        self.summaries = summaries
        self.tokenizer_func = tokenizer_func
        self.vocab = vocab

        # Pre-tokenize all texts and summaries
        self.tokenized_texts = [self.tokenizer_func(text, self.vocab) for text in self.texts]
        self.tokenized_summaries = [self.tokenizer_func(summary, self.vocab) for summary in self.summaries]

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retrieves the tokenized text and summary at the given index as tensors.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - text_tensor (torch.Tensor): The tokenized input text.
                - summary_tensor (torch.Tensor): The tokenized target summary.
        """
        text_tensor = torch.tensor(self.tokenized_texts[idx], dtype=torch.long)
        summary_tensor = torch.tensor(self.tokenized_summaries[idx], dtype=torch.long)
        return text_tensor, summary_tensor

def default_tokenizer_placeholder(sentence, vocab, max_len=MAX_SEQ_LEN, pad_token_id=PAD_TOKEN_ID):
    """
    A simple placeholder tokenizer function.

    This tokenizer converts a sentence into a list of token IDs by:
    1. Converting the sentence to lowercase.
    2. Splitting the sentence into words (tokens).
    3. Mapping words to their integer IDs using the provided `vocab`.
       Unknown words are mapped to an '<unk>' token ID.
    4. Padding the sequence with `pad_token_id` up to `max_len`.
    5. Truncating the sequence if it exceeds `max_len`.

    Args:
        sentence (str): The input sentence string.
        vocab (dict): A vocabulary mapping words to integer IDs.
                      It's expected to have '<unk>' and '<pad>' tokens.
        max_len (int, optional): The maximum length for the tokenized sequence.
                                 Defaults to `MAX_SEQ_LEN`.
        pad_token_id (int, optional): The ID used for padding.
                                      Defaults to `PAD_TOKEN_ID`.

    Returns:
        list of int: The list of token IDs representing the processed sentence.
    """
    tokens = sentence.lower().split()
    token_ids = [vocab.get(token, vocab.get('<unk>', 1)) for token in tokens] # Use <unk> for unknown words
    
    # Pad sequence
    padded_token_ids = token_ids[:max_len] + [pad_token_id] * (max_len - len(token_ids))
    
    # Truncate if longer
    if len(padded_token_ids) > max_len:
        padded_token_ids = padded_token_ids[:max_len]
        
    return padded_token_ids

def get_dummy_dataloaders(num_clients, batch_size, num_samples_per_client=100, fixed_max_seq_len=MAX_SEQ_LEN):
    """
    Generates dummy text data and creates PyTorch DataLoaders for simulated clients.

    This function creates a set of simple text and summary pairs, builds a basic
    vocabulary from them, and then distributes this data among the specified
    number of clients by creating a `TextDataset` and `DataLoader` for each.

    Args:
        num_clients (int): The number of client DataLoaders to generate.
        batch_size (int): The batch size for each DataLoader.
        num_samples_per_client (int, optional): The number of text-summary
                                                samples to generate for each client.
                                                Defaults to 100.
        fixed_max_seq_len (int, optional): The maximum sequence length to be used
                                           by the tokenizer. Defaults to `MAX_SEQ_LEN`.

    Returns:
        tuple: A tuple containing:
            - client_dataloaders (list of torch.utils.data.DataLoader): A list of
              DataLoader instances, one for each client.
            - vocab (dict): The vocabulary dictionary mapping tokens to integer IDs,
              generated from the dummy data.
    """
    all_texts = []
    all_summaries = []

    # Generate dummy data
    for i in range(num_clients * num_samples_per_client):
        all_texts.append(f"This is input sentence number {i+1}.")
        all_summaries.append(f"Summary for {i+1}.")

    # Create a vocabulary placeholder
    all_words = set()
    for sentence in all_texts + all_summaries:
        for word in sentence.lower().split():
            all_words.add(word)

    vocab = {'<pad>': PAD_TOKEN_ID, '<unk>': 1} # Add <pad> and <unk> tokens
    for i, word in enumerate(all_words):
        if word not in vocab: # Avoid reassigning <pad> or <unk>
            vocab[word] = len(vocab)


    # Define the tokenizer function to be used by TextDataset
    # It will use the vocab and fixed_max_seq_len defined here
    tokenizer_func = lambda sentence, v: default_tokenizer_placeholder(sentence, v, fixed_max_seq_len, PAD_TOKEN_ID)

    client_dataloaders = []
    for i in range(num_clients):
        start_idx = i * num_samples_per_client
        end_idx = (i + 1) * num_samples_per_client

        client_texts = all_texts[start_idx:end_idx]
        client_summaries = all_summaries[start_idx:end_idx]

        client_dataset = TextDataset(
            texts=client_texts,
            summaries=client_summaries,
            tokenizer_func=tokenizer_func, # Pass the tokenizer function
            vocab=vocab
        )

        client_dataloader = DataLoader(
            dataset=client_dataset,
            batch_size=batch_size,
            shuffle=True # Shuffle data for training
        )
        client_dataloaders.append(client_dataloader)

    return client_dataloaders, vocab # Also return vocab for potential use in model_config

if __name__ == '__main__':
    # Example usage:
    print("Testing data_utils.py...")
    num_dummy_clients = 2
    dummy_batch_size = 4
    dummy_samples_per_client = 10

    dataloaders, generated_vocab = get_dummy_dataloaders(
        num_clients=num_dummy_clients,
        batch_size=dummy_batch_size,
        num_samples_per_client=dummy_samples_per_client
    )

    print(f"Generated {len(dataloaders)} dataloaders.")
    print(f"Vocabulary size: {len(generated_vocab)}")
    # print(f"Vocabulary: {generated_vocab}")


    for i, dataloader in enumerate(dataloaders):
        print(f"\nClient {i} Dataloader:")
        for batch_idx, (texts_batch, summaries_batch) in enumerate(dataloader):
            print(f"  Batch {batch_idx + 1}:")
            print(f"    Texts shape: {texts_batch.shape}") # Expected: (batch_size, MAX_SEQ_LEN)
            print(f"    Summaries shape: {summaries_batch.shape}") # Expected: (batch_size, MAX_SEQ_LEN)
            # print(f"    Sample text tensor: {texts_batch[0]}")
            # print(f"    Sample summary tensor: {summaries_batch[0]}")
            if batch_idx >= 1: # Print only first two batches for brevity
                break
    
    print("\nSample of tokenization:")
    test_sentence = "This is a test sentence for the tokenizer."
    tokenized_test = default_tokenizer_placeholder(test_sentence, generated_vocab, MAX_SEQ_LEN, PAD_TOKEN_ID)
    print(f"Original: '{test_sentence}'")
    print(f"Tokenized & Padded: {tokenized_test}")

    print("\nChecking a few vocab entries:")
    for word in ["this", "is", "summary", "for", "nonexistentword", "<unk>", "<pad>"]:
        print(f"'{word}' -> {generated_vocab.get(word, generated_vocab.get('<unk>'))}")

    # Test edge case: empty sentence
    empty_sentence = ""
    tokenized_empty = default_tokenizer_placeholder(empty_sentence, generated_vocab, MAX_SEQ_LEN, PAD_TOKEN_ID)
    print(f"Original: '{empty_sentence}'")
    print(f"Tokenized & Padded: {tokenized_empty}")
    assert len(tokenized_empty) == MAX_SEQ_LEN

    # Test edge case: very long sentence
    long_sentence = " ".join([f"word{i}" for i in range(MAX_SEQ_LEN + 10)])
    tokenized_long = default_tokenizer_placeholder(long_sentence, generated_vocab, MAX_SEQ_LEN, PAD_TOKEN_ID)
    print(f"Original: '{long_sentence[:50]}...' (length: {len(long_sentence.split())})")
    print(f"Tokenized & Padded: {tokenized_long}")
    assert len(tokenized_long) == MAX_SEQ_LEN

    print("\ndata_utils.py test finished.")
