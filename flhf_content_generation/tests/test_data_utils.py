import unittest
import torch
from torch.utils.data import DataLoader as TorchDataLoader # Alias to avoid naming conflict
import sys
import os

# Adjust path to import from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from flhf_content_generation.src.data_utils import get_dummy_dataloaders, TextDataset, default_tokenizer_placeholder, MAX_SEQ_LEN, PAD_TOKEN_ID
except ModuleNotFoundError:
    # Fallback
    from src.data_utils import get_dummy_dataloaders, TextDataset, default_tokenizer_placeholder, MAX_SEQ_LEN, PAD_TOKEN_ID


class TestDataUtils(unittest.TestCase):
    def test_get_dummy_dataloaders(self):
        """Test the get_dummy_dataloaders function."""
        num_clients = 2
        batch_size = 4
        num_samples_per_client = 10
        fixed_max_seq_len = 12

        dataloaders, vocab = get_dummy_dataloaders(
            num_clients=num_clients,
            batch_size=batch_size,
            num_samples_per_client=num_samples_per_client,
            fixed_max_seq_len=fixed_max_seq_len
        )

        self.assertEqual(len(dataloaders), num_clients, f"Should create {num_clients} dataloaders.")
        self.assertIsInstance(dataloaders[0], TorchDataLoader, "Each item in dataloaders list should be a PyTorch DataLoader.")
        self.assertTrue(len(vocab) > 0, "Vocabulary should not be empty.")
        self.assertIn('<pad>', vocab, "Vocabulary should contain <pad> token.")
        self.assertIn('<unk>', vocab, "Vocabulary should contain <unk> token.")


        # Check if a sample batch can be drawn and has the correct structure
        try:
            sample_texts_batch, sample_summaries_batch = next(iter(dataloaders[0]))
        except StopIteration:
            self.fail("Dataloader is empty, cannot draw a sample batch.")
        
        self.assertIsNotNone(sample_texts_batch, "Texts batch should not be None.")
        self.assertIsNotNone(sample_summaries_batch, "Summaries batch should not be None.")
        
        self.assertEqual(sample_texts_batch.shape[0], batch_size, f"Texts batch size should be {batch_size}.")
        self.assertEqual(sample_texts_batch.shape[1], fixed_max_seq_len, f"Texts sequence length should be {fixed_max_seq_len}.")
        
        self.assertEqual(sample_summaries_batch.shape[0], batch_size, f"Summaries batch size should be {batch_size}.")
        self.assertEqual(sample_summaries_batch.shape[1], fixed_max_seq_len, f"Summaries sequence length should be {fixed_max_seq_len}.")

        self.assertEqual(sample_texts_batch.dtype, torch.long, "Texts tensor should be of type torch.long.")
        self.assertEqual(sample_summaries_batch.dtype, torch.long, "Summaries tensor should be of type torch.long.")


    def test_default_tokenizer_placeholder(self):
        """Test the placeholder tokenizer function."""
        vocab = {'<pad>': 0, '<unk>': 1, 'hello': 2, 'world': 3, 'test': 4}
        max_len = 5
        pad_id = 0

        # Test basic tokenization and padding
        sentence1 = "hello world"
        expected1 = [2, 3, pad_id, pad_id, pad_id]
        self.assertEqual(default_tokenizer_placeholder(sentence1, vocab, max_len, pad_id), expected1)

        # Test truncation
        sentence2 = "hello world this is a test"
        expected2 = [2, 3, 1, 1, 1] # 'this', 'is', 'a' are <unk>
        self.assertEqual(default_tokenizer_placeholder(sentence2, vocab, max_len, pad_id), expected2)

        # Test unknown words
        sentence3 = "unknown words here"
        expected3 = [1, 1, 1, pad_id, pad_id] # all <unk>
        self.assertEqual(default_tokenizer_placeholder(sentence3, vocab, max_len, pad_id), expected3)

        # Test empty sentence
        sentence4 = ""
        expected4 = [pad_id, pad_id, pad_id, pad_id, pad_id]
        self.assertEqual(default_tokenizer_placeholder(sentence4, vocab, max_len, pad_id), expected4)


    def test_text_dataset(self):
        """Test the TextDataset class."""
        texts = ["hello world", "another example"]
        summaries = ["summary one", "summary two here"]
        vocab = {'<pad>': 0, '<unk>': 1, 'hello': 2, 'world': 3, 'another': 4, 'example': 5, 'summary':6, 'one':7, 'two':8, 'here':9 }
        
        # Use the actual MAX_SEQ_LEN from data_utils for this test, or a local one
        test_max_len = 7 
        def tokenizer(sentence, v):
            return default_tokenizer_placeholder(sentence, v, test_max_len, PAD_TOKEN_ID)

        dataset = TextDataset(texts, summaries, tokenizer, vocab)

        self.assertEqual(len(dataset), 2, "Dataset length should be 2.")

        # Check first item
        text1_tensor, summary1_tensor = dataset[0]
        expected_text1 = [vocab['hello'], vocab['world']] + [PAD_TOKEN_ID] * (test_max_len - 2)
        expected_summary1 = [vocab['summary'], vocab['one']] + [PAD_TOKEN_ID] * (test_max_len - 2)
        
        self.assertTrue(torch.equal(text1_tensor, torch.tensor(expected_text1, dtype=torch.long)))
        self.assertTrue(torch.equal(summary1_tensor, torch.tensor(expected_summary1, dtype=torch.long)))
        
        # Check second item
        text2_tensor, summary2_tensor = dataset[1]
        expected_text2 = [vocab['another'], vocab['example']] + [PAD_TOKEN_ID] * (test_max_len - 2)
        expected_summary2 = [vocab['summary'], vocab['two'], vocab['here']] + [PAD_TOKEN_ID] * (test_max_len - 3)

        self.assertTrue(torch.equal(text2_tensor, torch.tensor(expected_text2, dtype=torch.long)))
        self.assertTrue(torch.equal(summary2_tensor, torch.tensor(expected_summary2, dtype=torch.long)))


if __name__ == '__main__':
    unittest.main()
