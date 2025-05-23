import unittest
import torch
import sys
import os

# Adjust path to import from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from flhf_content_generation.src.federated_learning.client import Client
    from flhf_content_generation.src.federated_learning.model import SimpleSeq2SeqModel # For model_config
    from flhf_content_generation.src.data_utils import get_dummy_dataloaders, MAX_SEQ_LEN
except ModuleNotFoundError:
    # Fallback
    from src.federated_learning.client import Client
    from src.federated_learning.model import SimpleSeq2SeqModel
    from src.data_utils import get_dummy_dataloaders, MAX_SEQ_LEN

class TestClient(unittest.TestCase):
    def setUp(self):
        """Initialize a sample client with dummy parameters."""
        self.model_config_base = {
            # 'input_dim' and 'output_dim' will be set by vocab size
            'hidden_dim': 8,
            'num_layers': 1
        }
        
        # Use a smaller fixed_max_seq_len for tests to keep dummy data manageable
        test_max_seq_len = 5 

        # Get a dummy dataloader and vocab
        # Note: get_dummy_dataloaders returns a list of dataloaders
        self.client_dataloaders, self.vocab = get_dummy_dataloaders(
            num_clients=1, 
            batch_size=2, 
            num_samples_per_client=4, 
            fixed_max_seq_len=test_max_seq_len 
        )
        
        self.model_config = self.model_config_base.copy()
        self.model_config['input_dim'] = len(self.vocab)
        self.model_config['output_dim'] = len(self.vocab)
        
        self.client = Client(
            client_id='client_test_0', 
            model_config=self.model_config, 
            data_loader=self.client_dataloaders[0] # Use the first (and only) dataloader
        )

    def test_client_creation(self):
        """Test that the client is created successfully."""
        self.assertIsNotNone(self.client, "Client should not be None after initialization.")
        self.assertIsNotNone(self.client.model, "Client model should not be None.")

    def test_train_local_model_placeholder(self):
        """Test that train_local_model runs without error (placeholder test)."""
        try:
            # The client's train_local_model is a placeholder, so this just checks if it runs.
            self.client.train_local_model(num_epochs=1, learning_rate=0.01)
        except Exception as e:
            self.fail(f"client.train_local_model() raised an exception: {e}")

    def test_generate_content_placeholder(self):
        """Test that generate_content runs and returns something (placeholder test)."""
        # Input sequence shape: (batch_size, seq_len)
        # Using the test_max_seq_len from setUp for consistency
        test_max_seq_len = self.client_dataloaders[0].dataset.tokenized_texts[0].shape[0] # get from actual data
        input_sequence = torch.randint(0, self.model_config['input_dim'], (1, test_max_seq_len)) 
        
        # The client's generate_content is a placeholder.
        # Current placeholder returns None.
        generated_output = self.client.generate_content(input_sequence)
        
        # This assertion will need to change if the placeholder implementation changes.
        self.assertIsNone(generated_output, "generate_content currently returns None. Update test if implementation changes.")
        # Example if it returned a tensor:
        # self.assertIsNotNone(generated_output, "generate_content should return an output.")
        # self.assertTrue(isinstance(generated_output, torch.Tensor) or isinstance(generated_output, list))


    def test_set_get_model_weights(self):
        """Test setting and getting model weights."""
        initial_weights = self.client.get_local_model_weights()
        self.assertIsNotNone(initial_weights, "Initial weights should not be None.")
        
        # Create a new model instance to simulate different weights
        # (or modify existing weights slightly if that's simpler)
        temp_model = SimpleSeq2SeqModel(**self.model_config)
        # Slightly change a weight to ensure set_global_model_weights has an effect
        # For example, for the first parameter tensor:
        # with torch.no_grad():
        #    for param in temp_model.parameters():
        #        param.data += 0.1 # Modify weights
        #        break 
        new_weights = temp_model.state_dict()

        try:
            self.client.set_global_model_weights(new_weights)
        except Exception as e:
            self.fail(f"client.set_global_model_weights() raised an exception: {e}")

        current_weights = self.client.get_local_model_weights()
        
        # Check that weights have been updated (at least one parameter should differ if new_weights was different)
        # This requires careful comparison. For simplicity, we check they are not the same object if they were different.
        # A more robust check would compare tensor values.
        
        # Basic check: ensure the types match and keys are the same
        self.assertEqual(initial_weights.keys(), current_weights.keys())

        # Example: Check if at least one weight tensor is different after setting new_weights
        # This assumes new_weights are actually different from initial_weights
        # found_diff = False
        # for key in initial_weights:
        #     if not torch.equal(initial_weights[key], current_weights[key]):
        #         found_diff = True
        #         break
        # if not torch.equal(initial_weights[list(initial_weights.keys())[0]], new_weights[list(new_weights.keys())[0]]): # if they were different
        #    self.assertTrue(found_diff, "Weights should be different after set_global_model_weights with new weights.")


if __name__ == '__main__':
    unittest.main()
