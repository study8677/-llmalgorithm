import unittest
import torch
from collections import OrderedDict
import sys
import os

# Adjust path to import from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from flhf_content_generation.src.federated_learning.server import Server
    from flhf_content_generation.src.federated_learning.model import SimpleSeq2SeqModel
except ModuleNotFoundError:
    # Fallback
    from src.federated_learning.server import Server
    from src.federated_learning.model import SimpleSeq2SeqModel


class TestServer(unittest.TestCase):
    def setUp(self):
        """Initialize a sample server with dummy parameters."""
        self.model_config = {
            'input_dim': 10,
            'output_dim': 10,
            'hidden_dim': 8,
            'num_layers': 1
        }
        self.server = Server(model_config=self.model_config)

    def test_server_creation(self):
        """Test that the server is created successfully."""
        self.assertIsNotNone(self.server, "Server should not be None after initialization.")
        self.assertIsNotNone(self.server.global_model, "Server's global model should not be None.")

    def test_get_global_model_weights(self):
        """Test that get_global_model_weights returns a state_dict."""
        weights = self.server.get_global_model_weights()
        self.assertIsNotNone(weights, "Global model weights should not be None.")
        self.assertIsInstance(weights, OrderedDict, "Weights should be an OrderedDict (state_dict).")

    def test_aggregate_model_updates_placeholder(self):
        """Test aggregate_model_updates (placeholder: runs without error and updates weights)."""
        # Get initial weights
        initial_weights = self.server.get_global_model_weights()
        # Create a deepcopy for comparison later if needed, or just compare one parameter
        # initial_param_value = list(initial_weights.values())[0].clone()


        # Create dummy client model weights
        client_model_1 = SimpleSeq2SeqModel(**self.model_config)
        client_model_2 = SimpleSeq2SeqModel(**self.model_config)

        # Optionally, modify weights slightly to ensure aggregation changes them
        with torch.no_grad():
            for param in client_model_1.parameters():
                param.data += 0.1
            for param in client_model_2.parameters():
                param.data += 0.2
        
        dummy_weights_list = [client_model_1.state_dict(), client_model_2.state_dict()]

        try:
            self.server.aggregate_model_updates(dummy_weights_list)
        except Exception as e:
            self.fail(f"server.aggregate_model_updates() raised an exception: {e}")

        updated_weights = self.server.get_global_model_weights()
        self.assertIsNotNone(updated_weights, "Updated global model weights should not be None.")

        # Check if weights have changed.
        # This is a basic check. A more rigorous test would verify the FedAvg logic.
        # For example, check if a specific parameter has a value consistent with averaging.
        # initial_first_param = list(initial_weights.values())[0]
        updated_first_param = list(updated_weights.values())[0]
        client1_first_param = list(dummy_weights_list[0].values())[0]
        client2_first_param = list(dummy_weights_list[1].values())[0]
        
        # Expected average for the first parameter of client_model_1 and client_model_2
        # (assuming initial server weights were 0, which is not true.
        # The aggregation averages the provided client_weights_list and loads it.
        # So, the new global model weights should be the average of client_model_1 and client_model_2)
        expected_param_value = (client1_first_param + client2_first_param) / 2.0
        
        self.assertTrue(torch.allclose(updated_first_param, expected_param_value),
                        f"Aggregated weights for the first parameter do not match the expected average. "
                        f"Expected: {expected_param_value}, Got: {updated_first_param}")


    def test_aggregate_model_updates_empty_list(self):
        """Test aggregate_model_updates with an empty list of weights."""
        initial_weights = self.server.get_global_model_weights()
        # Create a copy to ensure no change
        initial_weights_copy = {k: v.clone() for k, v in initial_weights.items()}

        try:
            self.server.aggregate_model_updates([]) # Pass an empty list
        except Exception as e:
            self.fail(f"server.aggregate_model_updates([]) raised an exception: {e}")

        updated_weights = self.server.get_global_model_weights()
        
        # Ensure weights did not change
        for key in initial_weights_copy:
            self.assertTrue(torch.equal(initial_weights_copy[key], updated_weights[key]),
                            f"Weights for key '{key}' should not change when an empty list is provided.")

if __name__ == '__main__':
    unittest.main()
