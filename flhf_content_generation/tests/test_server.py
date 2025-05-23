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
    from flhf_content_generation.src.federated_learning.model import AuxiliaryPromptStrategyModel
except ModuleNotFoundError:
    # Fallback
    from src.federated_learning.server import Server
    from src.federated_learning.model import AuxiliaryPromptStrategyModel


class TestServer(unittest.TestCase):
    def setUp(self):
        """Initialize a sample server with AuxiliaryPromptStrategyModel."""
        self.model_config = {
            'num_prompt_templates': 3,
            'num_fixed_keywords': 4,
            'input_features': 1 
        }
        self.server = Server(model_config=self.model_config)

    def test_server_creation(self):
        """Test that the server is created successfully with AuxiliaryPromptStrategyModel."""
        self.assertIsNotNone(self.server, "Server should not be None after initialization.")
        self.assertIsNotNone(self.server.global_model, "Server's global model should not be None.")
        self.assertIsInstance(self.server.global_model, AuxiliaryPromptStrategyModel, 
                              "Server's global model should be AuxiliaryPromptStrategyModel.")

    def test_get_global_model_weights(self):
        """Test that get_global_model_weights returns a state_dict from AuxiliaryPromptStrategyModel."""
        weights = self.server.get_global_model_weights()
        self.assertIsNotNone(weights, "Global model weights should not be None.")
        self.assertIsInstance(weights, OrderedDict, "Weights should be an OrderedDict (state_dict).")
        # Check if keys match those of an AuxiliaryPromptStrategyModel
        expected_keys = ['template_scorer.weight', 'template_scorer.bias', 
                         'keyword_scorer.weight', 'keyword_scorer.bias']
        for k in expected_keys:
            self.assertIn(k, weights.keys())


    def test_aggregate_model_updates(self):
        """Test aggregate_model_updates with AuxiliaryPromptStrategyModel weights."""
        # Create dummy client model weights from AuxiliaryPromptStrategyModel instances
        client_model_1 = AuxiliaryPromptStrategyModel(**self.model_config)
        client_model_2 = AuxiliaryPromptStrategyModel(**self.model_config)

        # Modify weights slightly to ensure aggregation changes them
        with torch.no_grad():
            for param in client_model_1.parameters():
                param.data.fill_(0.1) # Fill with a constant value for simplicity
            for param in client_model_2.parameters():
                param.data.fill_(0.3) # Fill with a different constant value
        
        dummy_weights_list = [client_model_1.state_dict(), client_model_2.state_dict()]

        try:
            self.server.aggregate_model_updates(dummy_weights_list)
        except Exception as e:
            self.fail(f"server.aggregate_model_updates() raised an exception: {e}")

        updated_weights = self.server.get_global_model_weights()
        self.assertIsNotNone(updated_weights, "Updated global model weights should not be None.")

        # Check if weights have been averaged correctly.
        # For example, the template_scorer.weight should be (0.1 + 0.3) / 2 = 0.2
        expected_value = 0.2 
        # Check one specific parameter (e.g., template_scorer.weight)
        # Note: This simple check assumes all params were filled with the same values.
        # A more robust check would iterate through parameters.
        first_param_name = list(updated_weights.keys())[0] # e.g. 'template_scorer.weight'
        
        self.assertTrue(torch.allclose(updated_weights[first_param_name], 
                                       torch.full_like(updated_weights[first_param_name], expected_value)),
                        f"Aggregated weights for '{first_param_name}' do not match the expected average. "
                        f"Expected all elements to be close to {expected_value}, "
                        f"Got: {updated_weights[first_param_name]}")


    def test_aggregate_model_updates_empty_list(self):
        """Test aggregate_model_updates with an empty list of weights."""
        initial_weights_od = self.server.get_global_model_weights()
        # Create a deep copy for comparison
        initial_weights_copy = OrderedDict((k, v.clone()) for k, v in initial_weights_od.items())


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
```
