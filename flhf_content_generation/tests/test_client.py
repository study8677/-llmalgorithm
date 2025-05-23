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
    from flhf_content_generation.src.federated_learning.model import AuxiliaryPromptStrategyModel
    from flhf_content_generation.src.llm_api_simulator import LLMAPISimulator
    # data_utils is not strictly needed for these client tests as data_loader is often None or mocked
except ModuleNotFoundError:
    # Fallback
    from src.federated_learning.client import Client
    from src.federated_learning.model import AuxiliaryPromptStrategyModel
    from src.llm_api_simulator import LLMAPISimulator

class TestClient(unittest.TestCase):
    def setUp(self):
        """Initialize a sample client for API-based FLHF."""
        self.templates = ["Template1: {input}", "Template2: {input} summary"]
        self.keywords = ["kw1", "kw2", "kw3"]
        
        self.model_config = {
            'num_prompt_templates': len(self.templates),
            'num_fixed_keywords': len(self.keywords),
            'input_features': 1 # Assuming a single scalar input for the auxiliary model
        }
        
        self.llm_sim = LLMAPISimulator(api_latency=0) # No latency for tests
        
        self.client = Client(
            client_id='client_test_0', 
            model_config=self.model_config, 
            data_loader=None, # data_loader is optional and not used in these specific tests
            learning_rate=0.01
        )

    def test_client_creation(self):
        """Test that the client is created successfully with AuxiliaryPromptStrategyModel."""
        self.assertIsNotNone(self.client, "Client should not be None after initialization.")
        self.assertIsNotNone(self.client.model, "Client model should not be None.")
        self.assertIsInstance(self.client.model, AuxiliaryPromptStrategyModel, "Client model should be AuxiliaryPromptStrategyModel.")
        self.assertIsNotNone(self.client.optimizer, "Client optimizer should be initialized.")

    def test_generate_content_with_llm(self):
        """Test the client's ability to generate content using the LLM simulator."""
        # Auxiliary model input features should match self.model_config['input_features']
        dummy_input = torch.ones(1, self.model_config['input_features']) 
        
        try:
            generated_text, template_scores, keyword_scores = self.client.generate_content_with_llm(
                client_input_data=dummy_input,
                llm_api_simulator=self.llm_sim,
                predefined_prompt_templates=self.templates,
                predefined_keywords=self.keywords
            )
        except Exception as e:
            self.fail(f"client.generate_content_with_llm() raised an exception: {e}")

        self.assertIsInstance(generated_text, str, "Generated text should be a string.")
        self.assertTrue(len(generated_text) > 0, "Generated text should not be empty.")
        
        self.assertIsInstance(template_scores, torch.Tensor, "Template scores should be a Tensor.")
        self.assertEqual(template_scores.shape, (1, len(self.templates)), "Template scores shape is incorrect.")
        
        self.assertIsInstance(keyword_scores, torch.Tensor, "Keyword scores should be a Tensor.")
        self.assertEqual(keyword_scores.shape, (1, len(self.keywords)), "Keyword scores shape is incorrect.")


    def test_train_local_aux_model(self):
        """Test that train_local_model (for auxiliary model) runs without error."""
        feedback_score = 0.8
        # Dummy scores as if they came from the auxiliary model's forward pass
        template_scores = torch.randn(1, len(self.templates)) 
        keyword_scores = torch.randn(1, len(self.keywords))

        try:
            # Note: The first argument `generated_text` was removed from client.train_local_model
            # as it's not directly used by the placeholder loss function.
            # If it were used (e.g., for more complex reward shaping), it would be needed here.
            self.client.train_local_model(
                feedback_score=feedback_score,
                template_scores=template_scores,
                keyword_scores=keyword_scores,
                num_epochs=1
            )
        except Exception as e:
            self.fail(f"client.train_local_model() raised an exception: {e}")


    def test_set_get_model_weights(self):
        """Test setting and getting model weights for AuxiliaryPromptStrategyModel."""
        initial_weights = self.client.get_local_model_weights()
        self.assertIsNotNone(initial_weights, "Initial weights should not be None.")
        
        # Create a new model instance to simulate different weights
        temp_model = AuxiliaryPromptStrategyModel(**self.model_config)
        # Slightly change a weight to ensure set_global_model_weights has an effect
        with torch.no_grad():
           for param in temp_model.parameters():
               param.data += 0.1 # Modify weights
               break 
        new_weights = temp_model.state_dict()

        self.client.set_global_model_weights(new_weights)
        current_weights = self.client.get_local_model_weights()
        
        self.assertEqual(initial_weights.keys(), current_weights.keys(), "Weight dictionary keys mismatch.")

        # Check if at least one weight tensor is different after setting new_weights
        found_diff = False
        for key in initial_weights:
            if not torch.equal(initial_weights[key], current_weights[key]):
                found_diff = True
                break
        # This check is meaningful only if new_weights are genuinely different from initial_weights
        # (which they are, due to the +0.1 modification)
        self.assertTrue(found_diff, "Weights should be different after set_global_model_weights with new weights.")


if __name__ == '__main__':
    unittest.main()
```
