import unittest
import torch
import sys
import os

# Adjust path to import from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from flhf_content_generation.src.federated_learning.model import AuxiliaryPromptStrategyModel
except ModuleNotFoundError:
    # Fallback if the above fails due to how tests might be discovered/run
    from src.federated_learning.model import AuxiliaryPromptStrategyModel


class TestAuxiliaryPromptStrategyModel(unittest.TestCase):
    def setUp(self):
        """Initialize a sample AuxiliaryPromptStrategyModel."""
        self.num_prompt_templates = 3
        self.num_fixed_keywords = 4
        self.input_features = 1

        self.model_config = {
            'num_prompt_templates': self.num_prompt_templates,
            'num_fixed_keywords': self.num_fixed_keywords,
            'input_features': self.input_features
        }
        self.model = AuxiliaryPromptStrategyModel(**self.model_config)

    def test_model_creation(self):
        """Test that the AuxiliaryPromptStrategyModel is created successfully."""
        self.assertIsNotNone(self.model, "Model should not be None after initialization.")
        self.assertEqual(self.model.template_scorer.out_features, self.num_prompt_templates)
        self.assertEqual(self.model.keyword_scorer.out_features, self.num_fixed_keywords)
        self.assertEqual(self.model.template_scorer.in_features, self.input_features)
        self.assertEqual(self.model.keyword_scorer.in_features, self.input_features)


    def test_model_forward_pass(self):
        """Test the model's forward pass with dummy input tensors."""
        # Test with batch_size = 1
        batch_size_1 = 1
        dummy_input_1 = torch.ones(batch_size_1, self.input_features)
        template_scores_1, keyword_scores_1 = self.model(dummy_input_1)

        self.assertIsNotNone(template_scores_1, "Template scores should not be None.")
        self.assertIsNotNone(keyword_scores_1, "Keyword scores should not be None.")
        self.assertIsInstance(template_scores_1, torch.Tensor, "Template scores should be a Tensor.")
        self.assertIsInstance(keyword_scores_1, torch.Tensor, "Keyword scores should be a Tensor.")

        self.assertEqual(template_scores_1.shape, (batch_size_1, self.num_prompt_templates),
                         f"Template scores shape mismatch for batch size {batch_size_1}.")
        self.assertEqual(keyword_scores_1.shape, (batch_size_1, self.num_fixed_keywords),
                         f"Keyword scores shape mismatch for batch size {batch_size_1}.")

        # Test with batch_size = 2
        batch_size_2 = 2
        dummy_input_2 = torch.rand(batch_size_2, self.input_features) # Use rand for variety
        template_scores_2, keyword_scores_2 = self.model(dummy_input_2)

        self.assertEqual(template_scores_2.shape, (batch_size_2, self.num_prompt_templates),
                         f"Template scores shape mismatch for batch size {batch_size_2}.")
        self.assertEqual(keyword_scores_2.shape, (batch_size_2, self.num_fixed_keywords),
                         f"Keyword scores shape mismatch for batch size {batch_size_2}.")

        # Test with more input features if model supports it (current default is 1)
        if self.input_features > 1:
            dummy_input_multi_feature = torch.rand(batch_size_1, self.input_features)
            ts_multi, ks_multi = self.model(dummy_input_multi_feature)
            self.assertEqual(ts_multi.shape, (batch_size_1, self.num_prompt_templates))
            self.assertEqual(ks_multi.shape, (batch_size_1, self.num_fixed_keywords))


if __name__ == '__main__':
    unittest.main()
```
