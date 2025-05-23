import torch
import torch.nn as nn

class AuxiliaryPromptStrategyModel(nn.Module):
    """
    An auxiliary model designed to learn a strategy for selecting prompt templates
    and keywords to guide a Large Language Model (LLM) via its API.

    This model is intended to be small and trainable on the client-side within an
    FLHF (Federated Learning with Human Feedback) setup. Its outputs (scores for
    templates and keywords) can be used by the client to formulate an effective
    prompt for the LLM.

    Attributes:
        input_features (int): The number of input features that will drive the selection.
                              This could represent client state, context, etc.
        num_prompt_templates (int): The number of predefined prompt templates the model
                                    can choose from or score.
        num_fixed_keywords (int): The number of predefined keywords the model can
                                  choose from or score.
        template_scorer (nn.Linear): A linear layer to output scores for each
                                     prompt template.
        keyword_scorer (nn.Linear): A linear layer to output scores for each
                                    predefined keyword.
    """
    def __init__(self, num_prompt_templates: int, num_fixed_keywords: int, input_features: int = 1):
        """
        Initializes the AuxiliaryPromptStrategyModel.

        Args:
            num_prompt_templates (int): The number of predefined prompt templates
                                        the model will learn to score.
            num_fixed_keywords (int): The number of predefined, fixed keywords
                                      the model will learn to score.
            input_features (int, optional): The dimensionality of the input tensor
                                            that will be fed into the model.
                                            Defaults to 1 (e.g., a dummy scalar input).
        """
        super(AuxiliaryPromptStrategyModel, self).__init__()
        self.input_features = input_features
        self.num_prompt_templates = num_prompt_templates
        self.num_fixed_keywords = num_fixed_keywords

        # Layer to score prompt templates
        self.template_scorer = nn.Linear(self.input_features, self.num_prompt_templates)

        # Layer to score a fixed set of predefined keywords
        self.keyword_scorer = nn.Linear(self.input_features, self.num_fixed_keywords)

    def forward(self, x_dummy_input: torch.Tensor):
        """
        Performs the forward pass of the model.

        This method takes a dummy input (placeholder for client state or context)
        and outputs scores for prompt templates and keywords.

        Args:
            x_dummy_input (torch.Tensor): A placeholder input tensor. Its shape should
                                          be (batch_size, self.input_features).
                                          For initial testing, this could be
                                          `torch.ones(1, self.input_features)`.

        Returns:
            tuple: A tuple containing:
                - template_scores (torch.Tensor): Scores for each prompt template.
                                                  Shape: (batch_size, self.num_prompt_templates).
                - keyword_scores (torch.Tensor): Scores for each predefined keyword.
                                                 Shape: (batch_size, self.num_fixed_keywords).
        """
        # Ensure input is of the correct shape if batch_size > 1
        # and input_features is 1, this handles scalar inputs per batch item.
        if self.input_features == 1:
            if x_dummy_input.ndim == 0 : # if it's a scalar e.g. from torch.tensor(1.0)
                x_dummy_input = x_dummy_input.view(1,-1) # make it (1,1) for a batch of 1
            elif x_dummy_input.ndim == 1: # if it's like torch.tensor([1.0, 2.0]) for batch_size=2
                 x_dummy_input = x_dummy_input.unsqueeze(-1) # Make it (batch_size, 1)


        template_scores = self.template_scorer(x_dummy_input)
        keyword_scores = self.keyword_scorer(x_dummy_input)

        # TODO: The selected template (e.g., by argmax on scores) and keywords
        # (e.g., by top-k scores or thresholding) will be used by the client
        # to formulate a prompt for the LLM API.

        return template_scores, keyword_scores

if __name__ == '__main__':
    # Example Usage
    num_templates = 5
    num_keywords = 10
    input_dim_scalar = 1 # Dummy input feature dimension (scalar)

    model_scalar_input = AuxiliaryPromptStrategyModel(
        num_prompt_templates=num_templates,
        num_fixed_keywords=num_keywords,
        input_features=input_dim_scalar
    )

    # Simulate a batch of dummy inputs (e.g., batch_size = 1, input_features = 1)
    dummy_input_s1 = torch.ones(1, input_dim_scalar) # Batch size 1
    ts_s1, ks_s1 = model_scalar_input(dummy_input_s1)

    print("AuxiliaryPromptStrategyModel Example (Scalar Input Features):")
    print(f"  Input (batch=1, features=1): {dummy_input_s1.shape}")
    print(f"  Template scores shape: {ts_s1.shape}") # Expected: (1, num_templates)
    print(f"  Keyword scores shape: {ks_s1.shape}")   # Expected: (1, num_keywords)
    print(f"  Sample Template Scores: {ts_s1.data}")
    print(f"  Sample Keyword Scores: {ks_s1.data}")

    # Example with batch_size > 1, input_features = 1
    dummy_input_s_batch = torch.rand(3, input_dim_scalar) # Batch size 3
    ts_s_batch, ks_s_batch = model_scalar_input(dummy_input_s_batch)
    print(f"\n  Input (batch=3, features=1): {dummy_input_s_batch.shape}")
    print(f"  Template scores shape: {ts_s_batch.shape}") # Expected: (3, num_templates)
    print(f"  Keyword scores shape: {ks_s_batch.shape}")   # Expected: (3, num_keywords)

    # Example with multiple input features
    input_dim_vector = 3 # Vector input feature dimension
    model_vector_input = AuxiliaryPromptStrategyModel(
        num_prompt_templates=num_templates,
        num_fixed_keywords=num_keywords,
        input_features=input_dim_vector
    )
    dummy_input_v_batch = torch.rand(2, input_dim_vector) # Batch size 2, 3 features
    ts_v_batch, ks_v_batch = model_vector_input(dummy_input_v_batch)
    print(f"\nAuxiliaryPromptStrategyModel Example (Vector Input Features):")
    print(f"  Input (batch=2, features=3): {dummy_input_v_batch.shape}")
    print(f"  Template scores shape: {ts_v_batch.shape}") # Expected: (2, num_templates)
    print(f"  Keyword scores shape: {ks_v_batch.shape}")   # Expected: (2, num_keywords)
```
