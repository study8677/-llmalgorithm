import torch # Keep torch if it's used by OrderedDict or model state_dict, otherwise it might be removed.
from .model import AuxiliaryPromptStrategyModel # Changed import
from collections import OrderedDict

class Server:
    """
    Represents the central server in an API-based Federated Learning with Human Feedback (FLHF) system.

    The server is responsible for managing the global auxiliary model (e.g.,
    `AuxiliaryPromptStrategyModel`). This auxiliary model is used by clients to
    formulate effective prompts for a central Large Language Model (LLM).
    The server aggregates updates for this auxiliary model from clients.

    Attributes:
        global_model (AuxiliaryPromptStrategyModel): The global instance of the
                                                     auxiliary prompt strategy model.
    """
    def __init__(self, model_config: dict):
        """
        Initializes the Server instance.

        Args:
            model_config (dict): Configuration dictionary for instantiating the
                                 global `AuxiliaryPromptStrategyModel`.
                                 Example: {'num_prompt_templates': 5, 'num_fixed_keywords': 10, 'input_features': 1}
        """
        self.global_model = AuxiliaryPromptStrategyModel(**model_config)

    def aggregate_model_updates(self, client_model_weights_list: list[OrderedDict]):
        """
        Aggregates auxiliary model updates from multiple clients using Federated Averaging (FedAvg).

        The weights (state_dict) from the clients' auxiliary models are averaged
        and loaded into the server's global auxiliary model.

        Args:
            client_model_weights_list (list of OrderedDict): A list where each
                element is a state dictionary from a client's `AuxiliaryPromptStrategyModel`.
        """
        # Federated Averaging (FedAvg) logic remains applicable
        if not client_model_weights_list:
            return

        # Initialize a dictionary to store the sum of weights
        aggregated_weights = OrderedDict()

        # Sum weights from all client models
        for client_weights in client_model_weights_list:
            for key, value in client_weights.items():
                if key in aggregated_weights:
                    aggregated_weights[key] += value
                else:
                    aggregated_weights[key] = value.clone() # Use .clone() to ensure a new tensor

        # Average the weights
        num_clients = len(client_model_weights_list)
        for key in aggregated_weights:
            aggregated_weights[key] /= num_clients

        # Update the global model's weights
        self.global_model.load_state_dict(aggregated_weights)

    def get_global_model_weights(self) -> OrderedDict:
        """
        Retrieves the weights of the server's global auxiliary model.

        This is typically called by clients at the beginning of an FL round
        to get the latest global auxiliary model parameters.

        Returns:
            OrderedDict: A state dictionary containing the weights of the
                         global `AuxiliaryPromptStrategyModel`.
        """
        return self.global_model.state_dict()
