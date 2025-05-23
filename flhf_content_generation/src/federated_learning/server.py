import torch
from .model import SimpleSeq2SeqModel
from collections import OrderedDict

class Server:
    """
    Represents the central server in the Federated Learning system.

    The server is responsible for managing the global model, aggregating model
    updates received from clients, and providing the updated global model back
    to the clients.

    Attributes:
        global_model (SimpleSeq2SeqModel): The global instance of the model.
    """
    def __init__(self, model_config):
        """
        Initializes the Server instance.

        Args:
            model_config (dict): Configuration dictionary for instantiating the
                                 global model. Passed to `SimpleSeq2SeqModel`.
        """
        self.global_model = SimpleSeq2SeqModel(**model_config)

    def aggregate_model_updates(self, client_model_weights_list):
        """
        Aggregates model updates from multiple clients using Federated Averaging (FedAvg).

        The weights from the client models are averaged and loaded into the
        server's global model.

        Args:
            client_model_weights_list (list of OrderedDict): A list where each
                element is a state dictionary (OrderedDict) representing the
                model weights from a client.
        """
        # Federated Averaging (FedAvg)
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

    def get_global_model_weights(self):
        """
        Retrieves the weights of the server's global model.

        This is typically called by clients at the beginning of an FL round
        to get the latest global model.

        Returns:
            OrderedDict: A state dictionary containing the weights of the global model.
        """
        return self.global_model.state_dict()
