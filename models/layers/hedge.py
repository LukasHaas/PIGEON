import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter
from preprocessing import haversine_matrix

# TODO: should be based on Geoguessr score

class HedgeLayer(nn.Module):
    """Deploys a Geoguessr guess hedging strategy.
    """
    def __init__(self, temperature: float=1):
        """Deploys a Geoguessr guess hedging strategy.

        Args:
            temperature (float, optional): variable determining how strongly
                guess probabilities should be modified to hedge against
                other players. A higher temperature corresponds to lower
                hedging. Defaults to 1.
        """
        super(HedgeLayer, self).__init__()
        self.temperature = Parameter(torch.tensor(float(temperature)))

    def forward(self, topk_locations: Tensor, topk_probs: Tensor) -> Tensor:
        """Computes forward pass to adjust prediction probabilities

        Args:
            topk_locations (Tensor): locations of topk predictions
            topk_probs (Tensor): probabilities of topk predictions

        Returns:
            Tensor: modified probability vector
        """
        # Compute centrality of guesses
        distances = haversine_matrix(topk_locations, topk_locations.t())
        centrality = 1 / (distances.mean(dim=0) / distances.mean())

        # Compute softmax over average distance to other points
        probs = self._temperature_softmax(centrality)

        # Redistribute probabilities
        initial_sum = topk_probs.sum()
        redist_probs = topk_probs * probs
        redist_probs = (redist_probs / redist_probs.sum()) * initial_sum
        return redist_probs.type('torch.cuda.FloatTensor')

    def _temperature_softmax(self, input: Tensor) -> Tensor:
        """Performs softmax with temperature.

        Args:
            input (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        ex = torch.exp(input / self.temperature)
        sum = torch.sum(ex, axis=0)
        return ex / sum