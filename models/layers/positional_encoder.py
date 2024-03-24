import math
import torch
from torch import nn

class PositionalEncoder(nn.Module):
    def __init__(self, dim_model: int, dropout_p: float = 0.1, max_len: int=1000):
        """Initializes the positional embedding layer to enrich data fed into transformers
           with positional information.
        Args:
            dim_model (int): model dimension
            dropout_p (float, optional): dropout for all embeddings. Defaults to 0.1.
            max_len (int, optional): determines how far the position can influence other tokens. Defaults to 1000.
        Note:
            This code is a modified version of: `<https://pytorch.org/tutorials/beginner/transformer_tutorial.html>`_.
        """
        super().__init__()

        # Dropout
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pos_encoding', nn.Parameter(pos_encoding, requires_grad=False))

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        """Generates positional embeddings.
        Args:
            token_embedding (torch.tensor): original embeddings
        Returns:
            torch.tensor: transformed embeddings
        """
        # Residual connection + positional encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])