import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    """
    Transform text into vector.

    Args:
        d_model: the dimension of the model in the paper  >> 512
        vocab_size: the vocabulary size

    Returns:
        a nn.Embedding layer
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Transformers don't have a built-in sense of sequence order, so positional encoding is
    introduced to provide information about the position of tokens in a sequence.

    The positional encoding is added element-wise to the input embeddings

    Args:
        d_model: the dimension of the embedding  >> 512
        seq_len: The max length of the sentence

    Returns:
        a nn.Positional Encoding layer
    """

    def __init__(self,d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = dropout

        # Create matrix of shape (seq_len, d_model)
        positional_encoding = torch.zeros(seq_len, d_model)

        # Create vector position with shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # unsqueeze: convert tensor to rank 2

