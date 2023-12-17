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

    It gives an indication when handle tokens that they are close ot distant from each other using mathematical formula
    So It's Calculated, not learned through training the model

    - We will calculate the full matrix for maximum seq_len, then we will take partion from it according to the sentence length
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
        self.dropout = nn.Dropout(dropout)

        # Create matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        
        
        # There's a formula in the paper to calculate the positional_encoding vector, we will use the simplified version
        # Create vector position with shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # unsqueeze: convert tensor to rank 2. example: tensor([[0.],[1.],[2.],[3.],..])
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -math.log((10000.0) / d_model))

        # Apply sin to even positions
        pe[:,0::2] = torch.sin(position * div_term)
        # Apply cos to odd positions

        pe[:,1::2] = torch.cos(position * div_term)

        # We will use it with batch of sentences, that's why we need to add additional dimention in the begineeing

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        # Saving variable in the buffer of the model enable us to save it when we save the model
        self.register_buffer('pe', pe)


    def forward(self, x):
        # x is the input tensor representing the embeddings of the input sequence.

        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)
