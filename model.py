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

class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    """
    Basically it's a fully connected layer

    According to the paper, we have to linear layer
    linear1: map from d_model = 512 to Dff = 2048
    linear2: map from d_ff = 2048 to d_model = 512

    """

    def __init__(self, d_model, d_ff, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # w1 and b1
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self,x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.linear2(x)

class MultiHeadAttentionBlock(nn.Module):

    """
    Check MultiHeadAttention architecture: images/multi-head-attention-archtictre.png

    """
    def __init__(self, d_model: int, h: int, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        # we will divide the embedding vector into h head >> So d_model should be divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(h * self.d_k, d_model)

        self.softmax = nn.Softmax()

    # staticmethod enable use to call the method without creating instance from the class, for ex: MultiHeadAttentionBlock.attention(...)
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # (Batch, h, seq_len, d_k) --> (Batch, h, seq_len, seq_len) -->
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)   # transpose last two dimensions

        if mask is not None:
            attention_score.masked_fill(mask==0, 1e-9)   # replace all values == 0 with 1e-9

        attention_score = attention_score.softmax(dim=-1)  # (Batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_score = dropout(attention_score)

        # attention_score @ value: (Batch, h, seq_len, d_k)
        return attention_score @ value, attention_score
    def forward(self, q, k, v, mask):
        """
        Mask is used if we want to prevent interaction between specific tokens by
        giving their intersection - inf (before applying softmax)
        """
        query = self.w_q(q)  # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        key = self.w_k(k)    # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        value = self.w_v(v)  # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)

        # divide each matrix into h heads

        """
        - We will keep batch and seq_len dimensions, as we want to split d_model
        - We will transpose as it's preferred to make h the second dimension for some reasons
        
        (Batch, seq_len, d_model) --> (Batch, seq_len, h, d_k)  --> (Batch, h, seq_len, d_k)
        
        """
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        # Calculate Attention
        # x: (Batch, h, seq_len, d_k)
        x, self.attention_score = self.attention(query, key, value, mask, self.dropout)

        # (Batch, h, seq_len, d_k) --> (Batch, seq_len, h, d_k)
        x = x.transpose(1,2).contiguous()  # create a new tensor that is contiguous in memory, needed when working with tensors that have undergone non-contiguous operations like transposition, slicing, or reshaping

        # (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_model)
        x = x.view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    """
    In Encoder, MultiHeadAttention is considered as self attention as we get the relation between each token in the sentence and other tokens in same sentence
    """
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x



class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)

class DecoderBlock(nn.Module):
    """
    MultiHeadAttentionBlock used as self_attention and cross_attention depending on the input
    """
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, tgt_mask))  # we have to use lambda as we pass a function to residual_connections
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output,encoder_output, src_mask))  # Query from Decoder, while key and value from encoder
        return self.residual_connections[2](x, self.feed_forward_block)

class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)

# We need to return the attention output back from d_model to position of the vocabs -> using linear layer

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.projection(x), dim=-1)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    # We will create  forward for each block for output reuseabilty and visualization

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, tgt, src_mask, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)

    # For Translation Transformer
    def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int,
                          d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1,
                          d_ff: int = 2048):
        # Create the embedding layers
        src_embed = InputEmbeddings(d_model, src_vocab_size)
        tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

        # Create the positional encoding layers
        src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
        tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

        # Create the encoder blocks
        encoder_blocks = []
        for _ in range(N):
            encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
            encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
            encoder_blocks.append(encoder_block)

        # Create the decoder blocks
        decoder_blocks = []
        for _ in range(N):
            decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
            decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
            decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block,
                                         feed_forward_block, dropout)
            decoder_blocks.append(decoder_block)

        # Create the encoder and decoder
        encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
        decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

        # Create the projection layer
        projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

        # Create the transformer
        transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

        # Initialize the parameters
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        return transformer