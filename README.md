
# Building Transformers from Scratch using PyTorch

This repository provides an overview of building a transformer model from scratch. Transformers are a popular architecture for various natural language processing tasks, including machine translation.



## Transformers Architecture

![Transformers.png](images%2FTransformers.png)


## Encoder

### Self-Attention

Self-attention provides a new embedding with the same shape as the input embedding but captures the relationships between tokens in a sentence. Each self-attention mechanism is considered one head.

### Multi-Head Attention

1. We take the input from positional encoding and create four copies of it. Three of them (Q, K, V) are input to multi-head attention, and one is sent to "Add & Norm."

2. Multiply Q, K, V (seq_len, d_model) with three weight matrices (d_model, d_model) respectively, resulting in Q`, K`, V` (seq_len, d_model).

3. Split Q`, K`, V` into smaller matrices, each of size (seq_len, d_k), where d_k is d_model / h, and h is the number of heads. For example, with 4 heads and d_model = 512, each matrix will be divided into 4 heads (seq_len, 128).

4. Calculate the attention between the smaller matrices, e.g., Q1`, K1`, V1`. The output will be head1. Repeat this process for other smaller matrices to obtain 4 heads.

5. Concatenate (head1, ..., head_h) to get (seq_len, h * d_v) where d_v = d_k.

6. Multiply the output by W_o, where W_o is (h * d_v, d_model).

Note: h * d_v should equal d_model if d_v = d_k.


![multi-head-attention-archtictre.png](images%2Fmulti-head-attention-archtictre.png)


### Layer Normalization (Add & Norm)

- There's a connection that takes a copy from positional encoding and directly feeds it into the Add & Norm layer.

- Calculate mean and variance for each vector.

- Replace each value in the vector with a normalized value using the mean and variance: X` = (X - mean) / (sqrt(variance - beta)).

- We have trainable parameters, beta and gamma. This make the freedom to the model to amplify some values

Then there are two steps:

1. Feed Forward: Fully connected Layer

2. (Add & Norm) layer with a copy from the previous (Add & Norm) layer.

## Decoder

1. Output Embedding

2. Positional Encoding

3. Masked Multi-Head Attention
   - Ensures that the output at a certain position depends only on words in previous positions.
   - Achieved by setting the softmax output for future words to -inf before multiplying by the value vector.

4. Add & Norm (taking a copy from the previous Masked Multi-Head Attention)

5. Multi-Head Attention
   - Takes key and value from the encoder and query from the decoder input (Masked Multi-Head >> Add & Norm).

6. Add & Norm (taking a copy from the previous Add & Norm)

7. Feed Forward: Fully Connected Layer

8. Add & Norm (taking a copy from the previous Add & Norm)

## Decision Block

1. Linear Layer: Maps (seq_len, d_model) to (seq_len, vocab_size) to obtain positions in the vocabulary related to the embedding.

2. Softmax: Produces (seq_len, vocab_size).

3. Generates the target sentence with <EOS> at the end.

## Training

1. Use labels and input sentences to calculate cross-entropy loss.

2. Backpropagation.

## Why Transformers are Optimized Compared to RNN?

Transformers are optimized because they perform translation in a single step, unlike RNNs, which require multiple time steps. In Transformers:

- The input sentence is given to the encoder, producing embeddings that include word embedding, positional information, and token relationships.

- The input sentence is also given to the decoder, which uses the encoder's output to produce output embeddings.

- Predictions are generated, cross-entropy loss is calculated, and the model is trained—all in one step, without the need for loops over tokens.

## Inference

Input Sentence: <SOS> He is a smart guy <EOS>

### Encoder

1. Input the sentence to the Encoder: Encoder Input >> Encoder >> Encoder output (seq_len, d_model).

### Decoder

In training, everything happens in one timestamp. In inference:

1. Timestamp 1: Input <SOS> to the decoder embedding and positional encoding. Get the query from "decoder input." Input the query and (key, value) from the Encoder output to the multi-head attention.

    - Decoder output >> Linear layer (seq_len, vocab_size) to get logits >> Softmax (to get token with the maximum value) >> First output token (هو).

2. Timestamp 2: Use the same encoder output. Use the predicted tokens up to Timestamp 1 <SOS> هو, and so on.

For selecting predicted tokens, there are two approaches:

1. Greedy Strategy: Select the token with the maximum probability from softmax.

2. Beam Search: Explores a broader range of options by considering the top B most probable sentences at each step.

   - Beam size (B) determines the number of candidates considered.

   - Evaluate all possible next words for each of the top B choices at each step.

   - Choose the top B most probable sentences based on their probabilities.

Example with B = 2:

Word1: 0.6
Word2: 0.3
Word3: 0.2
Word4: 0.1

1. Top B Candidates: Word1 (0.6), Word2 (0.3)

2. Expand candidates and repeat the process.

   ...

4. In every step, select the top B words and evaluate all possible next words for each of them to maintain the top B most probable sentences.

---
 
