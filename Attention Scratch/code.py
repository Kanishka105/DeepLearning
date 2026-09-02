import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
import nltk
import string
import math
nltk.download("punkt")
nltk.download("punkt_tab")
class Attention(nn.Module):
    def __init__(self, embedding_dim):

        super(Attention, self).__init__()

        self.embedding_dim = embedding_dim

        # Q, K, V learnable projections
        # Here we are creating a multilayer perceptron (MLP) for each of the Q, K, and V projections. Each MLP consists of a linear layer that maps the input embedding dimension to the same embedding dimension. This allows the model to learn different representations for queries, keys, and values from the same input embeddings.
        self.WQ = nn.Linear(
            embedding_dim,
            embedding_dim
        )
        self.WK = nn.Linear(
            embedding_dim,
            embedding_dim
        )
        self.WV = nn.Linear(
            embedding_dim,
            embedding_dim
        )
#         WQ/WK/WV = learnable parameters
#          Q/K/V = outputs produced using those parameters
# X @ WQ → Q
# X @ WK → K
# X @ WV → V


    def forward(self, X):
        # 1. Create Q, K, V
        Q = self.WQ(X)
        K = self.WK(X)
        V = self.WV(X)
        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )
        d_k = K.size(-1)
        scaled_scores = scores / math.sqrt(d_k)
        # Prevent the token from looking
        # at future tokens.
        sequence_length = X.size(1)
        # upper triangle matrix with 1s above the diagonal and 0s on and below the diagonal. This mask is used to prevent the model from attending to future tokens during training, ensuring that each token can only attend to itself and previous tokens.
#             I  love  deep  learning
#           ┌────────────────────────────
# I         │ 0    1     1      1
# love      │ 0    0     1      1
# deep      │ 0    0     0      1
# learning  │ 0    0     0      0
# Boolean form
# False  True   True   True
# False  False  True   True
# False  False  False  True
# False  False  False  False
# IT MEANS:
#              I      love    deep    learning
# I             KEEP   MASK    MASK    MASK
# love          KEEP   KEEP    MASK    MASK
# deep          KEEP   KEEP    KEEP   MASK
# learning      KEEP   KEEP    KEEP   KEEP
# 1 represents: Don't allow attention here. Because those positions are future tokens
        mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=X.device
            ),
            diagonal=1
        ).bool()
        scaled_scores = scaled_scores.masked_fill(
            mask,float("-inf"))
        # SOFTMAX-------------------------------
        attention_weights = torch.softmax(
            scaled_scores,
            dim=-1
        )
        # Value Output = summation of all the values weighted by their corresponding attention weights. This gives us a new representation for each token that incorporates information from other tokens in the sequence, based on their relevance as determined by the attention mechanism.
        attention_output = torch.matmul( attention_weights,V)
        # They tell us how much information to take from each Value vector.
        return attention_output, attention_weights
class AttentionModel(nn.Module):
    def __init__(
        self,vocab_size,embedding_dim):
        super().__init__()
        # Each token is represented by 4 learned numbers.
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)
        # embedding shape is [batch_size,tokens, embedding_dim]
        self.attention = Attention(embedding_dim)
        # attention _outputs and attention_weights
        self.output_layer = nn.Linear(
            embedding_dim,
            vocab_size
        )
    def forward(self, X):
        # Token IDs → Embeddings
        X = self.embedding(X)
        # Embeddings → Attention
        attention_output, attention_weights = self.attention(X)
        # Attention output → Vocabulary scores
        # logits are the raw score that theortically can be any real number. They are not probabilites[ between 0 and 1]
        logits = self.output_layer(attention_output)
        return logits, attention_weights


# Tokenization: --------lower-word_tokenize-remove punctuation-empty string remove-return tokens list
def tokenize(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [
        token
        for token in tokens
        if token not in string.punctuation]
    tokens = [
        token
        for token in tokens
        if token.strip()]
    return tokens
# Vocabulary Collection: Created from TRAINING DATA. It maps each unique token with unique ID. 
# IT HAS PAD_TOKEN[0]--padding and UNK_TOKEN[1]--unknown values.
def VocabCollection(tokenize_list):
    # PLACEHOLDERS
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_ID = 0
    UNK_ID = 1
    vocab = {
        PAD_TOKEN: PAD_ID,
        UNK_TOKEN: UNK_ID
    }
    for tokens in tokenize_list:
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
# Encodes: provide the token ids for each token in input.
def TokenID(vocab, tk):
    UNK_ID = 1
    train_encoded = []
    for tokens in tk:
        encoded = [
            vocab.get(token, UNK_ID)
            for token in tokens]
        train_encoded.append(encoded)
    return train_encoded
# INPUT DATA------
input_text = "The cat sat on the mat"
# Tokenization
tk = [tokenize(input_text)]
print("Tokens:")
print(tk)
# Vocabulary
vocab = VocabCollection(tk)
print("\nVocabulary:")
print(vocab)
# Token IDs
Token_id = TokenID(vocab,tk)
print("\nToken IDs:")
print(Token_id)
# --------------------------------------------------
# Training Task:Given the current/previous words, predict the next word.
# --------------------------------------------------

# Original:
#
# The cat sat on the mat
#
# Input:
#
# The cat sat on the
#
# Target:
#
# cat sat on the mat
# X=[the cat sat on the]
# Y=[cat sat on the mat] For every input position, we need a corresponding correct answer the-cat cat-sat sat-on on-the the-mat
# X and Y must have matching positions, with Y being the next token for each position in X.
# X=Y = 5tokens

X = torch.tensor(
    [Token_id[0][:-1]],
    dtype=torch.long
)

Y = torch.tensor(
    [Token_id[0][1:]],
    dtype=torch.long
)


print("\nX:")
print(X)

print("\nY:")
print(Y)


print("\nX shape:")
print(X.shape)


print("\nY shape:")
print(Y.shape)
# Embedding:
vocab_size = len(vocab)
EMBEDDING_DIM = 4
model = AttentionModel(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM
)
# LOSS_FUNCTION
loss_function = nn.CrossEntropyLoss()
# OPTIMIZER

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)
# TRAINING

epochs = 500


for epoch in range(epochs):
    logits, attention_weights = model(X)
    loss = loss_function(
        logits.reshape(-1, vocab_size),
        Y.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    logits, attention_weights = model(X)
    predictions = torch.argmax(logits,dim=-1)
id_to_token = { value: key for key, value in vocab.items()}
predicted_words = [
    id_to_token[token_id.item()]
    for token_id in predictions[0]]
actual_words = [
    id_to_token[token_id.item()]
    for token_id in Y[0]]

print("\nAttention Weights:")
print(attention_weights)


print("\nAttention Weights Shape:")
print(attention_weights.shape)
