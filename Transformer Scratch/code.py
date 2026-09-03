
from nltk.tokenize import word_tokenize
import string
import torch
import torch.nn as nn
import math
# 1. TOKENIZATION
inputSentence = "The cat sat on the mat"
def tokenize(text):
     text = text.lower()
     tokens = word_tokenize(text)
     tokens = [
        token
        for token in tokens
        if token not in string.punctuation
    ]
     tokens = [
        token
        for token in tokens
        if token.strip()
    ]
     return tokens
tokens = tokenize(inputSentence)
def VocabCollection(tokenize_list):
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
vocab = VocabCollection([tokens])
# 3. TOKEN → ID
def TokenID(vocab, tk):
    UNK_ID = 1
    encoded_data = []
    for tokens in tk:
        encoded = [
            vocab.get(token, UNK_ID)
            for token in tokens
        ]
        encoded_data.append(encoded)
    return encoded_data
token_ids = TokenID(vocab, [tokens])
# 4. ID → TOKEN
def decode_tokens(vocab, token_ids):
    id_to_token = {
        id: token
        for token, id in vocab.items()
    }
    decoded_tokens = []
    for ids in token_ids:
        tokens = [
            id_to_token.get(id, "<UNK>")
            for id in ids]
        decoded_tokens.append(tokens)
    return decoded_tokens
decoded_tokens = decode_tokens(vocab, token_ids)
print("Decoded:", decoded_tokens)
# 5. CREATE X AND Y
# tokens:
# the cat sat on the mat
# X:
# the cat sat on the
# Y:
# cat sat on the mat
X = torch.tensor(
    [token_ids[0][:-1]],
    dtype=torch.long)
Y = torch.tensor(
    [token_ids[0][1:]],
    dtype=torch.long)
print("X:", X)
print("Y:", Y)
print("X shape:", X.shape)
print("Y shape:", Y.shape)
# 6. FEED FORWARD NETWORK
class FeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim,hidden_dim)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim,embedding_dim)
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
# 7. MULTI-HEAD SELF-ATTENTION
class MultiHeadAttention(nn.Module):
    def __init__(self,embedding_dim,num_heads):
        super().__init__()
        assert embedding_dim % num_heads == 0
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = (embedding_dim // num_heads)
        # Q, K, V projections
        self.WQ = nn.Linear(embedding_dim,embedding_dim)
        self.WK = nn.Linear(embedding_dim,embedding_dim)
        self.WV = nn.Linear(embedding_dim,embedding_dim)
        # Output projection
        self.output_projection = nn.Linear(embedding_dim,embedding_dim)
    def forward(self, X):
        batch_size, sequence_length, _ = X.shape
        # 1. Create Q, K, V
        Q = self.WQ(X)
        K = self.WK(X)
        V = self.WV(X)
        # 2. Split into multiple heads // different heads can learn different types of relationships. That head has to learn one type of attention pattern. After that, we combine all heads back together to get the original embedding dimension.
        Q = Q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )
        K = K.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )
        V = V.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim)
        # 3. Move heads before sequence dimension
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        # Shape:
        # [batch, heads, sequence, head_dim]
        # 4. Calculate attention scores
        scores = torch.matmul(Q,K.transpose(-2, -1))
        # 5. Scale
        scores = scores / math.sqrt(self.head_dim)
        # 6. Causal mask
        mask = torch.triu(torch.ones(sequence_length,sequence_length,device=X.device),
            diagonal=1).bool()
        scores = scores.masked_fill(mask,float("-inf"))
        # 7. Softmax
        attention_weights = torch.softmax(scores,dim=-1)
         # 8. Weighted sum of V
        attention_output = torch.matmul(
            attention_weights,V)
        # 9. Join heads
        attention_output = attention_output.transpose(1,2)
        attention_output = (attention_output.contiguous().view(batch_size,sequence_length,self.embedding_dim))
        # 10. Final projection
        output = self.output_projection(attention_output)
        return output, attention_weights
# 8. TRANSFORMER BLOCK
class TransformerBlock(nn.Module):
    def __init__(
        self,embedding_dim,num_heads,hidden_dim):
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        # Feed Forward Network
        self.ffn = FeedForward(embedding_dim,hidden_dim)
        # Layer Normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
    def forward(self, X):
        # ATTENTION SUB-LAYER
        attention_output, attention_weights = (self.attention(X))
        # Residual connection
        X = X + attention_output
        # LayerNorm
        X = self.norm1(X)
        # FEED FORWARD SUB-LAYER
        ffn_output = self.ffn(X)
        # Residual connection
        X = X + ffn_output
        # LayerNorm
        X = self.norm2(X)
        return X, attention_weights
# 9. COMPLETE TRANSFORMER LANGUAGE MODEL
class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_heads,
        hidden_dim,
        max_sequence_length
    ):
        super().__init__()
        # Token Embedding
        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )
        # Positional Embedding
        self.position_embedding = nn.Embedding(
            max_sequence_length,
            embedding_dim
        )
        # Transformer Block
        self.transformer_block = TransformerBlock(
            embedding_dim,
            num_heads,
            hidden_dim
        )
        # LM HEAD
        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size
        )
    def forward(self, X):
        batch_size, sequence_length = X.shape
        # 1. TOKEN EMBEDDING
        token_embedding = self.token_embedding(X)
        # Shape:
        # [batch, sequence, embedding_dim]
        # 2. POSITION EMBEDDING
        positions = torch.arange(
            sequence_length,
            device=X.device
        )
        position_embedding = (
            self.position_embedding(positions)
        )
        # Shape:
        # [sequence, embedding_dim]
        # 3. ADD TOKEN + POSITION
        X = (
            token_embedding
            + position_embedding
        )
        # Shape:
        # [batch, sequence, embedding_dim]
        # 4. TRANSFORMER BLOCK
        X, attention_weights = (
            self.transformer_block(X)
        )
        # 5. LM HEAD
        logits = self.lm_head(X)
        # Shape:
        # [batch, sequence, vocab_size]
        return logits, attention_weights
# 10. MODEL CONFIGURATION
vocab_size = len(vocab)
embedding_dim = 4
num_heads = 2
hidden_dim = 16
max_sequence_length = 20
# 11. CREATE MODEL

model = TransformerModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    max_sequence_length=max_sequence_length
)


print("\nModel:")
print(model)


# 12. LOSS + OPTIMIZER

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)


# 13. TRAINING

epochs = 500

for epoch in range(epochs):

    # Forward pass

    logits, attention_weights = model(X)

    # Calculate loss

    loss = loss_function(
        logits.reshape(
            -1,
            vocab_size
        ),
        Y.reshape(-1)
    )

    # Clear gradients

    optimizer.zero_grad()

    # Backpropagation

    loss.backward()

    # Update parameters

    optimizer.step()

    # Print loss

    if (epoch + 1) % 50 == 0:

        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {loss.item():.4f}"
        )


# 14. EVALUATION

model.eval()

with torch.no_grad():

    logits, attention_weights = model(X)

    predictions = torch.argmax(
        logits,
        dim=-1
    )


# 15. DECODE PREDICTIONS

id_to_token = {
    value: key
    for key, value in vocab.items()
}


predicted_words = [
    id_to_token[token_id.item()]
    for token_id in predictions[0]
]


actual_words = [
    id_to_token[token_id.item()]
    for token_id in Y[0]
]


print("\nPredicted:", predicted_words)

print("Actual:   ", actual_words)


# 16. AUTOREGRESSIVE TEXT GENERATION

def generate_text(
    model,
    start_token,
    vocab,
    max_length=10
):

    model.eval()

    id_to_token = {
        id: token
        for token, id in vocab.items()
    }

    # Start token → ID

    input_ids = torch.tensor(
        [[vocab[start_token]]],
        dtype=torch.long
    )

    # Generate tokens

    with torch.no_grad():

        for _ in range(max_length - 1):

            # Prevent sequence from exceeding
            # positional embedding size

            if input_ids.shape[1] >= max_sequence_length:
                break

            # Forward pass

            logits, _ = model(input_ids)

            # Take logits of last token

            next_token_logits = (
                logits[:, -1, :]
            )

            # Select highest probability token

            next_token_id = torch.argmax(
                next_token_logits,
                dim=-1
            )

            # Add new token

            input_ids = torch.cat(
                [
                    input_ids,
                    next_token_id.unsqueeze(0)
                ],
                dim=1
            )
    # ID → Token
    

    generated_words = [
        id_to_token[token_id.item()]
        for token_id in input_ids[0]
    ]

    return " ".join(generated_words)


# 17. GENERATE TEXT

generated_text = generate_text(
    model=model,
    start_token="the",
    vocab=vocab,
    max_length=6
)

print("\nGenerated:", generated_text)
