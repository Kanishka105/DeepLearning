
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import string
import nltk
from torch.utils.data import TensorDataset, DataLoader
nltk.download("punkt")
nltk.download("punkt_tab")
class Models(nn.Module):
    def __init__(self,vocab_size,PAD_ID):
        super().__init__()
        # give [batch sequence features]
        # batch: how many reviews we process at one time. All batch review go through the model together.
        # Sequence: how many token positions are in each review -- max length : RNN process 200 times tokens for each review.
        # features: How many numbers represent each token.
        EMBEDDING_DIM = 128
        self.embedding = nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        padding_idx=PAD_ID)

        #input_size: at each time step, rnn receives 128 numbers == embedding dimensions=128.It does not mean 128 tokens
        # hidden size: the hidden layer in the starting have 128 zeroes so it shape is [1,128] for batch of 32 is [1,32,128]. Hidden state does not have to be 128 it could be 64 so each token-->128 numbers and rnn take those 128 numbers and then produces hidden state of 64 numbers.
        # batch_first: embedded data look like [batch,sequence,features] --[32:review,200:tokens,128:number per token]-- but pyTorch expects [seuence,batch,features] -- so it tells I am giving you a data in [batch,sequence,features] order.
        # Each token max length=200 it means we have h200 in each review
        self.rnn1=nn.RNN(input_size=128,hidden_size=128,batch_first=True)
        # self.lstm=nn.LSTM(input_size=128,hidden_size=128,batch_first=True)
        # self.gru=nn.GRU(input_size=128,hidden_size=128,batch_first=True)
        # hidden layer[1,32,128] convert to [32,128] by hidden.squeeze(0)
        self.fc=nn.Linear(in_features=128,out_features=1)
        # each review is now represented by 128 values, and we want one output for each review.
        # self.criteria=nn.BCEWithLogitsLoss()
    def forward(self,x):
        x=self.embedding(x)
        output,hidden=self.rnn1(x)
        hidden = hidden.squeeze(0)
        # hidden layer=[1[rnn layer],32[batch],128[hidden size]]
        # output layer=[32[batch],200[max length of each review encoder],128]
        x=self.fc(hidden)
        # convert 128 numbers to 1 number called logit[ not 0 nor 1 and not a probalility]
        # TIPS: Binary Cross Entropy:Compare my prediction with the actual 0/1 answer and calculate how wrong I am.
        # model give 2.5 but bce want to work between 0 and 1 so we need sigmoid 
        # Normal process linear--2.5--sigmoid--0.93--bce--loss
        # But pytorch has a shortcut : nn.BCEWithLogitsLoss(): sigmoid+bce
        # loss=self.criteria()
        return x
    


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
df = pd.read_csv(
    "./data/MovieREVIEW/IMDBDataset.csv")
# Tokenization -- word tokenize
df["tokenized_review"]=df["review"].apply(tokenize)
# Numerical classification 
df["label"] = df["sentiment"].map({
    "positive": 1,"negative": 0}).tolist()
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)
# Build Vocabulary
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1
vocab = {
    PAD_TOKEN: PAD_ID,
    UNK_TOKEN: UNK_ID
}
# add word to the vocab... to create fixed vocabulary during training. INCLUDE TRAINING DATA ONLY
for tokens in train_df["tokenized_review"]:
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
# Training Encoding: word convert to token id using vocabulary. Here we create the encoding list and store it in encode with each text of may be differnet length.
# Vocabulary size
vocab_size = len(vocab)
train_encoded = []

for tokens in train_df["tokenized_review"]:
    encoded = [
        vocab.get(token, UNK_ID)
        for token in tokens
    ]
    train_encoded.append(encoded)
val_encoded = []

for tokens in val_df["tokenized_review"]:
    encoded = [
        vocab.get(token, UNK_ID)
        for token in tokens
    ]
    val_encoded.append(encoded)

# encoded_texts = [
#     [2, 3, 4, 5, 6],   # length 5
#     [7, 8, 2, 9],       # length 4
#     [2, 5, 4, 10, 8, 6, 3]  # length 7
# ]
# We select the max length so each review is represent by exactly max length number.
MAX_LENGTH = 200
train_padded_texts = []
for sequence in train_encoded:
    # If length is less than max length than add padding 
    if len(sequence) < MAX_LENGTH:
        sequence = sequence + (
            [PAD_ID] *
            (MAX_LENGTH - len(sequence)))
    # If sequence is longer than MAX_LENGTH
    # Yes information got loss but if max length increase than more info retain and more compuation and memeory and rnn become slower.
    # Experiment with differ max length and compare validation losss and training time.
    else:
        sequence = sequence[:MAX_LENGTH]
    # if length is equal to the max length then retain the same encode
    train_padded_texts.append(sequence)
val_padded_texts = []
for sequence in val_encoded:
    # If length is less than max length than add padding 
    if len(sequence) < MAX_LENGTH:
        sequence = sequence + (
            [PAD_ID] *
            (MAX_LENGTH - len(sequence)))
    # If sequence is longer than MAX_LENGTH
    # Yes information got loss but if max length increase than more info retain and more compuation and memeory and rnn become slower.
    # Experiment with differ max length and compare validation losss and training time.
    else:
        sequence = sequence[:MAX_LENGTH]
    # if length is equal to the max length then retain the same encode
    val_padded_texts.append(sequence)
# converting your Python lists into PyTorch tensors so that PyTorch layer works
X_val = torch.tensor(
    val_padded_texts,
    dtype=torch.long)
X_train = torch.tensor(
    train_padded_texts,
    dtype=torch.long)
Y_train = torch.tensor(
    train_df["label"].values,
    dtype=torch.float32
)

Y_val = torch.tensor(
    val_df["label"].values,
    dtype=torch.float32
)
# create datasets
train_dataset = TensorDataset(
    X_train,
    Y_train
)

val_dataset = TensorDataset(
    X_val,
    Y_val
)
# load
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)
# Because X is your data, not something needed when creating the architecture.
model=Models(vocab_size,PAD_ID)
loss_function=nn.BCEWithLogitsLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-4)
num_epochs = 50

patience = 3
patience_counter = 0

best_val_loss = float("inf")

for epoch in range(num_epochs):
    model.train()
    for X, Y in train_loader:
        prediction=model(X)
        # give [32,1] pred.shape but dataloader give Y.shape=[32] so have to unsqueeze y 
        Y=Y.unsqueeze(1)
        loss=loss_function(prediction,Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, Y in val_loader:
            prediction = model(X)
            Y=Y.unsqueeze(1)
            loss = loss_function(prediction, Y)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
model.load_state_dict(torch.load("best_model.pth"))
def pad_sequence(sequence, max_length):

    if len(sequence) < max_length:
        sequence = sequence + (
            [PAD_ID] * (max_length - len(sequence))
        )
    else:
        sequence = sequence[:max_length]

    return sequence
def predict_sentiment(review, model, vocab):

    # 1. Tokenize
    tokens = tokenize(review)

    # 2. Encode using training vocabulary
    encoded = [
        vocab.get(token, UNK_ID)
        for token in tokens
    ]

    # 3. Pad / truncate
    encoded = pad_sequence(
        encoded,
        MAX_LENGTH
    )

    # 4. Convert to tensor
    X = torch.tensor(
        [encoded],
        dtype=torch.long
    )

    # 5. Prediction
    model.eval()

    with torch.no_grad():

        logit = model(X)

        probability = torch.sigmoid(logit).item()

    # 6. Classification
    if probability >= 0.5:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    return sentiment, probability

review = """
This movie was absolutely fantastic. The acting was brilliant,
the story was engaging and I enjoyed every single minute of it.
I would definitely recommend this movie.
"""

sentiment, probability = predict_sentiment(
    review,
    model,
    vocab
)

print("Sentiment:", sentiment)
print("Probability:", probability)






# IMDb DataFrame
#       ↓
# Train / Validation split
#       ↓
# ┌──────────────┬──────────────┐
# │ Train        │ Validation   │
# │              │              │
# │ Build vocab  │ Don't build  │
# │ from this    │ vocabulary   │
# └──────────────┴──────────────┘
#       ↓
# Encode both using TRAIN vocab
#       ↓
# Padding
#       ↓
# Tensor
#       ↓
# Dataset
#       ↓
# DataLoader
#       ↓
# Training

    #           IMDb
    #             ↓
    #       Tokenization
    #             ↓
    #     Train/Validation/Test
    #             ↓
    #      Training Vocabulary
    #             ↓
    #          Encoding
    #             ↓
    #       Padding 200
    #             ↓
    #          Tensor
    #             ↓
    #        DataLoader
    #             ↓
    #       ┌─────┼─────┐
    #       ↓     ↓     ↓
    #     RNN    LSTM   GRU
    #       ↓     ↓     ↓
    #       FC    FC    FC
    #       ↓     ↓     ↓
    #    Logit  Logit  Logit
    #       ↓     ↓     ↓
    #    Sigmoid Sigmoid Sigmoid
    #       ↓     ↓     ↓
    #    Sentiment prediction

