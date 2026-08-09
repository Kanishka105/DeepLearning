import numpy as np


def Relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# XOR dataset
arr = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

output = np.array([
    [0],
    [1],
    [1],
    [0]
])


# Biases
b1 = np.zeros((1, 2))
b2 = np.zeros((1, 1))


# Weights
wt1 = np.random.rand(2, 2)
wt2 = np.random.rand(2, 1)


learning_rate = 0.1
epochs = 10000


for e in range(epochs):

 

    z1 = np.dot(arr, wt1) + b1
    h1 = Relu(z1)

    z2 = np.dot(h1, wt2) + b2
    h2 = sigmoid(z2)



    loss = -np.mean(
        output * np.log(h2 + 1e-8) +
        (1 - output) * np.log(1 - h2 + 1e-8)
    )


    dZ2 = h2 - output

    dW2 = h1.T @ dZ2

    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ wt2.T

    dZ1 = dA1 * (z1 > 0)

    dW1 = arr.T @ dZ1

    db1 = np.sum(dZ1, axis=0, keepdims=True)



    wt2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    wt1 -= learning_rate * dW1
    b1 -= learning_rate * db1



    if e % 1000 == 0:
        print("Epoch:", e, "Loss:", loss)

predictions = (h2 >= 0.5).astype(int)

print("\nPredictions:")
print(predictions)
