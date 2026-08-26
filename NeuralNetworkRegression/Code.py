import torch
import torch.nn as nn
import sys
print(sys.executable)
print(sys.version)
X=torch.tensor([[0.],[1.],[2.],[3.],[4.],[5.]])
Y=torch.tensor([[2.],[5.],[8.],[11.],[14.],[17.]])
# Note Weight and Bias are floating point type so input must be compatible floating-point dtype.
# I am creating my own neural-network model using PyTorch's neural-network framework by using class
class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # 1 input → 3 neurons and nn.Linear(1,3) → 3 weights and 3 biases.
        self.layer1=nn.Linear(1,3)
        self.layer2=nn.Linear(3,3)
        self.layer3=nn.Linear(3,1)
    def forward(self,X):
        z1=self.layer1(X)
        # nonlinear activation functions
        a1=torch.relu(z1)
        # each neuron in the second layer receives all 3 outputs from the previous layer.
        z2=self.layer2(a1)
        a2=torch.relu(z2)
        # final neuron combines the three values and give predication
        z3=self.layer3(a2)
        y_pred= z3
        return y_pred
model=NeuralNetwork()
loss_function=nn.MSELoss()
# Use SGD to update all parameters of my model with learning rate 0.01
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
# training
# One epoch here means one complete pass over your six training examples. We are using full Batch Gradient Descent
for epoch in range(1000):
    prediction=model(X)
    # avg of loss of 6 examples
    loss=loss_function(prediction,Y)
    # optimizer.zero_grad()
    optimizer.zero_grad()
    # Calc new gradient
    loss.backward()
    # actually updates the weights and biases.
    optimizer.step()
    if epoch%100==0:
        print(
            "Epoch:",epoch,
            "Loss:",loss.item()
        )
# test using 6.0 value
test_input = torch.tensor([[6.0]])
prediction = model(test_input)

print("\nPrediction for 6:")
print(prediction)
print(X)
