import torch
import torch.nn as nn
import sys
print(sys.executable)
print(sys.version)
X=torch.tensor([[0.],[1.],[2.],[3.],[4.],[5.]])
Y=torch.tensor([[2.],[5.],[8.],[11.],[14.],[17.]])
class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(1,3)
        self.layer2=nn.Linear(3,3)
        self.layer3=nn.Linear(3,1)
    def forward(self,X):
        z1=self.layer1(X)
        a1=torch.relu(z1)
        z2=self.layer2(a1)
        a2=torch.relu(z2)
        z3=self.layer3(a2)
        y_pred= z3
        return y_pred
model=NeuralNetwork()
loss_function=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
# training
print(model)
print(X)
print("Shape:", X.shape)
print("Dtype:", X.dtype)
for epoch in range(1000):
    prediction=model(X)
    loss=loss_function(prediction,Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch%100==0:
        print(
            "Epoch:",epoch,
            "Loss:",loss.item()
        )
test_input = torch.tensor([[6.0]])

prediction = model(test_input)

print("\nPrediction for 6:")
print(prediction)
print(X)
