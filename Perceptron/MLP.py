import numpy as np
def ReLU(x):
    return np.maximum(0,x)
def stepFunction(x):
    if x>=0:
        return 1
    else:
        return 0
def calculateOutput(InputArr,WeightArr,BiasArr):
    return np.dot(InputArr,WeightArr)+BiasArr
    

InputArr=np.array([2,3])
BiasArr=np.array([0.1,0.2])
WeightArr=np.array([[0.5,0.4],[0.2,0.7]])
#neuron 1
z1=calculateOutput(InputArr,WeightArr[0],BiasArr[0])
z2=calculateOutput(InputArr,WeightArr[1],BiasArr[1])
ActArr=np.array([ReLU(z1),ReLU(z2)])
biasLayer2=-2
wtlayer2=np.array([0.6,0.8])
pred=stepFunction(calculateOutput(ActArr,wtlayer2,biasLayer2))
print(pred)

