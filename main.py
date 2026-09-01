import numpy as np
import pandas as pd
import torch

# ... код из предыдущего раздела здесь

def sigmoid(x):
    """Наша функция активации: f(x) = 1 / (1 + e^(-x))"""
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        # Взвешиваем входы, добавляем смещение, затем используем функцию активации
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

class OurNeuralNetwork:
    '''
    Нейронная сеть с:
    - 2 входами
    - скрытым слоем с 2 нейронами (h1, h2)
    - выходным слоем с 1 нейроном (o1)
    Каждый нейрон имеет одинаковые веса и смещение:
    - w = [0, 1]
    - b = 0
    '''
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)
  
    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        # Входы для o1 - это выходы из h1 и h2
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1

network = OurNeuralNetwork()
np_array = np.array([2, 3])
print(network.feedforward(np_array))

x = torch.rand(5, 3)
print(x)

x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_np) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_np, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")


tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=0) #y=0, x=1
print(t1)


data = np.load('tile_000002_height.npy') 
df = pd.DataFrame(data)

df.to_csv('file3.csv', index=False)
