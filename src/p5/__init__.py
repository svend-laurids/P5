import torch
from torch import nn
from .CLAF import *

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(12, 9),
            nn.ReLU(),
            nn.Linear(9, 6),
            CLAF(6, 10, sstep),
            nn.Linear(6, 3),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def main() -> None:
    model = NeuralNetwork()

    X = torch.rand(2, 12)
    logits = model(X)
    softmax = nn.Softmax(dim=1) 
    pred_prob = softmax(logits)

    print(pred_prob)

    


    
