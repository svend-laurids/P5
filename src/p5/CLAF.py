import torch

sstep = lambda n : (lambda x : torch.where(x <= 0.0, 0.0, torch.where(x >= 1.0, 1.0, x**(2*n)*(3-2*(x**n)))))
cos = lambda n : (lambda x : torch.where(x <= 0.0, 0.0, torch.where(x >= 1.0, 1.0, (1-cos(2*n*pi*x))/2)))

class CLAF(torch.nn.Module):
    def __init__(self, n_inputs, n_weights, function):
        super().__init__()
        self.n_weights = n_weights
        self.n_inputs = n_inputs
        self.function = function

        self.softmax = torch.nn.Softmax()
        self.weights = torch.rand(n_weights, requires_grad=True)

    def __transform_rows(self, X):
        return torch.stack([self.function(n+1)(X[n]) for n in range(self.n_weights)], dim=0)
    
    def forward(self, x):
        X = x.repeat(n_weights, 1) 
        self.__transform_rows(X)
        return self.softmax(weights) @ X


