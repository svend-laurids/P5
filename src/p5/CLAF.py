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
    
    def forward(self, X):
        Xpand = X.unsqueeze(2).repeat(1, 1, self.n_weights)
        Xview = Xpand.view(Xpand.shape[0]*Xpand.shape[1], Xpand.shape[2]).transpose(0, 1)
        for i in range(Xview.shape[0]):
            Xview[i] = self.function(i+1)(Xview[i])
        return Xpand @ self.softmax(self.weights)
