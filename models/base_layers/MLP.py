import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, base_neurons=[512, 512, 512], out_dim=10, softmax=False):
        """ 
        Multi-layer perceptron 
        """
        super().__init__()

        layers = []
        for layer_idx, (inp_neurons, out_neurons) in enumerate(
            zip(base_neurons[:-1], base_neurons[1:])
        ):
            layers.append(nn.Linear(inp_neurons, out_neurons))
            layers.append(nn.ReLU())
        self.final_layer = nn.Linear(out_neurons, out_dim)
        self.decoder = nn.Sequential(*layers)
        self.apply_softmax = softmax
        if self.apply_softmax:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        input = x
        decoded = self.decoder(input)
        out = self.final_layer(decoded)
        if self.apply_softmax:
            out = self.softmax(out)
        return out