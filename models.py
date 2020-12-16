import torch.nn as nn

class linearNN(nn.Module):
    def __init__(self, inp_size, out_size, widths, init=None):
        super().__init__()
        self.widths = widths
        self.inp_layer = nn.Linear(inp_size, widths[0], bias=False)
        self.layers = nn.ModuleList(nn.Linear(widths[i - 1], widths[i], bias=False)
                                    for i in range(1, len(widths)))
        self.out_layer = nn.Linear(widths[-1], out_size, bias=False)

    def forward(self, inp):
        x = self.inp_layer(inp)
        for i in range(len(self.widths) - 1):
            x = self.layers[i](x)
        x = self.out_layer(x)
        return x


class linearNN_with_activation(linearNN):
    def __init__(self, inp_size, out_size, widths, activation, init=None):
        super().__init__(inp_size, out_size, widths, init)
        self.activation = activation

    def forward(self, inp):
        x = self.inp_layer(inp)
        for i in range(len(self.widths) - 1):
            x = self.layers[i](self.activation(x))
        x = self.out_layer(self.activation(x))
        return x
