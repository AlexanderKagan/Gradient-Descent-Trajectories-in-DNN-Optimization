import torch.nn as nn
import torch


class linearNN(nn.Module):
    def __init__(self, inp_size, out_size, widths, init=None):
        super().__init__()

        self.widths = widths
        self.inp_layer = nn.Linear(inp_size, widths[0], bias=False)
        self.layers = nn.ModuleList(nn.Linear(widths[i - 1], widths[i], bias=False)
                                    for i in range(1, len(widths)))
        self.out_layer = nn.Linear(widths[-1], out_size, bias=False)

        def init_layer(init, layer):
            if isinstance(init, int):
                shape = layer.weight.shape
                assert init <= min(shape)
                new_weights = torch.zeros(shape)
                new_weights[:init, :init] = torch.randn((init, init))
                layer.weight.data = new_weights[torch.randperm(shape[0])]

            elif init == 'all_ones':
                layer.weight.data.fill_(1.)

            else:
                raise NameError('Method not implemented')

        if init is not None:
            init_layer(init, self.inp_layer)
            init_layer(init, self.out_layer)
            for i in range(len(self.layers)):
                init_layer(init, self.layers[i])

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
