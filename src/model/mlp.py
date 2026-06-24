import torch.nn as nn

def MLP(channels: list, do_bn=True, linear=False, dropout=False, p=0.5):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        if linear:
            layer = nn.Linear(channels[i - 1], channels[i])
        else:
            layer = nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True)
        layers.append(layer)

        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            if dropout:
                layers.append(nn.Dropout(p=p))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers) 