from torch import nn
from models.modified_resnet import ResnetEncoder


class SimCLRMain(nn.Module):
    def __init__(self,
                 encoder_dim=2048,
                 output_dim=128,
                 encoder_model='resnet50',
                 num_proj_layer=2):
        super(SimCLRMain, self).__init__()

        self.encoder_dim = encoder_dim
        self.output_dim = output_dim
        self.num_proj_layer = num_proj_layer

        if encoder_model == 'resnet50':
            self.f = ResnetEncoder()
        else:
            raise NotImplementedError

        proj_layers = [nn.Flatten()]
        for i in range(self.num_proj_layer):
            # For non-final layers, use bias and relu
            if i != self.num_proj_layer - 1:
                proj_layers.extend([
                    nn.Linear(self.encoder_dim, self.encoder_dim, bias=False),
                    nn.BatchNorm1d(self.encoder_dim),
                    nn.ReLU()
                ])
            else:
                proj_layers.extend([
                    nn.Linear(self.encoder_dim, self.output_dim, bias=False),
                    nn.BatchNorm1d(self.output_dim)
                ])
        self.g = nn.Sequential(*proj_layers)

    def forward(self, x):
        h = self.f(x)
        z = self.g(h)
        return h, z
