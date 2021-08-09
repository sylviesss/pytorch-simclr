from torch import nn
from models.resnets import ResnetEncoder
from models.resnets import Dropout, ResnetEncoderDropout


class SimCLRMain(nn.Module):
    def __init__(self,
                 low_quality_img,
                 configs,
                 encoder_model='no_dropout',
                 num_proj_layer=2):
        super(SimCLRMain, self).__init__()

        self.encoder_dim = configs["feature_dim"]
        self.output_dim = configs["compressed_dim"]
        self.num_proj_layer = num_proj_layer
        self.dropout = Dropout(p=configs['drop_prob'], inplace=False)

        if encoder_model == 'no_dropout':
            self.f = ResnetEncoder(low_quality_img=low_quality_img)
        elif encoder_model == 'dropout':
            self.f = ResnetEncoderDropout(drop_prob=configs['drop_prob'],
                                          low_quality_img=low_quality_img)
        else:
            raise NotImplementedError

        proj_layers = nn.Sequential()
        proj_layers.add_module("g_flatten", nn.Flatten())
        for i in range(self.num_proj_layer):
            # For non-final layers, use bias and relu
            if i != self.num_proj_layer - 1:
                proj_layers.add_module("g_linear" + str(i), nn.Linear(self.encoder_dim, self.encoder_dim))
                proj_layers.add_module("g_bn" + str(i), nn.BatchNorm1d(self.encoder_dim))
                proj_layers.add_module("g_relu" + str(i), nn.ReLU(inplace=True))
                if encoder_model == 'dropout':
                    proj_layers.add_module("g_dropout" + str(i), self.dropout)
            else:
                proj_layers.add_module("g_linear" + str(i), nn.Linear(self.encoder_dim, self.output_dim, bias=False))
                proj_layers.add_module("g_bn" + str(i), nn.BatchNorm1d(self.output_dim))

        self.g = proj_layers

    def forward(self, x):
        h = self.f(x)
        z = self.g(h)
        return h, z

    def forward(self, x):
        h = self.f(x)
        z = self.g(h)
        return h, z
