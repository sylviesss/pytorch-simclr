from torch import nn
from models.resnets import ResnetEncoder
from models.resnets import Dropout, ResnetEncoderDropout


class SimCLRMain(nn.Module):
    def __init__(self,
                 low_quality_img,
                 configs,
                 encoder_model='no_dropout',
                 num_proj_layer=2,
                 modified_network=False):
        super(SimCLRMain, self).__init__()

        self.encoder_dim = configs["feature_dim"]
        self.output_dim = configs["compressed_dim"]
        self.num_proj_layer = num_proj_layer

        if encoder_model == 'no_dropout':
            self.f = ResnetEncoder(low_quality_img=low_quality_img)
        elif encoder_model == 'dropout':
            self.f = ResnetEncoderDropout(drop_prob=configs['drop_prob'],
                                          low_quality_img=low_quality_img)
        else:
            raise NotImplementedError

        proj_layers = [nn.Flatten()]
        for i in range(self.num_proj_layer):
            if i != self.num_proj_layer - 1:
                # For non-final layers, use bias and relu
                proj_layers.extend([
                    nn.Linear(self.encoder_dim, self.encoder_dim, bias=True),
                    nn.BatchNorm1d(self.encoder_dim),
                    nn.ReLU(inplace=True)
                ])
                if encoder_model == 'dropout':
                    proj_layers.append(Dropout(p=configs['drop_prob']))
            else:
                # Final layer
                if not modified_network:
                    proj_layers.extend([
                        nn.Linear(self.encoder_dim, self.output_dim, bias=False),
                        nn.BatchNorm1d(self.output_dim)
                        ])
                else:
                    # Add Relu to ensure all raw logits are positive
                    proj_layers.extend([
                        nn.Linear(self.encoder_dim, self.output_dim, bias=False),
                        nn.BatchNorm1d(self.output_dim),
                        nn.ReLU()
                        ])
        self.g = nn.Sequential(*proj_layers)

    def forward(self, x):
        h = self.f(x)
        z = self.g(h)
        return h, z
