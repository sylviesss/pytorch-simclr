from models.simclr import SimCLRMain
import torch
import torch.nn as nn


class SimCLRFineTune(SimCLRMain):
    def __init__(self, pretrained_path, device, n_classes=10):
        super(SimCLRFineTune, self).__init__()
        self.n_classes = n_classes
        self.pretrained_path = pretrained_path
        self.device = device

        # Load pretrained model
        pretrained_model = torch.jit.load(self.pretrained_path,
                                          map_location=self.device)
        pretrained_params = pretrained_model.state_dict()
        self.load_state_dict(pretrained_params)

        self.supervised_head = nn.Sequential(nn.Flatten(),
                                             nn.Linear(self.encoder_dim, self.n_classes))

    def forward(self, x):
        h = self.f(x)
        score = self.supervised_head(h)
        return score