from data import get_augmented_dataloader
from models.simclr import SimCLRMain
from utils.model_utils import train_simclr
import torch
import matplotlib.pyplot as plt
import numpy as np
import json


# need to permute the numpy image in order to display it correctly
def show(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# Denormalize images for visualization
def denorm(x, channels=None, w=None, h=None, resize=False):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    if resize:
        if channels is None or w is None or h is None:
            print('Number of channels, width and height must be provided for resize.')
        x = x.view(x.size(0), channels, w, h)
    return x


if __name__ == '__main__':

    # Set a seed
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)

    # args.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open('utils/configs.json') as f:
        configs = json.load(f)

    # Get data
    loader_train_simclr = get_augmented_dataloader(batch_size=configs['batch_size_small'],
                                                   train_mode='pretrain')
    loader_train_clf, loader_eval_clf = get_augmented_dataloader(
        batch_size=configs['batch_size'],
        train_mode='lin_eval'
    )
    loader_test_clf = get_augmented_dataloader(batch_size=configs['batch_size'],
                                               train_mode='test')

    simclr_model = SimCLRMain()
    base_optim = torch.optim.Adam(simclr_model.parameters(), lr=configs['lr'], weight_decay=configs['wt_decay'])
    train_simclr(model=simclr_model,
                 optimizer=base_optim,
                 loader_train=loader_train_simclr,
                 device=device,
                 n_epochs=configs['n_epoch'],
                 accum_steps=configs['accum_steps'],
                 temperature=configs['temp']
                 )
    # TODO: Create a flexible training procedure, so we can choose among ['pretrain', 'lin_eval', 'fine_tune'] using
    #  args with one training file features_train, targets_train = feature_extraction( simclr_model=simclr_model,
    #  device=device, loader_lin_eval=loader_train_clf)
