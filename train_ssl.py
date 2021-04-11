from data import get_augmented_dataloader
from models.ssl import SimCLRFineTune
from utils.model_utils import train_ssl, test_ssl

import torch
import json


if __name__ == '__main__':
    # Set a seed.
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)

    # Args.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open('utils/configs.json') as f:
        configs = json.load(f)

    # According to the paper, the learning rate can be configured this way
    # TODO: stabilize this training process
    lr_ssl = 0.05 * configs['batch_size_small'] / 256

    # Load data.
    loader_train_ssl = get_augmented_dataloader(
        batch_size=configs['batch_size'],
        train_mode='fine_tune',
        ssl_label_size=configs['ssl_label_size'])
    loader_test = get_augmented_dataloader(
        batch_size=configs['batch_size'],
        train_mode='test'
    )

    simclr_ft = SimCLRFineTune('/content/simclr_model_bs512_nepoch1.pth', device=device)
    # SGD with Nesterov momentum
    fine_tune_optim = torch.optim.SGD(simclr_ft.parameters(), lr=lr_ssl, momentum=0.9, nesterov=True)

    train_ssl(
        simclr_ft=simclr_ft,
        optimizer=fine_tune_optim,
        n_epochs=configs['n_epoch_ssl'],
        device=device,
        loader_train=loader_train_ssl
    )

    test_ssl(
        simclr_ft=simclr_ft,
        device=device,
        loader_test=loader_test,
        return_loss_accuracy=False
    )