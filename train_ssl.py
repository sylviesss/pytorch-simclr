from data import AugmentedLoader
from models.ssl import SimCLRFineTune
from utils.model_utils import train_ssl, test_ssl
from argparse import ArgumentParser

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

    parser = ArgumentParser()
    # Can be a model pretrained on the same dataset or a different dataset for transfer learning.
    parser.add_argument('--model_path',
                        type=str,
                        help='Path of pretrained model')
    args = parser.parse_args()

    # According to the paper, the learning rate can be configured this way:
    # lr_ssl = 0.05 * configs['batch_size_small'] / 256

    # Load data.
    # Need to change ssl_label_size to 1 when doing transfer learning 
    loader_train_ssl = AugmentedLoader(dataset_name='cifar10',
                                       train_mode='fine_tune',
                                       batch_size=configs['batch_size_small'],
                                       cfgs=configs).loader
    loader_test = AugmentedLoader(dataset_name='cifar10',
                                  train_mode='test',
                                  batch_size=configs['batch_size_small'],
                                  cfgs=configs)
    simclr_ft = SimCLRFineTune(args.model_path, configs=configs, low_quality_img=True)
    # SGD with Nesterov momentum
    fine_tune_optim = torch.optim.SGD(simclr_ft.parameters(), lr=configs["lr_ssl"], momentum=configs['momentum_ssl'],
                                      nesterov=True)

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
