from torch.utils.data import DataLoader
from torchvision import datasets
import torch
import json
from argparse import ArgumentParser
from data import CIFAR10pair, compose_augmentation_train
from mine import MineNet, train_mine


def create_parser(configs):
    parser = ArgumentParser()
    parser.add_argument('--mine_bs',
                        default=configs['mine_bs'],
                        type=int,
                        help='batch size used for training the MINE.')
    parser.add_argument("--input_size",
                        default=configs["feature_dim"]*2,
                        type=int,
                        help="combined hidden dimension that is fed into the MINE.")
    parser.add_argument("simclr_model_path",
                        type=str,
                        help="path to the pretrained simclr model.")
    parser.add_argument("--mine_lr",
                        default=configs["mine_lr"],
                        type=float,
                        help="learning rate when training the MINE.")
    parser.add_argument("--mine_hidden_dim",
                        default=configs["mine_hidden_dim"],
                        type=int,
                        help="hidden dimension in the MINE.")
    return parser


if __name__ == "__main__":
    # args.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open('utils/configs.json') as f:
        configs = json.load(f)

    parser = create_parser(configs)
    args = parser.parse_args()

    mine_bs = args.mine_bs
    ds_mine_joint = CIFAR10pair(root=configs["data_dir"],
                                train=True,
                                transform=compose_augmentation_train(
                                    configs["cifar10_size"],
                                    mean_std=configs["cifar10_mean_std"]
                                ),
                                download=False)
    loader_mine_joint = DataLoader(dataset=ds_mine_joint,
                                   batch_size=mine_bs,
                                   shuffle=True)
    ds_mine_marginal = datasets.CIFAR10(root=configs["data_dir"],
                                        train=True,
                                        transform=compose_augmentation_train(
                                            configs["cifar10_size"],
                                            mean_std=configs["cifar10_mean_std"]
                                        ),
                                        download=False)
    loader_mine_marginal = DataLoader(dataset=ds_mine_marginal,
                                      batch_size=mine_bs,
                                      shuffle=True)

    mine_model = MineNet(input_size=args.input_size, hidden_size=args.mine_hidden_dim)
    mine_optim = torch.optim.Adam(mine_model.parameters(), lr=args.mine_lr)
    simclr_model = torch.jit.load(args.simclr_model_path)

    loss_mine, mi_mine = train_mine(
        loader_mine_joint,
        loader_mine_marginal,
        simclr_model,
        device,
        mine_model,
        mine_optim,
        n_iter=5000)

