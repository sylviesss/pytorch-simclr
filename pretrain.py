from data import get_augmented_dataloader
from models.simclr import SimCLRMain
from utils.model_utils import train_simclr, train_simclr_no_accum
import torch
import json
from argparse import ArgumentParser


def create_parser(configs):
    parser = ArgumentParser()
    parser.add_argument('--n_epoch',
                        default=configs['n_epoch'],
                        type=int,
                        help='number of epochs to train')
    parser.add_argument('--accum_steps',
                        default=configs['accum_steps'],
                        type=int,
                        help='number of gradient accumulation steps; total batch size is 64*accum_steps')
    parser.add_argument('--save_every',
                        default=configs['save_ckpt_every'],
                        type=int,
                        help='frequency for saving checkpoint during training')
    parser.add_argument('--batch_size',
                        default=configs['default_batch_size'],
                        type=int)
    return parser


if __name__ == '__main__':

    # Set a seed
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)

    # args.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open('utils/configs.json') as f:
        configs = json.load(f)

    parser = create_parser(configs)
    args = parser.parse_args()

    # Get data
    loader_train_simclr = get_augmented_dataloader(batch_size=configs['batch_size_small'],
                                                   train_mode='pretrain')

    simclr_model = SimCLRMain()
    base_optim = torch.optim.Adam(simclr_model.parameters(), lr=configs['lr'], weight_decay=configs['wt_decay'])
    train_simclr_no_accum(model=simclr_model,
                          optimizer=base_optim,
                          loader_train=loader_train_simclr,
                          device=device,
                          n_epochs=args.n_epoch,
                          save_every=args.save_every,
                          batch_size=args.batch_size,
                          temperature=configs['temp']
                          )
    # TODO: Create a flexible training procedure, so we can choose among ['pretrain', 'lin_eval', 'fine_tune'] using
    #  args with one training file features_train, targets_train = feature_extraction( simclr_model=simclr_model,
    #  device=device, loader_lin_eval=loader_train_clf)
