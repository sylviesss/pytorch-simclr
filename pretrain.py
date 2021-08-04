from data import AugmentedLoader
from models.simclr import SimCLRMain
from utils.model_utils import train_simclr
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
                        default=configs['batch_size_small'],
                        type=int)
    parser.add_argument('--dataset',
                        default='cifar10',
                        type=str,
                        help='Choose from "cifar10" or "stl10"')
    parser.add_argument('--path_for_saving',
                        default=configs['doc_path'],
                        type=str,
                        help='path for saving models and checkpoints (should include / at the end)')
    parser.add_argument('--resume_training_path',
                        default=None,
                        type=str,
                        help='path of the model that we want to resume training with')
    parser.add_argument('--encoder_model',
                        default='no_dropout',
                        type=str,
                        help='Indicate if we want to train a SimCLR model with dropouts')
    parser.add_argument('--temp',
                        default=configs['temp'],
                        type=float,
                        help='Temperature in NT-XENT loss')
    parser.add_argument('--modified_loss',
                        default=False,
                        type=bool,
                        help='Indicate if we want to use the modified loss during training')
    parser.add_argument('--save_ckpt',
                        default=True,
                        type=bool)
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
    loader_train_simclr = AugmentedLoader(dataset_name=args.dataset,
                                          train_mode='pretrain',
                                          batch_size=args.batch_size,
                                          cfgs=configs)

    simclr_model = SimCLRMain(low_quality_img=args.dataset == 'cifar10',
                              configs=configs,
                              encoder_model=args.encoder_model)
    base_optim = torch.optim.Adam(simclr_model.parameters(), lr=configs['lr'], weight_decay=configs['wt_decay'])
    train_simclr(model=simclr_model,
                 optimizer=base_optim,
                 loader_train=loader_train_simclr.loader,
                 loader_val=loader_train_simclr.valid_loader,
                 device=device,
                 n_epochs=args.n_epoch,
                 save_every=args.save_every,
                 temperature=args.temp,
                 accum_steps=args.accum_steps,
                 modified_loss=args.modified_loss,  # Use modified loss
                 save_ckpt=args.save_ckpt,
                 path_ext=configs['doc_path_modified_loss'],  # Change this to be more flexible
                 dataset_name=args.dataset,
                 checkpt_path=args.resume_training_path)

