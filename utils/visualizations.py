from sklearn import manifold
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import torch
import numpy as np
import pandas as pd
from data import AugmentedLoader


def plot_three_KDEs_positive(contrastive_metrics_h: dict,
                             contrastive_metrics_z: dict,
                             supervised_metrics: dict,
                             metric: str,
                             title="",
                             fig_size=(12, 8)):
    """
    Using metrics calculated with the function get_similarity_metrics_pairs
    (using POSITIVE PAIRS),
    show 3 kernel density estimation plots in one image.
    Args:
     contrastive_metrics_h: Metrics calculated with features taken from the
                            layer before the projection head in the pretrained
                            contrastive model.
     contrastive_metrics_z: Metrics calculated with features taken from the
                            layer after the projection head in the pretrained
                            contrastive model.
     supervised_metrics: Metrics calculated with features taken from a trained
                 supervised model.
     metric: choose from ['ned', 'cc', 'nmi', 'cos'].
     title: title of the plot.
     fig_size: size of matplotlib figure.
    """
    pos_metric = 'pos_' + metric

    sns.set_style('darkgrid')
    fig, ax = plt.subplots(figsize=fig_size)
    sns.kdeplot(contrastive_metrics_h[pos_metric], ax=ax, shade=True,
                label='contrastive - h')
    sns.kdeplot(contrastive_metrics_z[pos_metric], ax=ax, shade=True,
                label='contrastive - z')
    sns.kdeplot(supervised_metrics[pos_metric], ax=ax, shade=True,
                label='supervised')

    plt.legend()
    plt.title("KDE Plots - " + title)
    plt.show()


def plot_pos_neg_metrics(contrastive_metrics: dict,
                         supervised_metrics: dict,
                         metric: str,
                         fig_size=(14, 6),
                         title=""):
    """
    Plot KDEs of a chosen metric for positive and negative pairs for
    1) contrastive model,
    2) supervised model with outputs from the function
    get_similarity_metrics_pairs.
    Args:
     contrastive_metrics: first output from get_similarity_metrics_pairs.
     supervised_metrics: second output from get_similarity_metrics_pairs.
     metric: choose from ['ned', 'cc', 'nmi', 'cos'].
     title: title of the plot.
     fig_size: size of matplotlib figure.
    """
    pos_metric = 'pos_' + metric
    neg_metric = 'neg_' + metric
    contr_title = title + '- Contrastive'
    supv_title = title + '- Supervised'

    sns.set_style('darkgrid')
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=fig_size)
    sns.kdeplot(contrastive_metrics[pos_metric], ax=ax1, shade=True,
                label='contrastive - positive')
    sns.kdeplot(contrastive_metrics[neg_metric], ax=ax1, shade=True,
                label='contrastive - negative')
    ax1.set_title(contr_title)
    ax1.legend()

    sns.kdeplot(supervised_metrics[pos_metric], ax=ax2, shade=True,
                label='supervised - positive')
    sns.kdeplot(supervised_metrics[neg_metric], ax=ax2, shade=True,
                label='supervised - negative')
    ax2.set_title(supv_title)
    ax2.legend()
    plt.show()


def get_dataloader_tsne(train_mode,
                        batch_size,
                        dataset,
                        cfgs):
    """
    Helper function for get_tsne_representations and get_tsne_representations_simclr
    :param train_mode: "pretrain"/"valid"/"test"/"supervised"
    :param batch_size: number of datapoints we want to show in the TSNE plot
    :param dataset: "cifar10"/"stl10"
    :param cfgs: configurations (configs.json)
    :return: a dataloader
    """
    loader = AugmentedLoader(dataset_name=dataset,
                             train_mode=train_mode,
                             batch_size=batch_size,
                             cfgs=cfgs)
    return loader.loader


def get_tsne_representations_simclr(model,
                                    data_loader,
                                    device,
                                    perplexity=30,
                                    show_plot=True,
                                    feat_used='h'):
    """
    Calculate TSNE vectors using model and data from data_loader.
    :param model: pretrained model.
    :param data_loader: result from get_dataloader_tsne.
    :param device: "cuda"/"cpu".
    :param perplexity: (approx.) number of neighbors in the tsne plot
    :param show_plot: show plot if True.
    :param feat_used: "h" (hidden representations) / "z" (final representations)
    """
    # sample_inputs1, sample_inputs2, sample_label = next(iter(data_loader))
    sample_inputs1, _, sample_label = next(iter(data_loader))
    model = model.to(device=device)
    # Get latent representations of test data
    model.eval()
    with torch.no_grad():
        test_data = torch.Tensor(sample_inputs1)
        test_data = test_data.to(device=device, dtype=torch.float32)
        if feat_used == 'h':
            feat, _ = model(test_data)
        else:
            # use features from the final hidden layer
            _, feat = model(test_data)
        feat = torch.flatten(feat, start_dim=1, end_dim=-1)
    feat = feat.cpu().numpy()
    print("Obtained learned representations ... ")

    tsne = manifold.TSNE(n_components=2, init='pca', perplexity=perplexity,
                         random_state=0)
    X_tsne = tsne.fit_transform(feat)
    img_label = sample_label.numpy()
    print("Obtained T-SNE PCA embeddings...")

    # Plot images according to t-sne embedding
    if show_plot:
        fig, ax = plt.subplots(figsize=(12, 10))
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'grey', 'orange', 'purple']
        target_names = range(10)
        target_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        for i, c, label in zip(target_names, colors, target_labels):
            plt.scatter(X_tsne[img_label == i, 0], X_tsne[img_label == i, 1],
                        c=c, label=label)
        plt.legend(loc='best')
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"T-SNE plot: perplexity={perplexity}")
        plt.show()
    return X_tsne


def get_tsne_representations(model,
                             data_loader,
                             device,
                             perplexity=30,
                             show_plot=True):
    # sample_inputs1, sample_inputs2, sample_label = next(iter(data_loader))
    sample_inputs1, sample_label = next(iter(data_loader))
    model = model.to(device=device)
    # Get latent representations of test data
    model.eval()
    with torch.no_grad():
        test_data = torch.Tensor(sample_inputs1)
        test_data = test_data.to(device=device, dtype=torch.float32)
        feat = model(test_data)
        feat = torch.flatten(feat, start_dim=1, end_dim=-1)
    feat = feat.cpu().numpy()
    print("Obtained learned representations ... ")

    tsne = manifold.TSNE(n_components=2, init='pca', perplexity=perplexity,
                         random_state=0)
    X_tsne = tsne.fit_transform(feat)
    img_label = sample_label.numpy()
    print("Obtained T-SNE PCA embeddings...")

    # Plot images according to t-sne embedding
    if show_plot:
        fig, ax = plt.subplots(figsize=(12, 10))
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'grey', 'orange', 'purple']
        target_names = range(10)
        target_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        for i, c, label in zip(target_names, colors, target_labels):
            plt.scatter(X_tsne[img_label == i, 0], X_tsne[img_label == i, 1],
                        c=c, label=label)
        plt.legend(loc='best')
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"T-SNE plot: perplexity={perplexity}")
        plt.show()
    return X_tsne


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


def plot_loss_acc(loss: list,
                  accuracy: list,
                  fig_size=(14, 6),
                  title="",
                  save_plot=False):
    """
    Plot loss and accuracy by epoch side by side.
    """
    ep = [int(i) for i in list(range(1, len(loss) + 1))]
    df = pd.DataFrame({'Loss': loss, 'Accuracy': accuracy, 'Epoch': ep})
    sns.set_style('darkgrid')
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=fig_size)
    sns.lineplot(data=df, x='Epoch', y='Loss', ax=ax1)
    ax1.set_title('Loss vs. # of Epoch')

    sns.lineplot(data=df, x='Epoch', y='Accuracy', ax=ax2)
    ax2.set_title('Accuracy(%) vs. # of Epoch')
    plt.show()
    if save_plot:
        plt.savefig(title + ".png")
    plt.close()


def plot_grad_flow(named_parameters):
    """
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backward() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    """
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and (p.grad is not None) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig("grad_flow.png")
    plt.close()