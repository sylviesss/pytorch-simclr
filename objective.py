import torch
import torch.nn.functional as F
from torch import nn


def contrastive_loss(x_batch1,
                     x_batch2,
                     temperature=1.0,
                     normalize=True,
                     weight=None):
    """
    Calculate the NT-XENT loss.
    Args:
      x_batch* (tensor): minibatches of augmented samples of shape
                        (batch_size, out_dim).
      temperature (float): for temperature scaling.
      normalize (boolean):  indicate whether we normalize the input to the loss
                            function.
      weight (tensor): weights for loss of each sample in the minibatch.
    """
    VERY_LARGE_NUM = 1e9

    batch_size = x_batch1.shape[0]
    # L2 normalization along the axis with channels
    if normalize:
        x1 = F.normalize(x_batch1, p=2, dim=1)
        x2 = F.normalize(x_batch2, p=2, dim=1)
    else:
        x1 = x_batch1
        x2 = x_batch2
    labels = torch.arange(2 * batch_size, dtype=torch.long, device=x1.device)
    masks = torch.eye(batch_size, device=x1.device)

    # Calculate 4 sets of of cosine similarities and combine them
    logits_aa = (x1 @ x1.t()) / temperature
    logits_bb = (x2 @ x2.t()) / temperature
    # When calculating the loss, we do not want to include similarity of a representation with itself, so we set the
    # diagonal entries to a very negative number, to make their contributions to the nce loss almost 0
    logits_aa = logits_aa - masks * VERY_LARGE_NUM
    logits_bb = logits_bb - masks * VERY_LARGE_NUM
    # Negative samples
    logits_ab = (x1 @ x2.t()) / temperature
    logits_ba = (x2 @ x1.t()) / temperature

    # Use sum of losses to be consistent with the tf implementation
    # (Reduction.SUM_BY_NONZERO_WEIGHTS)
    loss_fn = nn.CrossEntropyLoss(weight=weight, reduction='mean')
    logits = torch.cat([torch.cat([logits_ab, logits_aa], dim=-1),
                        torch.cat([logits_bb, logits_ba], dim=-1)], dim=0)
    loss = loss_fn(logits, labels)
    _, predicted = logits.max(1)
    correct = predicted.eq(labels).sum().item()
    acc = 100. * correct / labels.shape[0]

    return loss, acc