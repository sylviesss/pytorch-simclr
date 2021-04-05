import torch
from objective import contrastive_loss
from tqdm import tqdm
from torch import nn


def train_simclr(model,
                 optimizer,
                 loader_train,
                 n_epochs,
                 device,
                 temperature=1.0):
    """
    Pretrain a SimCLR model with ResNet50 as the encoder.

    Args:
    model: A SimCLRMain model.
    optimizer: An Optimizer object we will use to train the model.
    loader_train (DataLoader): dataloader containing training data.
    temperature (float): temperature used in NT-XENT.
    epochs (int): the number of epochs to train for.
    device (torch.device): 'cuda' or 'cpu' depending on the availability of GPU.

    Returns: Nothing.
    """
    print_every = 100
    model = model.to(device=device)
    for e in range(n_epochs):
        for t, (x1, x2, _) in enumerate(loader_train):
            model.train()
            x1 = x1.to(device=device, dtype=torch.float32)
            x2 = x2.to(device=device, dtype=torch.float32)
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = contrastive_loss(z1.cpu(), z2.cpu(), temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % print_every == 0:
                print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))


def feature_extraction(simclr_model,
                       device,
                       loader_lin_eval):
    """
    Extract features from a pretrained SimCLR model using training dataset
    extracted in mode 'lin_eval'.
    Args:
      simclr_model: pretrained SimCLRMain.
      device (torch.device): 'cuda' or 'cpu' (depending on the availability of a gpu).
      loader_lin_eval: dataloader with data extracted with the function
                       get_augmented_dataloader in mode 'lin_eval'
                       (training set without augmentation)
    Returns: extracted features with corresponding labels for each image.
    """
    simclr_model.eval()
    features, targets = [], []
    with torch.no_grad():
        for img, tgt in tqdm(loader_lin_eval, desc='extracting features...'):
            targets.append(tgt)
            img = img.to(device=device, dtype=torch.float32)
            feature, _ = simclr_model(img)
            features.append(feature)
        features_all = torch.cat(features, dim=0)
        targets_all = torch.cat(targets, dim=0)
    return features_all, targets_all


def test_lin_eval(clf_model,
                  simclr_model,
                  valid_loader,
                  device,
                  n_iteration):
    """
    Using features extracted from simclr_model, evaluate clf_model.
    Args:
      clf_model: trained linear classifier.
      simclr_model: trained SimCLRMain model.
      valid_loader: dataloader with validation data.
      device (torch.device): 'cuda' or 'cpu'.
      n_iteration (int): only used for printing statements.
    Returns:
      loss: CE loss without the regularization term.
      top1_acc: accuracy.
      top5_acc: count if the correct class appears in the top 5 classes
                with largest probabilities.
    """
    features, targets = feature_extraction(simclr_model,
                                           device,
                                           valid_loader)
    clf_model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        preds = clf_model(features)
        loss = loss_fn(targets, preds)
        _, predicted = preds.max(1)
        top5_preds = torch.topk(preds, k=5, dim=1)
        correct = predicted.eq(targets).sum().item()
        top5_correct = sum([targets[i] in top5_preds[i, :] for i in range(targets.shape[0])])
        top1_acc = 100. * correct / targets.shape[0]
        top5_acc = 100. * top5_correct / targets.shape[0]
        print(
            'epoch {}: Validation Loss: {:.3f} | Validation Top 1 Accuracy: {:.3f}% | Validation Top 5 Accuracy {:.3f}%'.format(
                (n_iteration + 1), loss, top1_acc, top1_acc))


def train_lin_eval(features,
                   targets,
                   device,
                   simclr_model,
                   valid_loader,
                   representation_dim=2048,
                   n_class=10,
                   reg_weight=1e-3):
    """
    Train a linear classification model with LBFGS and L2 regularization using
    features extracted from the pretrained SimCLR model.
    Args:
      features (tensor): extracted features.
      targets (tensor): indexes of corresponding class of images from which
                       the features were extracted.
      device (torch.device): 'cuda' or 'cpu'.
      simclr_model: pretrained SimCLR model used in validation.
      valid_loader: dataloader with images for validation.
      representation_dim (int): dimension of features
                              (dimension of the inputs to the projection head).
      n_class (int): number of classes.
      reg_weight (float): weight of L2 regularization.
    Returns: the trained linear classifier.
    """
    linear_clf = nn.Linear(in_features=representation_dim,
                           out_features=n_class).to(device)
    linear_clf.train()
    lbfgs_optim = torch.optim.LBFGS(linear_clf.parameters())
    n_epoch = 1000
    loss_fn = torch.nn.CrossEntropyLoss()
    for i in range(n_epoch):
        # From Pytorch docs: Some optimization algorithms such as Conjugate Gradient
        # and LBFGS need to reevaluate the function multiple times, so you have to
        # pass in a closure that allows them to recompute your model. The closure
        # should clear the gradients, compute the loss, and return it.
        def closure():
            lbfgs_optim.no_grad()
            preds = linear_clf(features)
            loss = loss_fn(preds, targets)
            # add regularization
            loss += reg_weight * linear_clf.weight.pow(2).sum()
            loss.backward()

            _, predicted = preds.max(1)
            correct = predicted.eq(targets).sum().item()
            test_lin_eval(
                clf_model=linear_clf,
                simclr_model=simclr_model,
                valid_loader=valid_loader,
                device=device,
                n_iteration=i)
            print(
                'epoch {}: Loss: {:.3f} | Train Acc: {:.3f}%'.format((i + 1), loss, 100. * correct / targets.shape[0]))
            return loss

        linear_clf.step(closure)
    return linear_clf
