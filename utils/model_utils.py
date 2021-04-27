import torch
from objective import contrastive_loss
from tqdm import tqdm
from torch import nn
import json

with open('utils/configs.json') as f:
    configs = json.load(f)


def train_simclr(model,
                 optimizer,
                 loader_train,
                 n_epochs,
                 device,
                 accum_steps,
                 temperature,
                 checkpt_path=None):
    """
    Pretrain a SimCLR model with ResNet50 as the encoder.

    Args:
    model: A SimCLRMain model.
    optimizer: An Optimizer object we will use to train the model.
    loader_train (DataLoader): dataloader containing training data.
    temperature (float): temperature used in NT-XENT.
    epochs (int): the number of epochs to train for.
    accum_steps (int): number of steps to accumulate gradient for.
    device (torch.device): 'cuda' or 'cpu' depending on the availability of GPU.
    checkpt_path (str): if resuming training from a checkpoint, provide the path.

    Returns: Nothing.
    """
    if checkpt_path is not None:
        checkpoint = torch.load(checkpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        current_epoch = checkpoint['epoch']
    else:
        current_epoch = 0

    # For saving the model
    sample_inputs, _, _ = next(iter(loader_train))
    fixed_input = sample_inputs[:32, :, :, :]

    optimizer.zero_grad()
    print_every = 100
    # save_every = 20  # save model every x epochs
    model = model.to(device=device)
    for e in range(current_epoch, n_epochs):
        for t, (x1, x2, _) in enumerate(loader_train):
            model.train()
            x1 = x1.to(device=device, dtype=torch.float32)
            x2 = x2.to(device=device, dtype=torch.float32)
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = contrastive_loss(z1.cpu(), z2.cpu(), temperature)
            # Gradient accumulation
            loss.backward()
            if (t + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if t % print_every == 0:
                print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))

        if (e + 1) % configs["save_checkpt_every"] == 0:
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, "/vol/bitbucket/ss9920/checkpoints/simclr_ckpt_bs{}_nepoch{}.pth".format(
                configs["batch_size_small"] * configs["accum_steps"],
                (e + 1)))
    # save the model
    model.eval()
    with torch.no_grad():
        torch.jit.save(torch.jit.trace(model, fixed_input.to(device), check_trace=False),
                       "/vol/bitbucket/ss9920/project-results/simclr_model_bs{}_nepoch{}.pth".format(
                           configs["batch_size_small"] * configs["accum_steps"],
                           n_epochs))


def feature_extraction(simclr_model,
                       device,
                       loader):
    """
    Extract features from a pretrained SimCLR model using training dataset
    extracted in mode 'lin_eval'.
    Args:
        simclr_model: pretrained SimCLRMain.
        device (torch.device): 'cuda' or 'cpu' (depending on the availability of a gpu).
        loader: dataloader with data extracted with the function
                         get_augmented_dataloader in mode 'lin_eval'
                         (training set without augmentation)
    Returns: extracted features with corresponding labels for each image.
    """
    simclr_model.eval()
    features, targets = [], []
    with torch.no_grad():
        for img, tgt in tqdm(loader, desc='extracting features ...'):
            targets.append(tgt)
            img = img.to(device=device, dtype=torch.float32)
            feature, _ = simclr_model(img)
            features.append(feature)
        features_all = torch.cat(features, dim=0)
        targets_all = torch.cat(targets, dim=0)
    return features_all, targets_all


def test_lin_eval(clf_model,
                  simclr_model,
                  loader_test,
                  device):
    """
    Extract features with the pretrained simclr_model and test clf_model.
    Args:
        clf_model: trained linear classifier.
        simclr_model: pretrained simclr model.
        loader_test: data loader with test data.
        device (torch.device): 'cuda' or 'cpu'.
    Returns:
        loss: CE loss.
        top1_acc: accuracy.
    """
    features_valid, targets_valid = feature_extraction(
        simclr_model=simclr_model,
        device=device,
        loader=loader_test
    )
    clf_model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        preds = clf_model(features_valid)
        targets = targets_valid.to(device=device, dtype=torch.long)
        loss = loss_fn(preds, targets)
        _, predicted = preds.max(1)
        correct = predicted.eq(targets).sum().item()
        top1_acc = 100. * correct / targets.shape[0]
        # Do not print top5 accuracy for CIFAR10 because with only 10 classes it is always going to be quite high
        # top5_preds = torch.topk(preds, k=5, dim=1).indices
        # top5_correct = sum([targets[i] in top5_preds[i] for i in range(targets.shape[0])])
        # top5_acc = 100. * top5_correct / targets.shape[0]
    print(
        'Test Loss: {:.4f} | Test Top 1 Accuracy: {:.4f}%'.format(loss, top1_acc)
    )
    return loss, top1_acc


def train_lin_eval(features,
                   targets,
                   device,
                   representation_dim,
                   reg_weight,
                   n_step,
                   n_class=10
                   ):
    """
    Train a linear classification model with LBFGS and L2 regularization using
    features extracted from the pretrained SimCLR model.
    Args:
        features (tensor): extracted features.
        targets (tensor): indexes of corresponding class of images from which
                         the features were extracted.
        device (torch.device): 'cuda' or 'cpu'.
        representation_dim (int): dimension of features
                                  (dimension of the inputs to the projection head).
        reg_weight (float): weight of L2 regularization.
        n_step (int): number of epochs for training the linear evaluation.
        n_class (int): number of classes.
    Returns: the trained linear classifier.
    """
    linear_clf = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=representation_dim,
                  out_features=n_class)
    ).to(device)

    linear_clf.train()
    lbfgs_optim = torch.optim.LBFGS(linear_clf.parameters(), max_iter=configs['lbfgs_max_iter'])
    loss_fn = torch.nn.CrossEntropyLoss()
    iterator = tqdm(enumerate(range(n_step)), desc='Training the Linear Classification Model ...')
    for i, _ in iterator:
        # From Pytorch docs: Some optimization algorithms such as Conjugate Gradient
        # and LBFGS need to reevaluate the function multiple times, so you have to
        # pass in a closure that allows them to recompute your model. The closure
        # should clear the gradients, compute the loss, and return it.
        def closure():
            lbfgs_optim.zero_grad()
            preds = linear_clf(features)
            targets_device = targets.to(device=preds.device, dtype=torch.long)
            loss = loss_fn(preds, targets_device)
            # add regularization
            loss += reg_weight * linear_clf[1].weight.pow(2).sum()
            loss.backward()
            _, predicted = preds.max(1)
            correct = predicted.eq(targets_device).sum().item()
            iterator.set_description('epoch {}: Loss: {:.3f} | Train Acc: {:.3f}%'.format((i + 1), loss,
                                                                                          100. * correct /
                                                                                          targets_device.shape[0]))
            return loss

        lbfgs_optim.step(closure)

    return linear_clf


def train_ssl(simclr_ft,
              optimizer,
              n_epochs,
              device,
              loader_train
              ):
    """
    Fine tune a pretrained SimCLR model with a linear head.
    Args:
     simclr_ft: a SimCLRFineTune model.
     optimizer: optimizer.
     n_epochs (int): number of epochs to tune the model for.
     device (torch.device): 'cuda' or 'cpu'.
     loader_train: dataloader that contains the training dataset for
                   fine tuning.
    Returns: Nothing.
    """
    print_every = 20
    optimizer.zero_grad()
    simclr_ft = simclr_ft.to(device=device)
    simclr_ft.train()
    loss_fn = nn.CrossEntropyLoss()
    # TODO Add loss and accuracy plots
    for e in range(n_epochs):
        for i, (img, targets) in enumerate(loader_train):
            img = img.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.long)
            score = simclr_ft(img)
            loss = loss_fn(score, targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Calculate accuracy
            _, pred = score.max(1)
            correct = sum([pred[i] == targets[i] for i in range(targets.shape[0])])
            accuracy = 100. * correct / targets.shape[0]
            if (i + 1) % print_every == 0:
                print('epoch {}: | Train Loss: {:.3f} | Train Top 1 Accuracy: {:.3f}%'.format((e + 1), loss.item(),
                                                                                              accuracy))


def test_ssl(simclr_ft,
             device,
             loader_test,
             return_loss_accuracy=False):
    """
    Test fine tuned simclr model.
    Args:
     simclr_ft: a SimCLRFineTune model.
     device (torch.device): 'cuda' or 'cpu'.
     loader_test: dataloader that contains data for testing.
    Returns:
     (Optionally) Loss and accuracy (Top1).
    """
    simclr_ft = simclr_ft.to(device=device)
    simclr_ft.eval()
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    total_correct = 0
    total_sample = 0
    t = tqdm(enumerate(loader_test), desc='Testing the classification model ...')
    with torch.no_grad():
        for b, (img, targets) in t:
            total_sample += targets.shape[0]
            img = img.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.long)
            scores = simclr_ft(img)
            loss = loss_fn(scores, targets)
            losses.append(loss)
            _, pred = scores.max(1)
            correct = sum([pred[i] == targets[i] for i in range(targets.shape[0])])
            total_correct += correct
    acc = 100. * total_correct / total_sample
    print('Got %d / %d correct (%.2f)' % (total_correct, total_sample, acc))

    if return_loss_accuracy:
        return sum(losses) / len(losses), acc
