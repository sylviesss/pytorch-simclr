from models.modified_resnet import ResnetSupervised
from data import get_augmented_dataloader
from utils.model_utils import train_ssl, test_ssl
import torch
import torch.nn as nn
import json

# TODO: add a flexible path for saving the model; right now the path is for training on the DoC machines
if __name__ == '__main__':
    # Set a seed.
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)

    # Args.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open('utils/configs.json') as f:
        configs = json.load(f)

    resnet = ResnetSupervised()
    supervised_resnet_optim = torch.optim.Adam(resnet.parameters(),
                                               weight_decay=configs['wt_decay'])

    # For early stopping
    best_acc = 0
    patience = 4
    patience_counter = 0

    print_every = 100
    supervised_resnet_optim.zero_grad()
    resnet = resnet.to(device=device)
    loss_fn = nn.CrossEntropyLoss()

    # Create data loaders
    loader_train, loader_val = get_augmented_dataloader(
        batch_size=configs['batch_size_small'],
        train_mode='supervised_bm')
    loader_test = get_augmented_dataloader(
        batch_size=configs['batch_size_small'],
        train_mode='test')

    # # Get fixed inputs for saving the model
    # samples, _, _ = next(iter(loader_train))
    # fixed_input = samples[:batch_size_small, :, :, :]

    for e in range(configs['n_epoch']):
        for i, (img1, img2, targets) in enumerate(loader_train):
            resnet.train()
            images = [img1, img2]
            # Two augmented images per original image
            for img in images:
                img = img.to(device=device, dtype=torch.float32)
                targets = targets.to(device=device, dtype=torch.long)
                score = resnet(img)
                loss = loss_fn(score, targets)

                loss.backward()
                supervised_resnet_optim.step()
                supervised_resnet_optim.zero_grad()

                # Calculate accuracy
                _, pred = score.max(1)
                correct = sum([pred[i] == targets[i] for i in range(targets.shape[0])])
                accuracy = 100. * correct / targets.shape[0]

                if (i + 1) % print_every == 0:
                    print('epoch {}: | Train Loss: {:.3f} | Train Top 1 Accuracy: {:.3f}%'.format((e + 1), loss.item(),
                                                                                                  accuracy))

        # Early stopping
        val_loss, val_acc = test_ssl(simclr_ft=resnet,
                                     device=device,
                                     loader_test=loader_val,
                                     return_loss_accuracy=True)
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            print("Found a better model. Saving...")
            resnet.eval()
            with torch.no_grad():
                torch.save(resnet.state_dict(), "/content/supervised_bm_bs{}.pth".format(configs['batch_size_small']))
        else:
            patience_counter += 1
        if patience_counter == patience:
            print('Early stopping, reverting to the previous model ...')
            resnet = ResnetSupervised()
            resnet.load_state_dict(torch.load("/content/supervised_bm_bs{}.pth".format(configs['batch_size_small'])))
            break

    # Test
    resnet = ResnetSupervised()
    resnet.load_state_dict(torch.load("/content/supervised_bm_bs{}.pth".format(configs['batch_size_small'])))
    test_ssl(simclr_ft=resnet,
             device=device,
             loader_test=loader_test,
             return_loss_accuracy=False)

