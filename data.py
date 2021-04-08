import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, sampler
from PIL import Image
import json

with open('configs.json') as f:
    configs = json.load(f)


class CIFAR10pair(datasets.CIFAR10):
    def __init__(self, root, train, transform, download):
        super(CIFAR10pair, self).__init__(root=root,
                                          train=train,
                                          transform=transform,
                                          download=download)

    def __getitem__(self, idx):
        """
        Modify the method __getitem__
        (https://github.com/pytorch/vision/blob/19ad0bbc5e26504a501b9be3f0345381d6ba1efc/torchvision/datasets/cifar.py#L105)
        in class CIFAR10 to produce two transformed images per original image,
        along with their target.
        Args:
          idx (int): Index
        Returns:
          tuple: (augmented_img1, augmented_img2, target) where target is index
          of the target class.
        """
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)
        else:
            img1, img2 = img, img
        return img1, img2, target


def compose_augmentation_train(
        flip=True,
        color_distort_strength=configs['color_distort_strength'],
        color_drop_prob=configs['color_drop_prob'],
        img_size=configs['cifar10_size'],
        mean_std=None
):
    """
    Compose transformations for training image data.
    Args:
      flip (boolean): whether to randomly flip images.
      color_distort_strength: strength on color distortion.
      color_drop_prob: probability of randomly converting an image to gray scale.
      img_size (int/tuple): size of the original image in dataset.
      mean_std (dict): mean and standard deviation used to normalize dataset.
    Returns:
      A sequence of transformations.
    """

    gaussian_kernel_size = int(np.floor(0.1 * img_size))
    # Random crop+resize and flip
    transf = [transforms.RandomResizedCrop(size=img_size)]
    if flip:
        transf.append(transforms.RandomHorizontalFlip(p=0.5))
    # Color distortion
    color_jitter = transforms.ColorJitter(
        brightness=0.8 * color_distort_strength,
        contrast=0.8 * color_distort_strength,
        saturation=0.8 * color_distort_strength,
        hue=0.2 * color_distort_strength
    )
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    transf.append(rnd_color_jitter)
    transf.append(
        transforms.RandomGrayscale(p=color_drop_prob)
    )
    # Random Gaussian Blur
    transf.append(
        transforms.GaussianBlur(kernel_size=gaussian_kernel_size)
    )
    # Convert to tensor
    transf.append(transforms.ToTensor())
    # Normalize
    if mean_std is not None:
        transf.append(transforms.Normalize(mean=mean_std['mean'],
                                           std=mean_std['std']))

    return transforms.Compose(transf)


def compose_augmentation_fine_tune(flip=True,
                                   mean_std=None,
                                   img_size=configs['cifar10_size']):
    """
    Compose a transformation for fine tuning.
    Args:
     flip (boolean): whether to randomly flip images.
     mean_std (dict): mean and standard deviation used to normalize dataset.
     img_size (int/tuple): size of the original image in dataset.
    Returns:
     A sequence of transformations.
    """
    transf = [transforms.RandomResizedCrop(size=img_size)]
    if flip:
        transf.append(transforms.RandomHorizontalFlip(p=0.5))
    transf.append(transforms.ToTensor())
    if mean_std is not None:
        transf.append(transforms.Normalize(mean=mean_std['mean'],
                                           std=mean_std['std']))
    return transforms.Compose(transf)


def compose_augmentation_test(crop=False,
                              dim=None,
                              mean_std=None):
    """
    Compose transformations for testing image data, with the option to
    center crop images to a desired dimesion.
    Args:
      crop (boolean): indicate whether to crop.
      dim: can be a tuple of height and width, or an integer, in which case
           the center crop will crop a square.
      mean_std (dict): mean and standard deviation used to normalize dataset.
    Returns:
      A sequence of transformations.
    """
    transf = [transforms.ToTensor()]
    if crop and dim is not None:
        transf.append(transforms.CenterCrop(size=dim))
    if mean_std is not None:
        transf.append(transforms.Normalize(mean=mean_std['mean'],
                                           std=mean_std['std']))
    return transforms.Compose(transf)


def get_class_weights(ds,
                      return_wt=True):
    """
  Count the number of instances in each class in the dataset.
  Args:
   ds (torch.utils.data.datasets): torch dataset.
   return_wt (bool): if False, return the number count of each class instead
                     of weights (reciprocal) as a tensor.
  Returns:
   weights/class counts (tensors)
  """
    tgt = np.array(ds.targets)
    class_counts = [len(np.where(tgt == t)[0]) for t in np.unique(tgt)]
    class_counts_with_label = [(str(t), len(np.where(tgt == t)[0])) for t in np.unique(tgt)]
    samples_weight = np.array([class_counts[t] for t in tgt])
    if return_wt:
        return 1. / torch.from_numpy(samples_weight)
    else:
        return class_counts_with_label


def get_augmented_dataloader(batch_size,
                             root=None,
                             train_mode='pretrain',
                             normalize=True,
                             **kwargs):
    """
    Args:
     root (str): directory to put data in. If None, save data in a
                default directory.
     batch_size (int): size of minibatch.
     train_mode (str): choose a mode from ['pretrain', 'lin_eval', 'fine_tune', 'test'].
         - 'pretrain': two augmented images are created for each
                       original image. Returns one dataloader for training.
         - 'lin_eval': images in the training dataset are not
                       augmented since for linear evaluation we only use
                       pretrained simclr model to extract features. Returns two
                       dataloaders, one for training (0.8) and one for
                       validataion (0.2).
         - 'fine_tune': only random cropping with resizing and random left-to-right
                        flipping is used to preprocess images for fine tuning.
         - 'test': return the test dataloader.
     normalize (boolean): normalize the dataset if True.
     kwargs can contain ['ssl_label_size' (float)] if train_mode is fine_tune
    Returns:
     DataLoader(s): (img1, img2, target)
    """
    np.random.seed(42)
    ssl_label_size = kwargs.get('ssl_label_size', 1)  # 1%/10% from the paper
    if root is None:
        root = configs['data_dir']
    if normalize:
        mean_std = configs['cifar10_mean_std']
    else:
        mean_std = None
    # No validation set for pretrain because labels are not available
    if train_mode == 'pretrain':
        dataset = CIFAR10pair(root=root,
                              train=True,
                              transform=compose_augmentation_train(mean_std=mean_std),
                              download=True)
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2)
        return loader
    elif train_mode == 'lin_eval':
        train_dataset = datasets.CIFAR10(root=root,
                                         train=True,
                                         transform=compose_augmentation_test(mean_std=mean_std),
                                         download=True)
        # Return an additional validation dataloader
        valid_dataset = datasets.CIFAR10(root=root,
                                         train=True,
                                         transform=compose_augmentation_test(mean_std=mean_std),
                                         download=False)  # Because it was already downloaded
        num_train = len(valid_dataset)
        indices = list(range(num_train))
        split = int(np.floor(0.2 * num_train))
        # Shuffle indices before splitting into train and validation sets
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = sampler.SubsetRandomSampler(train_idx)
        valid_sampler = sampler.SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  sampler=train_sampler,
                                  num_workers=2,
                                  shuffle=False)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=batch_size,
                                  sampler=valid_sampler,
                                  num_workers=2,
                                  shuffle=False)
        return train_loader, valid_loader
    elif train_mode == 'fine_tune':
        dataset = datasets.CIFAR10(root=root,
                                   train=True,
                                   transform=compose_augmentation_fine_tune(mean_std=mean_std),
                                   download=True)
        num_train = len(dataset)
        # Q: What if in real life the labelled data is not uniformly distributed? Does it matter??
        # randomly select a subset with class weights for fine tuning
        n_training_samples = int(np.floor(ssl_label_size * num_train))
        wts = get_class_weights(dataset)
        weighted_sampler = sampler.WeightedRandomSampler(wts,
                                                         num_samples=n_training_samples)
        ssl_loader = DataLoader(dataset,
                                batch_size=batch_size,
                                sampler=weighted_sampler,
                                shuffle=False,
                                num_workers=2)
        return ssl_loader
    # This test set is for testing the classifier (linear evaluation / fine tuning)
    elif train_mode == 'test':
        dataset = datasets.CIFAR10(root=root,
                                   train=False,
                                   transform=compose_augmentation_test(mean_std=mean_std),
                                   download=True)
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=2)
        return loader

    else:
        raise NotImplementedError
