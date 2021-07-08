import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, sampler
from PIL import Image
import json

with open('utils/configs.json') as f:
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
        (https://pytorch.org/vision/stable/_modules/torchvision/datasets/cifar.html#CIFAR10)
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


class STL10pair(datasets.STL10):
    def __init__(self, root, split, transform, download):
        super(STL10pair, self).__init__(root=root,
                                        split=split,
                                        transform=transform,
                                        download=download)

    def __getitem__(self, idx):
        """
        Modifying the method __getitem__
        (https://pytorch.org/vision/stable/_modules/torchvision/datasets/stl10.html#STL10)
        in class STL10 to produce two transformed images per original image,
        along with their target.
        Args:
          idx (int): Index
        Returns:
          tuple: (augmented_img1, augmented_img2, target) where target is index
          of the target class.
        """
        img, target = self.data[idx], self.labels[idx]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)
        else:
            img1, img2 = img, img
        return img1, img2, target


def compose_augmentation_train(
        img_size,
        flip=True,
        color_distort_strength=configs['augmentation_params']['color_distort_strength'],
        color_drop_prob=configs['augmentation_params']['color_drop_prob'],
        mean_std=None
):
    """
    Compose transformations for training image data.
    Args:
      img_size (int/tuple): size of the original image in dataset.
      flip (boolean): whether to randomly flip images.
      color_distort_strength: strength on color distortion.
      color_drop_prob: probability of randomly converting an image to gray scale.
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


def compose_augmentation_fine_tune(img_size,
                                   flip=True,
                                   mean_std=None):
    """
    Compose a transformation for fine tuning.
    Args:
     img_size (int/tuple): size of the original image in dataset.
     flip (boolean): whether to randomly flip images.
     mean_std (dict): mean and standard deviation used to normalize dataset.
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


def compose_augmentation_supervised(mean_std=None):
    """
    Compose transformations for training the supervised benchmark.
    """
    # Color distortion
    color_jitter = transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_std['mean'],
            std=mean_std['std']
        ),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(size=img_size,padding=4)
        transforms.RandomAffine(degrees=0,
                                translate=(0.3, 0.3)),
        transforms.RandomApply([color_jitter], p=0.5)
    ])
    return transform


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


def get_cifar10_dataloader(img_size,
                           batch_size,
                           val_size,
                           train_mode,
                           root=None,
                           ssl_label_size=1,
                           mean_std=None):
    """
    Args:
     img_size (int): size of image (int*int).
     root (str): directory to put data in. If None, save data in a
                 default directory.
     batch_size (int): size of minibatch.
     val_size (float): size of validation set.
     train_mode (str): choose a mode from ['pretrain', 'lin_eval', 'fine_tune', 'test'].
         - 'pretrain': two augmented images are created for each
                       original image. Returns one dataloader for training.
         - 'lin_eval': images in the training dataset are not
                       augmented since for linear evaluation we only use
                       pretrained simclr model to extract features. Returns two
                       dataloaders, one for training (0.8) and one for
                       validation (0.2).
         - 'fine_tune': only random cropping with resizing and random left-to-right
                        flipping is used to preprocess images for fine tuning.
         - 'supervised_bm': dataset for training a supervised benchmark. Apply
                            the same augmentations as those that are applied
                            in contrastive learning.
         - 'test': return the test dataloader.
     ssl_label_size (float): how much label to use for ssl.
     mean_std (dict): dictionary with keys 'mean' and 'std'.
    Returns:
     DataLoader(s): (img1, img2, target)
    """
    np.random.seed(42)
    if root is None:
        root = configs['data_dir']

    # Include a validation set to evaluate accuracy of the auxiliary classification task
    if train_mode == 'pretrain':
        dataset = CIFAR10pair(root=root,
                              train=True,
                              transform=compose_augmentation_train(img_size=img_size,
                                                                   mean_std=mean_std),
                              download=True)
        num_all = len(dataset)
        num_val = int(val_size * num_all)
        train_ds, valid_ds = torch.utils.data.random_split(dataset, (num_all-num_val, num_val))
        train_loader = DataLoader(train_ds,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=2)
        valid_loader = DataLoader(valid_ds,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=2)
        return train_loader, valid_loader
    elif train_mode == 'lin_eval':
        dataset = datasets.CIFAR10(root=root,
                                         train=True,
                                         transform=compose_augmentation_test(mean_std=mean_std),
                                         download=True)
        # Validation
        # valid_dataset = datasets.CIFAR10(root=root,
        #                                  train=True,
        #                                  transform=compose_augmentation_test(mean_std=mean_std),
        #                                  download=False)  # Because it was already downloaded
        num_all = len(dataset)
        num_val = int(val_size * num_all)
        train_ds, valid_ds = torch.utils.data.random_split(dataset, (num_all-num_val, num_val))
        train_loader = DataLoader(train_ds,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=2)
        valid_loader = DataLoader(valid_ds,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=2)
        return train_loader, valid_loader
    elif train_mode == 'fine_tune':
        train_dataset = datasets.CIFAR10(root=root,
                                         train=True,
                                         transform=compose_augmentation_fine_tune(
                                             img_size=img_size,
                                             mean_std=mean_std),
                                         download=True)
        # Validation dataloader
        valid_dataset = datasets.CIFAR10(root=root,
                                         train=True,
                                         transform=compose_augmentation_test(mean_std=mean_std),
                                         download=False)
        num_train = len(train_dataset)
        if ssl_label_size == 1:
            indices = list(range(num_train))
            split = int(np.floor(val_size * num_train))
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
        else:
            # no validation set for ssl because portion of labelled data is already small
            n_training_samples = int(np.floor(ssl_label_size * num_train))
            wts = get_class_weights(train_dataset)
            weighted_sampler = sampler.WeightedRandomSampler(wts,
                                                             num_samples=n_training_samples)
            train_loader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      sampler=weighted_sampler,
                                      shuffle=False,
                                      num_workers=2)
            return train_loader
    elif train_mode == 'supervised_bm':
        train_dataset = CIFAR10pair(root=root,
                                    train=True,
                                    transform=compose_augmentation_supervised(mean_std=mean_std),
                                    download=True)
        # Return an additional validation dataloader
        valid_dataset = datasets.CIFAR10(root=root,
                                         train=True,
                                         transform=compose_augmentation_test(mean_std=mean_std),
                                         download=False)
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(val_size * num_train))
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


def get_stl10_dataloader(img_size,
                         batch_size,
                         train_mode,
                         val_size,
                         root=None,
                         mean_std=None):
    """
    Args:
     img_size (int/tuple): for resizing.
     batch_size (int): size of minibatch.
     root (str): directory to put data in. If None, save data in a default
                 directory.
     train_mode (str): choose a mode from ['pretrain', 'fine_tune', 'test'].
     val_size (float): size of validation set.
         - 'pretrain': two augmented images are created for each
                       original image. Returns one dataloader for training.
         - 'fine_tune': only random cropping with resizing and random left-to-right
                        flipping is used to preprocess images for fine tuning.
         - 'test': return the test dataloader.
     mean_std (dict): dictionary with keys 'mean' and 'std'.
    Returns:
     DataLoader(s): (img1, img2, target)
    """
    np.random.seed(42)
    if root is None:
        root = configs['data_dir']

    if train_mode == 'pretrain':
        dataset = STL10pair(root=root,
                            split='unlabeled',
                            transform=compose_augmentation_train(
                                img_size=img_size,
                                mean_std=mean_std
                            ),
                            download=True)
        dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2)
    elif train_mode == 'fine_tune':
        train_dataset = datasets.STL10(root=root,
                                       split='train',
                                       transform=compose_augmentation_fine_tune(
                                           img_size=img_size,
                                           mean_std=mean_std
                                       ),
                                       download=True)
        # Validation dataloader
        valid_dataset = datasets.CIFAR10(root=root,
                                         train=True,
                                         transform=compose_augmentation_test(mean_std=mean_std),
                                         download=False)
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(val_size * num_train))
        # Shuffle indices before splitting into train and validation sets
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = sampler.SubsetRandomSampler(train_idx)
        valid_sampler = sampler.SubsetRandomSampler(valid_idx)
        # TODO: check whether to shuffle; maybe not because they have 10 predefined folds
        train_dataloader = DataLoader(train_dataset, batch_size, sampler=train_sampler, shuffle=False, num_workers=2)
        valid_dataloader = DataLoader(valid_dataset, batch_size, sampler=valid_sampler, shuffle=False, num_workers=2)
        dataloader = (train_dataloader, valid_dataloader)
    elif train_mode == 'lin_eval':
        dataset = datasets.STL10(root=root,
                                 split='train',
                                 transform=compose_augmentation_test(
                                     mean_std=mean_std
                                 ),
                                 download=True)
        dataloader = DataLoader(dataset, batch_size, num_workers=2)
    elif train_mode == 'test':
        dataset = datasets.STL10(root=root,
                                 split='test',
                                 transform=compose_augmentation_test(mean_std=mean_std),
                                 download=True)
        dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=2)
    else:
        raise NotImplementedError
    return dataloader


class AugmentedLoader:
    def __init__(self,
                 dataset_name,
                 train_mode,
                 batch_size,
                 cfgs):
        """
        dataset_name: Select from ['cifar10', 'stl10']
        train_mode: select from ['pretrain', 'fine_tune', 'test',
                                'lin_eval', 'supervised_bm'] (last two are for cifar for now )
        batch_size: size of minibatchs in the dataloader
        cfgs: configurations loaded from configs.json
        """
        self.dataset = dataset_name
        self.train_mode = train_mode
        self.batch_size = batch_size

        # Train / test data depending on train_mode
        self.loader = None
        self.valid_loader = None

        self._load(cfgs)

    def _load(self, cfgs):
        """
        Load dataloaders.
        """
        img_size = cfgs[self.dataset + '_size']
        if len(cfgs[self.dataset + '_mean_std']) != 0:
            mean_std = cfgs[self.dataset + '_mean_std']
        else:
            mean_std = None
        if self.dataset == 'cifar10':
            loader = get_cifar10_dataloader(
                img_size=img_size,
                batch_size=self.batch_size,
                root=cfgs['data_dir'],
                train_mode=self.train_mode,
                ssl_label_size=cfgs['ssl_label_size'],
                mean_std=mean_std,
                val_size=0.2)
        elif self.dataset == 'stl10':
            loader = get_stl10_dataloader(img_size=img_size,
                                          batch_size=self.batch_size,
                                          train_mode=self.train_mode,
                                          root=cfgs['data_dir'],
                                          mean_std=mean_std,
                                          val_size=0.2)
        else:
            raise NotImplementedError

        if isinstance(loader, tuple):
            self.loader = loader[0]
            self.valid_loader = loader[1]
        else:
            self.loader = loader
