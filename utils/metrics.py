import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm as tqdm
from data import CIFAR10pair


# TODO: check calc_ned ** 2 == calc_nsed
def calc_ned(tensor_a, tensor_b):
    """
    For tensors of shape (batch_size, hidden_dim), calculate a normalized
    Euclidean distance for each batch by first normalizing the vectors to have
    norm = 1.
    Args:
     tensor_a: tensor of shape (batch_size, hidden_dim)
     tensor_b: tensor of shape (batch_size, hidden_dim)
    Returns:
     tensor of shape (batch_size, )
    """
    assert tensor_a.shape == tensor_b.shape, "Tensors must have the same shape"
    tensor_a = F.normalize(tensor_a, p=2, dim=1)
    tensor_b = F.normalize(tensor_b, p=2, dim=1)
    temp_tsr = torch.pow(tensor_a - tensor_b, 2)
    return torch.sqrt(torch.mean(temp_tsr, dim=1))


def calc_nsed(tensor_a, tensor_b):
    """
    Calculate the normalized squared euclidean distance between 2 vectors.
    """
    va = tensor_a - torch.mean(tensor_a, dim=1, keepdim=True)
    vb = tensor_b - torch.mean(tensor_b, dim=1, keepdim=True)
    # Squared Euclidean distance
    sed = torch.sum(torch.pow((va - vb), 2), dim=1)
    # Denominator is the sum of squared norms of the two vectors
    denom = torch.pow(torch.norm(va, dim=1), 2) + torch.pow(torch.norm(vb, dim=1), 2)
    return sed / denom


def calc_corrcoeff(tensor_a,
                   tensor_b):
    """
    Calculate correlation coefficients of vectors in two minibatches.
    Args:
     tensor_a: tensor of shape (batch_size, hidden_dim)
     tensor_b: tensor of shape (batch_size, hidden_dim)
    Returns:
     tensor of shape (batch_size, )
    """
    va = tensor_a - torch.mean(tensor_a, dim=1, keepdim=True)
    vb = tensor_b - torch.mean(tensor_b, dim=1, keepdim=True)
    cc = torch.sum(va * vb, dim=1) / (torch.sqrt(torch.sum(va ** 2, dim=1) * torch.sum(vb ** 2, dim=1)))
    return cc


def bucketize_feature_values(feat_vec):
    """
    Put values in a feature vector into buckets of size hidden_dim/2.
    Boundary of buckets for an example is based on its range
    of values.
    """
    n_buckets = feat_vec.shape[-1] / 2
    results = []
    bdr = torch.linspace(start=torch.min(feat_vec).numpy(),
                         end=torch.max(feat_vec).numpy(),
                         steps=int(n_buckets + 1),
                         device=feat_vec.device)
    results.append(torch.bucketize(feat_vec,
                                   boundaries=bdr,
                                   out_int32=True))
    return torch.stack(results)


def get_similarity_metrics_contrastive(contrastive_model,
                                       data_loader,
                                       use_hidden_feat,
                                       device):
    """
    Calculate the following metrics for a contrastive (SimCLR) model.
    - Normalized Euclidean Distance
    - Correlation Coefficient
    - Cosine Similarity
    Args:
     contrastive_model: pretrain / fine-tuned SimCLR model
     data_loader: dataloader with pairs of augmented images
     use_hidden_feat: indicate features from which layer of the contrastive
                      model will be used.
     device: cuda or cpu.
    """
    contr_metrics = {'pos_ned': [], 'pos_cc': [], 'pos_cos': [],
                     'neg_ned': [], 'neg_cc': [], 'neg_cos': []}

    cos_fn = nn.CosineSimilarity(dim=1)
    contrastive_model.to(device=device)
    contrastive_model.eval()
    t_contr = tqdm(data_loader, desc="Calculating similarity metrics (contrastive)...")
    # contrastive_model
    with torch.no_grad():
        for img1, img2, label in t_contr:
            img1 = img1.to(device=device, dtype=torch.float32)
            img2 = img2.to(device=device, dtype=torch.float32)

            if use_hidden_feat:
                contr_feat, _ = contrastive_model(img1)
                contr_feat_pos, _ = contrastive_model(img2)
            else:
                _, contr_feat = contrastive_model(img1)
                _, contr_feat_pos = contrastive_model(img2)

            contr_feat = torch.flatten(contr_feat, start_dim=1, end_dim=-1)
            # Positive samples
            contr_feat_pos = torch.flatten(contr_feat_pos, start_dim=1, end_dim=-1)
            # Negative samples - randomly shuffle samples in the minibatch
            contr_feat_neg = contr_feat_pos[torch.randperm(contr_feat_pos.shape[0])]
            contr_ned = calc_nsed(contr_feat, contr_feat_pos)
            contr_ned_neg = calc_nsed(contr_feat, contr_feat_neg)

            contr_metrics['pos_ned'].extend(contr_ned.flatten().cpu().tolist())
            contr_metrics['neg_ned'].extend(contr_ned_neg.flatten().cpu().tolist())
            contr_cc = calc_corrcoeff(contr_feat, contr_feat_pos)
            contr_cc_neg = calc_corrcoeff(contr_feat, contr_feat_neg)
            contr_metrics['pos_cc'].extend(contr_cc.flatten().cpu().tolist())
            contr_metrics['neg_cc'].extend(contr_cc_neg.flatten().cpu().tolist())

            contr_metrics['pos_cos'].extend(cos_fn(contr_feat, contr_feat_pos).flatten().cpu().tolist())
            contr_metrics['neg_cos'].extend(cos_fn(contr_feat, contr_feat_neg).flatten().cpu().tolist())
    return contr_metrics


def get_similarity_metrics_supv(supervised_model,
                                data_loader,
                                device):
    """
    Calculate the following metrics for a supervised (Resnet) model.
    - Normalized Euclidean Distance
    - Correlation Coefficient
    - Cosine Similarity
    Args:
     supervised_model: pretrain / fine-tuned supervised classification model
     data_loader: dataloader with pairs of augmented images
     device: cuda or cpu.
    """
    supv_metrics = {'pos_ned': [], 'pos_cc': [], 'pos_cos': [],
                    'neg_ned': [], 'neg_cc': [], 'neg_cos': []}

    cos_fn = nn.CosineSimilarity(dim=1)
    supervised_model = supervised_model.to(device=device)
    supervised_model.eval()
    # supervised model
    t_supv = tqdm(data_loader, desc="Calculating similarity metrics (supervised)...")
    with torch.no_grad():
        for img1, img2, label in t_supv:
            img1 = img1.to(device=device, dtype=torch.float32)
            img2 = img2.to(device=device, dtype=torch.float32)
            # supervised_model
            supv_feat = supervised_model(img1)
            # Positive samples
            supv_feat_pos = supervised_model(img2)
            # Negative samples
            supv_feat_neg = supv_feat_pos[torch.randperm(supv_feat_pos.shape[0])]

            supv_ned = calc_nsed(supv_feat, supv_feat_pos)
            supv_ned_neg = calc_nsed(supv_feat, supv_feat_neg)

            supv_metrics['pos_ned'].extend(supv_ned.flatten().cpu().tolist())
            supv_metrics['neg_ned'].extend(supv_ned_neg.flatten().cpu().tolist())

            supv_cc = calc_corrcoeff(supv_feat, supv_feat_pos)
            supv_cc_neg = calc_corrcoeff(supv_feat, supv_feat_neg)
            supv_metrics['pos_cc'].extend(supv_cc.flatten().cpu().tolist())
            supv_metrics['neg_cc'].extend(supv_cc_neg.flatten().cpu().tolist())
            supv_metrics['pos_cos'].extend(cos_fn(supv_feat, supv_feat_pos).flatten().cpu().tolist())
            supv_metrics['neg_cos'].extend(cos_fn(supv_feat, supv_feat_neg).flatten().cpu().tolist())
    return supv_metrics


def get_similarity_metrics_pairs(contrastive_model,
                                 supervised_model,
                                 data_loader_contr,
                                 data_loader_supv,
                                 use_hidden_feat,
                                 device):
    """
    Obtain the following similarity metrics for features of a) augmented pairs of
    images b) negative samples using 1) contrastive_model 2) supervised_model
    - Normalized Euclidean Distance
    - Correlation Coefficient
    - Cosine Similarity
    Args:
     contrastive_model: pretrain / fine-tuned SimCLR model
     supervised_model: trained supervised classfication model without the last
                       fc layer.
     data_loader_contr: dataloader with pairs of augmented images for the contrastive model
     data_loader_supv: dataloader with pairs of augmented images for the supervised model
     use_hidden_feat: indicate features from which layer of the contrastive
                      model will be used.
     device: cuda or cpu.
    Return:
     2 dictionaries with similarity metrics for each of the two models.
    """
    contr_metrics = get_similarity_metrics_contrastive(contrastive_model,
                                                       data_loader_contr,
                                                       use_hidden_feat,
                                                       device)
    supv_metrics = get_similarity_metrics_supv(supervised_model,
                                               data_loader_supv,
                                               device)
    return contr_metrics, supv_metrics


def get_loader_for_analogy_analysis(batch_size,
                                    dataset,
                                    cfgs):
    """
    Helper function for get_representation_analogy.
    Args:
     batch_size: batch size of the dataloader.
     dataset: "cifar10" / "stl10".
     cfgs: configurations.
    Return:
     A dataloader constructed from one of the customized dataset class
     (CIFAR10pair/STL10pair), where one image is the original image,
     and the other one is an augmented image. Obtained from function
     get_loader_for_analogy_analysis().
    """
    mean_std = cfgs[dataset + "_mean_std"]
    transf = [transforms.ToTensor(), transforms.RandomHorizontalFlip(p=1), transforms.Normalize(mean=mean_std['mean'],
                                                                                                std=mean_std['std'])]
    transf = transforms.Compose(transf)
    ds = CIFAR10pair(root=cfgs['data_dir'], train=True, transform=transf, download=True, anchor=True)
    loader = DataLoader(batch_size=batch_size, dataset=ds, shuffle=True)
    return loader


def get_representation_analogy(model1,
                               model2,
                               batch_size,
                               dataset,
                               cfgs):
    """
    Given two models, make forward passes with one batch of data from a
    dataloader and measure the difference between pairs.
    Args:
     batch_size: batch size of the dataloader.
     dataset: "cifar10" / "stl10".
     cfgs: configurations.
    """
    resu = {}
    loader = get_loader_for_analogy_analysis(batch_size, dataset, cfgs)
    # Get one batch of images; samples1 contains original images while samples2 contains augmented images
    samples1, samples2, _ = next(iter(loader))

    model1.eval()
    model2.eval()
    # make sure the two models are on the same device (cpu); not using gpu because only evaluating one batch
    model1.to(device='cpu')
    model2.to(device='cpu')
    L1_diff = nn.L1Loss(reduction='mean')
    L2_diff = nn.MSELoss(reduction='mean')
    with torch.no_grad():
        repr1_orig, _ = model1(samples1)
        repr1_augmented, _ = model1(samples2)
        repr2_orig, _ = model2(samples1)
        repr2_augmented, _ = model2(samples2)

        resu["model1_l1"] = L1_diff(repr1_orig, repr1_augmented).item()
        resu["model1_l2"] = L2_diff(repr1_orig, repr1_augmented).item()
        resu["model2_l1"] = L1_diff(repr2_orig, repr2_augmented).item()
        resu["model2_l2"] = L2_diff(repr2_orig, repr2_augmented).item()
    return resu