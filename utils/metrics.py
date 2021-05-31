from sklearn.metrics import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm as tqdm


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


def get_similarity_metrics_pairs(contrastive_model,
                                 supervised_model,
                                 data_loader,
                                 device,
                                 use_hidden_feat=True):
    """
    Obtain the following similarity metrics for features of a) augmented pairs of
    images b) negative samples using 1) contrastive_model 2) supervised_model
    - Normalized Euclidean Distance
    - Correlation Coefficient
    - NMI
    - Cosine Similarity
    Args:
     contrastive_model: pretrain / fine-tuned SimCLR model
     supervised_model: trained supervised classification model without the last
                       fc layer.
     data_loader: dataloader with pairs of augmented images
     use_hidden_feat: indicate features from which layer of the contrastive
                      model will be used.
     device: cuda or cpu.
    Return:
     2 dictionaries with similarity metrics for each of the two models.
    """
    contr_metrics = {'pos_ned': [], 'pos_cc': [], 'pos_nmi': [], 'pos_cos': [],
                     'neg_ned': [], 'neg_cc': [], 'neg_nmi': [], 'neg_cos': []}
    supv_metrics = {'pos_ned': [], 'pos_cc': [], 'pos_nmi': [], 'pos_cos': [],
                    'neg_ned': [], 'neg_cc': [], 'neg_nmi': [], 'neg_cos': []}

    cos_fn = nn.CosineSimilarity(dim=1)

    contrastive_model = contrastive_model.to(device=device)
    supervised_model = supervised_model.to(device=device)

    contrastive_model.eval()
    supervised_model.eval()
    t = tqdm(data_loader, desc="Calculating similarity metrics ...")
    with torch.no_grad():
        for img1, img2, label in t:
            img1 = img1.to(device=device, dtype=torch.float32)
            img2 = img2.to(device=device, dtype=torch.float32)

            # contrastive_model
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
            contr_nmi = [
                normalized_mutual_info_score(
                    bucketize_feature_values(contr_feat[i, :]).cpu().flatten().numpy(),
                    bucketize_feature_values(contr_feat_pos[i, :]).cpu().flatten().numpy()
                ) for i in range(contr_feat.shape[0]
                                 )
            ]
            contr_nmi_neg = [
                normalized_mutual_info_score(
                    bucketize_feature_values(contr_feat[i, :]).cpu().flatten().numpy(),
                    bucketize_feature_values(contr_feat_neg[i, :]).cpu().flatten().numpy()
                ) for i in range(contr_feat.shape[0]
                                 )
            ]
            contr_cc = calc_corrcoeff(contr_feat, contr_feat_pos)
            contr_cc_neg = calc_corrcoeff(contr_feat, contr_feat_neg)
            contr_metrics['pos_cc'].extend(contr_cc.flatten().cpu().tolist())
            contr_metrics['neg_cc'].extend(contr_cc_neg.flatten().cpu().tolist())
            contr_metrics['pos_nmi'].extend(contr_nmi)
            contr_metrics['neg_nmi'].extend(contr_nmi_neg)
            contr_metrics['pos_cos'].extend(cos_fn(contr_feat, contr_feat_pos).flatten().cpu().tolist())
            contr_metrics['neg_cos'].extend(cos_fn(contr_feat, contr_feat_neg).flatten().cpu().tolist())

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

            supv_nmi = [
                normalized_mutual_info_score(
                    bucketize_feature_values(supv_feat[i, :]).cpu().flatten().numpy(),
                    bucketize_feature_values(supv_feat_pos[i, :]).cpu().flatten().numpy()
                ) for i in range(supv_feat.shape[0]
                                 )
            ]
            supv_nmi_neg = [
                normalized_mutual_info_score(
                    bucketize_feature_values(supv_feat[i, :]).cpu().flatten().numpy(),
                    bucketize_feature_values(supv_feat_neg[i, :]).cpu().flatten().numpy()
                ) for i in range(supv_feat.shape[0]
                                 )
            ]
            supv_cc = calc_corrcoeff(supv_feat, supv_feat_pos)
            supv_cc_neg = calc_corrcoeff(supv_feat, supv_feat_neg)
            supv_metrics['pos_cc'].extend(supv_cc.flatten().cpu().tolist())
            supv_metrics['neg_cc'].extend(supv_cc_neg.flatten().cpu().tolist())
            supv_metrics['pos_nmi'].extend(supv_nmi)
            supv_metrics['neg_nmi'].extend(supv_nmi_neg)
            supv_metrics['pos_cos'].extend(cos_fn(supv_feat, supv_feat_pos).flatten().cpu().tolist())
            supv_metrics['neg_cos'].extend(cos_fn(supv_feat, supv_feat_neg).flatten().cpu().tolist())

    return contr_metrics, supv_metrics


