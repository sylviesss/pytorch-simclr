import torch
import torch.nn as nn


class MineNet(nn.Module):
    def __init__(self, input_size, hidden_size=1024):
        super(MineNet, self).__init__()

        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.out = nn.Linear(in_features=hidden_size, out_features=1)
        self.activation = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        out = self.out(x)
        return out


def sample_joint_marginal(loader_train_joint,
                          loader_train_marginal,
                          simclr_model,
                          device,
                          use_hidden_feat):
    """
    Args:
        loader_train_joint: dataloader with desired batch size and augmentations
                            derived from (dataset class) CIFAR10pair.
        loader_train_marginal: dataloader with desired batch size and augmentations
                                derived from (dataset class) CIFAR10.
        simclr_model: trained SimCLR model.
        device: torch device.
        use_hidden_feat: if true, return features before the projection head (h).
                         Else use the final features (z).
    """
    # dataloaders should be shuffled every time we use iter (if shuffle=True)
    joint1, joint2, _ = next(iter(loader_train_joint))
    marginal1, _ = next(iter(loader_train_marginal))
    marginal2, _ = next(iter(loader_train_marginal))
    joint1, joint2 = torch.tensor(joint1, dtype=torch.float32, device=device), torch.tensor(joint2, dtype=torch.float32,
                                                                                            device=device)
    marginal1, marginal2 = torch.tensor(marginal1, dtype=torch.float32, device=device), torch.tensor(marginal2,
                                                                                                     dtype=torch.float32,
                                                                                                     device=device)
    batch_size = joint1.shape[0]
    simclr_model.to(device=device)
    simclr_model.eval()
    with torch.no_grad():
        joint1_h, joint1_z = simclr_model(joint1)
        joint2_h, joint2_z = simclr_model(joint2)
        marg1_h, marg1_z = simclr_model(marginal1)
        marg2_h, marg2_z = simclr_model(marginal2)

        if use_hidden_feat:
            joint_all = torch.cat([joint1_h, joint2_h], dim=1).reshape(batch_size, -1)
            marginal_all = torch.cat([marg1_h, marg2_h], dim=1).reshape(batch_size, -1)
        else:
            joint_all = torch.cat([joint1_z, joint2_z], dim=1).reshape(batch_size, -1)
            marginal_all = torch.cat([marg1_z, marg2_z], dim=1).reshape(batch_size, -1)
    return joint_all, marginal_all


def mine_loss(mine_net, jnt_samples, mgn_samples, last_ma, ma_rate=1e-2):
    # First calculate the lower bound of MI
    t = mine_net(jnt_samples)
    et = torch.exp(mine_net(mgn_samples))
    mi_lower_bound = torch.mean(t) - torch.log(torch.mean(et))

    # Adjust for the biased stochastic gradient estimator
    if last_ma is None:
        ma_et = torch.mean(et)
    else:
        ma_et = (1 - ma_rate) * last_ma + ma_rate * torch.mean(et)
    loss = - (torch.mean(t) - torch.mean(et) / (torch.mean(ma_et).detach()))
    return loss, mi_lower_bound, ma_et


def train_mine(loader_train_joint,
               loader_train_marginal,
               simclr_model,
               device,
               mine_net,
               mine_optim,
               use_hidden_feat=True,
               n_iter=1000,
               ma_rate=1e-2):
    """
    Train an MI estimator to estimate I(tilde(X), Z).
    """
    loss_all = []
    mi_all = []
    mine_net.to(device=device)
    ma_et = None
    print_every = 50
    for i in range(n_iter):
        mine_net.train()
        joint_samples, marginal_samples = sample_joint_marginal(
            loader_train_joint,
            loader_train_marginal,
            simclr_model,
            device,
            use_hidden_feat
        )
        loss, mi, ma_et = mine_loss(
            mine_net,
            joint_samples,
            marginal_samples,
            ma_et,
            ma_rate)

        loss.backward()
        mine_optim.step()
        mine_optim.zero_grad()

        # For plotting
        loss_all.append(loss.item())
        mi_all.append(mi.item())

        if (i + 1) % print_every == 0:
            print("iteration #{}: loss: {:.4f} | MI: {:.4f}".format(i + 1, loss.item(), mi.item()))
    return loss_all, mi_all
