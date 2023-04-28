import copy

import torch


class BGRL(torch.nn.Module):
    r"""BGRL architecture for Graph representation learning.

    Args:
        encoder (torch.nn.Module): Encoder network to be duplicated and used in both online and target networks.
        predictor (torch.nn.Module): Predictor network used to predict the target projection from the online projection.

    .. note::
        `encoder` must have a `reset_parameters` method, as the weights of the target network will be initialized
        differently from the online network.
    """
    def __init__(self, encoder, predictor):
        super().__init__()
        # online network
        self.online_encoder = encoder
        self.predictor = predictor

        # target network
        self.target_encoder = copy.deepcopy(encoder)

        # reinitialize weights
        self.target_encoder.reset_parameters()
        # stop gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters()) + list(self.predictor.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.

        Args:
            mm (float): Momentum used in moving average update.
        """
        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def forward(self, online_x, target_x):
        # forward online network
        online_y = self.online_encoder(online_x)

        # prediction
        online_q = self.predictor(online_y)

        # forward target network
        with torch.no_grad():
            target_y = self.target_encoder(target_x).detach()
        return online_q, target_y


def load_trained_encoder(encoder, ckpt_path, device):
    r"""Utility for loading the trained encoder."""
    checkpoint = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(checkpoint['model'], strict=True)
    return encoder.to(device)


def compute_representations(net, dataset, device, return_dataset=False):
    r"""Pre-computes the representations for the entire dataset.

    Returns:
        [torch.Tensor, torch.Tensor]: Representations and labels.
    """
    net.eval()
    reps = []
    labels = []
    new_data_list = []

    for data in dataset:
        # forward
        with torch.no_grad():
            rep = net(data.to(device))
        labels.append(data.y)
        reps.append(rep)
        new_data = data.clone()
        new_data.x = rep
        new_data_list.append(new_data)

    reps = torch.cat(reps, dim=0)
    labels = torch.cat(labels, dim=0)

    if return_dataset:
        new_dataset = dataset.__class__(dataset.root)
        new_dataset.data, new_dataset.slices = dataset.collate(new_data_list)
        return new_dataset
    return [reps, labels]