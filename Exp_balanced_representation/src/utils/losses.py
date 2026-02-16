import torch
from torch.nn import functional as F
import numpy as np 


def compute_norm_mse_loss(ground_truth_outputs, predictions, active_entries, norm=1150):
    """
    Computes normed MSE Loss

    Args:
    outputs (torch.tensor): list of true outputs (ground_truth)
    predictions (torch.tensor): list of model predictions
    active_entries (torch.tensor): list of active entries
    norm (int): normalization constant

    Returns:
    mse_loss (float): normed mse loss value
    """
    mse_loss = torch.mean(
        (ground_truth_outputs - (predictions) / norm).pow(2) * active_entries,
    )
    return mse_loss


def compute_cross_entropy_loss(outputs, predictions, active_entries):
    """
    Computes cross entropy lossLoss

    Args:
    outputs (torch.tensor): list of true outputs (ground_truth)
    predictions (torch.tensor): list of model predictions
    active_entries (torch.tensor): list of active entries

    Returns:
    ce_loss (float): cross entropy value
    """

    ce_loss = torch.mean(
        -torch.sum(predictions * torch.log(F.softmax(outputs, dim=1)), dim=1),
    )
    return ce_loss


def gmm_nll_loss(pi, mu, sigma, target, active_entries):
	"""
	Compute negative log-likelihood for GMM
	Args:
	pi: mixture weights
	mu: means
	sigma: standard deviations
	target: target values
	"""

	# target /= 1150 # normalization
	target = target.unsqueeze(-1)

	pi = torch.clamp(pi, min=1e-12)
	pi = pi / pi.sum(dim=-1, keepdim=True)
	sigma = torch.clamp(sigma, min=1e-12)

	log_norm_const = 0.5 * np.log(2.0 * np.pi)
	z = (target - mu) / sigma 
	log_prob = -0.5 * z * z - torch.log(sigma) - log_norm_const

	log_mix = torch.logsumexp(torch.log(pi) + log_prob, dim=-1) * active_entries

	return -log_mix.mean()