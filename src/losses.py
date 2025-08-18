import torch
import math


# NLL

def compute_log_likelihood(y, pi, mu, sigma, eps=1e-6):
    """
    Compute log-likelihood for a batch of targets under a Mixture Density Network output.

    Parameters
    ----------
    y : torch.Tensor
        Ground truth tensor of shape [B, D] where D is number of dimensions (e.g., C, O, Si).
    pi : torch.Tensor
        Mixture weights of shape [B, M] where M is number of mixtures.
    mu : torch.Tensor
        Mixture means of shape [B, M, D].
    sigma : torch.Tensor
        Mixture std devs of shape [B, M, D].

    Returns
    -------
    log_likelihoods : torch.Tensor
        Log-likelihood per sample, shape [B]
    """
    B, M, D = mu.shape

    y = y.unsqueeze(1).expand(-1, M, -1)  # [B, M, D]

    normalizer = math.sqrt(2.0 * math.pi)
    coef = 1.0 / (normalizer * sigma + eps)
    exponent = -0.5 * ((y - mu) / (sigma + eps)) ** 2
    probs = coef * torch.exp(exponent)          # [B, M, D]
    probs = torch.prod(probs, dim=2)            # [B, M]

    weighted_probs = probs * pi                 # [B, M]
    total_prob = torch.sum(weighted_probs, dim=1) + eps  # [B]

    return torch.log(total_prob)                # [B]


def mdn_loss(y, pi, mu, sigma, eps=1e-6):
    """
    Mean negative log-likelihood loss for MDN.

    Returns
    -------
    scalar loss (float)
    """
    log_likelihood = compute_log_likelihood(y, pi, mu, sigma, eps)
    return -torch.mean(log_likelihood)


def mdn_loss_std(y, pi, mu, sigma, alpha=5.0, eps=1e-6, sample_weights=None):
    """
    MDN loss with optional sigma penalty and sample re-weighting.

    Parameters
    ----------
    alpha : float
        Penalty weight on predicted sigma (encourages uncertainty regularization).
    sample_weights : torch.Tensor or None
        Optional weights per sample (shape [B]).

    Returns
    -------
    scalar loss (float)
    """
    nll = mdn_loss(y, pi, mu, sigma, eps)

    if sample_weights is not None:
        nll = nll * sample_weights
    nll_loss = nll.mean()

    sigma_penalty = torch.mean(torch.log(1 + 1000 * sigma))
    return nll_loss + alpha * sigma_penalty



# Edge Sample Reweighting

def compute_sample_weights(targets, center=0.05, min_weight=1.0, max_weight=10.0):
    """
    Compute symmetric sample weights based on distance from a center point (e.g. 5%).

    Parameters
    ----------
    targets : torch.Tensor
        Target tensor of shape [B, D], where D is number of elements (C, O, Si, ...).
        We assume carbon (C) is the first dimension, i.e., targets[:, 0].
    center : float
        The center C-fraction value (typically 0.05 for 5%) around which weights are minimal.
    min_weight : float
        Weight assigned to samples at the center.
    max_weight : float
        Maximum weight assigned to samples at the edges (0 or 1 C-fraction).

    Returns
    -------
    weights : torch.Tensor
        1D tensor of shape [B], sample-wise weights between min_weight and max_weight.
    """
    carbon = targets[:, 0]  # assume C is at dim 0

    # Absolute distance from the center (e.g., 0.05)
    dist = torch.abs(carbon - center)

    # Normalize to [0, 1]
    max_dist = max(center, 1.0 - center)
    norm_dist = dist / max_dist

    # Linear interpolation between min and max weights
    weights = min_weight + (max_weight - min_weight) * norm_dist
    return weights


def compute_asymmetric_sample_weights(targets, center=0.05, left_scale=20, right_scale=10, base_weight=1.0):
    """
    Compute asymmetric sample weights based on distance from center, with different weights
    for low-C and high-C regions.

    Parameters
    ----------
    targets : torch.Tensor
        Tensor of shape [B, D], assumed to include C as the first dimension.
    center : float
        Central value (e.g. 0.05), which will receive the base weight.
    left_scale : float
        Scaling factor for weights to the *left* of the center (e.g. < 5%).
        Larger value → more weight on low-C samples.
    right_scale : float
        Scaling factor for weights to the *right* of the center (e.g. > 5%).
    base_weight : float
        Minimum weight at the center value.

    Returns
    -------
    weights : torch.Tensor
        Sample-wise weight vector of shape [B].
    """
    carbon = targets[:, 0]
    delta = carbon - center  # distance from center (positive: right, negative: left)

    weights = torch.zeros_like(carbon)

    # Left side (e.g. C < 5%)
    weights[delta < 0] = base_weight + left_scale * torch.abs(delta[delta < 0])

    # Right side (e.g. C > 5%)
    weights[delta >= 0] = base_weight + right_scale * torch.abs(delta[delta >= 0])

    return weights