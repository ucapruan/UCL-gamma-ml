import torch
import numpy as np
import matplotlib.pyplot as plt


def run_evaluation(model, val_loader, device, mdn_predict_mean, mdn_predict_std):
    """
    Run model on validation set and collect predictions, targets, stds, and MDN params.

    Parameters
    ----------
    model : torch.nn.Module
        Trained MDN model.
    val_loader : DataLoader
        Validation DataLoader.
    device : torch.device
        Device to run evaluation on.
    mdn_predict_mean : callable
        Function to compute MDN mixture mean prediction.
    mdn_predict_std : callable
        Function to compute MDN mixture standard deviation.

    Returns
    -------
    dict
        Dictionary with keys: preds, targets, stds, pi, mu, sigma (all as tensors).
    """
    model.eval()

    all_preds = []
    all_targets = []
    all_stds = []

    all_pi = []
    all_mu = []
    all_sigma = []

    with torch.no_grad():
        for val_X, val_Y in val_loader:
            val_X = val_X.to(device)
            val_Y = val_Y.to(device)

            pi, mu, sigma = model(val_X)            # [B, M], [B, M, D], [B, M, D]
            y_pred = mdn_predict_mean(pi, mu)       # [B, D]
            y_std = mdn_predict_std(pi, mu, sigma)  # [B, D]

            all_preds.append(y_pred.cpu())
            all_targets.append(val_Y.cpu())
            all_stds.append(y_std.cpu())

            all_pi.append(pi.cpu())
            all_mu.append(mu.cpu())
            all_sigma.append(sigma.cpu())

    return {
        "preds": torch.cat(all_preds, dim=0),
        "targets": torch.cat(all_targets, dim=0),
        "stds": torch.cat(all_stds, dim=0),
        "pi": torch.cat(all_pi, dim=0),
        "mu": torch.cat(all_mu, dim=0),
        "sigma": torch.cat(all_sigma, dim=0),
    }


def plot_mdn_distribution(idx, d, all_pi, all_mu, all_sigma, all_targets, gaussian_pdf, element_list):
    """
    idx: sample index
    d: dimension index (e.g. 0 = C, 1 = O, 2 = Si)
    """
    pi = all_pi[idx].numpy()              # [M]
    mu = all_mu[idx, :, d].numpy()        # [M]
    sigma = all_sigma[idx, :, d].numpy()  # [M]
    y_true = all_targets[idx, d].item()

    x = np.linspace(-0.02, 0.1, 500)
    pdf = np.zeros_like(x)

    for k in range(len(pi)):
        pdf += pi[k] * gaussian_pdf(x, mu[k], sigma[k])

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(x, pdf, label=f"Predicted PDF for Element {element_list[d]}", color="orange")
    plt.axvline(y_true, color="blue", linestyle="--", label="True Value")
    plt.title(f"Sample #{idx}: MDN Predicted Distribution for Element {element_list[d]}")
    plt.xlabel("Element Fraction")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()