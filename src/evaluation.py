import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import utils
import losses


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


def metrics(results):
    """
    Compute MAE and NLL from evaluation results.

    Parameters
    ----------
    results : dict
        Output from run_evaluation().

    Returns
    -------
    dict
        Dictionary with keys 'MAE' and 'NLL' (float).
    """
    mae = utils.compute_mae(results["preds"], results["targets"]).item()
    nll = losses.mdn_loss(results["targets"], results["pi"], results["mu"], results["sigma"]).item()
    return {"MAE": mae, "NLL": nll}


# Plot

def plot_sample_with_uncertainty(results, element_list, sample_index=0):
    """
    Plot prediction with uncertainty (±1 std) for a single sample.

    Parameters
    ----------
    results : dict
        Output dictionary from `run_evaluation()`, containing preds, targets, stds.
    element_list : list of str
        List of element names corresponding to each column.
    sample_index : int
        Index of the sample to visualize.
    """
    os.makedirs("../results", exist_ok=True)

    preds = results["preds"]
    targets = results["targets"]
    stds = results["stds"]

    x = torch.arange(len(element_list))

    plt.figure(figsize=(8, 5))
    plt.bar(x - 0.2, targets[sample_index], width=0.4, label="True")
    plt.bar(x + 0.2, preds[sample_index], width=0.4, yerr=stds[sample_index], capsize=5, label="Predicted ± 1 std")
    plt.xticks(x, element_list)
    plt.ylabel("Element Fraction")
    plt.title(f"Sample #{sample_index}: Prediction with Uncertainty")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("../results/sample_pred.svg", format="svg")
    plt.show()


def plot_mdn_distribution(results, element_list, sample_index=0, element_name="C"):
    """
    Plot MDN predicted distribution for one sample and one element.

    Parameters
    ----------
    results : dict
        Output from run_evaluation(), must contain keys: 'pi', 'mu', 'sigma', 'targets'.
    element_list : list of str
        List of element names, used for axis labels and index lookup.
    sample_index : int
        Index of the sample to visualize.
    element_name : str
        Element to visualize (e.g. "C", "O", "Si").
    """
    os.makedirs("../results", exist_ok=True)

    d = element_list.index(element_name)

    pi = results["pi"][sample_index].numpy()              # [M]
    mu = results["mu"][sample_index, :, d].numpy()        # [M]
    sigma = results["sigma"][sample_index, :, d].numpy()  # [M]
    y_true = results["targets"][sample_index, d].item()

    x = np.linspace(-0.02, 0.1, 500)
    pdf = np.zeros_like(x)

    for k in range(len(pi)):
        pdf += pi[k] * utils.gaussian_pdf(x, mu[k], sigma[k])

    plt.figure(figsize=(8, 5))
    plt.plot(x, pdf, label=f"Predicted PDF for {element_name}", color="orange")
    plt.axvline(y_true, color="blue", linestyle="--", label="True Value")
    plt.title(f"Sample #{sample_index}: MDN Distribution for {element_name}")
    plt.xlabel("Element Fraction")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../results/sample_carbon_pred_distribution.svg", format="svg")
    plt.show()


def plot_carbon_error_boxplot(results, element_to_index, bin_range="0-10"):
    """
    Generate a boxplot of absolute error grouped by carbon fraction bins.

    Parameters
    ----------
    results : dict
        Output dictionary from run_evaluation(). Should contain 'preds' and 'targets'.
    element_to_index : dict
        Mapping from element name to index. Must include 'C'.
    bin_range : str
        Either '0-10' for 0–10% range, or '0-100' for 0–100% full range.
    """
    os.makedirs("../results", exist_ok=True)

    carbon_index = element_to_index["C"]
    true_C = results["targets"][:, carbon_index].numpy()
    pred_C = results["preds"][:, carbon_index].numpy()
    abs_errors = np.abs(true_C - pred_C)


    if bin_range == '0-10':
        bins = np.linspace(0, 0.1, 11)
    elif bin_range == '0-100':
        bins = np.linspace(0, 1.0, 11)
    else:
        raise ValueError(" bin_range must be '0-10' or '0-100' ")


    bin_labels = [f"{int(b*100)}–{int(bins[i+1]*100)}%" for i, b in enumerate(bins[:-1])]
    bin_indices = np.digitize(true_C, bins) - 1


    grouped_errors = [[] for _ in range(len(bins) - 1)]
    for idx, err in zip(bin_indices, abs_errors):
        if 0 <= idx < len(grouped_errors):
            grouped_errors[idx].append(err)


    plt.figure(figsize=(10, 6))
    plt.boxplot(grouped_errors, labels=bin_labels, showfliers=True)
    plt.xlabel("True Carbon Fraction")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Error of Carbon Prediction")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig("../results/carbon_error_boxplot.svg", format="svg")
    plt.show()

def plot_abs_error_boxplot(results, element_list):
    """
    Plot boxplot of absolute error per element across the validation set.

    Parameters
    ----------
    results : dict
        Output dictionary from `run_evaluation()`, containing 'preds' and 'targets'.
    element_list : list of str
        List of element names used as tick labels.
    """
    os.makedirs("../results", exist_ok=True)

    preds = results["preds"]     # shape: [N, D]
    targets = results["targets"] # shape: [N, D]

    abs_errors = torch.abs(preds - targets).numpy()  # shape: [N, D]

    # Plot boxplot
    plt.figure(figsize=(8, 5))
    plt.boxplot(abs_errors, tick_labels=element_list, showfliers=False)
    plt.ylabel("Absolute Error")
    plt.xlabel("Element")
    plt.title("Boxplot of Absolute Error per Element")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("../results/abs_error_boxplot.svg", format="svg")
    plt.show()


def plot_true_vs_pred_scatter(results, element_name, element_to_index):
    """
    Plot scatter plot of true vs. predicted values for a given element.

    Parameters
    ----------
    results : dict
        Output dictionary from `run_evaluation()`, containing 'preds' and 'targets'.
    element_name : str
        Name of the element to visualize (e.g., "C").
    element_to_index : dict
        Mapping from element names to index.
    """
    os.makedirs("../results", exist_ok=True)

    idx = element_to_index[element_name]
    true_vals = results["targets"][:, idx].numpy()
    pred_vals = results["preds"][:, idx].numpy()

    plt.figure(figsize=(6, 5))
    plt.scatter(true_vals, pred_vals, alpha=0.5)
    plt.axline((0, 0), slope=1, color="red", linestyle="--")
    plt.xlabel(f"True {element_name} Fraction")
    plt.ylabel(f"Predicted {element_name} Fraction")
    plt.title(f"True vs. Predicted: {element_name}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"../results/{element_name}_true_vs_pred.svg", format="svg")
    plt.show()


def plot_residual_histogram(results, element_name, element_to_index, bins=30):
    """
    Plot histogram of residuals (true - predicted) for a given element.

    Parameters
    ----------
    results : dict
        Output dictionary from `run_evaluation()`, containing 'preds' and 'targets'.
    element_name : str
        Name of the element to visualize (e.g., "C").
    element_to_index : dict
        Mapping from element names to index.
    bins : int
        Number of histogram bins.
    """
    os.makedirs("../results", exist_ok=True)

    idx = element_to_index[element_name]
    true_vals = results["targets"][:, idx].numpy()
    pred_vals = results["preds"][:, idx].numpy()

    residuals = true_vals - pred_vals

    plt.figure(figsize=(6, 5))
    plt.hist(residuals, bins=bins, edgecolor="black")
    plt.xlabel("Residual (True - Predicted)")
    plt.ylabel("Frequency")
    plt.title(f"Residual Distribution for {element_name}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"../results/{element_name}_residual_hist.svg", format="svg")
    plt.show()


def plot_uncertainty_coverage(results, element_list, element_to_index):
    """
    Plot coverage rate of true values under predicted uncertainty intervals.

    Parameters
    ----------
    results : dict
        Output dictionary from run_evaluation(), must include 'preds', 'targets', and 'stds'.
    element_list : list of str
        List of element names to evaluate.
    element_to_index : dict
        Mapping from element names to column indices.
    """
    os.makedirs("../results", exist_ok=True)

    std_levels=[1.0, 1.68, 1.96]
    theoretical_coverage=[0.6827, 0.90, 0.95]

    all_preds = results["preds"]
    all_targets = results["targets"]
    all_stds = results["stds"]

    coverage_matrix = []

    for std_mult in std_levels:
        element_coverages = []
        for el in element_list:
            idx = element_to_index[el]
            true_vals = all_targets[:, idx].numpy()
            pred_vals = all_preds[:, idx].numpy()
            std_vals = all_stds[:, idx].numpy()

            lower = pred_vals - std_mult * std_vals
            upper = pred_vals + std_mult * std_vals
            within = (true_vals >= lower) & (true_vals <= upper)
            coverage = np.mean(within)
            element_coverages.append(coverage)
        coverage_matrix.append(element_coverages)

    coverage_matrix = np.array(coverage_matrix)  # shape: [len(std_levels), len(elements)]

    # Plotting
    x = np.arange(len(std_levels))
    width = 0.2
    colors = ["tab:blue", "tab:orange", "tab:green"]

    plt.figure(figsize=(10, 6))
    for i, el in enumerate(element_list):
        plt.bar(x + (i - 1) * width, coverage_matrix[:, i], width=width, label=el, color=colors[i])

    for i, (x_pos, ci) in enumerate(zip(x, theoretical_coverage)):
        plt.hlines(ci, x_pos - 0.35, x_pos + 0.35, colors="black", linestyles="dashed")
        plt.text(x_pos, ci + 0.01, f"{int(ci * 100)}%", ha="center", va="bottom", fontsize=10)

    plt.xticks(ticks=x, labels=[f"±{s} std" for s in std_levels])
    plt.ylabel("Coverage Rate")
    plt.ylim(0.5, 1.05)
    plt.title("Element-wise Prediction Coverage under Different Std Ranges")
    plt.legend(title="Element")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("../results/uncertainty_coverage.svg", format="svg")
    plt.show()