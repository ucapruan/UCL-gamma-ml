
import os
import numpy as np
import torch
import uproot
from torch.utils.data import TensorDataset, random_split

def scan_elements(root_files):
    """
    Scan all unique element across ROOT files.

    Parameters
    ----------
    root_files : list of str
        List of .root file paths.

    Returns
    -------
    element_list : list of str
        Sorted list of unique element.
    element_to_index : dict
        Mapping from element name to index.
    """
    all_elements = set()
    for path in root_files:
        with uproot.open(path) as file:
            df = file["Materials"].arrays(library="pd")
            all_elements.update(df["Element"].unique())

    element_list = sorted(list(all_elements))
    element_to_index = {el: i for i, el in enumerate(element_list)}
    return element_list, element_to_index


def load_dataset(root_dir, use_noise=True, val_ratio=0.2):
    """
    Load ROOT dataset from a directory and return torch Tensors.

    Parameters
    ----------
    root_dir : str
        Path to the directory containing .root files.
    use_noise : bool
        Whether to use the "Edep (noise)" smeared spectrum instead of "Edep".
    val_ratio : float
        Fraction of data to use for validation.

    Returns
    -------
    train_dataset : TensorDataset
    val_dataset : TensorDataset
    element_list : list of str
    element_to_index : dict
    """
    root_files = sorted([
        os.path.join(root_dir, f) for f in os.listdir(root_dir)
        if f.endswith(".root")
    ])

    # scan elements
    element_list, element_to_index = scan_elements(root_files)

    X = []
    Y = []

    for path in root_files:
        with uproot.open(path) as file:
            spectrum = file["Edep (noise)"].values() if use_noise else file["Edep"].values()

            # Min-max normalization
            spectrum_min = np.min(spectrum)
            spectrum_max = np.max(spectrum)
            if spectrum_max > spectrum_min:
                spectrum = (spectrum - spectrum_min) / (spectrum_max - spectrum_min)
            else:
                spectrum = np.zeros_like(spectrum)

            spectrum = spectrum.astype(np.float32)
            X.append(spectrum)

            df = file["Materials"].arrays(library="pd").drop_duplicates()   # drop duplication
            target = np.zeros(len(element_list), dtype=np.float32)
            for _, row in df.iterrows():
                idx = element_to_index[row["Element"]]
                target[idx] += row["Fraction"]
            Y.append(target)

    X = torch.tensor(np.stack(X)).unsqueeze(1)  # e.g. [N, 1, 1440]
    Y = torch.tensor(np.stack(Y))               # [N, num_elements]

    # Split dataset
    dataset = TensorDataset(X, Y)
    train_size = int((1 - val_ratio) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset, element_list, element_to_index