# Deep Learning Application on Gamma Spectrum Analysis

## Project Overview

This project investigated a hybrid architecture combining a **1d-CNN** and **MDN** was designed to predict elemental fractions while quantifying uncertainty, in order to predict soil elemental composition from neutron induced gamma spectra. Strategies such as customized loss regularization, oversampling, and normalization were introduced to enhance calibration and reduce edge bias.


## Environment Setup

This project requires **Python 3.10** and the following dependencies:

- os (standard library)
- numpy
- matplotlib
- uproot
- torch
- torchinfo (optional)

### Installation

You can install the required packages with:

```bash
pip install numpy matplotlib uproot torch torchinfo
```

## Repository Structure
```
├── Notebooks                           // Jupyter notebooks of project experiments
│   ├── ...
├── src                                 // source code
│   ├── data.py                         // data loading, preprocessing, and plot 
│   ├── evaluation.py                   // evaluation running, metrics and model performance plot
│   ├── losses.py                       // custom loss functions (NLL and sigma penalty)
│   ├── models.py                       // model structure (MLP, 1d-CNN+MDN)
│   └── utils.py                        // helper functions (computing mean, std, MAE, gaussian-pdf)
├── LICENSE                             // license file
├── MDN_3_mixture_with_penalty.pt       // pretrained MDN model on C: 0-10% dataset
├── MDN_no_softmax.pt                   // pretrained MDN model (without softmax constraint) on C: 0-10% dataset
├── MDN_pretrained_fullC.pt             // pretrained MDN model on C: 0-100% dataset
└── README.md                           //  documentation
```

## Model Overview

The model consists of:

- **1D-CNN**: Extracts spectral features from the gamma-ray input.
- **MDN Head**: Outputs a mixture of Gaussians per element, modeling \( p(\mathbf{y}|\mathbf{x}) \) and enabling uncertainty quantification.
- **Loss**: Negative log-likelihood (NLL) plus optional sigma penalty to regularize overconfidence.


## Key Features
- Data preprocess pipeline:
  - Search ROOT files
  - Create element list
  - Normalize spectrum data
  - Load and split dataset
- Probabilistic predictions with uncertainty
- Evaluation pipeline:
  - Run model on evaluation set
  - Metrics computation
  - Result visualization
  - Evaluation visualization
- Edge bias solving Strategies:
  - Normalization without Softmax
  - Loss reweighting
  - Oversampling

