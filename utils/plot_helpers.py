from . import DATA_DIR
import numpy as np
import xarray as xr
import pandas as pd
import json
import os,glob
import seaborn as sns
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
import matplotlib.cm as cm


#Make sure it can see the system path
from pathlib import Path
import sys

# Add parent directory of the notebook (the project root) to sys.path
ROOT = Path().resolve().parent   # X/
sys.path.insert(0, str(ROOT))


def plot_posterior(trace,var_name,ax=None,**kwargs):
    """
    Plot the posterior distribution on ax
    """
    if ax is None:
        ax=plt.subplot(111)
    data=getattr(trace.posterior,var_name).stack(sample=("chain","draw")).values
    sns.kdeplot(data,ax=ax,**kwargs)


def plot_posterior_hdi_mXb(
    trace,
    X,
    m_name="m",
    b_name="b",
    hdi_prob=0.95,
    ax=None,
    color="C0",
    label=None,
):
    """
    Plot posterior mean and HDI of m*X + b.

    Parameters
    ----------
    trace : arviz.InferenceData
        Posterior samples containing m and b.
    var_name : array
        array of X values at which to evaluate m*X + b.
    m_name : str
        Name of slope variable in the trace.
    b_name : str
        Name of intercept variable in the trace.
    hdi_prob : float
        HDI probability mass (default 0.95).
    ax : matplotlib axis or None
        Axis to plot on.
    color : str
        Line / band color.
    label : str or None
        Label for the mean line.

    Returns
    -------
    ax : matplotlib axis
    """

    if ax is None:
        fig, ax = plt.subplots()

    X = np.atleast_1d(X)
    X = np.sort(X)
    

    # Extract posterior samples and flatten chains
    m = trace.posterior[m_name].values.reshape(-1)
    b = trace.posterior[b_name].values.reshape(-1)

    # Compute posterior samples of m*X + b
    # shape: (nsamples, nX)
    Y = m[:, None] * X[None, :] + b[:, None]

    # Mean prediction
    Y_mean = Y.mean(axis=0)

    # HDI
    hdi = az.hdi(Y, hdi_prob=hdi_prob)

    # Plot
    ax.plot(X, Y_mean, color=color, label=label)
    ax.fill_between(
        X,
        hdi[:, 0],
        hdi[:, 1],
        color=color,
        alpha=0.3,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("m·X + b")

    return ax

