import pymc as pm
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import matplotlib.cm as cm
import pandas as pd
import pytensor.tensor as pt
import json


#Make sure it can see the system path

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from utils import load_data
####### FIXED EFFECTS MODEL ###########

def build_fixed_effects_model(evidence: xr.DataArray,
    priors={},
    var_name="X"):
    """
    Build a hierarchical random-effects model for CMIP ensemble data in PyMC.

    This model implements a Bayesian correlated random-effects framework for estimating a latent
    “true” quantity across multiple ESMs while 
    accounting for both structural differences between ESMs and internal variability 
    within an ESM. The model assumes:
    
    X_m = X_true + epsilon
    
    Where epsilon ~ N(0,sigma) is random measurement error
    
     Parameters
    ----------
    data : xr.DataArray
        CMIP ensemble data with dimensions that must include 'model'. Can have one or more
        ensemble members per model.
    priors : dict, optional
        Dictionary of callable prior constructors keyed by variable name. Supported keys:
        - 'X' or var_name: prior for the latent global mean.
        - 'sigma': prior for measurement error
        If a key is absent, default weakly informative priors are used.
    var_name : str, default "X"
        Name of the latent quantity being estimated (used for naming PyMC variables).
    """
    
    data = evidence.values
    with pm.Model() as model:
        # true value of the parameter we want to estimate
        if "X" in priors.keys():
            X_true = priors["X"](var_name)
        else:
            X_true=pm.Normal(var_name,0,100)
        if "sigma" in priors.keys():
            sigma = priors["sigma"]("sigma")
        else:
            sigma= pm.HalfNormal("sigma",1.0)
        pm.Normal("likelihood",X_true,sigma,observed=data)
    return model


####### RANDOM EFFECTS MODEL ###########

def use_noncentered(mu_iv, priors):
    # If IV is marginalized → non-centered is good
    if mu_iv is not None:
        return True

    # If sigma_iv is fixed to a small constant → centered is better
    if "sigma_iv" in priors:
        try:
            test = priors["sigma_iv"]("test")
            if np.isscalar(test) and test < 0.2:
                return False
        except Exception:
            pass

    # Default
    return True

def build_correlated_random_effects_model(data: xr.DataArray,
    priors={},
    var_name="X",
    mu_iv=None):
    
    """
    Build a hierarchical random-effects model for CMIP ensemble data in PyMC.

    This model implements a Bayesian correlated random-effects framework for estimating a latent
    “true” quantity across multiple ESMs while 
    accounting for both structural differences between ESMs and internal variability 
    within an ESM. The model supports:

    1. Latent correlated ESM means:
        - Each ESM mean X_m is assumed to be X_m = X_true + bias_m + bias_common
        - X_m is modeled as a multivariate normal (MVN) with
         correlation structure controlled by the shared parameter rho and a structural
         spread sigma_struct.  Rho is the fraction of variance explained by the common CMIP bias.
        - Non-centered parameterization is used to improve sampling efficiency in some cases

    2. Within-ESM internal variability:
       - If `mu_iv` is provided, internal variability sigma_iv is dominated completely by the prior
       and integrated out, yielding a Student-t likelihood:
           X_m^j ~ StudentT(nu, mu=X_m, sigma=exp(mu_iv))
         This is useful when each model has only 1 ensemble member.
       - Otherwise, sigma_iv is sampled explicitly  with optional
         per-model priors.  This is useful in e.g. detection and attribution where we have large model
         ensembles

    3. Correlation between model biases:
       - A common correlation rho is used to induce correlation among X_m values.
       - The correlation matrix is parameterized as (1-rho)*I + rho*J, where I is
         identity and J is a matrix of ones, and is factorized via Cholesky.

    Parameters
    ----------
    data : xr.DataArray
        CMIP ensemble data with dimensions that must include 'model'. Can have one or more
        ensemble members per model.
    priors : dict, optional
        Dictionary of callable prior constructors keyed by variable name. Supported keys:
        - 'X' or var_name: prior for the latent global mean.
        - 'sigma_struct': prior for structural spread across models.
        - 'sigma_iv': prior for internal variability (only used if mu_iv is None).
        - 'rho': prior for the correlation between model biases.
        If a key is absent, default weakly informative priors are used.
    var_name : str, default "X"
        Name of the latent quantity being estimated (used for naming PyMC variables).
    mu_iv : float or None, default None
        If provided, internal variability is fixed to mu_iv and
        the likelihood is Student-t, marginalizing sigma_iv. If None, sigma_iv is
        sampled explicitly using a HalfNormal or user-specified prior.
        Provide this if you think internal variability matters but you don't have sufficiently large ensembles
        If internal variability is known to be small (e.g. ECS) fix sigma_iv as a small constant instead.

    Returns
    -------
    vecmodel : pm.Model
        A PyMC model object representing the hierarchical random-effects model.

    Model Structure
    ---------------
    1. Global mean:
        X_true ~ Normal(0, 100) (default) or user-specified prior.
    
    2. Structural spread:
        sigma_struct ~ HalfNormal(1.0) (default) or user-specified prior.
    
    3. Correlation among ESM biases:
        rho ~ Beta(5, 5) (default) or user-specified prior.
    
    4. Latent ESM means:
        Data regime aware parameterization to ensure good geometry
        Both are statistically equivalent but trade off flexibility and speed
        
        (Centered, used when likelihood is sharp e.g. sigma_iv fixed or tiny)
        X_m ~ MVN(X_true, sigma_struct * (1-rho)I + rhoJ))
        
        (Non-centered, used when internal variability is weakly identified)
        z_m ~ Normal(0, 1)
        X_m = X_true + sigma_struct * Cholesky[(1-rho)*I + rho*J] @ z_m
    
    5. Within-model variability:
        - mu_iv is None: sigma_iv ~ HalfNormal(1.0) (per model) and
          likelihood: y_m ~ Normal(X_m, sigma_iv)
        - mu_iv provided: likelihood marginalizes sigma_iv:
          X_m^j ~ StudentT(nu=4, mu=X_m, sigma=exp(mu_iv))

    Notes
    -----
    - For single-member ensembles, mu_iv should be provided to use the Student-t likelihood.
    - Non-centered parameterization improves sampling efficiency for hierarchical X_m.
    - The Student-t likelihood encodes prior uncertainty in sigma_iv without slowing
      sampling in low-data regimes.
    """
    evidence_stacked = data.stack(run=("model",))
    
    # Unique models
    unique_models = np.unique(data.model.values)
    nmodels=len(unique_models)


    # Map model names to integers
    model_to_int = {name: i for i, name in enumerate(unique_models)}
    model_idx_int = np.array([model_to_int[m] for m in evidence_stacked["model"].values])
    with pm.Model(coords={"model": unique_models}) as vecmodel:
        
        # ======== Prior on the estimated quantity ============== #
        if var_name in priors.keys():
            X_true = priors[var_name](var_name)
        else:
            # Use a wide prior
            X_true = pm.Normal(var_name, 0, 100)
            
        # ======== Prior on CMIP structural spread ============== #
        if "sigma_struct" in priors.keys():
            sigma_struct = priors["sigma_struct"]("sigma_struct")
        else:
            sigma_struct = pm.HalfNormal("sigma_struct", 1.0)
            
         # ======== Prior on internal variability in each model =========#
        if mu_iv is None:
            # If mu_iv is specified, we assume sigma~iv,m ~ LogNormal(mu_iv,tau)
            # We then have y_m | X_m ~ StudentT
            #This is useful in cases where the uncertainty in iv is completely prior-dominated

            if "sigma_iv" in priors.keys():
                sigma_iv = priors["sigma_iv"]("sigma_iv")
            else:
                #Otherwise use a HalfNormal for sigma_iv
                sigma_iv = pm.HalfNormal("sigma_iv",
                                         1.0,
                                        dims="model")
            # Force it into a tensorvariable if it's a scalar
            sigma_iv = pt.as_tensor_variable(sigma_iv)


        # ======== Prior on correlation between biases =========#
        # for rho = constant, use priors["rho"] = lambda name: constant
        if "rho" in priors.keys():
            rho = priors["rho"]("rho")
        else:
            rho = pm.Beta("rho", alpha=5, beta=5)
        # Force it into a tensorvariable if it's a scalar
        rho = pt.as_tensor_variable(rho)

        # ======== Generate the correlation matrix =========#
        I = pt.eye(nmodels)
        J = pt.ones((nmodels, nmodels))
        corr = (1 - rho) * I + rho * J

        # Cholesky factor for speed
        #chol = sigma_struct * pt.linalg.cholesky(corr)
        chol_corr = pt.linalg.cholesky(corr)
        
        use_nc = use_noncentered(mu_iv, priors)

        # ======== Latent model means ======== #
        if use_nc:
            # ---- Non-centered ----
            z_m = pm.Normal("z_m", 0, 1, dims="model")

            X_m = pm.Deterministic(
                f"{var_name}_CMIP",
                X_true + sigma_struct * pt.dot(chol_corr, z_m),
                dims=("model",)
            )

        else:
            # ---- Centered ----
            X_m = pm.MvNormal(
                f"{var_name}_CMIP",
                mu=X_true * pt.ones(nmodels),
                chol=sigma_struct * chol_corr,
                dims=("model",)
            )

         # ======== Within-ensemble internal variability =========#
        # vectorized likelihood using integer index
        if mu_iv is not None:
            nu = 4

            pm.StudentT(
                f"{var_name}_lik",
                mu=X_m[model_idx_int],
                sigma=mu_iv,
                nu=nu,
                observed=evidence_stacked.values,
            )
        else:
            if len(sigma_iv.shape.eval()) > 0:
                pm.Normal(
                    f"{var_name}_lik",
                    mu=X_m[model_idx_int],
                    sigma=sigma_iv[model_idx_int],
                    observed=evidence_stacked.values,
                )
            else:
                pm.Normal(
                    f"{var_name}_lik",
                    mu=X_m[model_idx_int],
                    sigma=sigma_iv,
                    observed=evidence_stacked.values,
                )
    return vecmodel
    



def build_multiplicative_process(
    lookup_table,
    priors={},
    name="eta",
):
    """
    Build multiplicative process scaling for model i

    eta_eff_i = Prod_j eta_j ** L_{ji}

    Parameters
    ----------
    lookup_table : xr.DataArray, shape (n_process, n_model)
        Binary indicator of process presence.
    priors : dict
        Optional prior for `eta`.
    name : str
        Base name for multiplicative process parameters.

    Returns
    -------
    eta : TensorVariable, shape (n_process,)
        Process scaling parameters.
    eta_eff : TensorVariable, shape (n_model,)
        Effective scaling per model.
    """

    L = pt.as_tensor_variable(np.asarray(lookup_table).astype(float))
    nproc = L.shape[0]

    # ----- Per-process scaling -----
    if name in priors:
        eta = priors[name](name, dims="multiplicative_process")
    else:
        eta = pm.LogNormal(name, 0.0, 0.5, dims="multiplicative_process")

    # ----- Effective scaling (log-space) -----
    log_eta_eff = pt.dot(pt.log(eta), L)
    eta_eff = pm.Deterministic(
        f"{name}_eff",
        pt.exp(log_eta_eff),
        dims="model",
    )

    return eta, eta_eff


def build_additive_process(
    lookup_table,
    priors={},
    name="delta",
):
    """
    Build additive process contribution for model i

    delta_eff_i = Sum_j delta_j * L_{ji}
    """

    L = pt.as_tensor_variable(np.asarray(lookup_table).astype(float))
    nproc = L.shape[0]

    if name in priors:
        delta = priors[name](name, dims="additive_process")
    else:
        delta = pm.Normal(name, 0.0, 10.0, dims="additive_process")
        
    # delta_eff has dims "model"

    delta_eff = pm.Deterministic(
        f"{name}_eff",
        pt.dot(delta, L),
        dims="model",
    )

    return delta, delta_eff


def build_correlated_bias_model_with_processes(
    data: xr.DataArray,
    lookup_table,
    additive_processes=None,
    multiplicative_processes=None,
    priors={},
    var_name="X",
    mu_iv=None,
):
    """
    Correlated random-effects model with multiple multiplicative and additive processes.

    X_i^j ~ Normal(X_i, sigma_iv_i)
    X_i = eta_eff_i * X_i_unscaled + delta_eff_i
    eta_eff_i = Prod_j eta_j ** L_{ji}
    delta_eff_i = Sum_j delta_j * L_{ji}
    """

    # Stack ensemble dimension for vectorized likelihood
    evidence_stacked = data.stack(run=("model",))

    unique_models = np.unique(data.model.values)
    nmodels = len(unique_models)

    # Integer index for likelihood
    model_to_int = {name: i for i, name in enumerate(unique_models)}
    model_idx_int = np.array(
        [model_to_int[m] for m in evidence_stacked["model"].values]
    )
    # Reindex lookup_table to data.model:
    lookup_table = lookup_table.reindex(
    model=unique_models,
    fill_value=0,
)
    # Process lookup: (n_process, n_model)
    L = pt.as_tensor_variable(np.asarray(lookup_table).astype(float))
    nproc = L.shape[0]
    
    # normalize inputs
    if additive_processes is None:
        additive_processes = []
    if multiplicative_processes is None:
        multiplicative_processes = []

    coords = {"model": unique_models}

    if len(multiplicative_processes) > 0:
        coords["multiplicative_process"] = multiplicative_processes
    if len(additive_processes) > 0:
        coords["additive_process"] = additive_processes

    with pm.Model(coords=coords) as model:

        # ======== Prior on latent unscaled quantity ======== #
        if var_name in priors:
            X_true_unscaled = priors[var_name](f"{var_name}_unscaled")
        else:
            X_true_unscaled = pm.Normal(f"{var_name}_unscaled", 0, 100)

        # ======== Structural spread ======== #
        if "sigma_struct" in priors:
            sigma_struct = priors["sigma_struct"]("sigma_struct")
        else:
            sigma_struct = pm.HalfNormal("sigma_struct", 1.0)

        # ======== Process scalings ======== #
        if len(multiplicative_processes) > 0:
            eta, eta_eff = build_multiplicative_process(
                lookup_table=lookup_table.sel(process=multiplicative_processes),
                priors=priors,
                name="eta",
            )
        else:
            eta_eff = pt.ones(nmodels)

        if len(additive_processes) > 0:
            delta, delta_eff = build_additive_process(
                lookup_table=lookup_table.sel(process=additive_processes),
                priors=priors,
                name="delta",
            )
        else:
            delta_eff = pt.zeros(nmodels)

        # ======== Internal variability ======== #
        if mu_iv is None:
            if "sigma_iv" in priors:
                sigma_iv = priors["sigma_iv"]("sigma_iv")
            else:
                sigma_iv = pm.HalfNormal(
                    "sigma_iv", 1.0, dims="model"
                )
            sigma_iv = pt.as_tensor_variable(sigma_iv)

        # ======== Correlation between model biases ======== #
        if "rho" in priors:
            rho = priors["rho"]("rho")
        else:
            rho = pm.Beta("rho", 5, 5)
        rho = pt.as_tensor_variable(rho)

        # ======== Correlation matrix ======== #
        I = pt.eye(nmodels)
        J = pt.ones((nmodels, nmodels))
        corr = (1 - rho) * I + rho * J
        chol_corr = pt.linalg.cholesky(corr)

        use_nc = use_noncentered(mu_iv, priors)

        # ======== Latent correlated unscaled model means ======== #
        if use_nc:
            z_m = pm.Normal("z_m", 0, 1, dims="model")
            X_m_unscaled = (
                X_true_unscaled
                + sigma_struct * pt.dot(chol_corr, z_m)
            )
        else:
            X_m_unscaled = pm.MvNormal(
                f"{var_name}_CMIP_unscaled",
                mu=X_true_unscaled * pt.ones(nmodels),
                chol=sigma_struct * chol_corr,
                dims="model",
            )

        # ======== Apply effective process scaling ======== #
        X_m = pm.Deterministic(
            f"{var_name}_CMIP", 
            eta_eff * X_m_unscaled + delta_eff,
            dims="model"
        )

        # ======== Likelihood ======== #
        if mu_iv is not None:
            pm.StudentT(
                f"{var_name}_lik",
                mu=X_m[model_idx_int],
                sigma=mu_iv,
                nu=4,
                observed=evidence_stacked.values,
            )
        else:
            if len(sigma_iv.shape.eval()) > 0:
                pm.Normal(
                    f"{var_name}_lik",
                    mu=X_m[model_idx_int],
                    sigma=sigma_iv[model_idx_int],
                    observed=evidence_stacked.values,
                )
            else:
                pm.Normal(
                    f"{var_name}_lik",
                    mu=X_m[model_idx_int],
                    sigma=sigma_iv,
                    observed=evidence_stacked.values,
                )
        # ======== Global true quantity (process-level) ======== #
        if len(multiplicative_processes) > 0:
            eta_global = pt.prod(eta)
        else:
            eta_global = 1.0

        if len(additive_processes) > 0:
            delta_global = pt.sum(delta)
        else:
            delta_global = 0.0

        X_true = pm.Deterministic(
            f"{var_name}",
            eta_global * X_true_unscaled + delta_global,
        )


    return model
def simple_ec_model(constraint,priors={},var_name="ECS",observable_name="Y"):

    with pm.Model() as model:
        data=load_data.prep_EC_data(constraint)
        if var_name in priors.keys():
            X_true=priors[var_name](var_name)
        else:
            X_true = pm.Normal(var_name,0,100)
        if "m" in priors.keys():
            m=priors["m"]("m")
        else:
            m = pm.Normal("m",0,5)
        if "b" in priors.keys():
            b=priors["b"]("b")
        else:
            b = pm.Normal("b",0,5)
        if "sigma_reg" in priors.keys():
            sigma_reg=prior["sigma_reg"]("sigma_reg")
        else:
            sigma_reg = pm.HalfNormal("sigma_reg",1)
        # learn m,b
        mu_sim = m*data["X"].values+b
        pm.Normal(f"{observable_name}_sim",mu_sim,sigma_reg,observed=data["Y"].values)
        # emergent constraint
        mu_obs = m*X_true + b
        pm.Normal(f"{observable_name}_obs",mu_obs,data["sigma_Y"],observed=[data["Y_obs"]])
    return model



def add_emergent_constraint(
    model,
    latent_var_name,
    observable_var_name,
    observable_CMIP,
    observable_obs,
    observable_obs_sigma,
    priors={}
):
    """
    Add an emergent constraint submodel:
        observable_CMIP = m latent_var_CMIP + b
        observable_obs ~ Normal(m X_true + b, observable_obs_sigma)
    """

    with model:
        # Inherit X_true and X_m from the random effects model
        X_true=model[f"{latent_var_name}"]
        X_m = model[f"{latent_var_name}_CMIP"]
        
        # Emergent constraint may be calculated on a subset of models
        # Reindex to align
        observable_CMIP=observable_CMIP.reindex(
        model=model.coords["model"],
        fill_value=np.nan,
        )
        
        # Only grab the models that have simulated observables
        mask=~np.isnan(observable_CMIP.values)

        
        # ===== Regression parameters =====
        if "m" in priors:
            m = priors["m"](f"m_{observable_var_name}")
        else:
            m = pm.Normal(f"m_{observable_var_name}", 0, 5)

        if "b" in priors:
            b = priors["b"](f"b_{observable_var_name}")
        else:
            b = pm.Normal(f"b_{observable_var_name}", 0, 5)

        if "regression_sigma" in priors:
            sigma_Y = priors["regression_sigma"](f"regression_sigma_{observable_var_name}")
        else:
            sigma_Y = pm.HalfNormal(f"regression_sigma_{observable_var_name}", 1.0)

        # ===== Emergent relationship in the ensemble =====
        pm.Normal(
            f"{observable_var_name}_CMIP_lik",
            mu=m * X_m[mask] + b,
            sigma=sigma_Y,
            observed=observable_CMIP[mask].values,
        )

        # ===== Real-world emergent constraint =====
        pm.Normal(
            f"{observable_var_name}_obs_lik",
            mu=m * X_true + b,
            sigma=observable_obs_sigma,
            observed=observable_obs,
        )
