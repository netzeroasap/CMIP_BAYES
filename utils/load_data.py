from . import DATA_DIR
import numpy as np
import xarray as xr
import pandas as pd
import json
import os,glob


#Make sure it can see the system path
from pathlib import Path
import sys

# Add parent directory of the notebook (the project root) to sys.path
ROOT = Path().resolve().parent   # X/
sys.path.insert(0, str(ROOT))

# helper: inspect the directory structure
def print_tree(path=Path("."), prefix="", max_depth=4):
    if max_depth < 0:
        return
    entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(prefix + connector + entry.name)
        if entry.is_dir():
            extension = "    " if i == len(entries) - 1 else "│   "
            print_tree(entry, prefix + extension, max_depth - 1)

# prepare the data

def load_feedback(variable,generation="CMIP6"):
    variable_data = []
    variablemodels = []
    variablerips = []

    with open(
        DATA_DIR/'feedbacks/cmip56_forcing_feedback_ecs.json', 'r'
    ) as f:
        data = json.load(f)

    for model in data[generation]:
        for rip in data[generation][model]:
            variable_data.append(data[generation][model][rip][variable])
            variablemodels.append(model)
            variablerips.append(rip)

    
    return xr.DataArray(data=variable_data,coords={"model":variablemodels})

def feedback_dictionary(generation="CMIP6"):
    with open(
    DATA_DIR/'feedbacks/cmip56_forcing_feedback_ecs.json', 'r'
) as f:
            data=json.load(f)
            all_variables=list(next(iter(next(iter(data[generation].values())).values())).keys())
       # all_variables=data[generation]['ACCESS-CM2']['r1i1p1f1'].keys()
            evidence={}
            for vari in all_variables:
                evidence[vari]=load_feedback(vari,generation=generation)
            return evidence

def load_carbon_cycle(kind="4xCO2"):
    kind="4xCO2"
    betafile=DATA_DIR/"carbon_cycle/TCREsource_betagamma.csv"
    betagammadf=pd.read_csv(betafile)
    #cmip6=betagammadf[betagammadf.source=="CMIP6"]
    cmip6=betagammadf[betagammadf.source.isin(["CMIP6","CMIP6+"])]
    bgevidence={}
    bgevidence["betaL"]=xr.DataArray(data=getattr(cmip6,f"beta_L_{kind}").values,\
                          coords={"model":cmip6.model.values})
    bgevidence["gammaL"]=xr.DataArray(data=getattr(cmip6,f"gamma_L_{kind}").values,\
                          coords={"model":cmip6.model.values})
    return bgevidence




def get_temperature_anomaly(ds,year):
    annmeans=ds.tas.groupby("time.year").mean(dim="time")
    base_period=('1951','1980')
    base_clim=annmeans.sel(year=slice(*base_period)).mean()
    return float(annmeans.sel(year=year)-base_clim)

def load_temperature(year):
    damip_direc=DATA_DIR/"DAMIP/"
    damip_models=sorted(os.listdir(damip_direc))
    modelcoords=[]

    data=[]
    for model in damip_models:
        filenames=glob.glob(f"{damip_direc}/{model}/historical/*")
        for fil in filenames:
            with xr.open_dataset(fil) as ds:
                data+=[get_temperature_anomaly(ds,year)]
                modelcoords+=[model]

    return xr.DataArray(data=data,coords={"model":modelcoords})
