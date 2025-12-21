import json
import copy
import numpy as np
from dotmap import DotMap
import importlib.resources as pkg_resources
import math


def replace_nan_and_none(obj):
    """
    Recursively replaces:
        - None with np.nan
        - NaN (float or np.nan) with None
    """
    if isinstance(obj, list):
        return [replace_nan_and_none(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: replace_nan_and_none(v) for k, v in obj.items()}
    elif obj is None:
        return np.nan
    elif isinstance(obj, float) and (math.isnan(obj) or np.isnan(obj)):
        return None
    else:
        return obj


def save_profile(profile, suffix=""):
    profile.info.stock = profile.info.stock + suffix
    profile = copy.copy(profile)
    # convert to lists to make it json serializable
    profile.data.log_sensitivity = profile.data.log_sensitivity.tolist()
    profile.data.density_curves = profile.data.density_curves.tolist()
    profile.data.density_curves_layers = profile.data.density_curves_layers.tolist()
    profile.data.dye_density = profile.data.dye_density.tolist()
    profile.data.log_exposure = profile.data.log_exposure.tolist()
    profile.data.wavelengths = profile.data.wavelengths.tolist()
    package = pkg_resources.files("agx_emulsion.data.profiles")
    filename = profile.info.stock + ".json"
    resource = package / filename
    print("Saving to:", filename)
    profile_dict = profile.toDict()
    profile_dict = replace_nan_and_none(profile_dict)
    with resource.open("w") as file:
        json.dump(profile_dict, file, indent=4)


def load_profile(stock):
    package = pkg_resources.files("agx_emulsion.data.profiles")
    filename = stock + ".json"
    resource = package / filename
    profile = DotMap()
    with resource.open("r") as file:
        data = json.load(file)

    data = replace_nan_and_none(data)
    profile = DotMap(data)

    # Use asarray to avoid copy if already numpy arrays (though unlikely from JSON)
    profile.data.log_sensitivity = np.asarray(profile.data.log_sensitivity)
    profile.data.dye_density = np.asarray(profile.data.dye_density)
    profile.data.density_curves = np.asarray(profile.data.density_curves)
    profile.data.log_exposure = np.asarray(profile.data.log_exposure)
    profile.data.wavelengths = np.asarray(profile.data.wavelengths)
    profile.data.density_curves_layers = np.asarray(profile.data.density_curves_layers)
    return profile
