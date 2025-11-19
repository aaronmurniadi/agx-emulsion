import numpy as np
import scipy.ndimage
from opt_einsum import contract
from agx_emulsion.process.physics.density_curves import interpolate_exposure_to_density
from agx_emulsion.process.physics.couplers import compute_exposure_correction_dir_couplers, compute_dir_couplers_matrix, compute_density_curves_before_dir_couplers
from agx_emulsion.process.physics.grain import apply_grain_to_density, apply_grain_to_density_layers
from agx_emulsion.process.utils.fast_stats import fast_lognormal_from_mean_std
from agx_emulsion.process.utils.fast_interp import fast_interp

################################################################################
# AgXEmusion main class

def remove_viewing_glare_comp(le, dc, factor=0.2, density=1.0, transition=0.3):
    """
    Removes viewing glare compensation from the density curves of print paper.
    Parameters:
    le (numpy.ndarray): The log exposure values.
    dc (numpy.ndarray): density curves of the print paper. Shape (n,3).
    factor (float, optional): The factor by which to reduce the light exposure values of the shadows. (brighter shadows). Default is 0.1.
    density (float, optional): The density value of the transition point. Default is 1.2.
    transition (float, optional): The transition density range used for Gaussian filtering. Default is 0.3.
    Returns:
    numpy.ndarray: density curves with viewing glare compensation removed.
    """
    def _measure_slope(le, density_curve, le_center, range_ev=1):
        le_delta = np.log10(2**range_ev)/2
        le_0 = le_center - le_delta
        le_1 = le_center + le_delta
        density_0 = np.interp(le_0, le, density_curve)
        density_1 = np.interp(le_1, le, density_curve)
        slope = (density_1 - density_0)/(le_1 - le_0)
        return slope    
    
    dc_mean = np.mean(dc, axis=1)
    le_center = np.interp(density, dc_mean, le)
    slope = _measure_slope(le, dc_mean, le_center)
    le_step = np.mean(np.diff(le))
    dc_out = np.zeros_like(dc)
    for i in np.arange(3):
        le_nl = np.copy(le)
        le_nl[le>le_center] -= (le[le>le_center]-le_center)*factor
        le_transition = transition/slope
        le_nl = scipy.ndimage.gaussian_filter(le_nl, le_transition/le_step)
        dc_out[:,i] = np.interp(le_nl, le, dc[:,i])
    return dc_out

def lognorm_from_mean_std(M, S):
    """
    Returns a frozen lognormal distribution object (scipy.stats.rv_frozen)
    whose mean is M and std dev is S in linear space.
    """
    # 1. Compute sigma^2 in log-space
    sigma_sq = np.log(1.0 + (S**2) / (M**2))
    sigma = np.sqrt(sigma_sq)
    # 2. Compute mu in log-space
    mu = np.log(M) - 0.5 * sigma_sq
    # 3. In scipy.lognorm, 's' = sigma (the shape), and 'scale' = exp(mu)
    return scipy.stats.lognorm(s=sigma, scale=np.exp(mu))

def compute_random_glare_amount(amount, roughness, blur, shape):
    random_glare = fast_lognormal_from_mean_std(amount*np.ones(shape),
                                                roughness*amount*np.ones(shape))
    random_glare = scipy.ndimage.gaussian_filter(random_glare, blur)
    random_glare /= 100
    return random_glare

def compute_density_spectral(profile, density_cmy):
    density_spectral = contract('ijk, lk->ijl', density_cmy, profile.data.dye_density[:, 0:3])
    density_spectral += profile.data.dye_density[:, 3] * profile.data.tune.dye_density_min_factor
    return density_spectral

def develop_simple(profile, log_raw):
    density_curves = profile.data.density_curves
    log_exposure = profile.data.log_exposure
    gamma_factor = profile.data.tune.gamma_factor
    density_cmy = interpolate_exposure_to_density(log_raw, density_curves, log_exposure, gamma_factor)
    return density_cmy

class AgXEmulsion():
    def __init__(self, profile):
        self.sensitivity = 10**np.array(profile.data.log_sensitivity)
        self.dye_density = np.array(profile.data.dye_density)
        self.density_curves = np.array(profile.data.density_curves)
        self.density_curves_layers = np.array(profile.data.density_curves_layers)
        self.log_exposure = np.array(profile.data.log_exposure)
        self.wavelengths = np.array(profile.data.wavelengths)
        
        self.parametric = profile.parametric
        self.type = profile.info.type
        self.stock = profile.info.stock
        self.reference_illuminant = profile.info.reference_illuminant
        self.viewing_illuminant = profile.info.viewing_illuminant
        self.gamma_factor = profile.data.tune.gamma_factor
        self.dye_density_min_factor = profile.data.tune.dye_density_min_factor
        
        self.density_curves -= np.nanmin(self.density_curves, axis=0)
        self.sensitivity = np.nan_to_num(self.sensitivity) # replace nans with zeros
        self.midgray_value = 0.184 # in linear light value, no cctf applied
        self.midgray_rgb = np.array([[[self.midgray_value]*3]])
    
    ################################################################################
    # Generic methods

################################################################################

class Film(AgXEmulsion):
    def __init__(self, profile):
        super().__init__(profile)
        self.grain = profile.grain
        self.halation = profile.halation
        self.dir_couplers = profile.dir_couplers
        self.density_midscale_neutral = profile.info.density_midscale_neutral

    def develop(self, log_raw, pixel_size_um,
                bypass_grain=False,
                use_fast_stats=False,
                ):
        
        # 1. Interpolate density with curves
        density_cmy = interpolate_exposure_to_density(log_raw, self.density_curves, self.log_exposure, self.gamma_factor)

        # 2. Apply density correction dir couplers
        if self.dir_couplers.active:
            # compute inhibitors matrix with super a simplified diffusion model
            dir_couplers_amount_rgb = self.dir_couplers.amount * np.array(self.dir_couplers.ratio_rgb)
            M = compute_dir_couplers_matrix(dir_couplers_amount_rgb, self.dir_couplers.diffusion_interlayer)
            # compute density curves before dir couplers
            density_curves_0 = compute_density_curves_before_dir_couplers(self.density_curves, self.log_exposure, M, self.dir_couplers.high_exposure_shift)
            # compute exposure correction
            density_max = np.nanmax(self.density_curves, axis=0)
            diffusion_size_um = self.dir_couplers.diffusion_size_um
            diffusion_size_pixel = diffusion_size_um/pixel_size_um
            log_raw_0 = compute_exposure_correction_dir_couplers(log_raw, density_cmy, density_max, M, 
                                                                 diffusion_size_pixel, self.dir_couplers.high_exposure_shift)
            # interpolated with corrected curves
            density_cmy = interpolate_exposure_to_density(log_raw_0, density_curves_0, self.log_exposure, self.gamma_factor)

        # 3. Apply grain
        if self.grain.active and not bypass_grain:
            if not self.grain.sublayers_active:
                density_max = np.nanmax(self.density_curves, axis=0)
                density_cmy = apply_grain_to_density(density_cmy,
                                                    pixel_size_um=pixel_size_um,
                                                    agx_particle_area_um2=self.grain.agx_particle_area_um2,
                                                    agx_particle_scale=self.grain.agx_particle_scale,
                                                    density_min=self.grain.density_min,
                                                    density_max_curves=density_max,
                                                    grain_uniformity=self.grain.uniformity,
                                                    grain_blur=self.grain.blur,
                                                    n_sub_layers=self.grain.n_sub_layers)
            else:
                density_cmy_layers = interp_density_cmy_layers(density_cmy, self.density_curves, self.density_curves_layers)
                density_max_layers = np.nanmax(self.density_curves_layers, axis=0)
                density_cmy = apply_grain_to_density_layers(density_cmy_layers,
                                                            density_max_layers=density_max_layers,
                                                            pixel_size_um=pixel_size_um,
                                                            agx_particle_area_um2=self.grain.agx_particle_area_um2,
                                                            agx_particle_scale=self.grain.agx_particle_scale,
                                                            agx_particle_scale_layers=self.grain.agx_particle_scale_layers,
                                                            density_min=self.grain.density_min,
                                                            grain_uniformity=self.grain.uniformity,
                                                            grain_blur=self.grain.blur,
                                                            grain_blur_dye_clouds_um=self.grain.blur_dye_clouds_um,
                                                            grain_micro_structure=self.grain.micro_structure,
                                                            use_fast_stats=use_fast_stats)
        
        return density_cmy

    def get_density_mid(self):
        # assumes that dye density cmy are already scaled to fit the mid diffuse density
        d_mid = self.density_midscale_neutral
        density_spectral = np.sum(self.dye_density[:, :3] * d_mid, axis=1) + self.dye_density[:, 3]
        return density_spectral[None,None,:]


def interp_density_cmy_layers(density_cmy, density_curves, density_curves_layers):
    density_cmy_layers = np.zeros((density_cmy.shape[0], density_cmy.shape[1], 3, 3)) # x,y,layer,rgb
    for ch in np.arange(3):
            density_cmy_layers[:,:,:,ch] = fast_interp(np.repeat(density_cmy[:,:,ch,np.newaxis], 3, -1),
                                                     density_curves[:,ch], density_curves_layers[:,:,ch])
    return density_cmy_layers
    
################################################################################

if __name__=='__main__':
    from agx_emulsion.tests.test_main_simulation import test_main_simulation
    test_main_simulation()