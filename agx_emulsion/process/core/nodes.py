import numpy as np
import skimage.transform
import colour
from opt_einsum import contract

from agx_emulsion.process.core.pipeline import Node, PipelineContext
from agx_emulsion.process.utils.autoexposure import measure_autoexposure_ev
from agx_emulsion.process.utils.crop_resize import crop_image
from agx_emulsion.process.physics.emulsion import Film, compute_density_spectral, develop_simple, remove_viewing_glare_comp, compute_random_glare_amount
from agx_emulsion.process.physics.diffusion import apply_gaussian_blur_um, apply_halation_um, apply_unsharp_mask, apply_gaussian_blur
from agx_emulsion.process.physics.color_filters import color_enlarger, compute_band_pass_filter
from agx_emulsion.process.physics.illuminants import standard_illuminant
from agx_emulsion.process.utils.conversions import density_to_light
from agx_emulsion.process.utils.spectral_upsampling import rgb_to_raw_mallett2019, rgb_to_raw_hanatos2025
from agx_emulsion.process.utils.lut import compute_with_lut
from agx_emulsion.process.config import ENLARGER_STEPS, STANDARD_OBSERVER_CMFS

class AutoExposureNode(Node):
    def process(self, image, context: PipelineContext):
        params = context.params
        if params.camera.auto_exposure:
            input_color_space = params.io.input_color_space
            input_cctf = params.io.input_cctf_decoding
            method = params.camera.auto_exposure_method
            autoexposure_ev = measure_autoexposure_ev(image, input_color_space, input_cctf, method=method)
            exposure_ev = autoexposure_ev + params.camera.exposure_compensation_ev
        else:
            exposure_ev = params.camera.exposure_compensation_ev
        context.data['exposure_ev'] = exposure_ev
        return image

class CropAndRescaleNode(Node):
    def process(self, image, context: PipelineContext):
        params = context.params
        preview_resize_factor = params.io.preview_resize_factor
        upscale_factor = params.io.upscale_factor
        film_format_mm = params.camera.film_format_mm
        pixel_size_um = film_format_mm*1000 / np.max(image.shape)
        
        if params.io.crop:
            image = crop_image(image, center=params.io.crop_center, size=params.io.crop_size)
        if params.io.full_image:
            preview_resize_factor = 1.0
        if preview_resize_factor*upscale_factor != 1.0:
            image  = skimage.transform.rescale(image, preview_resize_factor*upscale_factor, channel_axis=2)
            pixel_size_um /= preview_resize_factor*upscale_factor
            
        context.data['preview_resize_factor'] = preview_resize_factor
        context.data['pixel_size_um'] = pixel_size_um
        return image

class ProfileChangesNode(Node):
    def process(self, image, context: PipelineContext):
        params = context.params
        if params.print_paper.glare.compensation_removal_factor > 0:
            le = params.print_paper.data.log_exposure
            dc = params.print_paper.data.density_curves
            dc_out = remove_viewing_glare_comp(le, dc,
                                      factor=params.print_paper.glare.compensation_removal_factor,
                                      density=params.print_paper.glare.compensation_removal_density,
                                      transition=params.print_paper.glare.compensation_removal_transition)
            params.print_paper.data.density_curves = dc_out
        
        if not params.io.full_image:
            params.negative.grain.active = False
            params.negative.halation.active = False
            
        return image

class FilmExposureNode(Node):
    def process(self, image, context: PipelineContext):
        params = context.params
        exposure_ev = context.data['exposure_ev']
        pixel_size_um = context.data['pixel_size_um']
        
        raw = self._rgb_to_film_raw(image, exposure_ev, params)
        raw = apply_gaussian_blur_um(raw, params.camera.lens_blur_um, pixel_size_um)
        raw = apply_halation_um(raw, params.negative.halation, pixel_size_um)
        
        if params.io.compute_film_raw:
            context.data['return_early'] = True
            return raw
            
        return raw

    def _rgb_to_film_raw(self, rgb, exposure_ev, params):
        sensitivity = 10**params.negative.data.log_sensitivity
        sensitivity = np.nan_to_num(sensitivity)
        
        if params.camera.filter_uv[0]>0 or params.camera.filter_ir[0]>0:
            band_pass_filter = compute_band_pass_filter(params.camera.filter_uv,
                                                        params.camera.filter_ir)
            sensitivity *= band_pass_filter[:,None]

        method = params.settings.rgb_to_raw_method
        raw = np.zeros_like(rgb)
        if method=='mallett2019':
            raw = rgb_to_raw_mallett2019(rgb,
                                         sensitivity,
                                         color_space=params.io.input_color_space,
                                         apply_cctf_decoding=params.io.input_cctf_decoding,
                                         reference_illuminant=params.negative.info.reference_illuminant)
        if method=='hanatos2025':
            raw = rgb_to_raw_hanatos2025(rgb,
                                         sensitivity,
                                         color_space=params.io.input_color_space,
                                         apply_cctf_decoding=params.io.input_cctf_decoding,
                                         reference_illuminant=params.negative.info.reference_illuminant)
        
        raw *= 2**exposure_ev
        return raw

class FilmDevelopmentNode(Node):
    supports_chunking = True
    def process(self, raw, context: PipelineContext):
        if context.data.get('return_early'): return raw
        
        params = context.params
        pixel_size_um = context.data['pixel_size_um']
        
        log_raw = np.log10(np.fmax(raw, 0.0) + 1e-10)
        film = Film(params.negative)
        density_cmy = film.develop(log_raw, pixel_size_um,
                                   use_fast_stats=params.settings.use_fast_stats)
        
        if params.debug.return_negative_density_cmy:
            context.data['return_early'] = True
            return density_cmy
            
        return density_cmy

class PrintExposureNode(Node):
    supports_chunking = True
    def process(self, density_cmy, context: PipelineContext):
        if context.data.get('return_early'): return density_cmy
        
        params = context.params
        if params.io.compute_negative:
            return density_cmy # Skip print steps if computing negative
            
        film_density_cmy_normalized = self._normalize_film_density(density_cmy, params)
        
        def spectral_calculation(density_cmy_n):
            density_cmy = self._denormalize_film_density(density_cmy_n, params)
            return self._film_density_cmy_to_print_log_raw(density_cmy, params)
            
        log_raw = self._spectral_lut_compute(film_density_cmy_normalized, spectral_calculation, params,
                                             use_lut=params.settings.use_enlarger_lut,
                                             save_enlarger_lut=True)
        return log_raw

    def _normalize_film_density(self, denisty_cmy, params):
        density_max = np.nanmax(params.negative.data.density_curves, axis=0)
        density_min = params.negative.grain.density_min
        density_max += density_min
        density_cmy_normalized = (denisty_cmy + density_min) / density_max
        return density_cmy_normalized
    
    def _denormalize_film_density(self, density_cmy_normalized, params):
        density_max = np.nanmax(params.negative.data.density_curves, axis=0)
        density_min = params.negative.grain.density_min
        density_max += density_min
        density_cmy = density_cmy_normalized * density_max - density_min
        return density_cmy

    def _film_density_cmy_to_print_log_raw(self, density_cmy, params):
        sensitivity = 10**params.print_paper.data.log_sensitivity
        sensitivity = np.nan_to_num(sensitivity)
        enlarger_light_source = standard_illuminant(params.enlarger.illuminant)
        raw = np.zeros_like(density_cmy)
        if not params.enlarger.just_preflash:
            density_spectral = compute_density_spectral(params.negative, density_cmy)
            print_illuminant = self._compute_print_illuminant(enlarger_light_source, params)
            light = density_to_light(density_spectral, print_illuminant)
            raw = contract('ijk, kl->ijl', light, sensitivity)
            raw *= params.enlarger.print_exposure
            raw_midgray_factor = self._compute_exposure_factor_midgray(sensitivity, print_illuminant, params)
            raw *= raw_midgray_factor
        raw_preflash = self._compute_raw_preflash(enlarger_light_source, sensitivity, params)
        raw += raw_preflash
        log_raw = np.log10(raw + 1e-10)
        return log_raw

    def _compute_print_illuminant(self, light_source, params):
        y_filter = params.enlarger.y_filter_neutral*ENLARGER_STEPS + params.enlarger.y_filter_shift
        m_filter = params.enlarger.m_filter_neutral*ENLARGER_STEPS + params.enlarger.m_filter_shift
        c_filter = params.enlarger.c_filter_neutral*ENLARGER_STEPS
        print_illuminant = color_enlarger(light_source, y_filter, m_filter, c_filter)
        return print_illuminant

    def _compute_preflash_illuminant(self, light_source, params):
        y_filter_preflash = params.enlarger.y_filter_neutral*ENLARGER_STEPS + params.enlarger.preflash_y_filter_shift
        m_filter_preflash = params.enlarger.m_filter_neutral*ENLARGER_STEPS + params.enlarger.preflash_m_filter_shift
        c_filter = params.enlarger.c_filter_neutral*ENLARGER_STEPS
        preflash_illuminant = color_enlarger(light_source, y_filter_preflash, m_filter_preflash, c_filter)
        return preflash_illuminant

    def _compute_raw_preflash(self, light_source, sensitivity, params):
        if params.enlarger.preflash_exposure > 0:
            preflash_illuminant = self._compute_preflash_illuminant(light_source, params)
            density_base = params.negative.data.dye_density[:, 3][None, None, :]
            light_preflash = density_to_light(density_base, preflash_illuminant)
            raw_preflash = contract('ijk, kl->ijl', light_preflash, sensitivity)
            raw_preflash *= params.enlarger.preflash_exposure
        else:
            raw_preflash = np.zeros((3))
        return raw_preflash

    def _compute_exposure_factor_midgray(self, sensitivity, print_illuminant, params):
        if params.enlarger.print_exposure_compensation:
            neg_exp_comp_ev = params.camera.exposure_compensation_ev
        else:
            neg_exp_comp_ev = 0.0
        rgb_midgray = np.array([[[0.184]*3]]) * 2**neg_exp_comp_ev
        
        # Need to call _rgb_to_film_raw logic here, but it's in FilmExposureNode. 
        # Duplicating for now or could make it a static utility.
        # Using a simplified version since we know inputs.
        
        # Re-using logic from FilmExposureNode via a temporary instance or utility would be better,
        # but for now I'll duplicate the minimal needed part.
        
        # Actually, let's just instantiate FilmExposureNode to use its method if possible, 
        # or better, move _rgb_to_film_raw to a utility function.
        # For now, I will duplicate the logic to avoid circular dependencies or complex refactoring mid-stream.
        
        sensitivity_neg = 10**params.negative.data.log_sensitivity
        sensitivity_neg = np.nan_to_num(sensitivity_neg)
        # Assuming no filters for midgray calc or same filters
        if params.camera.filter_uv[0]>0 or params.camera.filter_ir[0]>0:
             band_pass_filter = compute_band_pass_filter(params.camera.filter_uv, params.camera.filter_ir)
             sensitivity_neg *= band_pass_filter[:,None]
             
        raw_midgray = rgb_to_raw_hanatos2025(rgb_midgray, sensitivity_neg, color_space='sRGB', apply_cctf_decoding=False, reference_illuminant=params.negative.info.reference_illuminant)
        
        log_raw_midgray = np.log10(raw_midgray + 1e-10)
        density_cmy_midgray = develop_simple(params.negative, log_raw_midgray)
        density_spectral_midgray = compute_density_spectral(params.negative, density_cmy_midgray)
        light_midgray = density_to_light(density_spectral_midgray, print_illuminant)
        raw_midgray = contract('ijk, kl->ijl', light_midgray, sensitivity)
        factor = 1/raw_midgray[:,:,1]
        return factor

    def _spectral_lut_compute(self, data, spectral_calculation, params, use_lut=False, save_enlarger_lut=False, save_scanner_lut=False):
        steps = params.settings.lut_resolution
        if use_lut:
            data_out, lut = compute_with_lut(data, spectral_calculation, steps=steps)
            if save_enlarger_lut:
                params.debug.luts.enlarger_lut = lut
            if save_scanner_lut:
                params.debug.luts.scanner_lut = lut
        else:                                   
            data_out = spectral_calculation(data)
        return data_out

class PrintDevelopmentNode(Node):
    supports_chunking = True
    def process(self, log_raw, context: PipelineContext):
        if context.data.get('return_early'): return log_raw
        
        params = context.params
        if params.io.compute_negative:
             return log_raw # Pass through if computing negative (actually logic flow should handle this)
             
        density_cmy = develop_simple(params.print_paper, log_raw)
        
        if params.debug.return_print_density_cmy:
            context.data['return_early'] = True
            return density_cmy
            
        return density_cmy

class ScanSpectralNode(Node):
    supports_chunking = True
    def process(self, density_cmy, context: PipelineContext):
        if context.data.get('return_early'): return density_cmy
        
        params = context.params
        rgb = self._density_cmy_to_rgb(density_cmy, params)
        return rgb

    def _density_cmy_to_rgb(self, density_cmy, params):
        if params.io.compute_negative:
            density_cmy_n = self._normalize_film_density(density_cmy, params)
            profile = params.negative
        else:
            density_cmy_n = self._normalize_print_density(density_cmy, params)
            profile = params.print_paper
        scan_illuminant = standard_illuminant(profile.info.viewing_illuminant)
        normalization = np.sum(scan_illuminant * STANDARD_OBSERVER_CMFS[:, 1], axis=0)
        
        # spectral calculation
        def spectral_calculation(density_cmy_n):
            if params.io.compute_negative:
                density_cmy = self._denormalize_film_density(density_cmy_n, params)
            else:
                density_cmy = self._denormalize_print_density(density_cmy_n, params)
            density_spectral = compute_density_spectral(profile, density_cmy)
            light = density_to_light(density_spectral, scan_illuminant)            
            xyz = contract('ijk,kl->ijl', light, STANDARD_OBSERVER_CMFS[:]) / normalization
            log_xyz = np.log10(xyz + 1e-10)
            return log_xyz
            
        log_xyz = self._spectral_lut_compute(density_cmy_n, spectral_calculation, params,
                                             use_lut=params.settings.use_scanner_lut, save_scanner_lut=True)
        xyz = 10**log_xyz
        
        illuminant_xyz = contract('k,kl->l', scan_illuminant, STANDARD_OBSERVER_CMFS[:]) / normalization
        xyz = self.add_glare(xyz, illuminant_xyz, profile)
        illuminant_xy = colour.XYZ_to_xy(illuminant_xyz)
        rgb = colour.XYZ_to_RGB(xyz,
                                colourspace=params.io.output_color_space, 
                                apply_cctf_encoding=False,
                                illuminant=illuminant_xy)
        return rgb

    def add_glare(self, xyz, illuminant_xyz, profile):
        if profile.glare.active and profile.glare.percent>0:
            glare_amount = compute_random_glare_amount(profile.glare.percent,
                                                    profile.glare.roughness,
                                                    profile.glare.blur,
                                                    xyz.shape[:2])
            xyz += glare_amount[:,:,None] * illuminant_xyz[None,None,:]
        return xyz

    # Helper methods duplicated/adapted from PrintExposureNode/FilmExposureNode to avoid circular deps for now
    # Ideally these should be in a shared utility module
    def _normalize_film_density(self, denisty_cmy, params):
        density_max = np.nanmax(params.negative.data.density_curves, axis=0)
        density_min = params.negative.grain.density_min
        density_max += density_min
        density_cmy_normalized = (denisty_cmy + density_min) / density_max
        return density_cmy_normalized
    
    def _denormalize_film_density(self, density_cmy_normalized, params):
        density_max = np.nanmax(params.negative.data.density_curves, axis=0)
        density_min = params.negative.grain.density_min
        density_max += density_min
        density_cmy = density_cmy_normalized * density_max - density_min
        return density_cmy

    def _normalize_print_density(self, denisty_cmy, params):
        density_max = np.nanmax(params.print_paper.data.density_curves, axis=0)
        density_cmy_normalized = denisty_cmy / density_max
        return density_cmy_normalized
    
    def _denormalize_print_density(self, density_cmy_normalized, params):
        density_max = np.nanmax(params.print_paper.data.density_curves, axis=0)
        density_cmy = density_cmy_normalized * density_max
        return density_cmy

    def _spectral_lut_compute(self, data, spectral_calculation, params, use_lut=False, save_enlarger_lut=False, save_scanner_lut=False):
        steps = params.settings.lut_resolution
        if use_lut:
            data_out, lut = compute_with_lut(data, spectral_calculation, steps=steps)
            if save_enlarger_lut:
                params.debug.luts.enlarger_lut = lut
            if save_scanner_lut:
                params.debug.luts.scanner_lut = lut
        else:                                   
            data_out = spectral_calculation(data)
        return data_out

class ScanBlurNode(Node):
    supports_chunking = False
    def process(self, rgb, context: PipelineContext):
        if context.data.get('return_early'): return rgb
        params = context.params
        rgb = self._apply_blur_and_unsharp(rgb, params)
        rgb = self._apply_cctf_encoding_and_clip(rgb, params)
        return rgb

    def _apply_blur_and_unsharp(self, data, params):
        data = apply_gaussian_blur(data, params.scanner.lens_blur)
        unsharp_mask = params.scanner.unsharp_mask
        if unsharp_mask[0] > 0 and unsharp_mask[1] > 0:
            data = apply_unsharp_mask(data, sigma=unsharp_mask[0], amount=unsharp_mask[1])
        return data
    
    def _apply_cctf_encoding_and_clip(self, rgb, params):
        color_space = params.io.output_color_space
        if params.io.output_cctf_encoding:
            rgb = colour.RGB_to_RGB(rgb, color_space, color_space,
                    apply_cctf_decoding=False,
                    apply_cctf_encoding=True)
        rgb = np.clip(rgb, a_min=0, a_max=1)
        return rgb

class ScanNode(Node):
    # Wrapper for backward compatibility if needed, but we will replace usage in process.py
    # Or we can just implement process to call the two new nodes logic, but that defeats the chunking purpose if called as one node.
    # So we will deprecate this or leave it as is but unused in optimized pipeline.
    def process(self, density_cmy, context: PipelineContext):
        # Just delegate to new nodes logic sequentially without chunking benefit if used directly
        spectral_node = ScanSpectralNode()
        blur_node = ScanBlurNode()
        rgb = spectral_node.process(density_cmy, context)
        return blur_node.process(rgb, context)

class RescaleOutputNode(Node):
    supports_chunking = False
    def process(self, scan, context: PipelineContext):
        preview_resize_factor = context.data.get('preview_resize_factor', 1.0)
        if preview_resize_factor != 1.0:
            scan = skimage.transform.rescale(scan, 1/preview_resize_factor, channel_axis=2)
        return scan
