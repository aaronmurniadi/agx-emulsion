import numpy as np
import copy
from dotmap import DotMap

from agx_emulsion.pipeline.pipeline import Pipeline, PipelineContext
from agx_emulsion.pipeline.nodes import (
    AutoExposureNode, CropAndRescaleNode, ProfileChangesNode,
    FilmExposureNode, FilmDevelopmentNode, PrintExposureNode,
    PrintDevelopmentNode, ScanNode, RescaleOutputNode
)
from agx_emulsion.utils.io import read_neutral_ymc_filter_values
from agx_emulsion.profiles.io import load_profile
from agx_emulsion.utils.timings import timeit, plot_timings

ymc_filters = read_neutral_ymc_filter_values()

def photo_params(negative='kodak_portra_400_auc',
                 print_paper='kodak_portra_endura_uc',
                 ymc_filters_from_database=True):
    params = DotMap()
    params.negative = load_profile(negative)
    params.print_paper = load_profile(print_paper)
    params.camera = DotMap()
    params.enlarger = DotMap()
    params.scanner = DotMap()
    params.io = DotMap()
    
    params.camera.exposure_compensation_ev = 0.0
    params.camera.auto_exposure = True
    params.camera.auto_exposure_method = 'center_weighted'
    params.camera.lens_blur_um = 0.0 # about 5 um sigma for typical lenses, down to 2-4 um for high quality lenses, used for sharp simulations without lens blur.
    params.camera.film_format_mm = 35.0
    params.camera.filter_uv = (1, 410, 8)
    params.camera.filter_ir = (1, 675, 15)
    
    params.enlarger.illuminant = 'TH-KG3-L'
    params.enlarger.print_exposure = 1.0
    params.enlarger.print_exposure_compensation = True
    params.enlarger.y_filter_shift = 0.0
    params.enlarger.m_filter_shift = 0.0
    if ymc_filters_from_database:
        params.enlarger.y_filter_neutral = ymc_filters[print_paper][params.enlarger.illuminant][negative][0]
        params.enlarger.m_filter_neutral = ymc_filters[print_paper][params.enlarger.illuminant][negative][1]
        params.enlarger.c_filter_neutral = ymc_filters[print_paper][params.enlarger.illuminant][negative][2]
    else:
        params.enlarger.y_filter_neutral = 0.9
        params.enlarger.m_filter_neutral = 0.5
        params.enlarger.c_filter_neutral = 0.35
    params.enlarger.lens_blur = 0.0
    params.enlarger.preflash_exposure = 0.0
    params.enlarger.preflash_y_filter_shift = 0.0
    params.enlarger.preflash_m_filter_shift = 0.0
    params.enlarger.just_preflash = False
    
    params.scanner.lens_blur = 0.55
    params.scanner.unsharp_mask = (0.7,1.0)

    params.io.input_color_space = 'ProPhoto RGB'
    params.io.input_cctf_decoding = False
    params.io.output_color_space = 'sRGB'
    params.io.output_cctf_encoding = True
    params.io.crop = False
    params.io.crop_center = (0.5,0.5)
    params.io.crop_size = (0.1, 1.0)
    params.io.preview_resize_factor = 1.0
    params.io.upscale_factor = 1.0
    params.io.full_image = False
    params.io.compute_negative = False
    params.io.compute_film_raw = False
    
    params.debug.deactivate_spatial_effects = False
    params.debug.deactivate_stochastic_effects = False
    params.debug.input_negative_density_cmy = False
    params.debug.return_negative_density_cmy = False
    params.debug.return_print_density_cmy = False
    params.debug.print_timings = False
    
    params.settings.rgb_to_raw_method = 'hanatos2025'
    params.settings.use_camera_lut = True
    params.settings.use_enlarger_lut = False
    params.settings.use_scanner_lut = False
    params.settings.lut_resolution = 17
    params.settings.use_fast_stats = False
    
    return params

class AgXPhoto():
    def __init__(self, params):
        self._params = copy.deepcopy(params)
        # main components
        self.camera = params.camera
        self.negative = params.negative
        self.enlarger = params.enlarger
        self.print_paper = params.print_paper
        self.scanner = params.scanner
        # auxiliary and special
        self.io = params.io
        self.debug = params.debug
        self.settings = params.settings
        self.timings = {} # dictionary to hold timing info
        self._apply_debug_switches()

    def _apply_debug_switches(self):
        if self.debug.deactivate_spatial_effects:
            self.negative.halation.size_um = [0,0,0]
            self.negative.halation.scattering_size_um = [0,0,0]
            self.negative.dir_couplers.diffusion_size_um = 0
            self.negative.grain.blur = 0.0
            self.negative.grain.blur_dye_clouds_um = 0.0
            self.print_paper.glare.blur = 0
            self.camera.lens_blur_um = 0.0
            self.enlarger.lens_blur = 0.0
            self.scanner.lens_blur = 0.0
            self.scanner.unsharp_mask = (0.0, 0.0)

        if self.debug.deactivate_stochastic_effects:
            self.negative.grain.active = False
            self.negative.glare.active = False
            self.print_paper.glare.active = False

    def process(self, image):
        image = np.double(np.array(image)[:,:,0:3])
        
        # Initialize pipeline
        pipeline = Pipeline()
        pipeline.add_node(AutoExposureNode())
        pipeline.add_node(CropAndRescaleNode())
        pipeline.add_node(ProfileChangesNode())
        pipeline.add_node(FilmExposureNode())
        pipeline.add_node(FilmDevelopmentNode())
        pipeline.add_node(PrintExposureNode())
        pipeline.add_node(PrintDevelopmentNode())
        pipeline.add_node(ScanNode())
        pipeline.add_node(RescaleOutputNode())
        
        # Create context
        context = PipelineContext(self._params)
        
        # Run pipeline
        result = pipeline.run(image, context)
        
        # Update timings if needed (not fully implemented in nodes yet, but keeping structure)
        # self.timings = context.timings 
        
        return result
        


def photo_process(image, params):
    photo = AgXPhoto(params)
    image_out = photo.process(image)
    if params.debug.print_timings:
        print(photo.timings)
        plot_timings(photo.timings)
    return image_out
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from agx_emulsion.utils.io import load_image_oiio
    from agx_emulsion.utils.numba_warmup import warmup
    warmup()
    # image = load_image_oiio('img/targets/cc_halation.png')
    # image = plt.imread('img/targets/it87_test_chart_2.jpg')
    # image = np.double(image[:,:,:3])/255
    image = load_image_oiio('img/test/portrait_leaves_32bit_linear_prophoto_rgb.tif')
    # image = [[[0.184,0.184,0.184]]]
    # image = [[[0,0,0], [0.184,0.184,0.184], [1,1,1]]]
    params = photo_params(print_paper='kodak_portra_endura_uc')
    params.io.input_cctf_decoding = True
    params.print_paper.glare.active = False
    params.debug.deactivate_stochastic_effects = False
    params.camera.exposure_compensation_ev = 0
    params.camera.auto_exposure = True
    params.io.preview_resize_factor = 0.3
    params.io.upscale_factor = 3.0
    params.io.full_image = False
    params.io.compute_negative = False
    params.negative.grain.agx_particle_area_um2 = 1
    params.enlarger.preflash_exposure = 0.0
    params.enlarger.print_exposure_compensation = True
    params.enlarger.print_exposure = 1.0
    params.negative.grain.active = False
    params.debug.return_negative_density_cmy = False
    params.debug.return_print_density_cmy = False
    
    params.settings.use_fast_stats = True
    params.settings.use_camera_lut = True
    params.settings.use_enlarger_lut = True
    params.settings.use_scanner_lut = True
    params.settings.lut_resolution = 32
    params.debug.print_timings = True
    image = photo_process(image, params)
    # plt.imshow(image[:,:,1])
    plt.imshow(image)
    plt.show()