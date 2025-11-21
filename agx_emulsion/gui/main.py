import numpy as np
import napari
import json
from enum import Enum
from napari.layers import Image
from napari.types import ImageData
from napari.settings import get_settings
from magicgui import magicgui

from pathlib import Path
from dotmap import DotMap
from magicclass import magicclass, field, vfield, set_design, set_options, magicmenu, magiccontext
from napari.qt.threading import thread_worker

from agx_emulsion.process.config import ENLARGER_STEPS
from agx_emulsion.process.utils.io import load_image_oiio
from agx_emulsion.process.core.process import  photo_params, photo_process
from agx_emulsion.process.physics.stocks import FilmStocks, PrintPapers, Illuminants
from agx_emulsion.process.profiles.factory import swap_channels
from agx_emulsion.process.utils.numba_warmup import warmup

# precompile numba functions
warmup()

class RGBColorSpaces(Enum):
    sRGB = 'sRGB'
    DCI_P3 = 'DCI-P3'
    DisplayP3 = 'Display P3'
    AdobeRGB = 'Adobe RGB (1998)'
    ITU_R_BT2020 = 'ITU-R BT.2020'
    ProPhotoRGB = 'ProPhoto RGB'
    ACES2065_1 = 'ACES2065-1'

class RGBtoRAWMethod(Enum):
    hanatos2025 = 'hanatos2025'
    mallett2019 = 'mallett2019'

class AutoExposureMethods(Enum):
    median = 'median'
    center_weighted = 'center_weighted'

@magicclass(labels=False, name="Run")
class Run:
    def __init__(self, ui):
        self._ui = ui

    def run_simulation(self):
        """Run the simulation."""
        self._ui._run_simulation()

@magicclass(widget_type="scrollable")
class AgXEmulsionConfiguration:
    def __init__(self):
        self._viewer = None

    @magicclass(name="Input Image")
    class Input:
        filename = field(Path("./"), label="File", options={"mode": "r"})
        
        def load_image(self):
            """Load the image from the specified file."""
            if not self.filename.value.exists() or self.filename.value.is_dir():
                return
            img_array = load_image_oiio(str(self.filename.value))
            img_array = img_array[..., :3]

            # Get the viewer from the parent
            viewer = self.__magicclass_parent__._viewer
            if viewer:
                viewer.add_image(img_array, name="Input Image")

    @magicclass(name="Basic Options")
    class BasicOptions:
        compute_full_image = vfield(False, label="Compute Full Image", options={"tooltip": "Do not apply preview resize, compute full resolution image. Keeps the crop if active."})
        preview_resize_factor = vfield(0.3, label="Preview Resize", options={"tooltip": "Scale image size down (0-1) to speed up preview processing"})
        upscale_factor = vfield(1.0, label="Upscale Factor", options={"tooltip": "Scale image size up to increase resolution"})
        crop = vfield(False, label="Crop", options={"tooltip": "Crop image to a fraction of the original size to preview details at full scale"})
        crop_center = vfield((0.50, 0.50), label="Crop Center", options={"tooltip": "Center of the crop region in relative coordinates in x, y (0-1)"})
        crop_size = vfield((0.1, 0.1), label="Crop Size", options={"tooltip": "Normalized size of the crop region in x, y (0,1), as fraction of the long side."})
        input_color_space = vfield(RGBColorSpaces.ProPhotoRGB, label="Color Space", options={"tooltip": "Color space of the input image, will be internally converted to sRGB and negative values clipped"})
        apply_cctf_decoding = vfield(False, label="Apply CCTF Decoding", options={"tooltip": "Apply the inverse cctf transfer function of the color space"})
        spectral_upsampling_method = vfield(RGBtoRAWMethod.hanatos2025, label="Spectral Upsampling", options={"tooltip": "Method to upsample the spectral resolution of the image, hanatos2025 works on the full visible locus, mallett2019 works only on sRGB (will clip input)."})
        filter_uv = vfield((1, 410, 8), label="Filter UV", options={"tooltip": "Filter UV light, (amplitude, wavelength cutoff in nm, sigma in nm). It mainly helps for avoiding UV light ruining the reds. Changing this enlarger filters neutral will be affected."})
        filter_ir = vfield((1, 675, 15), label="Filter IR", options={"tooltip": "Filter IR light, (amplitude, wavelength cutoff in nm, sigma in nm). Changing this enlarger filters neutral will be affected."})
        

    @magicclass(name="Film Simulation")
    class Film:
        film_stock = vfield(FilmStocks.kodak_vision3_500t, label="Film Stock", options={"tooltip": "Film stock to simulate"})
        film_format_mm = vfield(35.0, label="Format (mm)", options={"tooltip": "Long edge of the film format in millimeters, e.g. 35mm or 60mm"})
        camera_lens_blur_um = vfield(0.0, label="Lens Blur (um)", options={"tooltip": "Sigma of gaussian filter in um for the camera lens blur. About 5 um for typical lenses, down to 2-4 um for high quality lenses, used for sharp input simulations without lens blur."})
        exposure_compensation_ev = vfield(0.0, label="Exposure Comp (EV)", options={"min": -100, "max": 100, "step": 0.5, "tooltip": "Exposure compensation value in ev of the negative"})
        auto_exposure = vfield(False, label="Auto Exposure", options={"tooltip": "Automatically adjust exposure based on the image content"})
        auto_exposure_method = vfield(AutoExposureMethods.center_weighted, label="Auto Exposure Method")

        @magicclass(name="Grain")
        class Grain:
            active = vfield(True, label="Active", options={"tooltip": "Add grain to the negative"})
            sublayers_active = vfield(True, label="Sublayers Active")
            particle_area_um2 = vfield(0.1, label="Particle Area (um2)", options={"step": 0.1, "tooltip": "Area of the particles in um2, relates to ISO. Approximately 0.1 for ISO 100, 0.1 for ISO 200, 0.4 for ISO 400 and so on."})
            particle_scale = vfield((0.8, 1.0, 2), label="Particle Scale", options={"tooltip": "Scale of particle area for the RGB layers, multiplies particle_area_um2"})
            particle_scale_layers = vfield((2.5, 1.0, 0.5), label="Particle Scale Layers", options={"tooltip": "Scale of particle area for the sublayers in every color layer, multiplies particle_area_um2"})
            density_min = vfield((0.07, 0.08, 0.12), label="Density Min", options={"tooltip": "Minimum density of the grain, typical values (0.03-0.06)"})
            uniformity = vfield((0.97, 0.97, 0.99), label="Uniformity", options={"tooltip": "Uniformity of the grain, typical values (0.94-0.98)"})
            blur = vfield(0.65, label="Blur", options={"tooltip": "Sigma of gaussian blur in pixels for the grain, to be increased at high magnifications, (should be 0.8-0.9 at high resolution, reduce down to 0.6 for lower res)."})
            blur_dye_clouds_um = vfield(1.0, label="Blur Dye Clouds (um)", options={"tooltip": "Scale the sigma of gaussian blur in um for the dye clouds, to be used at high magnifications, (default 1)"})
            micro_structure = vfield((0.1, 30), label="Micro Structure", options={"tooltip": "Parameter for micro-structure due to clumps at the molecular level, [sigma blur of micro-structure / ultimate light-resolution (0.10 um default), size of molecular clumps in nm (30 nm default)]. Only for insane magnifications."})

        @magicclass(name="Halation")
        class Halation:
            active = vfield(True, label="Active")
            scattering_strength = vfield((1.0, 2.0, 4.0), label="Scattering Strength", options={"tooltip": "Fraction of scattered light (0-100, percentage) for each channel [R,G,B]"})
            scattering_size_um = vfield((30, 20, 15), label="Scattering Size (um)", options={"tooltip": "Size of the scattering effect in micrometers for each channel [R,G,B], sigma of gaussian filter."})
            halation_strength = vfield((10.0, 7.30, 7.1), label="Halation Strength", options={"tooltip": "Fraction of halation light (0-100, percentage) for each channel [R,G,B]"})
            halation_size_um = vfield((200, 200, 200), label="Halation Size (um)", options={"tooltip": "Size of the halation effect in micrometers for each channel [R,G,B], sigma of gaussian filter."})

        @magicclass(name="Couplers")
        class Couplers:
            active = vfield(True, label="Active")
            dir_couplers_amount = vfield(1.0, label="Amount", options={"step": 0.05, "tooltip": "Amount of coupler inhibitors, control saturation, typical values (0.8-1.2)."})
            dir_couplers_ratio = vfield((1.0, 1.0, 1.0), label="Ratio")
            dir_couplers_diffusion_um = vfield(10, label="Diffusion (um)", options={"step": 5, "tooltip": "Sigma in um for the diffusion of the couplers, (5-20 um), controls sharpness and affects saturation."})
            diffusion_interlayer = vfield(2.0, label="Diffusion Interlayer", options={"tooltip": "Sigma in number of layers for diffusion across the rgb layers (typical layer thickness 3-5 um, so roughly 1.0-4.0 layers), affects saturation."})
            high_exposure_shift = vfield(0.0, label="High Exposure Shift")

    @magicclass(name="Print Simulation")
    class Print:
        print_paper = vfield(PrintPapers.kodak_supra_endura, label="Print Paper", options={"tooltip": "Print paper to simulate"})
        print_illuminant = vfield(Illuminants.lamp, label="Illuminant", options={"tooltip": "Print illuminant to simulate"})
        print_exposure = vfield(1.0, label="Exposure", options={"step": 0.05, "tooltip": "Exposure value for the print (proportional to seconds of exposure, not ev)"})
        print_exposure_compensation = vfield(False, label="Exposure Comp", options={"tooltip": "Apply exposure compensation from negative exposure compensation ev, allow for changing of the negative exposure compensation while keeping constant print time."})
        print_y_filter_shift = vfield(0, label="Y Filter Shift", options={"min": -ENLARGER_STEPS, "max": ENLARGER_STEPS, "tooltip": "Y filter shift of the color enlarger from a neutral position, enlarger has 170 steps"})
        print_m_filter_shift = vfield(0, label="M Filter Shift", options={"min": -ENLARGER_STEPS, "max": ENLARGER_STEPS, "tooltip": "M filter shift of the color enlarger from a neutral position, enlarger has 170 steps"})

        @magicclass(name="Preflashing")
        class Preflashing:
            exposure = vfield(0.0, label="Exposure", options={"step": 0.005, "tooltip": "Preflash exposure value in ev for the print"})
            y_filter_shift = vfield(0, label="Y Filter Shift", options={"min": -ENLARGER_STEPS, "tooltip": "Shift the Y filter of the enlarger from the neutral position for the preflash, typical values (-20-20), enlarger has 170 steps"})
            m_filter_shift = vfield(0, label="M Filter Shift", options={"min": -ENLARGER_STEPS, "tooltip": "Shift the M filter of the enlarger from the neutral position for the preflash, typical values (-20-20), enlarger has 170 steps"})
            just_preflash = vfield(False, label="Just Preflash", options={"tooltip": "Only apply preflash to the print, to visualize the preflash effect"})

        @magicclass(name="Glare")
        class Glare:
            active = vfield(True, label="Active", options={"tooltip": "Add glare to the print"})
            percent = vfield(0.10, label="Percent", options={"step": 0.05, "tooltip": "Percentage of the glare light (typically 0.1-0.25)"})
            roughness = vfield(0.4, label="Roughness", options={"tooltip": "Roughness of the glare light (0-1)"})
            blur = vfield(0.5, label="Blur", options={"tooltip": "Sigma of gaussian blur in pixels for the glare"})
            compensation_removal_factor = vfield(0.0, label="Comp Removal Factor", options={"step": 0.05, "tooltip": "Factor of glare compensation removal from the print, e.g. 0.2=20% underexposed print in the shadows, typical values (0.0-0.2). To be used instead of stochastic glare (i.e. when percent=0)."})
            compensation_removal_density = vfield(1.2, label="Comp Removal Density", options={"tooltip": "Density of the glare compensation removal from the print, typical values (1.0-1.5)."})
            compensation_removal_transition = vfield(0.3, label="Comp Removal Transition", options={"tooltip": "Transition density range of the glare compensation removal from the print, typical values (0.1-0.5)."})

    @magicclass(name="Scanner")
    class Scanner:
        scan_lens_blur = vfield(0.00, label="Lens Blur", options={"step": 0.05, "tooltip": "Sigma of gaussian filter in pixel for the scanner lens blur"})
        scan_unsharp_mask = vfield((0.7, 0.7), label="Unsharp Mask", options={"tooltip": "Apply unsharp mask to the scan, [sigma in pixel, amount]"})
        output_color_space = vfield(RGBColorSpaces.sRGB, label="Output Color Space", options={"tooltip": "Color space of the output image"})
        output_cctf_encoding = vfield(True, label="Output CCTF Encoding", options={"tooltip": "Apply the cctf transfer function of the color space. If false, data is linear."})
        compute_negative = vfield(False, label="Compute Negative", options={"tooltip": "Show a scan of the negative instead of the print"})

    @magicclass(name="Advanced")
    class Advanced:
        film_channel_swap = vfield((0, 1, 2), label="Film Channel Swap")
        film_gamma_factor = vfield(1.0, label="Film Gamma Factor", options={"tooltip": "Gamma factor of the density curves of the negative, < 1 reduce contrast, > 1 increase contrast"})
        print_channel_swap = vfield((0, 1, 2), label="Print Channel Swap")
        print_gamma_factor = vfield(1.0, label="Print Gamma Factor", options={"step": 0.05, "tooltip": "Gamma factor of the print paper, < 1 reduce contrast, > 1 increase contrast"})
        print_density_min_factor = vfield(0.4, label="Print Density Min Factor", options={"min": 0, "max": 1, "step": 0.2, "tooltip": "Minimum density factor of the print paper (0-1), make the white less white"})

    def _run_simulation(self):
        input_layer = None
        if self._viewer:
            for layer in self._viewer.layers:
                if isinstance(layer, Image):
                    input_layer = layer
                    break
        
        if input_layer is None:
            print("No image layer found.")
            return

        # Gather parameters
        params = photo_params(self.Film.film_stock.value, self.Print.print_paper.value)
        
        # Special
        if self.Advanced.film_channel_swap != (0, 1, 2):
            params.negative = swap_channels(params.negative, self.Advanced.film_channel_swap)
        if self.Advanced.print_channel_swap != (0, 1, 2):
            params.print_paper = swap_channels(params.print_paper, self.Advanced.print_channel_swap)
        
        params.negative.data.tune.gamma_factor = self.Advanced.film_gamma_factor
        params.print_paper.data.tune.gamma_factor = self.Advanced.print_gamma_factor
        params.print_paper.data.tune.dye_density_min_factor = self.Advanced.print_density_min_factor
        
        # Glare
        params.print_paper.glare.active = self.Print.Glare.active
        params.print_paper.glare.percent = self.Print.Glare.percent
        params.print_paper.glare.roughness = self.Print.Glare.roughness
        params.print_paper.glare.blur = self.Print.Glare.blur
        params.print_paper.glare.compensation_removal_factor = self.Print.Glare.compensation_removal_factor
        params.print_paper.glare.compensation_removal_density = self.Print.Glare.compensation_removal_density
        params.print_paper.glare.compensation_removal_transition = self.Print.Glare.compensation_removal_transition

        # Camera
        params.camera.lens_blur_um = self.Film.camera_lens_blur_um
        params.camera.exposure_compensation_ev = self.Film.exposure_compensation_ev
        params.camera.auto_exposure = self.Film.auto_exposure
        params.camera.auto_exposure_method = self.Film.auto_exposure_method.value
        params.camera.film_format_mm = self.Film.film_format_mm
        params.camera.filter_uv = self.BasicOptions.filter_uv
        params.camera.filter_ir = self.BasicOptions.filter_ir
        
        # IO
        params.io.preview_resize_factor = self.BasicOptions.preview_resize_factor
        params.io.upscale_factor = self.BasicOptions.upscale_factor
        params.io.crop = self.BasicOptions.crop
        params.io.crop_center = self.BasicOptions.crop_center
        params.io.crop_size = self.BasicOptions.crop_size
        params.io.input_color_space = self.BasicOptions.input_color_space.value
        params.io.input_cctf_decoding = self.BasicOptions.apply_cctf_decoding
        params.io.output_color_space = self.Scanner.output_color_space.value
        params.io.output_cctf_encoding = self.Scanner.output_cctf_encoding
        params.io.full_image = self.BasicOptions.compute_full_image
        params.io.compute_negative = self.Scanner.compute_negative
        
        # Halation
        params.negative.halation.active = self.Film.Halation.active
        params.negative.halation.strength = np.array(self.Film.Halation.halation_strength)/100
        params.negative.halation.size_um = np.array(self.Film.Halation.halation_size_um)
        params.negative.halation.scattering_strength = np.array(self.Film.Halation.scattering_strength)/100
        params.negative.halation.scattering_size_um = np.array(self.Film.Halation.scattering_size_um)
        
        # Grain
        params.negative.grain.active = self.Film.Grain.active
        params.negative.grain.sublayers_active = self.Film.Grain.sublayers_active
        params.negative.grain.agx_particle_area_um2 = self.Film.Grain.particle_area_um2
        params.negative.grain.agx_particle_scale = self.Film.Grain.particle_scale
        params.negative.grain.agx_particle_scale_layers = self.Film.Grain.particle_scale_layers
        params.negative.grain.density_min = self.Film.Grain.density_min
        params.negative.grain.uniformity = self.Film.Grain.uniformity
        params.negative.grain.blur = self.Film.Grain.blur
        params.negative.grain.blur_dye_clouds_um = self.Film.Grain.blur_dye_clouds_um
        params.negative.grain.micro_structure = self.Film.Grain.micro_structure
        
        # Couplers
        params.negative.dir_couplers.active = self.Film.Couplers.active
        params.negative.dir_couplers.amount = self.Film.Couplers.dir_couplers_amount 
        params.negative.dir_couplers.ratio_rgb = self.Film.Couplers.dir_couplers_ratio
        params.negative.dir_couplers.diffusion_size_um = self.Film.Couplers.dir_couplers_diffusion_um
        params.negative.dir_couplers.diffusion_interlayer = self.Film.Couplers.diffusion_interlayer
        params.negative.dir_couplers.high_exposure_shift = self.Film.Couplers.high_exposure_shift
        
        # Enlarger
        params.enlarger.illuminant = self.Print.print_illuminant.value
        params.enlarger.print_exposure = self.Print.print_exposure
        params.enlarger.print_exposure_compensation = self.Print.print_exposure_compensation
        params.enlarger.y_filter_shift = self.Print.print_y_filter_shift
        params.enlarger.m_filter_shift = self.Print.print_m_filter_shift
        params.enlarger.preflash_exposure = self.Print.Preflashing.exposure
        params.enlarger.preflash_y_filter_shift = self.Print.Preflashing.y_filter_shift
        params.enlarger.preflash_m_filter_shift = self.Print.Preflashing.m_filter_shift
        params.enlarger.just_preflash = self.Print.Preflashing.just_preflash
        
        # Scanner
        params.scanner.lens_blur = self.Scanner.scan_lens_blur
        params.scanner.unsharp_mask = self.Scanner.scan_unsharp_mask
        
        # Settings
        params.settings.rgb_to_raw_method = self.BasicOptions.spectral_upsampling_method.value
        params.settings.use_camera_lut = False
        params.settings.use_enlarger_lut = True
        params.settings.use_scanner_lut = True
        params.settings.lut_resolution = 32
        params.settings.use_fast_stats = True

        image = np.double(input_layer.data[:,:,:3])
        
        # Run async
        worker = self._process_image(image, params)
        worker.returned.connect(self._on_process_finished)
        worker.start()

    @thread_worker
    def _process_image(self, image, params):
        scan = photo_process(image, params)
        scan = np.uint8(scan*255)
        return scan

    def _on_process_finished(self, scan):
        if self._viewer:
            layer_name = "Simulation Result"
            if layer_name in self._viewer.layers:
                self._viewer.layers[layer_name].data = scan
            else:
                self._viewer.add_image(scan, name=layer_name)

def main():
    # create a viewer
    viewer = napari.Viewer(title='AGX-Emulsion')
    viewer.window._qt_viewer.dockLayerControls.setVisible(False)
    viewer.window._qt_viewer.dockLayerList.setVisible(False)
    layer_list = viewer.window._qt_viewer.dockLayerList
    
    settings = get_settings()
    settings.appearance.theme = 'system'

    # Instantiate the GUI
    configuration = AgXEmulsionConfiguration()
    configuration._viewer = viewer
    
    # Instantiate Run Control
    run = Run(configuration)

    # Add widgets to viewer
    viewer.window.add_dock_widget(configuration, area="right", name="Configuration", tabify=False)
    viewer.window.add_dock_widget(layer_list, area="right", name="Layers", tabify=False)
    viewer.window.add_dock_widget(run, area="right", name="Run", tabify=False)

    napari.run()

if __name__ == "__main__":
    main()
