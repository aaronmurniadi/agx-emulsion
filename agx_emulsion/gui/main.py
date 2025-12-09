import numpy as np
import napari
import json
from enum import Enum
from napari.layers import Image
from napari.types import ImageData
from napari.settings import get_settings
from napari.utils import progress
from qtpy.QtCore import Qt, QObject, Signal
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QCheckBox, QComboBox, QDoubleSpinBox, QSpinBox, 
    QTabWidget, QScrollArea, QFileDialog, QGroupBox, QFormLayout
)

from pathlib import Path
from dotmap import DotMap
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

class WorkerSignals(QObject):
    progress = Signal(str, int, int)

class AgXEmulsionConfiguration(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self._viewer = viewer
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Input Section
        input_layout = QHBoxLayout()
        
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setWordWrap(True)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.load_image)
        
        input_layout.addWidget(QLabel("File:"))
        input_layout.addWidget(self.file_path_label)
        input_layout.addWidget(browse_btn)
        layout.addLayout(input_layout)

        # Run Section
        run_layout = QHBoxLayout()
        
        run_btn = QPushButton("Run Simulation")
        run_btn.clicked.connect(self.run_simulation)
        self.compute_full_image = QCheckBox("Compute Full Image")
        self.compute_full_image.setToolTip("Do not apply preview resize, compute full resolution image. Keeps the crop if active.")
        
        run_layout.addWidget(run_btn)
        run_layout.addWidget(self.compute_full_image)
        layout.addLayout(run_layout)

        # Settings Tabs
        self.tabs = QTabWidget()
        
        # Initialize control references containers
        self.film_controls = DotMap()
        self.print_controls = DotMap()
        self.scanner_controls = DotMap()
        self.advanced_controls = DotMap()
        self.misc_controls = DotMap()

        self.setup_film_tab()
        self.setup_print_tab()
        self.setup_scanner_tab()
        self.setup_advanced_tab()
        self.setup_misc_tab()

        layout.addWidget(self.tabs)
        
        # Add stretch to keep widgets at top
        # layout.addStretch() # remove stretch to let tab expand? or use scroll area inside tabs

    def create_scroll_area(self, widget):
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        return scroll

    def add_spin(self, layout, label, value, min_val=0.0, max_val=100.0, step=1.0, decimals=2, tooltip="", key=None, storage=None):
        if isinstance(value, float):
            spin = QDoubleSpinBox()
            spin.setDecimals(decimals)
            spin.setSingleStep(step)
            spin.setRange(min_val, max_val)
        else:
            spin = QSpinBox()
            spin.setSingleStep(int(step))
            spin.setRange(int(min_val), int(max_val))
        
        spin.setValue(value)
        spin.setToolTip(tooltip)
        
        layout.addRow(label, spin)
        if key and storage is not None:
            storage[key] = spin
        return spin

    def add_combo(self, layout, label, enum_class, default_value, tooltip="", key=None, storage=None):
        combo = QComboBox()
        for i, item in enumerate(enum_class):
            combo.addItem(item.value, item)
            if item == default_value:
                combo.setCurrentIndex(i)
        
        combo.setToolTip(tooltip)
        layout.addRow(label, combo)
        if key and storage is not None:
            storage[key] = combo
        return combo

    def add_checkbox(self, layout, label, value, tooltip="", key=None, storage=None):
        cb = QCheckBox()
        cb.setChecked(value)
        cb.setToolTip(tooltip)
        layout.addRow(label, cb)
        if key and storage is not None:
            storage[key] = cb
        return cb

    def add_tuple_spin(self, layout, label, value_tuple, min_val=0.0, max_val=100.0, step=1.0, tooltip="", key=None, storage=None, value_labels=None):
        container = QWidget()
        v_layout = QVBoxLayout(container)
        v_layout.setContentsMargins(0, 0, 0, 0)
        
        spins = []
        for i, val in enumerate(value_tuple):
            if isinstance(val, float):
                s = QDoubleSpinBox()
                s.setDecimals(2)
                s.setSingleStep(step)
                s.setRange(min_val, max_val)
            else:
                s = QSpinBox()
                s.setSingleStep(int(step))
                s.setRange(int(min_val), int(max_val))
            s.setValue(val)
            s.setToolTip(tooltip)
            
            if value_labels and i < len(value_labels):
                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)
                lbl = QLabel(value_labels[i])
                lbl.setStyleSheet("font-size: 80%; color: #888;")
                row_layout.addWidget(lbl)
                row_layout.addWidget(s)
                v_layout.addWidget(row_widget)
            else:
                v_layout.addWidget(s)
                
            spins.append(s)
            
        layout.addRow(label, container)
        if key and storage is not None:
            storage[key] = spins
        return spins

    def setup_film_tab(self):
        # Main widget for the Film tab
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # General Settings Section
        general_group = QGroupBox()
        general_layout = QFormLayout(general_group)
        s = self.film_controls
        
        self.add_combo(general_layout, "Film Stock", FilmStocks, FilmStocks.kodak_vision3_500t, "Film stock to simulate", "film_stock", s)
        self.add_spin(general_layout, "Format (mm)", 35.0, 1.0, 1000.0, 1.0, 1, "Long edge of the film format in millimeters", "film_format_mm", s)
        self.add_spin(general_layout, "Lens Blur (um)", 0.0, 0.0, 100.0, 0.1, 2, "Sigma of gaussian filter in um for the camera lens blur", "camera_lens_blur_um", s)
        self.add_spin(general_layout, "Exposure Comp (EV)", 0.0, -100.0, 100.0, 0.5, 2, "Exposure compensation value in ev of the negative", "exposure_compensation_ev", s)
        self.add_checkbox(general_layout, "Auto Exposure", False, "Automatically adjust exposure based on the image content", "auto_exposure", s)
        self.add_combo(general_layout, "Auto Exposure Method", AutoExposureMethods, AutoExposureMethods.center_weighted, "", "auto_exposure_method", s)
        
        main_layout.addWidget(general_group)
        
        # Nested Tabs for Sub-sections
        sub_tabs = QTabWidget()
        
        # --- Grain Tab ---
        grain_tab = QWidget()
        grain_layout = QFormLayout(grain_tab)
        # s is still self.film_controls, sharing the same storage
        self.add_checkbox(grain_layout, "Sublayers Active", True, "", "sublayers_active", s)
        self.add_spin(grain_layout, "Particle Area (um2)", 0.1, 0.0, 100.0, 0.1, 2, "Area of the particles in um2, relates to ISO", "particle_area_um2", s)
        self.add_tuple_spin(grain_layout, "Particle Scale", (0.8, 1.0, 2.0), 0.0, 100.0, 0.1, "Scale of particle area for the RGB layers", "particle_scale", s, value_labels=("R", "G", "B"))
        self.add_tuple_spin(grain_layout, "Particle Scale Layers", (2.5, 1.0, 0.5), 0.0, 100.0, 0.1, "Scale of particle area for the sublayers", "particle_scale_layers", s, value_labels=("R", "G", "B"))
        self.add_tuple_spin(grain_layout, "Density Min", (0.07, 0.08, 0.12), 0.0, 1.0, 0.01, "Minimum density of the grain", "density_min", s, value_labels=("R", "G", "B"))
        self.add_tuple_spin(grain_layout, "Uniformity", (0.97, 0.97, 0.99), 0.0, 1.0, 0.01, "Uniformity of the grain", "uniformity", s, value_labels=("R", "G", "B"))
        self.add_spin(grain_layout, "Blur", 0.65, 0.0, 100.0, 0.05, 2, "Sigma of gaussian blur in pixels for the grain", "blur", s)
        self.add_spin(grain_layout, "Blur Dye Clouds (um)", 1.0, 0.0, 100.0, 0.1, 2, "Scale the sigma of gaussian blur in um for the dye clouds", "blur_dye_clouds_um", s)
        self.add_tuple_spin(grain_layout, "Micro Structure", (0.1, 30), 0.0, 100.0, 0.1, "Parameter for micro-structure", "micro_structure", s, value_labels=("Sigma", "Size"))
        sub_tabs.addTab(self.create_scroll_area(grain_tab), "Grain")
        
        # --- Halation Tab ---
        halation_tab = QWidget()
        halation_layout = QFormLayout(halation_tab)
        self.add_tuple_spin(halation_layout, "Scattering Strength", (1.0, 2.0, 4.0), 0.0, 100.0, 0.1, "Fraction of scattered light (0-100)", "scattering_strength", s, value_labels=("R", "G", "B"))
        self.add_tuple_spin(halation_layout, "Scattering Size (um)", (30, 20, 15), 0.0, 1000.0, 1.0, "Size of the scattering effect in micrometers", "scattering_size_um", s, value_labels=("R", "G", "B"))
        self.add_tuple_spin(halation_layout, "Halation Strength", (10.0, 7.30, 7.1), 0.0, 100.0, 0.1, "Fraction of halation light (0-100)", "halation_strength", s, value_labels=("R", "G", "B"))
        self.add_tuple_spin(halation_layout, "Halation Size (um)", (200, 200, 200), 0.0, 2000.0, 1.0, "Size of the halation effect in micrometers", "halation_size_um", s, value_labels=("R", "G", "B"))
        sub_tabs.addTab(self.create_scroll_area(halation_tab), "Halation")

        # --- Couplers Tab ---
        couplers_tab = QWidget()
        couplers_layout = QFormLayout(couplers_tab)
        self.add_checkbox(couplers_layout, "Active", True, "", "couplers_active", s)
        self.add_spin(couplers_layout, "Amount", 1.0, 0.0, 10.0, 0.05, 2, "Amount of coupler inhibitors", "dir_couplers_amount", s)
        self.add_tuple_spin(couplers_layout, "Ratio", (1.0, 1.0, 1.0), 0.0, 10.0, 0.1, "", "dir_couplers_ratio", s, value_labels=("R", "G", "B"))
        self.add_spin(couplers_layout, "Diffusion (um)", 10, 0, 1000, 5, 0, "Sigma in um for the diffusion of the couplers", "dir_couplers_diffusion_um", s)
        self.add_spin(couplers_layout, "Diffusion Interlayer", 2.0, 0.0, 100.0, 0.1, 2, "Sigma for diffusion across rgb layers", "diffusion_interlayer", s)
        self.add_spin(couplers_layout, "High Exposure Shift", 0.0, -10.0, 10.0, 0.1, 2, "", "high_exposure_shift", s)
        sub_tabs.addTab(self.create_scroll_area(couplers_tab), "Couplers")
        
        main_layout.addWidget(sub_tabs)
        
        self.tabs.addTab(main_widget, "Film")

    def setup_print_tab(self):
        # Main widget for the Print tab
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # General Settings Section
        general_group = QGroupBox()
        general_layout = QFormLayout(general_group)
        s = self.print_controls

        self.add_combo(general_layout, "Print Paper", PrintPapers, PrintPapers.kodak_supra_endura, "Print paper to simulate", "print_paper", s)
        self.add_combo(general_layout, "Illuminant", Illuminants, Illuminants.lamp, "Print illuminant to simulate", "print_illuminant", s)
        self.add_spin(general_layout, "Exposure", 1.0, 0.0, 100.0, 0.05, 2, "Exposure value for the print", "print_exposure", s)
        self.add_checkbox(general_layout, "Exposure Comp", False, "Apply exposure compensation from negative", "print_exposure_compensation", s)
        self.add_spin(general_layout, "Y Filter Shift", 0, -ENLARGER_STEPS, ENLARGER_STEPS, 1, 0, "Y filter shift", "print_y_filter_shift", s)
        self.add_spin(general_layout, "M Filter Shift", 0, -ENLARGER_STEPS, ENLARGER_STEPS, 1, 0, "M filter shift", "print_m_filter_shift", s)

        main_layout.addWidget(general_group)

        # Nested Tabs
        sub_tabs = QTabWidget()

        # --- Preflashing Tab ---
        preflash_tab = QWidget()
        preflash_layout = QFormLayout(preflash_tab)
        self.add_spin(preflash_layout, "Exposure", 0.0, 0.0, 100.0, 0.005, 3, "Preflash exposure value", "preflash_exposure", s)
        self.add_spin(preflash_layout, "Y Filter Shift", 0, -ENLARGER_STEPS, ENLARGER_STEPS, 1, 0, "Shift Y filter for preflash", "preflash_y_filter_shift", s)
        self.add_spin(preflash_layout, "M Filter Shift", 0, -ENLARGER_STEPS, ENLARGER_STEPS, 1, 0, "Shift M filter for preflash", "preflash_m_filter_shift", s)
        self.add_checkbox(preflash_layout, "Just Preflash", False, "Only apply preflash", "just_preflash", s)
        sub_tabs.addTab(self.create_scroll_area(preflash_tab), "Preflashing")

        # --- Glare Tab ---
        glare_tab = QWidget()
        glare_layout = QFormLayout(glare_tab)
        self.add_checkbox(glare_layout, "Active", True, "Add glare to the print", "glare_active", s)
        self.add_spin(glare_layout, "Percent", 0.10, 0.0, 1.0, 0.05, 2, "Percentage of the glare light", "percent", s)
        self.add_spin(glare_layout, "Roughness", 0.4, 0.0, 1.0, 0.05, 2, "Roughness of the glare light", "roughness", s)
        self.add_spin(glare_layout, "Blur", 0.5, 0.0, 100.0, 0.1, 2, "Sigma of gaussian blur", "blur", s)
        self.add_spin(glare_layout, "Comp Removal Factor", 0.0, 0.0, 1.0, 0.05, 2, "Factor of glare compensation removal", "compensation_removal_factor", s)
        self.add_spin(glare_layout, "Comp Removal Density", 1.2, 0.0, 10.0, 0.1, 2, "Density of the glare compensation removal", "compensation_removal_density", s)
        self.add_spin(glare_layout, "Comp Removal Transition", 0.3, 0.0, 10.0, 0.1, 2, "Transition density range", "compensation_removal_transition", s)
        sub_tabs.addTab(self.create_scroll_area(glare_tab), "Glare")

        main_layout.addWidget(sub_tabs)

        self.tabs.addTab(main_widget, "Print")

    def setup_scanner_tab(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        s = self.scanner_controls

        self.add_spin(layout, "Lens Blur", 0.00, 0.0, 100.0, 0.05, 2, "Sigma of gaussian filter in pixel", "scan_lens_blur", s)
        self.add_tuple_spin(layout, "Unsharp Mask", (0.7, 0.7), 0.0, 100.0, 0.1, "Apply unsharp mask [sigma, amount]", "scan_unsharp_mask", s, value_labels=("Sigma", "Amount"))
        self.add_combo(layout, "Output Color Space", RGBColorSpaces, RGBColorSpaces.sRGB, "Color space of the output image", "output_color_space", s)
        self.add_checkbox(layout, "Output CCTF Encoding", True, "Apply the cctf transfer function", "output_cctf_encoding", s)
        self.tabs.addTab(self.create_scroll_area(widget), "Scanner")

    def setup_advanced_tab(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        s = self.advanced_controls

        self.add_tuple_spin(layout, "Film Channel Swap", (0, 1, 2), 0, 2, 1, "", "film_channel_swap", s, value_labels=("0", "1", "2"))
        self.add_spin(layout, "Film Gamma Factor", 1.0, 0.0, 10.0, 0.05, 2, "Gamma factor of the density curves", "film_gamma_factor", s)
        self.add_tuple_spin(layout, "Print Channel Swap", (0, 1, 2), 0, 2, 1, "", "print_channel_swap", s, value_labels=("0", "1", "2"))
        self.add_spin(layout, "Print Gamma Factor", 1.0, 0.0, 10.0, 0.05, 2, "Gamma factor of the print paper", "print_gamma_factor", s)
        self.add_spin(layout, "Print Density Min Factor", 0.4, 0.0, 1.0, 0.2, 2, "Minimum density factor", "print_density_min_factor", s)

        self.tabs.addTab(self.create_scroll_area(widget), "Advanced")

    def setup_misc_tab(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        s = self.misc_controls

        self.add_spin(layout, "Preview Resize", 0.3, 0.0, 1.0, 0.1, 2, "Scale image size down", "preview_resize_factor", s)
        self.add_spin(layout, "Upscale Factor", 1.0, 0.1, 10.0, 0.1, 2, "Scale image size up", "upscale_factor", s)
        self.add_checkbox(layout, "Crop", False, "Crop image", "crop", s)
        self.add_tuple_spin(layout, "Crop Center", (0.50, 0.50), 0.0, 1.0, 0.01, "Center of the crop region", "crop_center", s, value_labels=("X", "Y"))
        self.add_tuple_spin(layout, "Crop Size", (0.1, 0.1), 0.0, 1.0, 0.01, "Normalized size of the crop region", "crop_size", s, value_labels=("X", "Y"))
        self.add_combo(layout, "Color Space", RGBColorSpaces, RGBColorSpaces.ProPhotoRGB, "Color space of the input image", "input_color_space", s)
        self.add_checkbox(layout, "Apply CCTF Decoding", False, "Apply the inverse cctf transfer function", "apply_cctf_decoding", s)
        self.add_combo(layout, "Spectral Upsampling", RGBtoRAWMethod, RGBtoRAWMethod.hanatos2025, "Method to upsample the spectral resolution", "spectral_upsampling_method", s)
        self.add_tuple_spin(layout, "Filter UV", (1, 410, 8), 0, 1000, 1, "Filter UV light", "filter_uv", s, value_labels=("Amp", "Wavelen", "Sigma"))
        self.add_tuple_spin(layout, "Filter IR", (1, 675, 15), 0, 1000, 1, "Filter IR light", "filter_ir", s, value_labels=("Amp", "Wavelen", "Sigma"))

        self.tabs.addTab(self.create_scroll_area(widget), "Misc.")

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Image", 
            str(Path.cwd()), 
            "Images (*.tif *.tiff *.jpg *.jpeg *.png *.arw *.cr2 *.nef *.dng)"
        )
        
        if not file_path:
            return

        path = Path(file_path)
        self.file_path_label.setText(path.name)
        
        ext = path.suffix.lower()
        img_array = None
        
        # Check for RAW file extensions
        raw_extensions = ['.arw', '.cr2', '.nef', '.dng']
        if ext in raw_extensions:
            try:
                import sys
                import importlib.util
                
                # Locate the scripts/prepare_input.py file
                current_dir = Path(__file__).parent
                script_path = current_dir.parent.parent / "scripts" / "prepare_input.py"
                
                if not script_path.exists():
                    print(f"Error: Could not find prepare_input.py at {script_path}")
                    return

                # Dynamic import
                spec = importlib.util.spec_from_file_location("prepare_input", script_path)
                prepare_input = importlib.util.module_from_spec(spec)
                sys.modules["prepare_input"] = prepare_input
                spec.loader.exec_module(prepare_input)
                
                print(f"Loading RAW image: {file_path}")
                # Process the raw image
                img_array = prepare_input.process_raw_image(file_path)
                
            except Exception as e:
                print(f"Error loading RAW file: {e}")
                import traceback
                traceback.print_exc()
                return
        else:
            img_array = load_image_oiio(file_path)
            img_array = img_array[..., :3]

        if img_array is not None:
            if self._viewer:
                # Remove existing Input Image layer if present
                if "Input Image" in self._viewer.layers:
                    self._viewer.layers.remove("Input Image")
                    
                self._viewer.add_image(img_array, name="Input Image")
                self._viewer.reset_view()

    def get_tuple_val(self, widgets):
        # helper to extract tuple from list of spinboxes/widgets
        return tuple(w.value() for w in widgets)

    def run_simulation(self):
        input_layer = None
        if self._viewer:
            for layer in self._viewer.layers:
                if isinstance(layer, Image) and layer.name == "Input Image": # Better robustness? Or just any image
                    input_layer = layer
                    break
            if input_layer is None and len(self._viewer.layers) > 0 and isinstance(self._viewer.layers[0], Image):
                 # Fallback to first image layer
                 input_layer = self._viewer.layers[0]
        
        if input_layer is None:
            print("No image layer found.")
            return

        # Helper to get value
        def v(control):
            if isinstance(control, (QComboBox)):
                return control.currentData()
            elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
                return control.value()
            elif isinstance(control, QCheckBox):
                return control.isChecked()
            elif isinstance(control, list):
                return self.get_tuple_val(control)
            return None

        # Gather parameters
        params = photo_params(v(self.film_controls.film_stock).value, v(self.print_controls.print_paper).value)
        
        # Special
        film_swap = v(self.advanced_controls.film_channel_swap)
        if film_swap != (0, 1, 2):
            params.negative = swap_channels(params.negative, film_swap)
        
        print_swap = v(self.advanced_controls.print_channel_swap)
        if print_swap != (0, 1, 2):
            params.print_paper = swap_channels(params.print_paper, print_swap)
        
        params.negative.data.tune.gamma_factor = v(self.advanced_controls.film_gamma_factor)
        params.print_paper.data.tune.gamma_factor = v(self.advanced_controls.print_gamma_factor)
        params.print_paper.data.tune.dye_density_min_factor = v(self.advanced_controls.print_density_min_factor)
        
        # Glare
        params.print_paper.glare.active = v(self.print_controls.glare_active)
        params.print_paper.glare.percent = v(self.print_controls.percent)
        params.print_paper.glare.roughness = v(self.print_controls.roughness)
        params.print_paper.glare.blur = v(self.print_controls.blur)
        params.print_paper.glare.compensation_removal_factor = v(self.print_controls.compensation_removal_factor)
        params.print_paper.glare.compensation_removal_density = v(self.print_controls.compensation_removal_density)
        params.print_paper.glare.compensation_removal_transition = v(self.print_controls.compensation_removal_transition)

        # Camera
        params.camera.lens_blur_um = v(self.film_controls.camera_lens_blur_um)
        params.camera.exposure_compensation_ev = v(self.film_controls.exposure_compensation_ev)
        params.camera.auto_exposure = v(self.film_controls.auto_exposure)
        params.camera.auto_exposure_method = v(self.film_controls.auto_exposure_method).value
        params.camera.film_format_mm = v(self.film_controls.film_format_mm)
        params.camera.filter_uv = v(self.misc_controls.filter_uv)
        params.camera.filter_ir = v(self.misc_controls.filter_ir)
        
        # IO
        params.io.preview_resize_factor = v(self.misc_controls.preview_resize_factor)
        params.io.upscale_factor = v(self.misc_controls.upscale_factor)
        params.io.crop = v(self.misc_controls.crop)
        params.io.crop_center = v(self.misc_controls.crop_center)
        params.io.crop_size = v(self.misc_controls.crop_size)
        params.io.input_color_space = v(self.misc_controls.input_color_space).value
        params.io.input_cctf_decoding = v(self.misc_controls.apply_cctf_decoding)
        params.io.output_color_space = v(self.scanner_controls.output_color_space).value
        params.io.output_cctf_encoding = v(self.scanner_controls.output_cctf_encoding)
        params.io.full_image = self.compute_full_image.isChecked()
        params.io.compute_negative = v(self.scanner_controls.compute_negative)
        
        # Halation
        # Note: 'active' for halation/grain was shared in original as Settings.Film.active.
        # I renamed it to couplers_active in setup_film_tab. 
        # But wait, looking at original:
        # Settings.Film.active was defined in [Couplers] section but used for halation, grain, and couplers.
        # I should probably use that same checkbox for all if that was the intent.
        # In my setup I made "couplers_active", "glare_active".
        # Let's see original: 
        # Line 148: active = vfield(True, label="Active") -> Under [Couplers] label. 
        # Line 265: params.negative.halation.active = self.Settings.Film.active
        # Line 272: params.negative.grain.active = self.Settings.Film.active
        # Line 284: params.negative.dir_couplers.active = self.Settings.Film.active
        # So yes, one active flag controls Halation, Grain, and Couplers.
        # I should rename my 'couplers_active' to something more general or just use it.
        
        general_active = v(self.film_controls.couplers_active)

        params.negative.halation.active = general_active
        params.negative.halation.strength = np.array(v(self.film_controls.halation_strength))/100
        params.negative.halation.size_um = np.array(v(self.film_controls.halation_size_um))
        params.negative.halation.scattering_strength = np.array(v(self.film_controls.scattering_strength))/100
        params.negative.halation.scattering_size_um = np.array(v(self.film_controls.scattering_size_um))
        
        # Grain
        params.negative.grain.active = general_active
        params.negative.grain.sublayers_active = v(self.film_controls.sublayers_active)
        params.negative.grain.agx_particle_area_um2 = v(self.film_controls.particle_area_um2)
        params.negative.grain.agx_particle_scale = v(self.film_controls.particle_scale)
        params.negative.grain.agx_particle_scale_layers = v(self.film_controls.particle_scale_layers)
        params.negative.grain.density_min = v(self.film_controls.density_min)
        params.negative.grain.uniformity = v(self.film_controls.uniformity)
        params.negative.grain.blur = v(self.film_controls.blur)
        params.negative.grain.blur_dye_clouds_um = v(self.film_controls.blur_dye_clouds_um)
        params.negative.grain.micro_structure = v(self.film_controls.micro_structure)
        
        # Couplers
        params.negative.dir_couplers.active = general_active
        params.negative.dir_couplers.amount = v(self.film_controls.dir_couplers_amount)
        params.negative.dir_couplers.ratio_rgb = v(self.film_controls.dir_couplers_ratio)
        params.negative.dir_couplers.diffusion_size_um = v(self.film_controls.dir_couplers_diffusion_um)
        params.negative.dir_couplers.diffusion_interlayer = v(self.film_controls.diffusion_interlayer)
        params.negative.dir_couplers.high_exposure_shift = v(self.film_controls.high_exposure_shift)
        
        # Enlarger
        params.enlarger.illuminant = v(self.print_controls.print_illuminant).value
        params.enlarger.print_exposure = v(self.print_controls.print_exposure)
        params.enlarger.print_exposure_compensation = v(self.print_controls.print_exposure_compensation)
        params.enlarger.y_filter_shift = v(self.print_controls.print_y_filter_shift)
        params.enlarger.m_filter_shift = v(self.print_controls.print_m_filter_shift)
        params.enlarger.preflash_exposure = v(self.print_controls.preflash_exposure)
        params.enlarger.preflash_y_filter_shift = v(self.print_controls.preflash_y_filter_shift)
        params.enlarger.preflash_m_filter_shift = v(self.print_controls.preflash_m_filter_shift)
        params.enlarger.just_preflash = v(self.print_controls.just_preflash)
        
        # Scanner
        params.scanner.lens_blur = v(self.scanner_controls.scan_lens_blur)
        params.scanner.unsharp_mask = v(self.scanner_controls.scan_unsharp_mask)
        
        # Settings
        params.settings.rgb_to_raw_method = v(self.misc_controls.spectral_upsampling_method).value
        params.settings.use_camera_lut = False
        params.settings.use_enlarger_lut = True
        params.settings.use_scanner_lut = True
        params.settings.lut_resolution = 32
        params.settings.use_fast_stats = True

        image = np.double(input_layer.data[:,:,:3])
        
        # Create signals and progress bar
        self.signals = WorkerSignals()
        pbr = progress(total=0)
        
        def on_progress(name, step, total):
            pbr.total = total
            pbr.set_description(f"Running: {name}")
            pbr.update(1)
            
        self.signals.progress.connect(on_progress)
        
        def on_finished(scan):
            pbr.close()
            self._on_process_finished(scan)
            if self._viewer:
                self._viewer.reset_view()
            
        def on_error(e):
            pbr.close()
            print(f"Error during simulation: {e}")
            import traceback
            traceback.print_exc()

        # Run async
        worker = self._process_image(image, params, progress_signal=self.signals.progress)
        worker.returned.connect(on_finished)
        worker.errored.connect(on_error)
        worker.start()

    @thread_worker
    def _process_image(self, image, params, progress_signal):
        def cb(name, step, total):
            progress_signal.emit(name, step, total)
            
        scan = photo_process(image, params, progress_callback=cb)
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
    configuration = AgXEmulsionConfiguration(viewer)
    
    # Add widgets to viewer
    viewer.window.add_dock_widget(configuration, area="right", name="Configuration", tabify=False)
    viewer.window.add_dock_widget(layer_list, area="right", name="Layers", tabify=False)

    napari.run()

if __name__ == "__main__":
    main()
