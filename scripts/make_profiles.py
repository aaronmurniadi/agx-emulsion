import matplotlib.pyplot as plt
import numpy as np
from agx_emulsion.process.profiles.factory import create_profile, process_negative_profile, process_paper_profile, plot_profile, replace_fitted_density_curves, adjust_log_exposure
from agx_emulsion.process.profiles.io import save_profile, load_profile
from agx_emulsion.process.utils.io import read_neutral_ymc_filter_values, save_ymc_filter_values
from agx_emulsion.process.profiles.correct import correct_negative_curves_with_gray_ramp, align_midscale_neutral_exposures
from agx_emulsion.process.utils.fit_print_filters import fit_print_filters
from agx_emulsion.process.core.process import photo_params

process_print_paper = False
process_negative = True

def calculate_and_save_ymc_filters(negative_stock, print_paper_stock, illuminant='TH-KG3-L'):
    """
    Calculate YMC filter values for neutral balance and save to database.
    This is used to populate the YMC filters database with new film-paper combinations.
    """
    print(f'Calculating YMC filters for {negative_stock} on {print_paper_stock}...')

    # Load existing YMC filters
    ymc_filters = read_neutral_ymc_filter_values()

    # Check if already exists
    if (print_paper_stock in ymc_filters and
        illuminant in ymc_filters[print_paper_stock] and
        negative_stock in ymc_filters[print_paper_stock][illuminant]):
        print(f'YMC filters already exist for {negative_stock} on {print_paper_stock}')
        return ymc_filters[print_paper_stock][illuminant][negative_stock]

    # Start with a basic parameter set
    params = photo_params(negative_stock, print_paper_stock, ymc_filters_from_database=False)

    # Modify settings for fitting
    params.enlarger.print_exposure_compensation = False  # Disable for fitting
    params.camera.auto_exposure = False
    params.debug.deactivate_spatial_effects = True
    params.debug.deactivate_stochastic_effects = True
    params.print_paper.glare.active = False
    params.settings.rgb_to_raw_method = 'mallett2019'
    params.io.input_cctf_decoding = False
    params.io.input_color_space = 'sRGB'
    params.io.output_cctf_encoding = False
    params.io.resize_factor = 1.0

    # Fit the filters
    y_filter, m_filter, _ = fit_print_filters(params, iterations=5)

    ymc_values = [y_filter, m_filter, params.enlarger.c_filter_neutral]
    print(f'Calculated YMC filters: [{ymc_values[0]:.3f}, {ymc_values[1]:.3f}, {ymc_values[2]:.3f}]')

    # Save to database
    if print_paper_stock not in ymc_filters:
        ymc_filters[print_paper_stock] = {}
    if illuminant not in ymc_filters[print_paper_stock]:
        ymc_filters[print_paper_stock][illuminant] = {}

    ymc_filters[print_paper_stock][illuminant][negative_stock] = ymc_values
    save_ymc_filter_values(ymc_filters)
    print(f'Saved YMC filters to database')

    return ymc_values

print('----------------------------------------')
print('Paper profiles')
#               label,                               name,                               ref_illu        illu    sens, curv, dye,  dom
paper_info = [('kodak_ektacolor_edge',              'Kodak Ektacolor Edge',              'TH-KG3-L',  'D50',  None, None, None, 1.0),
              ('kodak_ultra_endura',                'Kodak Professional Ultra Endura',   'TH-KG3-L',  'D50',  None, None, None, 1.0),
              ('kodak_endura_premier',              'Kodak Professional Endura Premier', 'TH-KG3-L',  'D50',  None, None, None, 1.0),
              ('kodak_portra_endura',               'Kodak Professional Portra Endura',  'TH-KG3-L',  'D50',  None, None, None, 1.0),
              ('kodak_supra_endura',                'Kodak Professional Supra Endura',   'TH-KG3-L',  'D50',  'kodak_portra_endura', None, 'kodak_portra_endura', 1.0),
              ('fujifilm_crystal_archive_typeii',   'Fujifilm Crystal Archive Type II',  'TH-KG3-L',  'D50',  None, 'kodak_supra_endura', None, 1.0),
              ('kodak_2393',                        'Kodak Vision Premier 2393',         'TH-KG3-L',  'K75P', None, None, None, 1.0),
              ('kodak_2383',                        'Kodak Vision 2383',                 'TH-KG3-L',  'K75P', None, None, None, 1.0),
]

if process_print_paper:
    for label, name, ref_illu, illu, sens, curv, dye, dom in paper_info:
        profile = create_profile(stock=label,
                                name=name,
                                type='paper',
                                log_sensitivity_donor=sens,
                                denisty_curves_donor=curv,
                                dye_density_cmy_donor=dye,
                                densitometer='status_A',
                                reference_illuminant=ref_illu,
                                viewing_illuminant=illu,
                                log_sensitivity_density_over_min=dom)
        save_profile(profile)
        plot_profile(profile)
        profile = process_paper_profile(profile)
        save_profile(profile, '_uc')


print('----------------------------------------')
print('Negative profiles')

#               label,                    name,                       suffix   dye_donor,   ls_donor            ddmm_donor           d_over_min, ref_ill target_paper,                align_mid_exp  trustability proc?
stock_info = [
              ('kodak_vision3_50d',      'Kodak Vision3 50D',         '',      None       , None,               None,                0.2,        'D55',  'kodak_2383_uc',             None,          0.3,         False),
              ('kodak_vision3_250d',     'Kodak Vision3 250D',        '',      None       , None,               None,                0.2,        'D55',  'kodak_2383_uc',             None,          0.3,         False),
              ('kodak_vision3_200t',     'Kodak Vision3 200T',        '',      None       , None,               None,                0.2,        'T',    'kodak_2383_uc',             None,          0.3,         False),
              ('kodak_vision3_500t',     'Kodak Vision3 500T',        '',      None       , None,               None,                0.2,        'T',    'kodak_2383_uc',             None,          0.3,         False),
              ('kodak_ektar_100',        'Kodak Ektar 100',           '',      'generic_a', None,               None,                0.2,        'D55',  'kodak_portra_endura_uc',    None,          1.0,         False),
              ('kodak_pro_image_100',    'Kodak Pro Image 100',       '',      'generic_a', None,               None,                0.2,        'D55',  'kodak_portra_endura_uc',    None,          1.0,         True),
              ('kodak_portra_160',       'Kodak Portra 160',          '',      'generic_a', None,               None,                0.2,        'D55',  'kodak_portra_endura_uc',    None,          1.0,         False),
              ('kodak_portra_400',       'Kodak Portra 400',          '',      'generic_a', None,               None,                0.2,        'D55',  'kodak_portra_endura_uc',    None,          1.0,         False),
              ('kodak_portra_800',       'Kodak Portra 800',          '',      'generic_a', None,               None,                0.2,        'D55',  'kodak_portra_endura_uc',    None,          1.0,         False),
              ('kodak_portra_800_push1', 'Kodak Portra 800 (Push 1)', '',      'generic_a', 'kodak_portra_800', 'kodak_portra_800',  0.2,        'D55',  'kodak_portra_endura_uc',    None,          1.0,         False),
              ('kodak_portra_800_push2', 'Kodak Portra 800 (Push 2)', '',      'generic_a', 'kodak_portra_800', 'kodak_portra_800',  0.2,        'D55',  'kodak_portra_endura_uc',    None,          1.0,         False),
              ('kodak_gold_200',         'Kodak Gold 200',            '',      'generic_a', None,               None,                0.2,        'D55',  'kodak_portra_endura_uc',    None,          1.0,         False),
              ('kodak_ultramax_400',     'Kodak Ultramax 400',        '',      'generic_a', None,               None,                0.2,        'D55',  'kodak_portra_endura_uc',    None,          1.0,         False),
              ('fujifilm_pro_400h',      'Fujifilm Pro 400H',         '',      'generic_a', None,               None,                1.0,        'D55',  'kodak_portra_endura_uc',    'mid',         0.3,         False),
              ('fujifilm_xtra_400',      'Fujifilm X-Tra 400',        '',      'generic_a', None,               None,                1.0,        'D55',  'kodak_portra_endura_uc',    None,          0.3,         False),
              ('fujifilm_c200',          'Fujifilm C200',             '',      'generic_a', None,               None,                1.0,        'D55',  'kodak_portra_endura_uc',    'green',       0.3,         False),
              ]

if process_negative:
    for label, name, suff, dye, ls_donor, ddmm_donor, d_over_min, ref_ill, target_paper, align_mid_exp, trustability, proc in stock_info:
        if not proc:
            continue
        profile = create_profile(stock=label,
                                 name=name,
                                 type='negative',
                                 densitometer='status_M',
                                 dye_density_cmy_donor=dye,
                                 log_sensitivity_donor=ls_donor,
                                 dye_density_min_mid_donor=ddmm_donor,
                                 reference_illuminant=ref_ill,
                                 log_sensitivity_density_over_min=d_over_min)
        save_profile(profile)
        suffix = '_'+suff
        if dye=='generic_a':
            suffix += 'a'
        profile = process_negative_profile(profile)
        save_profile(profile, suffix+'u')
        if align_mid_exp is not None:
            profile = align_midscale_neutral_exposures(profile, reference_channel=align_mid_exp)
        
        # Ensure paper profile has log_sensitivity before using it
        try:
            paper_profile = load_profile(target_paper)
            log_sens = getattr(paper_profile.data, 'log_sensitivity', None)
            needs_fix = False
            # Check if log_sensitivity is None or invalid
            if log_sens is None:
                # Set a default NaN array to prevent crash
                if hasattr(paper_profile.data, 'wavelengths') and paper_profile.data.wavelengths is not None:
                    wl_arr = np.asarray(paper_profile.data.wavelengths)
                    wl_len = len(wl_arr) if wl_arr.size > 0 else 1
                    paper_profile.data.log_sensitivity = np.full((wl_len, 3), np.nan)
                else:
                    paper_profile.data.log_sensitivity = np.full((1, 3), np.nan)
                print(f'Warning: Paper profile {target_paper} had None log_sensitivity, set to NaN array')
                needs_fix = True
            elif isinstance(log_sens, np.ndarray) and log_sens.dtype == object:
                # Handle object array with None values
                if hasattr(paper_profile.data, 'wavelengths') and paper_profile.data.wavelengths is not None:
                    wl_arr = np.asarray(paper_profile.data.wavelengths)
                    wl_len = len(wl_arr) if wl_arr.size > 0 else 1
                    paper_profile.data.log_sensitivity = np.full((wl_len, 3), np.nan)
                else:
                    paper_profile.data.log_sensitivity = np.full((1, 3), np.nan)
                print(f'Warning: Paper profile {target_paper} had invalid log_sensitivity dtype, set to NaN array')
                needs_fix = True
            
            # Save the fixed profile so it's used when correct_negative_curves_with_gray_ramp loads it
            if needs_fix:
                # Save with the target_paper name (which may include _uc suffix)
                original_stock = paper_profile.info.stock
                paper_profile.info.stock = target_paper
                # save_profile adds the suffix to stock, so we need to remove it first if present
                if target_paper.endswith('_uc'):
                    paper_profile.info.stock = target_paper[:-3]
                    save_profile(paper_profile, '_uc')
                else:
                    paper_profile.info.stock = target_paper
                    save_profile(paper_profile, '')
                paper_profile.info.stock = original_stock
            
            profile = correct_negative_curves_with_gray_ramp(profile, 
                                                            target_paper=target_paper, 
                                                            data_trustability=trustability)
        except Exception as e:
            print(f'Warning: Could not load or validate paper profile {target_paper}: {e}')
            print('Skipping correction step')
        
        profile = replace_fitted_density_curves(profile)
        profile = adjust_log_exposure(profile)
        save_profile(profile, 'c')
        plot_profile(profile)

        # Calculate and save YMC filters for this film-paper combination
        try:
            final_profile_name = label + suffix + 'uc'  # The final corrected profile name
            calculate_and_save_ymc_filters(final_profile_name, target_paper)
        except Exception as e:
            print(f'Warning: Could not calculate YMC filters for {final_profile_name} on {target_paper}: {e}')

print('----------------------------------------')
print('Populating missing YMC filters')

# List of common paper stocks to check against
common_papers = ['kodak_portra_endura_uc', 'kodak_supra_endura_uc', 'kodak_endura_premier_uc',
                 'kodak_ektacolor_edge_uc', 'kodak_ultra_endura_uc', 'fujifilm_crystal_archive_typeii_uc',
                 'kodak_2383_uc', 'kodak_2393_uc']

# Get all negative profiles that end with 'auc' (corrected profiles)
import os
import glob
profile_dir = os.path.join(os.path.dirname(__file__), '..', 'agx_emulsion', 'data', 'profiles')
json_files = glob.glob(os.path.join(profile_dir, '*.json'))

for json_file in json_files:
    filename = os.path.basename(json_file)
    if filename.endswith('auc.json'):  # corrected negative profiles
        film_stock = filename[:-5]  # remove .json
        for paper_stock in common_papers:
            try:
                calculate_and_save_ymc_filters(film_stock, paper_stock)
            except Exception as e:
                # Skip if profile doesn't exist or other errors
                pass

plt.show()