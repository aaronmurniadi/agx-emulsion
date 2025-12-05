import matplotlib.pyplot as plt
from agx_emulsion.process.profiles.factory import create_profile, process_negative_profile, process_paper_profile, plot_profile, replace_fitted_density_curves, adjust_log_exposure
from agx_emulsion.process.profiles.io import save_profile
from agx_emulsion.process.profiles.correct import correct_negative_curves_with_gray_ramp, align_midscale_neutral_exposures
from agx_emulsion.process.physics.stocks import fit_print_filters, PrintPapers, Illuminants
from agx_emulsion.process.core.process import photo_params
from agx_emulsion.process.utils.io import read_neutral_ymc_filter_values, save_ymc_filter_values

process_print_paper = False
process_negative = True
fit_ymc_filters = True  # Set to True to fit YMC filters for processed films

print('----------------------------------------')
print('Paper profiles')
#               label,                               name,                               ref_illu     illu    sens, curv, dye,  dom
paper_info = [
              ('kodak_ektacolor_edge',              'Kodak Ektacolor Edge',              'TH-KG3-L',  'D50',  None, None, None, 1.0),
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

#             proc,   type,      label,                    name,                       suffix   dye_donor,   ls_donor            ddmm_donor           d_over_min, ref_ill, target_paper,               align_mid_exp, trustability, densitometer
stock_info = [
              (False,  'negative', 'kodak_vision3_50d',      'Kodak Vision3 50D',         '',      None       , None,               None,                0.2,        'D55',  'kodak_2383_uc',             None,          0.3,         'status_A'),
              (False,  'negative', 'kodak_vision3_250d',     'Kodak Vision3 250D',        '',      None       , None,               None,                0.2,        'D55',  'kodak_2383_uc',             None,          0.3,         'status_A'),
              (False,  'negative', 'kodak_vision3_200t',     'Kodak Vision3 200T',        '',      None       , None,               None,                0.2,        'T',    'kodak_2383_uc',             None,          0.3,         'status_A'),
              (False,  'negative', 'kodak_vision3_500t',     'Kodak Vision3 500T',        '',      None       , None,               None,                0.2,        'T',    'kodak_2383_uc',             None,          0.3,         'status_A'),
              (True,   'negative', 'kodak_100t_5247',        'Kodak 100T 5247',           '',      None       , None,               None,                0.2,        'D55',  'kodak_portra_endura_uc',    None,          0.3,         'status_A'),
              (False,  'negative', 'kodak_ektar_100',        'Kodak Ektar 100',           '',      'generic_a', None,               None,                0.2,        'D55',  'kodak_portra_endura_uc',    None,          1.0,         'status_A'),
              (False,  'negative', 'kodak_portra_160',       'Kodak Portra 160',          '',      'generic_a', None,               None,                0.2,        'D55',  'kodak_portra_endura_uc',    None,          1.0,         'status_A'),
              (False,  'negative', 'kodak_portra_400',       'Kodak Portra 400',          '',      'generic_a', None,               None,                0.2,        'D55',  'kodak_portra_endura_uc',    None,          1.0,         'status_A'),
              (False,  'negative', 'kodak_portra_800',       'Kodak Portra 800',          '',      'generic_a', None,               None,                0.2,        'D55',  'kodak_portra_endura_uc',    None,          1.0,         'status_A'),
              (False,  'negative', 'kodak_portra_800_push1', 'Kodak Portra 800 (Push 1)', '',      'generic_a', 'kodak_portra_800', 'kodak_portra_800',  0.2,        'D55',  'kodak_portra_endura_uc',    None,          1.0,         'status_A'),
              (False,  'negative', 'kodak_portra_800_push2', 'Kodak Portra 800 (Push 2)', '',      'generic_a', 'kodak_portra_800', 'kodak_portra_800',  0.2,        'D55',  'kodak_portra_endura_uc',    None,          1.0,         'status_A'),
              (False,  'negative', 'kodak_gold_200',         'Kodak Gold 200',            '',      'generic_a', None,               None,                0.2,        'D55',  'kodak_portra_endura_uc',    None,          1.0,         'status_A'),
              (False,  'negative', 'kodak_ultramax_400',     'Kodak Ultramax 400',        '',      'generic_a', None,               None,                0.2,        'D55',  'kodak_portra_endura_uc',    None,          1.0,         'status_A'),
              (False,  'negative', 'fujifilm_pro_400h',      'Fujifilm Pro 400H',         '',      'generic_a', None,               None,                1.0,        'D55',  'kodak_portra_endura_uc',    'mid',         0.3,         'status_A'),
              (False,  'negative', 'fujifilm_xtra_400',      'Fujifilm X-Tra 400',        '',      'generic_a', None,               None,                1.0,        'D55',  'kodak_portra_endura_uc',    None,          0.3,         'status_A'),
              (False,  'negative', 'fujifilm_c200',          'Fujifilm C200',             '',      'generic_a', None,               None,                1.0,        'D55',  'kodak_portra_endura_uc',    'green',       0.3,         'status_A'),
              ]

if process_negative:
    # Load YMC filters database
    if fit_ymc_filters:
        ymc_filters = read_neutral_ymc_filter_values()
    
    for proc, type, label, name, suff, dye, ls_donor, ddmm_donor, d_over_min, ref_ill, target_paper, align_mid_exp, trustability, densitometer in stock_info:
        if not proc:
            continue
        profile = create_profile(stock=label,
                                 name=name,
                                 type=type,
                                 densitometer=densitometer,
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
        profile = correct_negative_curves_with_gray_ramp(profile, 
                                                        target_paper=target_paper, 
                                                        data_trustability=trustability)
        profile = replace_fitted_density_curves(profile)
        profile = adjust_log_exposure(profile)
        save_profile(profile, 'c')
        plot_profile(profile)
        
        # Fit YMC filters for all print papers and illuminants
        if fit_ymc_filters:
            # Use the final stock name from the profile (already updated by save_profile)
            final_stock_name = profile.info.stock  # e.g., 'kodak_100t_5247_uc'
            print('----------------------------------------')
            print(f'Fitting YMC filters for {final_stock_name}')
            print('----------------------------------------')
            
            for paper in PrintPapers:
                print(f'  Paper: {paper.value}')
                for illuminant in Illuminants:
                    print(f'    Illuminant: {illuminant.value}')
                    try:
                        # Create photo_params with the processed profile
                        params = photo_params(negative=final_stock_name, 
                                            print_paper=paper.value, 
                                            ymc_filters_from_database=False)
                        params.enlarger.illuminant = illuminant.value
                        
                        # Use default starting values or values from similar film
                        # Try to get values from similar film (e.g., vision3_500t_uc) if available
                        try:
                            similar_film = 'kodak_vision3_500t_uc'  # Similar tungsten Vision3 film
                            if similar_film in ymc_filters.get(paper.value, {}).get(illuminant.value, {}):
                                y0, m0, c0 = ymc_filters[paper.value][illuminant.value][similar_film]
                            else:
                                y0, m0, c0 = [0.90, 0.70, 0.35]  # Default starting values
                        except:
                            y0, m0, c0 = [0.90, 0.70, 0.35]  # Default starting values
                        
                        params.enlarger.y_filter_neutral = y0
                        params.enlarger.m_filter_neutral = m0
                        params.enlarger.c_filter_neutral = c0
                        
                        # Fit the filters
                        filter_y, filter_m, residues = fit_print_filters(params, iterations=10)
                        
                        # Store the fitted values
                        if paper.value not in ymc_filters:
                            ymc_filters[paper.value] = {}
                        if illuminant.value not in ymc_filters[paper.value]:
                            ymc_filters[paper.value][illuminant.value] = {}
                        
                        ymc_filters[paper.value][illuminant.value][final_stock_name] = [filter_y, filter_m, c0]
                        print(f'      Fitted: [{filter_y:.4f}, {filter_m:.4f}, {c0:.4f}]')
                        
                    except Exception as e:
                        print(f'      Error fitting filters: {e}')
                        # Use default values as fallback
                        if paper.value not in ymc_filters:
                            ymc_filters[paper.value] = {}
                        if illuminant.value not in ymc_filters[paper.value]:
                            ymc_filters[paper.value][illuminant.value] = {}
                        ymc_filters[paper.value][illuminant.value][final_stock_name] = [y0, m0, c0]
                        print(f'      Using defaults: [{y0:.4f}, {m0:.4f}, {c0:.4f}]')
    
    # Save updated YMC filters database once after all films are processed
    if fit_ymc_filters and process_negative:
        print('----------------------------------------')
        print('Saving updated YMC filters database...')
        save_ymc_filter_values(ymc_filters)
        print('Done!')

plt.show()