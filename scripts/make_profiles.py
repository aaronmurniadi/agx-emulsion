import matplotlib.pyplot as plt
import numpy as np
import scipy
import copy
from agx_emulsion.process.profiles.factory import create_profile, process_negative_profile, process_paper_profile, plot_profile, replace_fitted_density_curves, adjust_log_exposure
from agx_emulsion.process.profiles.io import save_profile
from agx_emulsion.process.profiles.correct import correct_negative_curves_with_gray_ramp, align_midscale_neutral_exposures
from agx_emulsion.process.core.process import photo_params
from agx_emulsion.process.physics.stocks import FilmStocks, PrintPapers, Illuminants
from agx_emulsion.process.utils.fit_print_filters import fit_print_filters
from agx_emulsion.process.physics.illuminants import standard_illuminant
from agx_emulsion.process.utils.io import save_ymc_filter_values

process_print_paper = True
process_negative = True
process_ymc_filters = True

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
              ('kodak_100t_5247',        'Kodak 100T 5247',           '',      None       , None,               None,                0.2,        'T',  'kodak_portra_endura_uc',    None,          0.3,         True),
              ('kodak_pro_image_100',    'Kodak Pro Image 100',       '',      'generic_a', None,               None,                0.2,        'D55',  'kodak_portra_endura_uc',    None,          0.3,         True),
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
        profile = correct_negative_curves_with_gray_ramp(profile, 
                                                        target_paper=target_paper, 
                                                        data_trustability=trustability)
        profile = replace_fitted_density_curves(profile)
        profile = adjust_log_exposure(profile)
        save_profile(profile, 'c')
        plot_profile(profile)


print('----------------------------------------')
print('YMC Filter calculation')

def make_ymc_filters_dictionary(PrintPapers, Illuminants, FilmStocks):
    ymc_filters_0 = {}
    residues = {}
    for paper in PrintPapers:
        ymc_filters_0[paper.value] = {}
        residues[paper.value] = {}
        for light in Illuminants:
            ymc_filters_0[paper.value][light.value] = {}
            residues[paper.value][light.value] = {}
            for film in FilmStocks:
                ymc_filters_0[paper.value][light.value][film.value] = [0.90, 0.70, 0.35]
                residues[paper.value][light.value][film.value] = 0.184
    ymc_filters = copy.copy(ymc_filters_0)
    save_ymc_filter_values(ymc_filters)
    return ymc_filters, residues

def fit_all_stocks(ymc_filters, residues, iterations=5, randomess_starting_points=0.5):
    ymc_filters_out = copy.deepcopy(ymc_filters)
    r = randomess_starting_points
    
    for paper in PrintPapers:
        print(' '*20)
        print('#'*20)
        print(paper.value)
        for light in Illuminants:
            print('-'*20)
            print(light.value)
            for stock in FilmStocks:
                if residues[paper.value][light.value][stock.value] > 5e-4:
                    y0 = ymc_filters[paper.value][light.value][stock.value][0]
                    m0 = ymc_filters[paper.value][light.value][stock.value][1]
                    c0 = ymc_filters[paper.value][light.value][stock.value][2]
                    y0 = np.clip(y0, 0, 1)*(1-r) + np.random.uniform(0,1)*r
                    m0 = np.clip(m0, 0, 1)*(1-r) + np.random.uniform(0,1)*r
                    
                    p = photo_params(negative=stock.value, print_paper=paper.value, ymc_filters_from_database=False)
                    p.enlarger.illuminant = light.value
                    p.enlarger.y_filter_neutral = y0
                    p.enlarger.m_filter_neutral = m0
                    p.enlarger.c_filter_neutral = c0
            
                    yf, mf, res = fit_print_filters(p, iterations=iterations)
                    ymc_filters_out[paper.value][light.value][stock.value] = [yf, mf, c0]
                    residues[paper.value][light.value][stock.value] = np.sum(np.abs(res))
    return ymc_filters_out

if process_ymc_filters:
    plot_data = False
    density_midgray_test = False
    print_filter_test = False
    spread = 0.2
    
    d55 = standard_illuminant(type='D55', return_class=True)
    
    ymc_filters, residues = make_ymc_filters_dictionary(PrintPapers, Illuminants, FilmStocks)
    ymc_filters = fit_all_stocks(ymc_filters, residues, iterations=20)
    save_ymc_filter_values(ymc_filters)

    if print_filter_test:
        for paper in PrintPapers:
            for stock in FilmStocks:
                for light in Illuminants:
                    if stock.value.type=='negative':
                        YMC = ymc_filters[paper.value][light.value][stock.value]
                        paper.value.print_filter_test(stock.value, light.value[:], d55[:],
                                                        y_filter=YMC[0],
                                                        m_filter=YMC[1],
                                                        c_filter=YMC[2],
                                                        y_filter_spread=spread,
                                                        m_filter_spread=spread)

# plt.show()
