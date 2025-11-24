from filter_closed_form_backprop import build_objective_context, OBJECTIVES
import matplotlib.pyplot as plt
from matplotlib import use as mpluse
import numpy as np
import optimization_util
import util

import sys
sys.path.append('../kalman_filter_bank')
from filter_bank import run_filter_bank

mpluse('Qt5Agg')


def pos_mse_precontext(time_bounds, max_freq_fft=2.0, spectrum_thresh=None) -> dict:

    # connect to DB
    conn = util.connect('btc', db_root='../data')

    # build the selection
    selection = util.selector()
    selection.start_time = time_bounds[0]
    selection.stop_time = time_bounds[1]

    # fetch data
    price_df = util.fetch_price_space(conn=conn, selection=selection)

    # convert to rate space; set dt
    z = (price_df.Open.values - price_df.Open.values[0]) / price_df.Open.values[0]
    dt = 1/(24*60)  # len(train_price_df.Date.unique()) / len(train_price_df)

    # pad front/back of signal for stronger FFT estimation of signal edges
    pad_len = util.compute_pad_length(z)
    z_pad, pad_bounds = util.pad_signal(z, L=pad_len)

    # produce truth training data
    if spectrum_thresh is None:
        fft_dict_train = util.extract_low_pass_components(z_pad, dt, max_freq=max_freq_fft)
    else:
        fft_dict_train = util.extract_low_pass_components_cdf_thresh(z_pad, dt, cdf_thresh=spectrum_thresh)
    fft_dict_train['truth'] = fft_dict_train['truth'][pad_bounds[0]:pad_bounds[1]]  # fft reconstruction w/o pad

    # store everything important
    fft_dict_train['raw'] = z
    fft_dict_train['dt'] = dt

    return fft_dict_train


def vel_mse_precontext(time_bounds, max_freq_fft=2.0, spectrum_thresh=None) -> dict:
    return pos_mse_precontext(time_bounds, max_freq_fft=max_freq_fft, spectrum_thresh=spectrum_thresh)


def build_objective_precontext(selection,
                               obj_name='pos_mse',
                               max_freq=2.0,
                               spectrum_thresh=None,
                               cluster_cdf=.9,
                               omegas=None):
    # build the appropriate data class for the objective function
    if obj_name == 'pos_mse':
        pre_ctx = pos_mse_precontext(selection, max_freq, spectrum_thresh=spectrum_thresh)
    elif obj_name == 'vel_mse':
        pre_ctx = vel_mse_precontext(selection, max_freq)
    elif obj_name == 'anova_loss':
        pre_ctx = anova_precontext(selection, omegas=omegas, max_freq_fft=max_freq, cluster_cdf_threshold=cluster_cdf)
    else:
        pre_ctx = None

    return pre_ctx


def run_validation(filter_bank, selections, objective, truth_extraction_params=None):
    if truth_extraction_params is None:
        truth_extraction_params = {'max_freq': 5.0,
                                   'spectrum_thresh': None,
                                   'cluster_cdf': .9,
                                   'omegas': filter_bank.omegas}
    losses = []
    for s in selections:
        start_str, stop_str = (datetime.fromtimestamp(s[0]).strftime("%D"),
                               datetime.fromtimestamp(s[1]).strftime("%D"))
        print(f'\nAnalyzing {start_str} - {stop_str}')

        # create this segment's precontext
        pre_ctx = build_objective_precontext(s,
                                             objective,
                                             max_freq=truth_extraction_params['max_freq'],
                                             spectrum_thresh=truth_extraction_params['spectrum_thresh'],
                                             cluster_cdf=truth_extraction_params['cluster_cdf'],
                                             omegas=truth_extraction_params['cluster_cdf'])

        # run the filter bank on this selection
        filter_dict = run_filter_bank(filter_bank, pre_ctx['raw'])
        filter_bank.reset_states()

        # compute loss
        ctx = build_objective_context(pre_ctx, filter_dict, objective)
        loss = OBJECTIVES[objective]['fn'](ctx)

        # store
        print(f'\t{objective}: {loss}')
        losses.append(loss)

    return np.mean(losses)


if __name__ == '__main__':
    from datetime import datetime
    import pickle

    validation_set = [[datetime(2025, 1, 5).timestamp(), datetime(2025, 1, 15).timestamp()],
                      [datetime(2025, 2, 5).timestamp(), datetime(2025, 2, 15).timestamp()],
                      [datetime(2025, 3, 5).timestamp(), datetime(2025, 3, 15).timestamp()],
                      [datetime(2025, 4, 5).timestamp(), datetime(2025, 4, 15).timestamp()],
                      [datetime(2025, 5, 5).timestamp(), datetime(2025, 5, 15).timestamp()]]
    objective_func = 'vel_mse'
    training_timestamps = {'pos_mse': '2025-11-09_19-13-25',
                           'vel_mse': '2025-11-08_19-28-14'}
    bank_root = 'C:\\Users\\cwass\\OneDrive\\Desktop\\Drexel\\2025\\4_Fall\\CS-591\\training_sessions'
    bank_path = f'{bank_root}\\{objective_func}\\{training_timestamps[objective_func]}\\filter_bank.pkl'
    skfb = pickle.load(open(bank_path, 'rb'))
    skfb.reset_states()

    mean_loss = run_validation(skfb, validation_set, objective_func)
    print(f'Mean Loss [{objective_func}]: {mean_loss}')
