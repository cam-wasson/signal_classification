from filter_bank_gradient_descent import build_objective_context, OBJECTIVES
import matplotlib.pyplot as plt
from matplotlib import use as mpluse
import numpy as np
import optimization_util
import util

import sys
sys.path.append('../kalman_filter_bank')
from filter_bank import run_filter_bank
from Labelling.ExtremaCluster import compute_cluster_dict

mpluse('Qt5Agg')

BANK_ROOT = 'C:\\Users\\cwass\\OneDrive\\Desktop\\Drexel\\2025\\4_Fall\\CS-591\\training_sessions'
# BANK_ROOT = 'C:\\Users\\cwass\\Desktop\\Drexel\\Capstone\\training_sessions'
dt = 1/(24*60)


def pos_mse_precontext(time_bounds, max_freq_fft=2.0, spectrum_thresh=None, cluster_cdf_threshold=0.9) -> dict:

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
    
    fft_dict_train['cluster_dict'] = compute_cluster_dict(z, max_freq_fft, cluster_cdf_threshold, dt)

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
    # elif obj_name == 'anova_loss':
    #     pre_ctx = anova_precontext(selection, omegas=omegas, max_freq_fft=max_freq, cluster_cdf_threshold=cluster_cdf)
    else:
        pre_ctx = None
    print(pre_ctx.keys())
    return pre_ctx


def run_validation(filter_bank, selections, objective, truth_extraction_params=None):
    if truth_extraction_params is None:
        truth_extraction_params = {'max_freq': 5.0,
                                   'spectrum_thresh': None,
                                   'cluster_cdf': .9,
                                   'omegas': filter_bank.omegas}
    losses = []
    filter_outputs = []
    prectxs = []
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
        prectxs.append(pre_ctx)

        # run the filter bank on this selection
        filter_dict = run_filter_bank(filter_bank, pre_ctx['raw'])
        filter_bank.reset_states()
        filter_outputs.append(filter_dict)

        # compute loss
        ctx = build_objective_context(pre_ctx, filter_dict, objective)
        loss = OBJECTIVES[objective]['fn'](ctx)

        # store
        print(f'\t{objective}: {loss}')
        losses.append(loss)

    # fig = plot_estimation(filter_outputs, prectxs, objective)
    fig = plot_feature_space(filter_outputs, prectxs, objective)
    # fig.savefig(f'{BANK_ROOT}\\{objective}_estimation_quadplot.png')

    return np.mean(losses), filter_outputs


def plot_estimation(filter_output, pre_ctxs, obj_str):
    fig, ax = plt.subplots(2, len(filter_output))
    fig.suptitle(obj_str)

    for i in range(len(filter_output)):
        # plot positions
        t_plot = np.arange(len(pre_ctxs[i]['raw']))*pre_ctxs[i]['dt']
        ax[0, i].scatter(t_plot, pre_ctxs[i]['raw'], label='meas', s=5, color='k', alpha=.75)
        ax[0, i].plot(t_plot, filter_output[i]['x'][:, 0], label='filter', linewidth=3)
        ax[0, i].plot(t_plot, pre_ctxs[i]['truth'], label='truth', alpha=.5)
        ax[0, i].legend()
        ax[0, i].set_title('position')

        # plot velocities
        vel_truth = np.gradient(pre_ctxs[i]['truth']) / pre_ctxs[i]['dt']
        ax[1, i].plot(t_plot, filter_output[i]['x'][:, 1], label='filter', linewidth=3)
        ax[1, i].plot(t_plot, vel_truth, alpha=.5, label='truth')
        ax[1, i].legend()
        ax[1, i].set_title('velocity')

    return fig
    
    
def plot_feature_space(filter_output_sets, pre_contexts, objective):
    n_features = 4  # spread, emp vel, emp acc, amp vel, phi vel
    n_sets = len(filter_output_sets)
    fig, ax = plt.subplots(n_sets, n_features)
    fig.suptitle(objective)
    
    for s_idx in range(len(filter_output_sets)):
        # extract info for plotting
        filter_x = filter_output_sets[s_idx]['x']
        amps = filter_output_sets[s_idx]['amp']
        phis = filter_output_sets[s_idx]['phi']
        cluster_dict = pre_contexts[s_idx]['cluster_dict']
        z = pre_contexts[s_idx]['raw']
        t_plot = np.arange(z.shape[0]) * dt
        
        # construct the label array from the cluster dictionary
        max_idx = np.concatenate(cluster_dict['cluster_max']['x_points'])
        min_idx = np.concatenate(cluster_dict['cluster_min']['x_points'])
        label_arr = np.zeros_like(z)
        label_arr[max_idx] = 1
        label_arr[min_idx] = -1
        combined_anova = 0
        
        # plot spread
        spread = z - filter_x[:, 0]
        ax[s_idx, 0].scatter(t_plot, spread, color='k', alpha=.5, s=3)
        ax[s_idx, 0].scatter(t_plot[min_idx], spread[min_idx], color='green', s=3)
        ax[s_idx, 0].scatter(t_plot[max_idx], spread[max_idx], color='red', s=3)
        ax[s_idx, 0].set_title('spread')
        combined_anova += optimization_util.anova_1d(spread, label_arr)
        
        # plot empirical velocity
        emp_vel = np.diff(np.concatenate(([0], filter_x[:,0]))) / dt
        ax[s_idx, 1].scatter(t_plot, emp_vel, color='k', alpha=.5, s=3)
        ax[s_idx, 1].scatter(t_plot[min_idx], emp_vel[min_idx], color='green', s=3)
        ax[s_idx, 1].scatter(t_plot[max_idx], emp_vel[max_idx], color='red', s=3)
        ax[s_idx, 1].set_title('empirical vel')
        combined_anova += optimization_util.anova_1d(emp_vel, label_arr)
        
        # plot empirical acceleration
        emp_acc = np.diff(np.concatenate(([0], filter_x[:, 1]))) / dt
        ax[s_idx, 2].scatter(t_plot, emp_acc, color='k', alpha=.5, s=3)
        ax[s_idx, 2].scatter(t_plot[min_idx], emp_acc[min_idx], color='green', s=3)
        ax[s_idx, 2].scatter(t_plot[max_idx], emp_acc[max_idx], color='red', s=3)
        ax[s_idx, 2].set_title('empirical acc')
        combined_anova += optimization_util.anova_1d(emp_acc, label_arr)
        
        # plot amplitude velocity
        amp_vel = np.tanh(np.diff(np.vstack((np.zeros(amps.shape[1]), amps)), axis=0) / dt)[:, 0].reshape(-1, 1)
        t_repeat = np.repeat(t_plot.reshape(-1, 1), amp_vel.shape[1], axis=1)
        ax[s_idx, 3].scatter(t_repeat, amp_vel, color='k', alpha=.5, s=3)
        ax[s_idx, 3].scatter(t_repeat[min_idx], amp_vel[min_idx], color='green', s=3)
        ax[s_idx, 3].scatter(t_repeat[max_idx], amp_vel[max_idx], color='red', s=3)
        ax[s_idx, 3].set_title('amp vel (tanh scaled)')
        combined_anova += optimization_util.anova_1d(amp_vel, label_arr)
        
        # plot phase velocity
        # phi_vel = np.tanh(np.diff(np.vstack((np.zeros(phis.shape[1]), phis)), axis=0) / dt)[:, 0].reshape(-1, 1)
        # ax[s_idx, 4].scatter(t_repeat, phi_vel, color='k', alpha=.5, s=3)
        # ax[s_idx, 4].scatter(t_repeat[min_idx], phi_vel[min_idx], color='green', s=3)
        # ax[s_idx, 4].scatter(t_repeat[max_idx], phi_vel[max_idx], color='red', s=3)
        # ax[s_idx, 4].set_title('phi vel (tanh scaled)')
        print(f'Set {s_idx} Combined ANOVA: {combined_anova}')

    # compute spread anova
    # compute emp vel anova
    # compute emp acc anova
    # combine anovas
    plt.show(block=True)


if __name__ == '__main__':
    from datetime import datetime
    import pickle
    import time

    tStart = time.time()
    # validation_set = [[datetime(2025, 1, 5).timestamp(), datetime(2025, 1, 15).timestamp()],
    #                   [datetime(2025, 2, 5).timestamp(), datetime(2025, 2, 15).timestamp()],
    #                   [datetime(2025, 3, 5).timestamp(), datetime(2025, 3, 15).timestamp()],
    #                   [datetime(2025, 4, 5).timestamp(), datetime(2025, 4, 15).timestamp()],
    #                   [datetime(2025, 5, 5).timestamp(), datetime(2025, 5, 15).timestamp()]]
    validation_set = [[datetime(2025, 7, 31).timestamp(), datetime(2025, 8, 10).timestamp() - 1],
                      [datetime(2025, 8, 11).timestamp(), datetime(2025, 8, 12).timestamp() - 1]]
    objective_func = 'vel_mse'
    training_timestamps = {'pos_mse': '2025-11-09_19-13-25',
                           'vel_mse': '2025-11-08_19-28-14'}
    bank_path = f'{BANK_ROOT}\\{objective_func}\\{training_timestamps[objective_func]}\\filter_bank.pkl'
    skfb = pickle.load(open(bank_path, 'rb'))
    skfb.reset_states()

    truth_extraction = {'max_freq': 5.0,
                        'spectrum_thresh': None,
                        'cluster_cdf': .9,
                        'omegas': skfb.omegas}

    mean_loss, filter_dicts = run_validation(skfb, validation_set, objective_func, truth_extraction)
    print(f'Mean Loss [{objective_func}]: {mean_loss}')
    print(f'\n{time.time() - tStart} seconds')
