import itertools
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from tqdm import tqdm

import sys
sys.path.append('../')
import util


def plot_label_over_signal(signal, labels):
    if len(labels.shape) > 1:
        labels = labels[:, -1]

    # get the important indices
    buy_idx, sell_idx = labels == -1, labels == 1

    # make the plot
    t = np.arange(signal.shape[0])
    plt.figure()
    plt.plot(t, signal)
    plt.scatter(t[buy_idx], signal[buy_idx], color='green', label='buy')
    plt.scatter(t[sell_idx], signal[sell_idx], color='red', label='sell')

    # label the plot
    plt.title('Labels over Signal')
    plt.xlabel('Time')
    plt.ylabel('Rate')
    plt.legend()
    plt.show()


def pad_signal(sig, l=None):
    if l is None:
        # find the nearest power of 2 greater than this segment length
        l = get_pad_length(sig)

    # determine sizes of the pads
    N = sig.shape[0]
    pad_total = l - N
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    # create the padded arrays and concatenate
    pad_left_values = np.full(pad_left, sig[0])
    pad_right_values = np.full(pad_right, sig[-1])
    padded = np.concatenate([pad_left_values, sig, pad_right_values])
    return padded, [len(pad_left_values), l - len(pad_right_values)]


def get_pad_length(sig):
    for p in range(16):
        if len(sig) < 2 ** p:
            return 2 ** p
    return 0


def compute_cluster_dict(signal_segment, fft_cutoff, cdf_thresh, dt):
    '''    cluster = {
                    "x_points": [x_values],                 # indices of the labeled region
                    "y_points": [y_values],            # signal values of points x
                    "x_center": np.mean(x_values),           # mean index (for ordering clusters)
                    "y_center": np.mean(y_values)    # Mean signal value within the labeled region
                } '''
    # empty vars
    cluster = {'cluster_min': {"x_points": [],
                               "x_center": [],
                               "y_points": [],
                               "y_center": []},
               'cluster_max': {"x_points": [],
                               "x_center": [],
                               "y_points": [],
                               "y_center": []}}
    t = np.arange(signal_segment.shape[0])

    # compute the clean signal from the segment
    fft_dict = util.extract_low_pass_components(signal_segment, dt, max_freq=fft_cutoff)
    cluster['fft_dict'] = fft_dict
    
    # get min/max values of clean signal
    clean_signal_min, clean_signal_max = (argrelextrema(fft_dict['truth'], np.less)[0],
                                          argrelextrema(fft_dict['truth'], np.greater)[0])

    # handle first index
    if clean_signal_min[0] < clean_signal_max[0]:
        # add 0 as a local min if the next extrema is a local max AND the clean signal is increasing
        clean_signal_min = np.concatenate(([0], clean_signal_min))
    else:
        # add 0 as a local max if the next extrema is a local min AND the clean signal is decreasing
        clean_signal_max = np.concatenate(([0], clean_signal_max))

    # handle last index
    if clean_signal_min[-1] > clean_signal_max[-1]:
        # add -1 as a local min if the next extrema is a local max AND the clean signal is decreasing
        clean_signal_min = np.concatenate((clean_signal_min, [signal_segment.shape[0] - 1]))
    else:
        # add -1 as a local man if the next extrema is a local min AND the clean signal is increasing
        clean_signal_max = np.concatenate((clean_signal_max, [signal_segment.shape[0] - 1]))

    # label the LEFT edge (beginning)
    if clean_signal_max[0] < clean_signal_min[0] and (abs(clean_signal_max[0] - clean_signal_min[0]) > 2):
        # filter on this subset
        raw_signal_slice = signal_segment[clean_signal_max[0]: clean_signal_min[0]]
        t_slice = t[clean_signal_max[0]: clean_signal_min[0]]

        # label sell region
        extrema_idx_slice = compute_extreme_values_cdf(raw_signal_slice, rel_peak=cdf_thresh, side='max')

        # store in cluster dictionary
        cluster['cluster_max']['x_points'].append(t_slice[extrema_idx_slice])
        cluster['cluster_max']['x_center'].append(np.mean(t_slice[extrema_idx_slice]))
        cluster['cluster_max']['y_points'].append(raw_signal_slice[extrema_idx_slice])
        cluster['cluster_max']['y_center'].append(np.mean(raw_signal_slice[extrema_idx_slice]))
    elif (clean_signal_max[0] > clean_signal_min[0]) and (abs(clean_signal_max[0] - clean_signal_min[0]) > 2):
        # filter on this subset
        raw_signal_slice = signal_segment[clean_signal_min[0]: clean_signal_max[0]]
        t_slice = t[clean_signal_min[0]: clean_signal_max[0]]

        # label buy region
        extrema_idx_slice = compute_extreme_values_cdf(raw_signal_slice, rel_peak=cdf_thresh, side='min')

        # store in cluster dictionary
        cluster['cluster_min']['x_points'].append(t_slice[extrema_idx_slice])
        cluster['cluster_min']['x_center'].append(np.mean(t_slice[extrema_idx_slice]))
        cluster['cluster_min']['y_points'].append(raw_signal_slice[extrema_idx_slice])
        cluster['cluster_min']['y_center'].append(np.mean(raw_signal_slice[extrema_idx_slice]))

    # label the RIGHT edge (end of segment)
    if (clean_signal_max[-1] > clean_signal_min[-1]) and (abs(clean_signal_max[-1] - clean_signal_min[-1]) > 2):
        # filter on this subset
        raw_signal_slice = signal_segment[clean_signal_min[-1]: clean_signal_max[-1]]
        t_slice = t[clean_signal_min[-1]: clean_signal_max[-1]]

        # label sell region
        extrema_idx_slice = compute_extreme_values_cdf(raw_signal_slice, rel_peak=cdf_thresh, side='max')

        # store in cluster dictionary
        cluster['cluster_max']['x_points'].append(t_slice[extrema_idx_slice])
        cluster['cluster_max']['x_center'].append(np.mean(t_slice[extrema_idx_slice]))
        cluster['cluster_max']['y_points'].append(raw_signal_slice[extrema_idx_slice])
        cluster['cluster_max']['y_center'].append(np.mean(raw_signal_slice[extrema_idx_slice]))
    elif (clean_signal_max[-1] < clean_signal_min[-1]) and (abs(clean_signal_max[-1] - clean_signal_min[-1]) > 2):
        # filter on this subset
        raw_signal_slice = signal_segment[clean_signal_max[-1]: clean_signal_min[-1]]
        t_slice = t[clean_signal_max[-1]: clean_signal_min[-1]]

        # label buy region
        extrema_idx_slice = compute_extreme_values_cdf(raw_signal_slice, rel_peak=cdf_thresh, side='min')

        # store in cluster dictionary
        cluster['cluster_min']['x_points'].append(t_slice[extrema_idx_slice])
        cluster['cluster_min']['x_center'].append(np.mean(t_slice[extrema_idx_slice]))
        cluster['cluster_min']['y_points'].append(raw_signal_slice[extrema_idx_slice])
        cluster['cluster_min']['y_center'].append(np.mean(raw_signal_slice[extrema_idx_slice]))

    # iterate over clean signal local min indices to find true signal max (sell) regions
    for i in range(len(clean_signal_min) - 1):
        # filter on this slice
        slice_index = np.arange(clean_signal_min[i], clean_signal_min[i + 1])
        raw_signal_slice = signal_segment[slice_index]
        t_slice = t[slice_index]

        # find indices of values that are greater than {rel_peak} of the maximum value
        extrema_idx_slice = compute_extreme_values_cdf(raw_signal_slice, rel_peak=cdf_thresh, side='max')

        # store in cluster dictionary
        cluster['cluster_max']['x_points'].append(t_slice[extrema_idx_slice])
        cluster['cluster_max']['x_center'].append(np.mean(t_slice[extrema_idx_slice]))
        cluster['cluster_max']['y_points'].append(raw_signal_slice[extrema_idx_slice])
        cluster['cluster_max']['y_center'].append(np.mean(raw_signal_slice[extrema_idx_slice]))

    # iterate over clean signal local max indices to find true signal min (buy) regions
    for i in range(len(clean_signal_max) - 1):
        # filter on this slice
        slice_index = np.arange(clean_signal_max[i], clean_signal_max[i + 1])
        raw_signal_slice = signal_segment[slice_index]
        t_slice = t[slice_index]

        # find indices of values that are less than {rel_peak} of the minimum value
        extrema_idx_slice = compute_extreme_values_cdf(raw_signal_slice, rel_peak=cdf_thresh, side='min')

        # store in cluster dictionary
        if len(t[slice_index][extrema_idx_slice]) < 2:
            continue
        cluster['cluster_min']['x_points'].append(t_slice[extrema_idx_slice])
        cluster['cluster_min']['x_center'].append(np.mean(t_slice[extrema_idx_slice]))
        cluster['cluster_min']['y_points'].append(raw_signal_slice[extrema_idx_slice])
        cluster['cluster_min']['y_center'].append(np.mean(raw_signal_slice[extrema_idx_slice]))

    return cluster


def compute_extreme_values_cdf(signal, rel_peak, side='max'):
    # compute ecdf
    f, x = compute_cdf(signal, bins=len(signal) + 1)

    # find extreme values
    if side == 'min':
        if not np.any(f <= 1 - rel_peak):
            # return value of the minimum indices if the slice is too small
            top_idx = (signal <= max(x[f <= np.min(f) + 10 ** -8]))
        else:
            # find indices of values less than the "rel peak" on the cdf
            top_idx = signal <= max(x[f <= 1 - rel_peak])
    else:
        if not np.any(f >= rel_peak):
            # return value of the maximum indices if the slice is too small
            top_idx = (signal >= max(x[f <= np.max(f) - 10 ** -8]))
        else:
            # find indices of values greater than the "rel peak" on the cdf
            top_idx = (signal >= min(x[f >= rel_peak]))
    return top_idx
    
 
def compute_cdf(values, bins=1000):
    heights, edges = np.histogram(values, bins=bins)
    cdf_f, cdf_x = np.cumsum(heights) / sum(heights), edges[:-1]
    return cdf_f.flatten(), cdf_x.flatten()
 

