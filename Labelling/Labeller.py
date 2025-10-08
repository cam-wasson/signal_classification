import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import argrelextrema
from Utilities.Interpolator import interpEq as ie
from Utilities.Interpolator import interpFFT as ifft
import Utilities.math_shortcuts as ms
import TradeSystem.Classifier.qnn_input_generator as qnn_ig


def datamine_extrema(raw_signal, extrema_idxs, side=None):
    if side is None:
        print("Please set 'side' parameter equal to 'min' or 'max'")
        return

    # find side's points in between opposite side extrema in the clean signal
    raw_extrema_idx_list = []
    for i in range(0, len(extrema_idxs)):
        # filter raw signal to region between these peaks
        this_extrema_idx = extrema_idxs[i]
        try:
            next_extrema_idx = extrema_idxs[i + 1]
        except IndexError:
            next_extrema_idx = len(raw_signal)
        signal_subset = list(raw_signal[this_extrema_idx: next_extrema_idx + 1])

        # grab top 10% or top 3 of the extreme signal values in this region AFTER the most extreme point
        if side.lower() == 'min':
            raw_extrema_idx = signal_subset.index(min(signal_subset)) + this_extrema_idx
        elif side.lower() == 'max':
            raw_extrema_idx = signal_subset.index(max(signal_subset)) + this_extrema_idx
        else:
            raw_extrema_idx = None

        raw_extrema_idx_list.append(raw_extrema_idx)

    # t = np.arange(raw_signal.shape[0])
    # plt.plot(t, raw_signal)
    # if side.lower() == 'min':
    #     color = 'green'
    # elif side.lower() == 'max':
    #     color = 'red'
    # plt.scatter(t[raw_extrema_idx_list], raw_signal[raw_extrema_idx_list], color=color)

    return np.unique(np.sort(raw_extrema_idx_list))


def create_label_spaces(price_space, enhanced_method=False, diff_noise_floor=.5):
    # dates = price_space.Date.unique()
    print('computing label space...')
    unknown_label_value = 2
    growth_label_value = 1
    decay_label_value = 0

    y = unknown_label_value * np.ones((price_space.shape[0], 2))
    y[:, 0] = price_space.EpochTime.values
    y_idxs = np.arange(0, y.shape[0])
    dates = qnn_ig.create_date_pairs(price_space.Date.unique(), num_days=1)

    ctr = 0
    for date in dates:  # price_space['Date'].unique():

        times = price_space.EpochTime.loc[(price_space.Date == date[0]) |
                                          (price_space.Date == date[1])].values
        current_date_df = price_space.loc[(price_space.EpochTime.values >= times[0]) &
                                          (price_space.EpochTime.values <= times[-1])]

        # denoise the price signal
        function_df = ie.driver(current_date_df, 390, num_components=13)
        clean_closing_vals = function_df['S(t)'].values

        # label periods of growth and decay
        # signal_differentials = np.gradient(clean_closing_vals)
        # growth_idxs = np.where(signal_differentials > growth_threshold)[0]
        # decay_idxs = np.where(signal_differentials < -1*growth_threshold)[0]
        # growth_idxs = np.where(signal_differentials > 0)[0]
        # decay_idxs = np.where(signal_differentials < 0)[0]

        # find local max and min indices
        mins, maxs = (argrelextrema(clean_closing_vals, np.less)[0],
                      argrelextrema(clean_closing_vals, np.greater)[0])

        if enhanced_method:
            # insert edge cases where applicable
            if mins[0] > maxs[0]:
                mins = np.concatenate(([0], mins.flatten()))
            else:
                maxs = np.concatenate(([0], maxs.flatten()))
            if mins[-1] > maxs[-1]:
                maxs = np.concatenate((maxs.flatten(), [clean_closing_vals.shape[0] - 1]))
            else:
                mins = np.concatenate((mins.flatten(), [clean_closing_vals.shape[0] - 1]))

            # find true local mins/maxes in raw signal
            raw_mins, raw_maxs = (datamine_extrema(current_date_df.Close.values, maxs, side='min'),
                                  datamine_extrema(current_date_df.Close.values, mins, side='max'))
            # combine edge indices w/ local max/mins into one sorted array
            extrema_idxs = np.sort(
                np.concatenate(([0], raw_mins, raw_maxs, [clean_closing_vals.shape[0] - 1])).flatten())
        else:
            # combine edge indices w/ local max/mins into one sorted array
            extrema_idxs = np.sort(np.concatenate(([0], mins, maxs, [clean_closing_vals.shape[0] - 1])).flatten())

        # extrema difference noise floor; price difference between peaks must be larger than $0.50
        # diff_noise_floor = .5  # np.mean(np.abs(np.gradient(clean_closing_vals[extrema_idxs])))

        # iterate over extrema indices, label periods w/ high amounts of growth/decay
        growth_idxs, decay_idxs = [], []
        this_extrema = extrema_idxs[0]
        for i in range(1, extrema_idxs.shape[0] - 1):
            if enhanced_method:
                next_extrema = extrema_idxs[i]
            else:
                this_extrema, next_extrema = extrema_idxs[i], extrema_idxs[i + 1]
            this_raw_val, next_raw_val = (current_date_df.Close.values[this_extrema],
                                          current_date_df.Close.values[next_extrema])
            if np.abs(next_raw_val - this_raw_val) > diff_noise_floor:
                label_idxs = list(range(this_extrema, next_extrema))
                if (next_raw_val - this_raw_val) > 0:
                    growth_idxs += label_idxs
                else:
                    decay_idxs += label_idxs
                this_extrema = next_extrema

        # update primary label array
        updates_idxs = y_idxs[np.in1d(y[:, 0], current_date_df.EpochTime.values)]
        y[updates_idxs[growth_idxs], 1] = growth_label_value
        y[updates_idxs[decay_idxs], 1] = decay_label_value

        ctr += 1
        if ctr % 10 == 0:
            # plt.show()
            print(f"\tPoints mined for {date} \t|\t {ctr}/{len(dates)} dates")

    # drop unlabeled points
    # y = y[y[:, -1] != -1]
    #
    # # format y into onehot style encoding
    # y_onehot = qnn_ig.convert_to_onehot(y)
    return y  # , y_onehot


def create_label_spaces_sliding_window(price_space, window_len=390*10, window_stride=30, n_comps=25):
    print(f'Computing Label Space w/ Window {window_len} and Stride {window_stride}')
    growth_label_value = 1
    decay_label_value = -1

    # instantiate the label dictionary
    y_dict = dict()

    ctr = 0
    for t in range(window_len, price_space.shape[0], window_stride):  # price_space['Date'].unique():

        times = price_space.EpochTime.iloc[t-window_len: t].values
        current_date_df = price_space.iloc[t-window_len: t]
        y_slice = np.zeros((times.shape[0], 2))
        y_slice[:, 0] = times

        # denoise the price signal
        function_df = ie.driver(current_date_df, 390, num_components=n_comps)

        return_list = compute_inc_dec_idx(function_df, cdf_filtering=False)
        decay_idxs, growth_idxs = return_list[0], return_list[1]

        # update primary label array
        y_slice[growth_idxs, 1] = growth_label_value
        y_slice[decay_idxs, 1] = decay_label_value
        y_dict = update_y_dict(y_dict, y_slice)

        ctr += 1
        if ctr % 100 == 0:
            # plt.show()
            print(f"\tPoints mined for {price_space.Date.iloc[t]} \t|\t {t}/{price_space.shape[0]} minutes")

    # convert the dictionary to a unidimensional label space
    y = convert_dict_to_labels(y_dict, threshold=.75)

    return y  # , y_onehot


def compute_maintenance_classifier_labels(price_space_df, cdf_thresh=.955, col_str='Open'):
    dates = price_space_df.Date.unique()
    y = np.zeros((price_space_df.shape[0], 2))
    y[:, 0] = price_space_df.EpochTime.values
    print(f'Computing labels for {len(dates)} days')
    for d in range(len(dates)):
        # print(f'analyzing {d}')
        if d % 100 == 0:
            print(f'\t{d}/{len(dates)} days labelled')

        # filter on this day
        price_slice = price_space_df.loc[price_space_df.Date == dates[d]]
        y_slice = np.zeros((price_slice.shape[0], 2))
        y_slice[:, 0] = price_slice.EpochTime.values

        # produce clean signal
        function_df = ie.driver_v2(price_slice, 390, cdf_thresh=cdf_thresh, col_str=col_str)
        # function_df, component_df = ie.driver_v2(price_slice,
        #                                          PPD=PPD,
        #                                          cdf_thresh=cdf_thresh,
        #                                          col_str=col_str,
        #                                          return_components=True)

        # compute increasing/decreasing idxs
        [decay_idxs, growth_idxs] = compute_inc_dec_idx(function_df=function_df)

        # put labels into the slice
        y_slice[decay_idxs, 1] = -1
        y_slice[growth_idxs, 1] = 1

        # plot_signals(price_slice, y_slice)

        # insert slice into the return array
        insert_idxs = np.in1d(y[:, 0], y_slice[:, 0])
        y[insert_idxs] = y_slice

    return y, function_df


def compute_inc_dec_idx(function_df, cdf_filtering=False):
    # extract signals from DF
    raw_signal = function_df['raw_signal'].values
    clean_signal = function_df['S(t)'].values

    # locate indices of clean signal extrema
    clean_signal_min, clean_signal_max = (argrelextrema(clean_signal, np.less)[0],
                                          argrelextrema(clean_signal, np.greater)[0])
    if clean_signal_min[0] < clean_signal_max[0]:
        clean_signal_max = np.concatenate(([0], clean_signal_max))
    else:
        clean_signal_min = np.concatenate(([0], clean_signal_min))
    if clean_signal_min[-1] < clean_signal_max[-1]:
        clean_signal_min = np.concatenate((clean_signal_min, [clean_signal.shape[0] - 1]))
    else:
        clean_signal_max = np.concatenate((clean_signal_max, [clean_signal.shape[0] - 1]))

    # get indices of raw signal extrema via opposite extrema iteration
    raw_min_idx = datamine_extrema(raw_signal, clean_signal_max, side='min')
    raw_max_idx = datamine_extrema(raw_signal, clean_signal_min, side='max')

    # compile into one array
    extrema_idx = np.sort(np.concatenate(([0],
                                          raw_min_idx,
                                          raw_max_idx,
                                          [clean_signal.shape[0] - 1])).flatten())

    # determine price delta screening from the CDF of extremas
    price_delta_thresh = 0
    if cdf_filtering:
        cdf_thresh = .1
        extrema_deltas = raw_signal[extrema_idx[1:]] - raw_signal[extrema_idx[:-1]]
        extrema_delta_cdf_y, extrema_delta_cdf_x = ms.compute_cdf(np.abs(extrema_deltas))
        if len(extrema_delta_cdf_x[extrema_delta_cdf_y < cdf_thresh]) > 0:
            price_delta_thresh = extrema_delta_cdf_x[extrema_delta_cdf_y < cdf_thresh][-1]

    # iterate over extrema and label according to signal behavior
    growth_idxs, decay_idxs = [], []
    for i in range(0, extrema_idx.shape[0] - 1):
        # get adjacent extrema
        this_extrema, next_extrema = extrema_idx[i], extrema_idx[i + 1]
        this_raw_extrema_val, next_raw_extrema_val = (raw_signal[this_extrema],
                                                      raw_signal[next_extrema])

        # find the bounds of this extrema window
        label_idxs = list(range(this_extrema, next_extrema))

        # assign growth/decay labels based on the difference between these extrema values
        if (next_raw_extrema_val - this_raw_extrema_val) > price_delta_thresh:
            growth_idxs += label_idxs
            decay_idxs += [next_extrema]
        else:
            decay_idxs += label_idxs
            growth_idxs += [next_extrema]

    return [np.unique(decay_idxs), np.unique(growth_idxs)]


def create_tuning_labels(tune_price_df, n_comps=25, price_delta_thresh=.1):
    dates = tune_price_df.Date.unique()
    y = np.zeros((tune_price_df.shape[0], 2))
    y[:, 0] = tune_price_df.EpochTime.values
    i = 0
    for d in range(len(dates)):
        # filter on this day
        tune_price_slice = tune_price_df.loc[tune_price_df.Date == dates[d]]
        y_slice = np.zeros((tune_price_slice.shape[0], 2))
        y_slice[:, 0] = tune_price_slice.EpochTime.values

        # produce clean signal
        function_df = ie.driver(tune_price_slice, 390, num_components=n_comps)
        clean_signal = function_df['S(t)'].values

        # locate indices of extrema in clean signal
        extrema_idx = np.sort(np.concatenate(([0],
                                              argrelextrema(clean_signal, np.less)[0],
                                              argrelextrema(clean_signal, np.greater)[0],
                                              [clean_signal.shape[0] - 1])).flatten())

        # iterate over extrema and label according to signal behavior
        growth_idxs, decay_idxs = [], []
        for i in range(0, extrema_idx.shape[0] - 1):
            this_extrema, next_extrema = extrema_idx[i], extrema_idx[i + 1]
            this_raw_extrema_val, next_raw_extrema_val = (tune_price_slice.Close.values[this_extrema],
                                                          tune_price_slice.Close.values[next_extrema])

            label_idxs = list(range(this_extrema, next_extrema))
            # if abs(next_raw_extrema_val - this_raw_extrema_val) <= price_delta_thresh:
            #     continue
            if (next_raw_extrema_val - this_raw_extrema_val) > 0:
                growth_idxs += label_idxs
            else:
                decay_idxs += label_idxs

        # put labels into the slice
        y_slice[decay_idxs, 1] = -1
        y_slice[growth_idxs, 1] = 1

        # insert slice into the return array
        y[i: i+tune_price_slice.shape[0]] = y_slice

        # update indexer
        i += tune_price_slice.shape[0]

    return y


def convert_dict_to_labels(y_dict, threshold=.75):
    et_keys = list(y_dict.keys())
    y = np.zeros((len(et_keys), 2))
    for i in range(len(et_keys)):
        # get this time's label values
        labels = y_dict[et_keys[i]]
        # get number of occurrences for each label
        class_val, counts = np.unique(labels, return_counts=True)
        # compute the probability of the most frequent label
        label_prob = class_val[np.argmax(counts)] * counts[np.argmax(counts)] / len(labels)
        # store the max label probability in the array
        y[i, :] = [float(et_keys[i]), label_prob]

    # set high probs to their respective label, low probability times to 0 label
    y[y[:, 1] > 0, 1] = 1
    y[y[:, 1] < 0, 1] = -1
    y[np.abs(y[:, 1]) < threshold] = 0
    return y


def plot_signals(current_date_df, y_arr, clean_closing_vals=None):
    tplot = np.arange(0, current_date_df.shape[0])
    # create plot
    plt.figure()
    plt.title(f'{current_date_df.Date.values[0]} - {current_date_df.Date.values[-1]}')
    plt.xlabel('Time (mins)')
    plt.ylabel('Asset Price ($)')

    # plot price signals
    plt.plot(tplot, current_date_df.Close.values, label='raw_signal')
    if clean_closing_vals is not None:
        plt.plot(tplot, clean_closing_vals, label='clean_signal')

    # plot classifications
    growth_idxs = np.in1d(current_date_df.EpochTime.values, y_arr[y_arr[:, 1] > 0, 0])
    decay_idxs = np.in1d(current_date_df.EpochTime.values, y_arr[y_arr[:, 1] < 0, 0])
    plt.scatter(tplot[growth_idxs],
                current_date_df.Close.values[growth_idxs],
                color='green',
                alpha=.5)
    plt.scatter(tplot[decay_idxs],
                current_date_df.Close.values[decay_idxs],
                color='red',
                alpha=.5)
    plt.legend()


def update_y_dict(y_dict, times_label_arr):
    for i in range(times_label_arr.shape[0]):
        if f'{int(times_label_arr[i, 0])}' not in y_dict.keys():
            y_dict[f'{int(times_label_arr[i, 0])}'] = [int(times_label_arr[i, 1])]
        else:
            y_dict[f'{int(times_label_arr[i, 0])}'] += [int(times_label_arr[i, 1])]

    return y_dict


def order_classifier_labels(price_space, n_comps=8, half_peak_labelling=False, print_statements=False):
    y = -999*np.ones((price_space.shape[0], 2))
    y[:, 0] = price_space.EpochTime.values

    # produce labels one day at a time
    dates = price_space.Date.unique()
    for d in range(dates.shape[0]):
        if print_statements:
            print(f'\tLabelling Date: {dates[d]}')
        # filter on this day
        price_slice = price_space.loc[price_space.Date == dates[d]]
        y_slice = np.zeros((price_slice.shape[0], 2))
        y_slice[:, 0] = price_slice.EpochTime.values

        # produce clean signal with LOW number of components
        function_df = ie.driver(price_slice, 390, num_components=n_comps)
        clean_signal = function_df['S(t)'].values

        # get local extrema indices
        local_min_idx = argrelextrema(clean_signal, np.less)[0]
        local_max_idx = argrelextrema(clean_signal, np.greater)[0]

        # add edge cases
        local_max_idx = np.concatenate(([0], local_max_idx))
        local_min_idx = np.concatenate(([0], local_min_idx))
        local_max_idx = np.concatenate((local_max_idx, [price_slice.shape[0] - 1]))
        local_min_idx = np.concatenate((local_min_idx, [price_slice.shape[0] - 1]))

        # get indices for sell labels through opposite extrema iteration
        for i in range(local_min_idx.shape[0] - 1):
            # filter on all values between local minima
            this_idx, next_idx = local_min_idx[i], local_min_idx[i + 1]
            raw_signal_subset = price_slice.Close.values[this_idx: next_idx]

            # find indices of the highest raw prices in the subset
            sort_idx = np.argsort(raw_signal_subset)

            # label top 20% of values as sell
            sell_idx = sort_idx[-int(len(sort_idx) * .2):]

            # filter down the sell index to label everything PAST the HIGHEST price
            if half_peak_labelling and len(sell_idx) > 3:
                sell_idx = np.sort(sell_idx)
                peak_idx = np.arange(sell_idx.shape[0], dtype=int)[sell_idx == np.argmax(raw_signal_subset)][0]
                sell_idx = sell_idx[peak_idx-1:]

            # convert to slice index and store in slice array
            slice_idx = sell_idx + this_idx
            y_slice[slice_idx, 1] = 1

        # get indices for buy labels through opposite extrema iteration
        for i in range(local_max_idx.shape[0] - 1):
            this_idx, next_idx = local_max_idx[i], local_max_idx[i + 1]
            raw_signal_subset = price_slice.Close.values[this_idx: next_idx]

            # find indices of the highest raw prices in the subset
            sort_idx = np.argsort(raw_signal_subset)

            # label top 20% of values as sell
            buy_idx = sort_idx[:int(len(sort_idx) * .2)]

            # filter down the buy index to label everything PAST the LOWEST price
            if half_peak_labelling and len(buy_idx) > 3:
                buy_idx = np.sort(buy_idx)
                peak_idx = np.arange(buy_idx.shape[0], dtype=int)[buy_idx == np.argmin(raw_signal_subset)][0]
                buy_idx = buy_idx[peak_idx-1:]

            # convert to slice index and store in slice array
            slice_idx = buy_idx + this_idx
            y_slice[slice_idx, 1] = -1

        # # plot
        # t = np.arange(price_slice.shape[0])
        # plt.figure()
        # plt.plot(t, price_slice.Close.values)
        # plt.scatter(t[y_slice[:, 1] == -1], price_slice.Close.values[y_slice[:, 1] == -1], color='green')
        # plt.scatter(t[y_slice[:, 1] == 1], price_slice.Close.values[y_slice[:, 1] == 1], color='red')
        # plt.show()

        # store the label slice in the big array
        y[np.in1d(y[:, 0], y_slice[:, 0])] = y_slice

    return y


if __name__ == '__main__':
    import Utilities.query_shortcuts as qs
    import time
    from datetime import datetime

    timerStart = time.time()
    conn = qs.connect('QQQ')
    et_start = datetime(2024, 1, 1, 9).timestamp()
    et_stop = datetime(2024, 3, 31, 16).timestamp()
    price_df = qs.fetch_price_space(conn, et_pres=et_stop, et_start=et_start)
    # label_arr = create_label_spaces_sliding_window(price_df)
    label_arr = compute_maintenance_classifier_labels(price_space_df=price_df, n_comps=13, cdf_thresh=None, col_str='Open')

    print(f'{(time.time() - timerStart)/60} mins')
