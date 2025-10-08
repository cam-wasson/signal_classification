import numpy as np


def pad_signal(sig, L=512):
    # determine sizes of the pads
    N = sig.shape[0]
    pad_total = L - N
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    # create the padded arrays and concatenate
    pad_left_values = np.full(pad_left, sig[0])
    pad_right_values = np.full(pad_right, sig[-1])
    padded = np.concatenate([pad_left_values, sig, pad_right_values])
    return padded, [len(pad_left_values), L - len(pad_right_values)]


def compute_pad_length(sig):
    # find smallest power of 2 greater than
    return int(2**np.ceil(np.log2(len(sig))))


def moving_avg(x, l):
    xbar = x[0]*np.ones_like(x)
    for i in range(1, l):
        xbar[i] = np.mean(x[:i])
    avg = np.convolve(x, np.ones(l) / l, 'valid')
    xbar[-len(avg):] = avg
    return xbar


def moving_std(x, l):
    x_sig = np.std(x)*np.ones_like(x)
    for i in range(l, len(x)):
        x_sig[i] = np.std(x[i-l:i])
    return x_sig


def moving_max(x, l=10):
    x_max = np.zeros_like(x)
    for i in range(1, len(x)):
        if i < l:
            x_max[i] = max(x[:i])
        else:
            x_max[i] = max(x[i-l:i])
    return x_max


def sinusoidal_f_matrix(omega, dt):
    return np.array([
        [np.cos(omega * dt), np.sin(omega * dt) / omega],
        [-omega * np.sin(omega * dt), np.cos(omega * dt)]
    ])


def sinusoidal_q_matrix(omega, dt, sigma_xi):
    q = sigma_xi ** 2 * dt
    Q = q * np.array([
        [(2 * omega * dt - np.sin(2 * omega * dt)) / (4 * omega ** 3),
         np.sin(omega * dt) ** 2 / (2 * omega ** 2)],
        [np.sin(omega * dt) ** 2 / (2 * omega ** 2),
         dt / 2 + np.sin(2 * omega * dt) / (4 * omega)]
    ])
    return Q


def r_matrix(meas_shape, rho):
    # return identity matrix of R values
    return rho*np.eye(meas_shape)


def h_matrix(x_shape, z_shape):
    H = np.zeros((z_shape, x_shape))
    H[np.diag_indices(min(H.shape))] = 1.0
    return H


def extract_low_pass_components(signal, dt, max_freq=20):
    """
    Analyzes a signal to extract approximate frequency bins (omegas),
    amplitudes, and phases. Returns a dictionary of arrays.
    """
    # Example placeholder: use FFT to get peaks
    freqs = np.fft.fftfreq(len(signal), d=dt)
    fft_vals = np.fft.fft(signal)
    amps = 2 * np.abs(fft_vals) / len(signal)
    phases = np.angle(fft_vals)

    # pick strongest bins
    idx = (abs(freqs) < max_freq)
    clean_signal = np.real(np.fft.ifft(fft_vals * idx))

    # compute omega, amplitude, and phase
    omegas = 2 * np.pi * freqs[idx]
    amplitudes = amps[idx]
    phases_out = phases[idx]
    low_pass_freq = freqs[idx]

    return {
        'freq': low_pass_freq[np.abs(omegas) > 0],
        'omega': omegas[np.abs(omegas) > 0],
        'amp': amplitudes[np.abs(omegas) > 0],
        'phi': phases_out[np.abs(omegas) > 0],
        'raw': signal,
        'truth': clean_signal,
        'dt': dt,
        'fhat_real': np.real(fft_vals[idx]),
        'fhat_imag': np.imag(fft_vals[idx]),
    }
    

def full_fft_extract(signal, dt):
    freqs = np.fft.fftfreq(len(signal), d=dt)
    fft_vals = np.fft.fft(signal)
    amps = 2 * np.abs(fft_vals) / len(signal)
    phases = np.angle(fft_vals)
    
    return {
        'freq': freqs,
        'amp': amps,
        'phi': phases,
        'raw': signal,
        'dt': dt,
        'fhat_real': np.real(fft_vals),
        'fhat_imag': np.imag(fft_vals)
    }


def extract_low_pass_components_cdf_thresh(signal, dt, cdf_thresh=.95):
    # Example placeholder: use FFT to get peaks
    freqs = np.fft.fftfreq(len(signal), d=dt)
    fft_vals = np.fft.fft(signal)
    amps = 2 * np.abs(fft_vals) / len(signal)
    phases = np.angle(fft_vals)

    # pick strongest bins
    cdf_x, cdf_y = compute_cdf(amps, bins=min(1000, len(signal)))
    amp_thresh = np.mean((min(cdf_x[cdf_y >= cdf_thresh]), max(cdf_x[cdf_y <= cdf_thresh])))
    idx = (amps > amp_thresh)

    # compute the clean signal
    clean_signal = np.real(np.fft.ifft(fft_vals * idx))

    # compute omega, amplitude, and phase
    omegas = 2 * np.pi * freqs[idx]
    amplitudes = amps[idx]
    phases_out = phases[idx]
    low_pass_freq = freqs[idx]

    return {
        'freq': low_pass_freq[np.abs(omegas) > 0],
        'omega': omegas[np.abs(omegas) > 0],
        'amp': amplitudes[np.abs(omegas) > 0],
        'phi': phases_out[np.abs(omegas) > 0],
        'truth': clean_signal,
        'fhat_real': np.real(fft_vals),
        'fhat_imag': np.imag(fft_vals),
    }


def compute_omegas(in_signal, pad=True, max_freq=10):
    # pad the signal
    if pad:
        in_signal, true_bounds = pad_signal(in_signal)

    dt = 1 / len(in_signal)
    sig_dict = extract_low_pass_components(in_signal, 1 / dt, max_freq=max_freq)


def compute_cdf(values, bins=1000):
    heights, edges = np.histogram(values, bins=bins)
    cdf_f, cdf_x = np.cumsum(heights) / sum(heights), edges[:-1]
    return cdf_x.flatten(), cdf_f.flatten()

