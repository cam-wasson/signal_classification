from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import StandardScaler


def transform_theta(theta, scale=10):
    return theta*scale


def inverse_transform_theta(theta, scale=10):
    return theta/scale


@dataclass
class PositionErrorContext:
    filter_state: np.array
    truth_position: np.array


def position_error(
        context: PositionErrorContext
) -> float:
    """Compute the mean squared error (MSE) between estimates and truth.

    The filter bank is run over the measurements to produce state
    estimates.  The squared error between the estimated and true
    positions and velocities is accumulated across time and filters.

    Parameters
    ----------
    context: position_error_context
        Data class containing the necessary metadata to compute the loss function
    Returns
    -------
    float
        Loss accounting for filter's estimation of true position and overall smoothness
    """

    # extract values from input dictionaries
    filter_state = context.filter_state
    truth_position = context.truth_position

    # scale
    scaler = StandardScaler()
    scaled_truth = scaler.fit_transform(truth_position.reshape(-1, 1)).flatten()
    scaled_filter = scaler.transform(filter_state[:, 0].reshape(-1, 1)).flatten()

    # compute overall accuracy
    mse = np.mean(np.square(np.abs(scaled_truth - scaled_filter)))

    # compute smoothness of the filter estimates
    sigma_vel = np.std(np.diff(scaled_filter))

    # compute metrics
    smooth_loss = mse + sigma_vel
    return smooth_loss


@dataclass
class VelocityErrorContext:
    filter_state: np.array
    truth_position: np.array
    truth_velocity: None
    dt: float


def velocity_error(
        context: VelocityErrorContext
) -> float:
    """Compute the mean squared error (MSE) between estimates and truth.

    The filter bank is run over the measurements to produce state
    estimates.  The squared error between the estimated and true
    positions and velocities is accumulated across time and filters.

    Parameters
    ----------
    context: velocity_error_context
        Data class containing the necessary metadata to compute the loss function
    Returns
    -------
    float
        Loss accounting for filter's estimation of true velocity and overall smoothness
    """

    # extract values from input dictionaries
    filter_state = context.filter_state
    if context.truth_velocity is None:
        truth_position = context.truth_position
        dt = context.dt
        truth_velocity = np.gradient(truth_position) / dt
    else:
        truth_velocity = context.truth_velocity

    # scale
    scaler = StandardScaler()
    scaled_truth = scaler.fit_transform(truth_velocity.reshape(-1, 1))
    scaled_filter = scaler.transform(filter_state[:, 1].reshape(-1, 1))

    # compute overall accuracy
    mse = np.mean(np.square(np.abs(scaled_truth - scaled_filter)))

    # compute smoothness of the filter estimates
    tv = np.std(np.diff(filter_state[:, 1]))

    # compute metrics
    smooth_loss = mse + tv
    return smooth_loss


@dataclass
class SpreadMaxContext:
    measurement: np.array
    filter_state: np.array
    cluster_dictionary: dict


def spread_max(context: SpreadMaxContext):

    # extract data
    filter_state = context.filter_state
    measurements = context.measurement
    cluster_dict = context.cluster_dictionary

    # scale the data
    scaler = StandardScaler()
    scaled_truth = scaler.fit_transform(measurements.reshape(-1, 1))
    scaled_filter = scaler.transform(filter_state[:, 0].reshape(-1, 1))

    # compute spread
    spread = scaled_truth - scaled_filter

    # compute the total values of the spread for each cluster type
    cluster_max_sum = np.sum(spread[np.concatenate(cluster_dict['cluster_max']['x_points'])])
    cluster_min_sum = np.sum(spread[np.concatenate(cluster_dict['cluster_min']['x_points'])])

    # compute the total spread distance
    total_spread = cluster_max_sum - cluster_min_sum

    # convert to loss -- as total spread increases, the loss decreases
    return 1 / total_spread


class PhaseAlignmentContext:
    filter_state = np.array
    truth_position = np.array
    max_frequency = float
    cdf_thresh = None
    dt = float


def phase_alignment(context: PhaseAlignmentContext):
    # run FFT on the raw data
    fhat_truth = np.fft.fft(context.truth_position)
    fft_freq = np.fft.fftfreq(fhat_truth)
    keep_idx = np.abs(fft_freq) <= context.max_frequency

    # run FFT on the kf
    fhat_kf = np.fft.fft(context.filter_state[:, 0])

    # perform cosine similarity of FFT spectra
    dot = np.sum(fhat_truth[keep_idx] * fhat_kf[keep_idx], axis=1)  # element-wise dot product on rows
    mag1 = np.linalg.norm(fhat_truth[keep_idx])
    mag2 = np.linalg.norm(fhat_kf[keep_idx])
    cosine_sim = dot / (mag1 * mag2)

    # promote cosine sims close to 1, -1; punish cosine sim ~0
    return 1 / (np.abs(cosine_sim) - 10**-8)
