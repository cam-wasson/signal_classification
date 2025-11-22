from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import StandardScaler


def transform_theta(theta, scale=10):
    return theta * scale


def inverse_transform_theta(theta, scale=10):
    return theta / scale


def anova_1d(values, labels):
    # compute initial and easy stuff
    label_values = np.unique(labels)
    dof_between, dof_within = (label_values.shape[0] - 1, labels.shape[0] - label_values.shape[0])

    # calculate per class stuff
    ss_between, ss_within = 0, 0
    for i in range(label_values.shape[0]):
        # grab indices for this class/group/label value
        lab_idx = (labels == label_values[i])

        # calculate sum of the squares of feature values between classes
        ss_between += len(values[lab_idx]) * np.square(np.abs(np.mean(values[lab_idx]) - np.mean(values)))

        # calculate sum of the squares of feature values  within classes
        ss_within += np.sum(np.square(np.abs(values[lab_idx] - np.mean(values[lab_idx]))))

    # compute F score
    f_score = (ss_between / dof_between) / (ss_within / dof_within)
    return f_score


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

    def __init__(self, filter_state=np.array([]), truth_position=np.array([]), truth_velocity=None, dt=0.0):
        self.filter_state = filter_state
        self.truth_position = truth_position
        self.truth_velocity = truth_velocity
        self.dt = dt


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

    def __init__(self, measurement=np.array([]), filter_state=np.array([]), cluster_dictionary=dict):
        self.measurement = measurement
        self.filter_state = filter_state
        self.cluster_dictionary = cluster_dictionary


def spread_max(context: SpreadMaxContext, penalty=1.2):
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
    max_cluster_values = spread[np.concatenate(cluster_dict['cluster_max']['x_points'])]
    min_cluster_values = spread[np.concatenate(cluster_dict['cluster_min']['x_points'])]
    bad_max_idx = max_cluster_values < 0
    bad_min_idx = min_cluster_values > 0

    # compute the total spread distance
    good_score = sum(max_cluster_values[~bad_max_idx]) + sum(np.abs(min_cluster_values[~bad_min_idx]))
    bad_score = penalty * (sum(np.abs(max_cluster_values[bad_max_idx])) +
                           sum(np.abs(min_cluster_values[bad_min_idx])))

    # convert to loss -- as total spread increases, the loss decreases
    return 1 / max((good_score - bad_score), 10**-8)


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
    return 1 / (np.abs(cosine_sim) - 10 ** -8)


@dataclass
class ANOVAContext:
    measurement: np.array
    label_arr: np.array
    filter_dict: dict
    dt: float
    omegas: np.array

    def __init__(self):
        pass


def anova_loss(context: ANOVAContext):
    # extract values
    raw = context.measurement
    filter_state = context.filter_dict['x']
    # amps = context.filter_dict['amp']
    # phis = context.filter_dict['phi']
    dt = context.dt
    label_arr = context.label_arr[context.label_arr != 0]
    # omegas = context.omegas

    # compute tracking feature values
    spread = raw - filter_state[:, 0]
    # velocity_filter = filter_state[:, 1]
    velocity_empirical = np.diff(np.concatenate(([0], filter_state[:, 0]))) / dt
    acceleration_empirical = np.diff(np.concatenate(([0], filter_state[:, 1]))) / dt
    # velocity_analytical = np.sum(amps*omegas * np.cos(omegas*dt + phis), axis=1)
    # acceleration_analytical = np.sum(-1*amps*np.square(omegas) * np.sin(omegas*dt + phis), axis=1)

    # compute system dynamics feature values
    # amp_dot = np.diff(np.vstack(([0]*amps.shape[1], amps)), axis=0) / dt
    # phi_dot = np.diff(np.vstack(([0]*amps.shape[1], phis)), axis=0) / dt
    # amp_dot_dot = np.diff(np.vstack(([0]*amp_dot.shape[1], amp_dot)), axis=0) / dt
    # phi_dot_dot = np.diff(np.vstack(([0]*phi_dot.shape[1], phi_dot)), axis=0) / dt

    # compute final loss
    total_anova = 0

    total_anova += anova_1d(spread[context.label_arr != 0], label_arr)
    # total_anova += anova_1d(velocity_filter[context.label_arr != 0], label_arr)
    total_anova += anova_1d(velocity_empirical[context.label_arr != 0], label_arr)
    total_anova += anova_1d(acceleration_empirical[context.label_arr != 0], label_arr)

    # total_anova += anova_1d(amp_dot, label_arr)
    # total_anova += anova_1d(phi_dot, label_arr)
    # total_anova += anova_1d(amp_dot_dot, label_arr)
    # total_anova += anova_1d(phi_dot_dot, label_arr)

    # log scale the total ANOVA and
    if total_anova > 1:
        return 1 / np.log10(total_anova)
    else:
        return 1 / total_anova
