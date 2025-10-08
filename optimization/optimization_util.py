import numpy as np
from sklearn.preprocessing import StandardScaler


def transform_theta(theta, scale=10):
    return theta*scale


def inverse_transform_theta(theta, scale=10):
    return theta/scale


def position_error(
        training_data: dict,
        filter_dict: dict,
) -> float:
    """Compute the mean squared error (MSE) between estimates and truth.

    The filter bank is run over the measurements to produce state
    estimates.  The squared error between the estimated and true
    positions and velocities is accumulated across time and filters.

    Parameters
    ----------
    training_data : np.array
        Dictionary containing the ground truth position that the filter should be estimating.
    filter_dict : ndarray
        The filter's estimation of the position/velocity measurements
    Returns
    -------
    float
        Loss accounting for filter's estimation of true position and overall smoothness
    """
    scaler = StandardScaler()

    # extract values from input dictionaries
    filter_state = filter_dict['x']
    truth_position = training_data['truth']

    # scale
    scaled_truth = scaler.fit_transform(truth_position.reshape(-1, 1))
    scaled_filter = scaler.transform(filter_state[:, 0].reshape(-1, 1))

    # compute overall accuracy
    # mse = np.mean(np.square(np.abs(truth_position - filter_state[:, 0])))
    mse = np.mean(np.square(np.abs(scaled_truth - scaled_filter)))

    # compute smoothness of the filter estimates
    sigma_vel = np.std(np.diff(scaled_filter[:, 0]))

    # compute metrics
    smooth_loss = mse + sigma_vel
    return smooth_loss


def velocity_error(
        training_data: dict,
        filter_dict: dict,
) -> float:
    """Compute the mean squared error (MSE) between estimates and truth.

    The filter bank is run over the measurements to produce state
    estimates.  The squared error between the estimated and true
    positions and velocities is accumulated across time and filters.

    Parameters
    ----------
    training_data : dict
        Dictionary containing the ground truth position that the filter should be estimating.
    filter_dict : ndarray
        The filter's estimation of the position/velocity measurements
    Returns
    -------
    float
        Loss accounting for filter's estimation of true velocity and overall smoothness
    """

    # extract values from input dictionaries
    filter_state = filter_dict['x']
    truth_position = training_data['truth']
    dt = training_data['dt']

    # compute true velocity
    truth_velocity = np.gradient(truth_position) / dt

    # compute overall accuracy
    mse = np.mean(np.square(np.abs(truth_velocity - filter_state[:, 1])))

    # compute smoothness of the filter estimates
    acc = np.diff(filter_state[:, 1]) / dt
    tv_acc = np.mean(np.abs(np.diff(acc)))  # total variation of velocity

    # compute metrics
    smooth_loss = mse + tv_acc
    return smooth_loss
