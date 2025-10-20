import numpy as np
from filterpy.kalman import KalmanFilter


class SinusoidalKalmanFilter(KalmanFilter):
    def __init__(self, dim_x=2, dim_z=1, omega=np.pi/4, dt=0.0, sigma_xi=0.1, rho=1e-2):
        super().__init__(dim_x=dim_x, dim_z=dim_z)
        # set state and dynamics info
        self.x = np.zeros(dim_x)
        self.dim_z = dim_z
        self.dt = dt
        self.omega = omega
        self.sigma_xi = sigma_xi
        self.rho = rho

        # set filter transition matrices
        self.F = sinusoidal_f_matrix(omega, dt)
        self.P = np.eye(dim_x) * 1e5
        self.H = h_matrix(dim_x, dim_z)
        self.set_r(rho)
        self.set_q(sigma_xi)

    def amplitude(self):
        return np.sqrt(self.x[0]**2 + (self.x[1]/self.omega)**2)

    def phase(self):
        return np.arctan2(self.x[0], self.x[1]/self.omega)

    def set_q(self, sigma_xi):
        self.Q = sinusoidal_q_matrix(self.omega, self.dt, sigma_xi)
        self.sigma_xi = sigma_xi

    def set_r(self, rho):
        self.R = r_matrix(self.dim_z, rho)
        self.rho = rho


def r_matrix(meas_shape, rho):
    # return identity matrix of R values
    return rho*np.eye(meas_shape)


def h_matrix(x_shape, z_shape):
    H = np.zeros((z_shape, x_shape))
    H[np.diag_indices(min(H.shape))] = 1.0
    return H
    
    
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