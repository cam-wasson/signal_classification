import numpy as np
from filter_bank import NarrowbandTrackingFilterBank


class SentinelPipeline:
    def __init__(self, tracker: NarrowbandTrackingFilterBank, discriminators: dict):
        # store the track/discrim filter banks
        self.tracker = tracker
        self.spread_discriminator = discriminators['spread']
        self.vel_discriminator = discriminators['emp_vel']
        self.acc_discriminator = discriminators['emp_acc']

        # build the feature containers
        self.time_cache = []
        self.track_cache = {'spread': [],
                            'meas': [],
                            'filter_pos': [],
                            'filter_vel': [],
                            'emp_vel': [],
                            'emp_acc': []}
        self.discrim_spread_cache = {'combined': {'spread': [],
                                                  'filter_vel': [],
                                                  'emp_vel': [],
                                                  'emp_acc': []}}
        self.discrim_emp_vel_cache = {'spread': [],
                                      'filter_vel': [],
                                      'emp_vel': [],
                                      'emp_acc': []}
        self.discrim_emp_acc_cache = {'spread': [],
                                      'filter_vel': [],
                                      'emp_vel': [],
                                      'emp_acc': []}

    def step(self, z):
        # put the measurement through the tracker
        tracker_step_x, _ = self.tracker.step(z)

        # update internal tracker cache
        self.update_tracker_cache(z, tracker_step_x)

        # run discrimination models
        disc_spread_out = self.spread_discriminator.step(self.track_cache['spread'][-1])
        disc_vel_out = self.vel_discriminator.step(self.track_cache['emp_vel'][-1])
        disc_acc_out = self.acc_discriminator.step(self.track_cache['acc_vel'][-1])

        # update discrim filter bank caches
        pass

    def update_tracker_cache(self, z, x):
        # store observables
        self.track_cache['meas'].append(z)
        self.track_cache['spread'].append(z - x[0])
        self.track_cache['filter_pos'].append(x[0])
        self.track_cache['filter_vel'].append(x[1])

        # compute features, store
        emp_vel = np.diff(np.concatenate(([0], self.track_cache['filter_pos'])))
        emp_acc = np.diff(np.concatenate(([0], self.track_cache['filter_vel'])))
        self.track_cache['emp_vel'].append(emp_vel[-1])
        self.track_cache['emp_acc'].append(emp_acc[-1])

    def run(self, z_arr):
        pass


if __name__ == '__main__':
    print('hellO!@')
