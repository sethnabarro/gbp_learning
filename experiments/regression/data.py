# coding=utf-8
import numpy as np

from core.utils.data import ReinitDataIterator


def load_data(n_tr_points=90, n_te_points=225, seed=411):
    np.random.seed(seed)

    def gt(x):
        return np.sin(x * 5. / 0.75) * 5. + (10. * x) ** 2. * 0.15

    xs_obs = np.sort(np.random.uniform(-1., 1., size=(n_tr_points, 1)))
    std_nonstat = 1.5 + 1. * xs_obs ** 2.
    ys_obs = gt(xs_obs) + np.random.normal(scale=std_nonstat)
    y_mean_tr, y_std_tr = np.mean(ys_obs), np.std(ys_obs)
    ys_obs = (ys_obs - y_mean_tr) / (2. * y_std_tr)
    train_data = ReinitDataIterator(list(zip(xs_obs[:, None, None, None].astype(np.float32), ys_obs.astype(np.float32))))
    xs_obs_te = np.linspace(-1.1, 1.1, n_te_points)[:, None]
    ys_obs_te = gt(xs_obs_te) + np.random.normal(size=(n_te_points, 1)) * 1.5
    ys_obs_te = (ys_obs_te - y_mean_tr) / y_std_tr
    test_data = ReinitDataIterator(list(zip(xs_obs_te[:, None, None, None].astype(np.float32), ys_obs_te.astype(np.float32))))
    return train_data, test_data
