import scipy.signal
import numpy as np
from numpy import linalg
from sklearn.preprocessing import StandardScaler


def bandpass(trials, args):
    ntrials = trials.shape[0]
    nchannels = trials.shape[1]
    nsamples = trials.shape[2]
    sample_rate = 100
    lo = args.lo
    hi = args.hi

    a, b = scipy.signal.iirfilter(6, [lo / (sample_rate / 2.0), hi / (sample_rate / 2.0)])
    trials_filt = np.zeros((ntrials, nchannels, nsamples))
    for i in range(ntrials):
        trials_filt[i, :, :] = scipy.signal.filtfilt(a, b, trials[i, :, :], axis=1)
    return trials_filt


def logvar(trials):
    return np.log(np.var(trials, axis=2))


class CSP(object):
    def __init__(self):
        super(CSP, self).__init__()
        self.W = None
        self.trained = False

    def cov(self, trials):
        # trials [ntrials, nchannels, nsamples]
        return np.mean((trials @ np.transpose(trials, [0, 2, 1])) / trials.shape[2], axis=0)

    def whitening(self, sigma):
        U, l, _ = linalg.svd(sigma)
        return U @ np.diag(l ** -0.5)

    def fit(self, x, y):
        cov0 = self.cov(x[y == 0])
        cov1 = self.cov(x[y == 1])
        P = self.whitening(cov0 + cov1)
        B, _, _ = linalg.svd(P.T @ cov1 @ P)
        self.W = P @ B
        self.trained = True

    def predict(self, x):
        assert self.trained
        return np.einsum('ij,kjl->kil', self.W.T, x)


def csp_transform(x, y, x_test, y_test, args):
    x_filt = bandpass(x, args)
    x_test_filt = bandpass(x_test, args)
    csp = CSP()
    csp.fit(x_filt, y)
    x_train = logvar(csp.predict(x_filt)[:, [0, -1], :])
    x_test = logvar(csp.predict(x_test_filt)[:, [0, -1], :])
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    return x_train, y, x_test, y_test