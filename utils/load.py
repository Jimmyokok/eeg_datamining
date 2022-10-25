import numpy as np
import scipy.io


def load_eeg_data(args):
    y = []
    x = []

    m = scipy.io.loadmat('data/{}/BCICIV_calib_ds1{}.mat'.format(args.data, args.id), struct_as_record=True)
    EEG = m['cnt'].T
    cl_lab = [s[0] for s in m['nfo']['classes'][0][0][0]]
    sample_rate = m['nfo']['fs'][0][0][0][0]
    nchannels = EEG.shape[0]

    nsamples = EEG.shape[1]
    event_onsets = m['mrk'][0][0][0]
    event_codes = m['mrk'][0][0][1]
    labels = np.zeros((1, nsamples), int)
    labels[0, event_onsets] = event_codes

    win = np.arange(int(args.trial_begin * sample_rate), int(args.trial_end * sample_rate))
    for cl, code in zip(cl_lab, np.unique(event_codes)):
        cl_onsets = event_onsets[event_codes == code]
        for i, onset in enumerate(cl_onsets):
            x.append(EEG[:, win + onset])
            y.append(cl_lab.index(cl))
    x = np.concatenate([x], axis=0)
    y = np.array(y).astype(int)
    assert x.shape[0] == y.shape[0]
    # Print some information
    if args.verbose:
        print('\tTotal trials:', x.shape[0])
        print('\tSample rate:', sample_rate)
        print('\tNumber of channels:', nchannels)
        print('\tClass labels:', cl_lab)
    return x, y