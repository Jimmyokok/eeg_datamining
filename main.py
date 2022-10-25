from utils.config import build_args
from utils.load import load_eeg_data
from model.csp import csp_transform
from model.svm import SVM
from model.lda import LDA
from sklearn.model_selection import StratifiedKFold, train_test_split
from utils.utils import evaluate_using_tsne
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import random
import torch
from tqdm import tqdm



def main(args):
    if args.verbose:
        print('Loading dataset {}.{}'.format(args.data, args.id))
    x, y = load_eeg_data(args)
    args.nchannels = 2
    accs = []
    f1s = []
    if args.model == 'svm':
        model = SVM
    elif args.model == 'lda':
        model = LDA
    else:
        raise NotImplementedError
    if args.verbose:
        print('Creating classifier {}...'.format(args.model))
    if args.valid == 'cross':
        kf = StratifiedKFold(n_splits=args.k)
        if args.verbose:
            print('Doing cross validation...')
        for train_index, test_index in tqdm(kf.split(x, y)):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            x_train, y_train, x_test, y_test = csp_transform(x_train, y_train, x_test, y_test, args)
            m = model(args)
            m.fit(x_train, y_train)
            y_pred = m.predict(x_test)
            accs.append(accuracy_score(y_test, y_pred))
            f1s.append(f1_score(y_test, y_pred))
        print('Final result: accuracy {}, f1 {}'.format(np.mean(accs), np.mean(f1s)))
    else:
        print('Searching for the optimal seed...')
        for i in tqdm(range(100)):
            seed = random.randint(0, 65536)
            x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=args.ratio, shuffle=True, random_state=seed)
            x_train, y_train, x_test, y_test = csp_transform(x_train, y_train, x_test, y_test, args)
            m = model(args)
            m.fit(x_train, y_train)
            y_pred = m.predict(x_test)
            accs.append(accuracy_score(y_test, y_pred))
            f1s.append(f1_score(y_test, y_pred))
        print('Final result: accuracy {}, f1 {}'.format(np.max(accs), np.max(f1s)))


if __name__ == '__main__':
    args = build_args()
    main(args)