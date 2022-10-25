from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


class SVM(object):
    def __init__(self, args):
        super(SVM, self).__init__()
        self.trained = False

    def fit(self, x, y):
        svc = SVC()
        params = {
            'C': [0.01, 0.1, 1.0, 10.0]
        }
        self.gs = GridSearchCV(svc, param_grid=params)
        self.gs.fit(x, y)
        self.trained = True

    def predict(self, x):
        assert self.trained
        return self.gs.predict(x)