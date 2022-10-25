from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class LDA(object):
    def __init__(self, args):
        super(LDA, self).__init__()
        self.trained = False

    def fit(self, x, y):
        self.lda = LinearDiscriminantAnalysis()
        self.lda.fit(x, y)
        self.trained = True

    def predict(self, x):
        assert self.trained
        return self.lda.predict(x)