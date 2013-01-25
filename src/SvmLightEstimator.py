from sklearn.base import BaseEstimator
import svmlight as svm

class SvmLightEstimator(BaseEstimator):
    def __init__(self, cost=None, cost_ratio=1.0, biased_hyperplane=False):
        self.svm = svm.Learner()
        if cost:
            self.svm.cost = cost
        self.svm.biased_hyperplane = biased_hyperplane
        self.svm.cost_ratio = cost_ratio

    def get_params(self, deep=True):
        return {'cost': self.svm.cost,
                'cost_ratio': self.svm.cost_ratio,
                'biased_hyperplane': self.svm.biased_hyperplane}
    
    def set_params(self, **params):
        if 'cost' in params:
            self.svm.cost = params['cost']
        if 'cost_ratio' in params:
            self.svm.cost_ratio = params['cost_ratio']
        if 'biased_hyperplane' in params:
            self.svm.biased_hyperplane = params['biased_hyperplane']
        return self

    def fit(self, data, class_values):
        self.model = self.svm.learn(data, class_values)
    
    def predict(self, data):
        return [1 if self.model.classify(d) > 0 else -1
                for d in data]
