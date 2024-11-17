import numpy as np

class LogisticRegression(object):
    """"Logistic Regression classifier using Gradient Descent
    Parameters
    -----------
    eta : float
        Learning rate (between 0.00 and 1.0)
    n_iter : int
        passes over traning data
    random_state : int
        Random number generator seed for weights initialization

    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting
    b_ : scalar
        bias after fitting
    cost_ : list
        Logistic cost function value in each epoch
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data
        Parameters
        -----------
        X : {array-like}, shape [n_samples, n_features]

        y : array_like, shape [n_samples]
        
        Returns
        --------
        self : object
        """
        rgen = np.random.RandomState(random_state=self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size = X.shape[1])
        self.b_ = np.float_(.0)
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            error = (y-output)
            self.w_ += self.eta * X.T.dot(error)
            self.b_ += self.eta * error.sum()
            cost = (-y.dot(np.log(output)) - (1-y).dot(np.log(1-output)))
            self.cost_.append(cost)


    def net_input(self, X):
        """Forward pass"""
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
    
    def predict(self, X):
        """predict the label class"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)