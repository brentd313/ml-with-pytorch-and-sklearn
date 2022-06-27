# Object-oriented perceptron API (Chapter 2)
import numpy as np

class RMCPPerceptron:
    """
    Perceptron classifier. Implements a McCulloch-Pitts neuron and uses
    Rosenblatt's perceptron learning rule in order to perform binary
    classification.

    Parameters
    -----------
    lr: float
        Learning rate (between 0.0 and 1.0)
    n_epochs: int
        Number of passes over the training data.
    random_seed: int
        Seed for randomly initializing weight vector.

    Attributes
    -----------
    w_: 1d-array
        Fitted weights.
    b_: scalar
        Fitted bias.
    nerrors_: list of int
        Number of misclassifications (i.e. updates) in each epoch.
    """
    def __init__(lr=0.01, n_epochs=50, random_seed=1):
        self.lr = lr
        self.n_epochs = n_epochs
        self.random_seed = random_seed
    
    def fit(self, X, y):
        """
        Fit perceptron parameters to training data.

        Parameters
        -----------
        X: {array-like}, shape = [n_examples, n_features]
            Feature matrix for training data, where n_examples is the number of
            examples and n_features is the number of features.
        y: array-like, shape = [n_examples]
            Target values for training set.
        
        Returns
        --------
        self: RMCPPerceptron
        """
        rgen = np.random.RandomState(self.random_seed)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float(0.)
        self.nerrors_ = []
        for _ in range(self.n_epochs):
            n_errors = 0
            for xi, yi in zip(X, y):
                update = self.lr * (yi - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                n_errors += int(update != 0.0)
            self.nerrors_.append(n_errors)
        return self

    def net_input(self, X):
        """
        Computes the net input for the given feature matrix, where the net input
        is defined as the weighted sum of the feature vectors and the bias.
        """
        return np.dot(X, self.w_) + self.b_
    
    def predict(self, X):
        """
        Computes the predicted class labels for the given feature matrix.
        """
        return np.where(self.net_input(X) >= 0.0, 1, 0)
                