from __future__ import annotations
from typing import Callable
from typing import NoReturn
from base_estimator import BaseEstimator
import numpy as np
from loss_functions import misclassification_error


def default_callback(fit: Perceptron, x: np.ndarray, y: int):
    pass


class Perceptron(BaseEstimator):
    """
    Perceptron half-space classifier

    Finds a separating hyperplane for given linearly separable data.

    Attributes
    ----------
    include_intercept: bool, default = True
        Should fitted model include an intercept or not

    max_iter_: int, default = 1000
        Maximum number of passes over training data

    coefs_: ndarray of shape (n_features,) or (n_features+1,)
        Coefficients vector fitted by Perceptron algorithm. To be set in
        `Perceptron.fit` function.

    callback_: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response
    """

    def __init__(self,
                 include_intercept: bool = True,
                 max_iter: int = 1000,
                 callback: Callable[[Perceptron, np.ndarray, int], None] = default_callback):
        """
        Instantiate a Perceptron classifier

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        max_iter: int, default = 1000
            Maximum number of passes over training data

        callback: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response
        """
        super().__init__()
        self.include_intercept_ = include_intercept
        self.max_iter_ = max_iter
        self.callback_ = callback
        self.coefs_ = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a halfspace to to given samples. Iterate over given data as long as there exists a sample misclassified
        or that did not reach `self.max_iter_`

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.fit_intercept_`
        """
        if self.include_intercept_:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.coefs_ = np.zeros(X.shape[1])
        for iteration in range(self.max_iter_):
            misclassified = False
            for xi, target in zip(X, y):
                if np.dot(self.coefs_, xi) * target <= 0:
                    self.coefs_ += xi * target
                    misclassified = True
                    if self.callback_:
                        self.callback_(self, xi, target)
                    break
            if not misclassified:
                break

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if self.include_intercept_:
            b = np.ones((X.shape[0], 1))
            X = np.hstack((b, X))

        return np.sign(X @ self.coefs_)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """

        return misclassification_error(y, self._predict(X))


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Initialize means and priors
        self.mu_ = np.zeros((n_classes, n_features))
        self.pi_ = np.zeros(n_classes)

        # Compute the mean vectors and prior probabilities for each class
        for idx, label in enumerate(self.classes_):
            X_class = X[y == label]
            self.mu_[idx, :] = np.mean(X_class, axis=0)
            self.pi_[idx] = X_class.shape[0] / X.shape[0]

        # Compute the common covariance matrix
        self.cov_ = np.zeros((n_features, n_features))
        for idx, label in enumerate(self.classes_):
            X_class = X[y == label]
            centered = X_class - self.mu_[idx, :]
            self.cov_ += np.dot(centered.T, centered)

        self.cov_ /= X.shape[0] - n_classes
        self._cov_inv = np.linalg.inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        discriminants = (X @ self._cov_inv @ self.mu_.T - 0.5 *
                         np.sum(self.mu_ @ self._cov_inv *
                                self.mu_,
                                axis=1) + np.log(self.pi_))
        return self.classes_[np.argmax(discriminants, axis=1)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        likelihoods = np.zeros((n_samples, n_classes))

        for idx, label in enumerate(self.classes_):
            mean_vec = self.mu_[idx]
            diff = X - mean_vec
            exponent = -0.5 * np.sum(diff @ self._cov_inv * diff, axis=1)
            coefficient = 1 / np.sqrt((2 * np.pi) ** X.shape[1] * np.linalg.det(self.cov_))
            likelihoods[:, idx] = coefficient * np.exp(exponent)

        return likelihoods

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(self.predict(X), y)


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.mu_ = np.zeros((n_classes, n_features))
        self.vars_ = np.zeros((n_classes, n_features))
        self.pi_ = np.zeros(n_classes)

        for idx, label in enumerate(self.classes_):
            X_class = X[y == label]
            self.mu_[idx, :] = np.mean(X_class, axis=0)
            self.vars_[idx, :] = np.var(X_class, axis=0)
            self.pi_[idx] = X_class.shape[0] / X.shape[0]

        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling predict")

        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        class_probs = np.zeros((n_samples, n_classes))

        for idx, label in enumerate(self.classes_):
            pdf = np.prod(self.gaussian_pdf(X, self.mu_[idx], self.vars_[idx]), axis=1)
            class_probs[:, idx] = pdf * self.pi_[idx]
        predictions = np.argmax(class_probs, axis=1)
        return self.classes_[predictions]

    def gaussian_pdf(self, x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:

        return np.exp(-0.5 * ((x - mu) ** 2 / var)) / np.sqrt(2 * np.pi * var)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        # Initialize array to store likelihoods for each sample and class
        likelihoods = np.zeros((n_samples, n_classes))

        # Calculate likelihood for each sample and class
        for i in range(n_samples):
            for j in range(n_classes):
                # Calculate the product of individual feature probabilities
                product = np.prod(self.gaussian_pdf(X[i], self.mu_[j], self.vars_[j]))
                # Multiply by the class prior
                likelihoods[i, j] = product * self.pi_[j]

        return likelihoods

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self._predict(X))
