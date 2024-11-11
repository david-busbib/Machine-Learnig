import loss_functions
from classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import matplotlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
import numpy as np

def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():

    for n, f in [
        ("Linearly Separable", "linearly_separable.npy")
        ,
                 ("Linearly Inseparable", "linearly_inseparable.npy")
                 ]:
        # Load dataset
        x, y = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def callback_func(fit: Perceptron, x_: np.ndarray, y_: int):
            losses.append(fit._loss(x, y))

        d = list(range(1, 1001))
        p = Perceptron(callback=callback_func)
        p._fit(x, y)

        # Plot figure of loss as function of fitting iteration
        # if len(d) != len(losses):
        #     for _ in range(len(d) - len(losses)):
        #         losses.append(0)

        fig = px.line(x=range(len(losses)), y=losses, markers=True,
                      title=f"Perceptron Training Error  - Loss on fitting iteration"
                            f"on {f} "). \
            update_layout(xaxis_title=" ITERATIONS", yaxis_title="ERROR")
        # fig.update_xaxes(range=[0, 55])

        fig.write_image(f"perceptron_loss_{f}.png")
        # fig.show()

def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """

    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    and create comparative visualizations.
    """
    datasets = ["gaussian1.npy", "gaussian2.npy"]

    for dataset in datasets:
        # Load dataset
        X, y = load_dataset(dataset)

        # Initialize and fit classifiers
        lda = LDA()
        gnb = GaussianNaiveBayes()
        gnb._fit(X, y)
        lda._fit(X, y)

        # Predict using classifiers
        gnb_predictions = gnb._predict(X)
        lda_predictions = lda._predict(X)

        # Calculate accuracies
        gnb_accuracy = loss_functions.accuracy(y, gnb_predictions)
        lda_accuracy = loss_functions.accuracy(y, lda_predictions)

        # Create subplots
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.01, vertical_spacing=0.9,
                            subplot_titles=[f"Classifier: Gaussian Naive Bayes  ||  Accuracy: {gnb_accuracy:.2f}",
                                            f"Classifier: LDA  ||  Accuracy: {lda_accuracy:.2f}"])

        # Plot for Gaussian Naive Bayes
        plot_classifier_predictions(fig, X, y, gnb_predictions, gnb, row=1, col=1)

        # Plot for LDA
        plot_classifier_predictions(fig, X, y, lda_predictions, lda, row=1, col=2, showlegend=False)

        # Update layout and save the figure
        fig.update_layout(title_text=f"{dataset} Dataset", title_x=0.5,
                          font=dict(family="Courier New, monospace", size=16, color="RebeccaPurple"))
        fig.write_html(f"{dataset}.Fit.Gaussian.And.LDA.html")


def plot_classifier_predictions(fig, X, y, predictions, classifier, row, col, showlegend=True):
    """
    Helper function to plot classifier predictions.
    """
    for c in np.unique(y):
        # Plot true values
        true_points = X[y == c]
        fig.add_trace(go.Scatter(x=true_points[:, 0], y=true_points[:, 1], mode="markers", name=f"True value - {c}",
                                 marker=dict(symbol=c, size=10, line=dict(color="black", width=2), color='white'),
                                 showlegend=showlegend), row=row, col=col)

        # Plot predicted values
        predicted_points = X[predictions == c]
        colors = ['red', 'yellow', 'blue']
        fig.add_trace(go.Scatter(x=predicted_points[:, 0], y=predicted_points[:, 1], mode="markers",
                                 name=f"Predicted value - {c}",
                                 marker=dict(size=10, line=dict(color="black", width=2), color=colors[int(c)]),
                                 showlegend=showlegend), row=row, col=col)

    # Add means
    fig.add_trace(go.Scatter(x=classifier.mu_[:, 0], y=classifier.mu_[:, 1], mode="markers", name='Mean',
                             marker=dict(symbol='x', size=13, line=dict(color="black", width=2), color='black'),
                             showlegend=showlegend), row=row, col=col)

    # Add ellipses for covariance (if applicable)
    if hasattr(classifier, 'cov_'):
        for i, mean in enumerate(classifier.mu_):
            fig.add_trace(get_ellipse(mean, classifier.cov_), row=row, col=col)
    elif hasattr(classifier, 'vars_'):
        for i, mean in enumerate(classifier.mu_):
            cov = np.diag(classifier.vars_[i])
            fig.add_trace(get_ellipse(mean, cov), row=row, col=col)


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
