import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type
from sklearn.metrics import roc_curve, auc
from cross_validate import cross_validate
import plotly
from base_module import BaseModule
from gradient_descent import GradientDescent
from learning_rate import FixedLR
from loss_functions import misclassification_error
from utils import custom
from modules import L1, L2
from logistic_regression import LogisticRegression
from utils import split_train_test
import plotly.graph_objects as go

def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    v = []
    w = []

    def foo(mdl, **kwargs):
        v.append(kwargs['val'])
        w.append(kwargs['weights'])
        return

    return foo, v, w


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: tuple = (1, 0.1, 0.01, 0.001)):
    # plotly.offline.init_notebook_mode()

    fig1 = go.Figure(layout=dict(title="L1 Norm as a function of the GD iteration for all specified learning rates"))
    fig2 = go.Figure(layout=dict(title="L2 Norm as a function of the GD iteration for all specified learning rates"))

    for eta in etas:
        # L2 Norm
        l2 = L2(init.copy())
        callback_l2 = get_gd_state_recorder_callback()
        gd_l2 = GradientDescent(learning_rate=FixedLR(eta), out_type="best", callback=callback_l2[0])
        weights_l2 = gd_l2.fit(l2, X=None, y=None)

        if eta == 0.01:
            plotly.offline.plot(plot_descent_path(L2, np.array(callback_l2[2]), title=f"L2 Norm | eta={eta}"))

        fig2.add_trace(go.Scatter(x=np.arange(len(callback_l2[1])), y=np.array(callback_l2[1]).flatten(),
                                  mode="lines", name=f"eta = {eta}"))

        l2.weights = weights_l2.copy()
        print(f"eta: {eta}")
        print(f"L2 Norm, lowest error: {l2.compute_output()}")

        # L1 Norm
        l1 = L1(init.copy())
        callback_l1 = get_gd_state_recorder_callback()
        gd_l1 = GradientDescent(learning_rate=FixedLR(eta), out_type="best", callback=callback_l1[0])
        weights_l1 = gd_l1.fit(l1, X=None, y=None)

        if eta == 0.01:
            plotly.offline.plot(plot_descent_path(L1, np.array(callback_l1[2]), title=f"L1 Norm | eta={eta}"))

        fig1.add_trace(go.Scatter(x=np.arange(len(callback_l1[1])), y=np.array(callback_l1[1]).flatten(),
                                  mode="lines", name=f"eta = {eta}"))

        l1.weights = weights_l1.copy()
        print(f"L1 Norm, lowest error: {l1.compute_output()}")

    fig1.update_layout(xaxis_title="GD Iteration", yaxis_title="Norm")
    plotly.offline.plot(fig1)

    fig2.update_layout(xaxis_title="GD Iteration", yaxis_title="Norm")
    plotly.offline.plot(fig2)


def load_data(path: str = "SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
    X_test, y_test = X_test.to_numpy(), y_test.to_numpy()
    # Plotting convergence rate of logistic regression over SA heart disease data
    callback, descent_path, values = get_gd_state_recorder_callback()
    gd = GradientDescent(callback=callback, learning_rate=FixedLR(1e-4), max_iter=20000, out_type="last")
    lg = LogisticRegression(solver=gd)
    lg._fit(X_train, y_train)
    # plot ROC curve - taken from LAB 04
    fpr, tpr, thresholds = roc_curve(y_train, lg.predict_proba(X_train))
    c = [custom[0], custom[-1]]
    plotly.offline.plot(go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         marker_color=c[1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=f"{{ROC Curve Of Fitted Model - Logistic Regression}}={auc(fpr, tpr):.6f}",
                         xaxis=dict(title="False Positive Rate (FPR)"),
                         yaxis=dict(title="True Positive Rate (TPR)"))))
    a_star = round(thresholds[np.argmax(tpr - fpr)], 2)
    lg.alpha_ = a_star
    print("-----------")
    print(f"a* = {a_star}")
    print(f"model test error: {lg._loss(X_test, y_test)}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    s = np.zeros((len(lambdas), 2))
    p="l1"
    train_errors, validation_errors = [], []
    b = 0
    for i in lambdas:
        train_error, valid_error = cross_validate(LogisticRegression(solver=gd, penalty=p, lam=i),
                                                  X_train, y_train, misclassification_error)
        s[b] = train_error, valid_error
        b += 1
        train_errors.append(train_error), validation_errors.append(valid_error)
    best_lam = lambdas[np.argmin(validation_errors)]
    model = LogisticRegression(solver=gd, penalty=p, lam=best_lam).fit(X_train, y_train)
    print("-----------")
    print(f"module {p.capitalize()} | lambda chosen: {best_lam}")
    print(f"model test error: {model.loss(X_test, y_test)}")
    if p=="l1":
        fig = go.Figure([go.Scatter(x=lambdas, y=s[:, 0], name="Train Error"),
                         go.Scatter(x=lambdas, y=s[:, 1], name="Validation Error")],
                        layout=go.Layout(
                            title="Train and Validation errors (averaged over the k-folds)",
                            xaxis=dict(title=r"lambda$", type="log"),
                            yaxis_title="Error Value"))

        plotly.offline.plot(fig)
    else:
        fig1 = go.Figure([go.Scatter(x=lambdas, y=s[:, 0], name="Train Error"),
                         go.Scatter(x=lambdas, y=s[:, 1], name="Validation Error")],
                        layout=go.Layout(
                            title="Train and Validation errors (averaged over the k-folds)",
                            xaxis=dict(title="lambda", type="log"),
                            yaxis_title="Error Value"))

        plotly.offline.plot(fig1)

if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    # fit_logistic_regression()