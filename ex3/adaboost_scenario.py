import numpy as np
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from adaboost import AdaBoost
from decision_stump import DecisionStump


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


#
# def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
#     (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)
#
#     # Question 1: Train- and test errors of AdaBoost in noiseless case
#     Adaboost = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
#
#     # Question 2: Plotting decision surfaces
#     T = [5, 50, 100, 250]
#     lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
#     raise NotImplementedError()
#
#     # Question 3: Decision surface of best performing ensemble
#     raise NotImplementedError()
#
#     # Question 4: Decision surface with weighted samples
#     raise NotImplementedError()


custom = ['#636EFA', '#EF553B']  # Placeholder for colorscale


def plot_errors(train_err, test_err, n_learners, noise):
    x = list(range(1, n_learners + 1))
    plt.plot(x, train_err, label='Train Error')
    plt.plot(x, test_err, label='Test Error')
    plt.title(f"Train and Test Errors as a function of the number of fitted learners with noise ratio: {noise}",
              fontsize=8)
    plt.xlabel("Fitted Learners")
    plt.ylabel("Error")
    plt.legend()
    plt.grid()
    plt.show()


def plot_decision_surfaces(adaboost, train_X, test_X, test_y, T, noise):
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.02, vertical_spacing=0.2,
                        subplot_titles=[f"{t} Classifiers" for t in T])
    err = []
    for j, t in enumerate(T):
        fig.add_traces([decision_surface(lambda X: adaboost.partial_predict(X, t), *lims, showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y, colorscale=custom,
                                               line=dict(width=0.5, color="DarkSlateGrey")))
                        ],
                       rows=(j // 2) + 1, cols=(j % 2) + 1)
        err.append(adaboost.partial_loss(test_X, test_y, t))
    print(err)
    fig.update_layout(title=f"Decision Boundary, noise: {noise}", margin_t=100)
    fig.update_xaxes(visible=False).update_yaxes(visible=False)
    fig.write_image(f"adaboost.decision.boundary.all.iterations.noise.{noise}.png")
    fig.show()


def plot_best_ensemble(adaboost, test_X, test_y, test_err, noise):
    min_err = min(test_err)
    min_size = test_err.index(min_err) + 1
    lims = np.array([np.r_[test_X].min(axis=0), np.r_[test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = go.Figure([decision_surface(lambda X: adaboost.partial_predict(X, min_size), *lims, showscale=False),
                     go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                marker=dict(color=test_y, colorscale=custom))
                     ])
    fig.update_layout(
        title=f"Decision Boundary | Best ensemble size: {min_size} | noise: {noise} | accuracy = {1 - min_err}")
    fig.update_xaxes(visible=False).update_yaxes(visible=False)
    fig.write_image(f"adaboost.decision.boundary.best.size.ensemble.noise.{noise}.png")
    # fig.show()


# def plot_weighted_samples(adaboost, train_X, train_y, noise):
#     lims = np.array([np.r_[train_X].min(axis=0), np.r_[train_X].max(axis=0)]).T + np.array([-.1, .1])
#     print(adaboost.D_, "GRFfrbfihf")
#     # D = (adaboost.D_ / adaboost.D_.max()) * 5
#
#     fig1 = go.Figure([decision_surface(adaboost.predict, *lims, showscale=False),
#                       go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
#                                  marker=dict(color=train_y, colorscale=[custom[0], custom[-1]], size=adaboost.D_,
#                                              sizeref=5 * np.max(adaboost.D_) / ((45 if noise == 0 else adaboost.D_) ** 2),
#                                              sizemode="area", sizemin=0.5, line=dict(width=1, color="DarkSlateGrey")))
#                       ])
#
#     fig1.update_layout(title=f"noise={noise}: Decision Bounda with weighted train ensemble")
#     fig1.update_xaxes(visible=False)
#     fig1.update_yaxes(visible=False)
#     fig1.write_html(f"weighted_samples_decision_boundary_noise_{noise}.html")
#
#     fig1.write_image(f"adaboost.decision.boundary.weighted.noise.{noise}.png")
#     fig1.show()




def plot_weighted_samples1(adaboost, train_X, train_y, lims, noise):

    D = 5 * (adaboost.D_ / adaboost.D_.max())

    # Create a mesh grid for the decision surface
    xx, yy = np.meshgrid(np.linspace(lims[0][0], lims[0][1], 500),
                         np.linspace(lims[1][0], lims[1][1], 500))
    Z = adaboost.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 10))
    plt.contourf(xx, yy, Z, alpha=0.3, levels=[-1, 0, 1], colors=['blue', 'red'])

    # Scatter plot with point sizes proportional to sample weights
    scatter = plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, s=D * 25, edgecolor='k', cmap='coolwarm', alpha=0.6)

    # Add color bar
    cbar = plt.colorbar(scatter, ticks=[])
    cbar.set_label('Label', rotation=270, labelpad=15)
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Final AdaBoost Sample Distribution with Noise Level {noise}")

    plt.show()
    plt.grid(True)

    plt.savefig(f"adaboost.decision.boundary.weighted.noise.{noise}", format='png')
    plt.close()


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)
    adaboost = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)

    train_err = [adaboost.partial_loss(train_X, train_y, t) for t in range(1, n_learners + 1)]
    test_err = [adaboost.partial_loss(test_X, test_y, t) for t in range(1, n_learners + 1)]

    # plot_errors(train_err, test_err, n_learners, noise)
    # plot_decision_surfaces(adaboost, train_X, test_X, test_y, [5, 50, 100, 250], noise)
    # plot_best_ensemble(adaboost, test_X, test_y, test_err, noise)
    # plot_weighted_samples(adaboost, train_X, train_y, noise)
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    plot_weighted_samples1(adaboost, train_X, train_y, lims, noise)


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
