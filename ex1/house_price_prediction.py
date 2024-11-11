from typing import NoReturn
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
    preprocess training data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data
    y: pd.Series

    Returns  df["recently_renovated"] = np.where(df["yr_renovated"] >= np.percentile(df.yr_renovated.unique(), 70), 1, 0)
    df = df.drop("yr_renovated", 1)
    -------
    A clean, preprocessed version of the data
    """
    df_ = X.drop(["id", "lat", "long", "date", "sqft_lot15", "sqft_living15"], axis=1)
    df_.fillna(df_.mean(), inplace=True)
    df_["decade_home_built"] = (df_["yr_built"] / 10).astype(int)
    df_ = pd.get_dummies(df_, prefix="new_decade_home_built_", columns=['decade_home_built'])

    df_["new_renovated_data_"] = (
        np.where(df_["yr_renovated"] >= np.percentile(df_.yr_renovated.unique(), 75), 1, 0))

    df_ = df_.drop(["yr_built", "yr_renovated"], axis=1)

    df_ = df_[(df_["bedrooms"] < 21) & (df_["sqft_lot"] < 900000)]
    y_df = y.loc[df_.index].dropna()
    df_ = df_.loc[y_df.index]
    return df_, y.loc[df_.index].dropna()


def preprocess_test(X: pd.DataFrame):
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """
    X_pre_test_ = X.drop(["id", "lat", "long", "date", "sqft_lot15", "sqft_living15"], axis=1)
    if "yr_renovated" in X_pre_test_.columns and np.any(X_pre_test_['yr_renovated'].notna()):
        recent_renovation_threshold = np.percentile(
            X_pre_test_.loc[X_pre_test_['yr_renovated'].notna(), 'yr_renovated'].unique(), 75)
        X_pre_test_["new_renovated_data_"] = np.where(X_pre_test_["yr_renovated"] >= recent_renovation_threshold, 1, 0)
    else:
        X_pre_test_["new_renovated_data_"] = 0
    # Convert year built to decade built
    if "yr_built" in X_pre_test_.columns:
        X_pre_test_["decade_home_built"] = (X_pre_test_["yr_built"] / 10).astype(int)
    else:
        X_pre_test_["decade_home_built"] = 0  # Default value if yr_built is somehow missing
    # Drop columns that were dropped in the training set
    X_pre_test_ = X_pre_test_.drop(["yr_built", "yr_renovated"], axis=1)

    X_pre_test_ = pd.get_dummies(X_pre_test_, prefix="new_decade_home_built_", columns=['decade_home_built'])

    return X_pre_test_


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # Ensure the output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)


        X = X.drop("intercept", axis=1)
        # P = ["condition"]
        for col in X:

            # calculate pearson corr
            rho = np.cov(X[col], y)[0, 1] / (np.std(X[col]) * np.std(y))

            fig = px.scatter(pd.DataFrame({'x': X[col], 'y': y}), x="x", y="y", trendline="ols",
                             title=f"Correlation Between {col} Values and Response <br>Pearson Correlation {rho}",
                             labels={"x": f"{col} Values", "y": "Response Values"})
            fig.write_image(f"pearson.correlation.{col}.png")
    pass


def plot_qes_6(results, p_values,flag=None):
    mean_losses = np.mean(results, axis=1)
    std_losses = np.std(results, axis=1)
    plt.figure(figsize=(10, 5))
    plt.errorbar(p_values, mean_losses, yerr=2 * std_losses, fmt='o', capsize=5,
                 capthick=2, ecolor='gray', markersize=5,  # Smaller markersize
                 mfc='blue', mec='blue', errorevery=1,  # Blue color for markers
                 linestyle='dotted',
                 linewidth=2, label='Mean Loss ± 2 STD')
    plt.plot(p_values, mean_losses, linestyle='solid', linewidth=2, color='blue')

    plt.title('Mean Loss vs. Training Set Size')
    plt.xlabel('Percentage of Training Data Used (%)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_size_vs_loss.png")
    plt.show()
    # frames =[]
    # frames.append(go.Frame(
    # data=[
    #     go.Scatter(x=x, y=y_plot, mode="markers+lines",
    #                name="Real Points", marker=dict(color="black", opacity=.7))],

def train_test_split(X, y, test_size=0.25, random_state=None):
    if random_state:
        np.random.seed(random_state)

        # Generate random indices
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    # Calculate number of samples for the test set
    test_samples = int(len(X) * test_size)

    # Split indices into train and test
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]

    # Create training and testing sets
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")
    X = df.drop('price', axis=1)
    y = df['price']
    # X_train1, X_test1, y_train1, y_test1 = sk.train_test_split(X, y, test_size=0.25, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train_processed, y_train_processed = preprocess_train(X_train, y_train)
    feature_evaluation(X_train_processed, y_train_processed, output_path="feature_plots")
    X_test_processed = preprocess_test(X_test)

    print("Test data preprocessing successful.")

    p_values = range(10, 101)
    results = np.zeros((len(p_values), 10))
    y_plot=[]
    y_std=[]
    for index, p in enumerate(p_values):
        for trial in range(10):
            sample_X = X_train_processed.sample(frac=(p / 100), random_state=trial)
            sample_y = y_train_processed.loc[sample_X.index]
            our_model_reg_ = LinearRegression(include_intercept=True)
            our_model_reg_.fit(sample_X, sample_y)
            # y_pred = our_model_reg_.predict(X_test_processed)
            mse = our_model_reg_.loss(X_test_processed, y_test)
            y_plot.append(np.mean(mse))
            y_std.append(np.std(mse))
            results[index, trial] = mse

    std_pred = np.array(y_std)
    y_plot = np.array(y_plot)
    lst =list(p_values)
    plot_qes_6(results,lst)
