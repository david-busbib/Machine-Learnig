import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from polynomial_fitting import PolynomialFitting
from house_price_prediction import train_test_split
import plotly.express as px
def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df =pd.read_csv(filename,parse_dates=['Date']).dropna().drop_duplicates()

    df["Month"]=df["Month"].astype(int)
    df["Year"] = df["Year"].astype(str)
    df["Day"] = df["Day"].astype(int)
    df=df[(df['Temp'] >= -50)&(df['Temp'] <= 50)&(df['Day']>=1)&(df['Day']<=31 )]

    df = df[1 <= df['Month']]
    df = df[12 >= df['Month']]
    df["DayOfYear"]=df["Date"].dt.dayofyear
    return df


def qes_3():
    global israel_data
    israel_data = df[df["Country"] == "Israel"]
    scatter_plot = px.scatter(israel_data, x="DayOfYear", y="Temp", color="Year",
                              title="Average Daily Temperature | the Day of the year ")
    scatter_plot.write_image("israel_daily_avg_temp_change.png")
    monthly_temp_std = israel_data.groupby('Month')['Temp'].std()
    bar_plot = px.bar(monthly_temp_std, x=list(range(1, 13)), y='Temp',
                      title="Standard Deviation of Daily Temperatures for Each Month",
                      labels={"x": "Month", "y": "Standard Deviation of Temperature"})
    bar_plot.write_image("israel_monthly_temp_std.png")


def qes_4():
    global df_3
    df_3 = df.groupby(['Country', 'Month'], as_index=False).agg({'Temp': ['mean', 'std']})
    df_3.columns = ['Country', 'Month', 'Temp_mean', 'Temp_std']
    plt.figure(figsize=(12, 6))
    # Iterate over each country to plot separately
    for country in df_3['Country'].unique():
        country_data = df_3[df_3['Country'] == country]
        plt.errorbar(country_data['Month'], country_data['Temp_mean'], yerr=country_data['Temp_std'],
                     label=country, capsize=5, marker='o', linestyle='-', markersize=5)
    plt.title("Israel Standard Deviation of Daily Temperatures by Month ")
    plt.xlabel("Month")
    plt.ylabel("Temperature (°C)")
    plt.legend(title="Country")
    plt.grid(True)
    plt.savefig("israel_monthly_avg_temp_with_std.png")
    plt.show()


def qe_5():
    global pol, fig
    x_isreal = israel_data['DayOfYear']
    y_israel = israel_data['Temp']
    train_X, test_X, train_y, test_y = train_test_split(x_isreal, y_israel)
    lost_list = []
    for k in range(1, 11):
        pol = PolynomialFitting(k)
        pol.fit(train_X.to_numpy(), train_y.to_numpy())
        re = round(pol.loss(test_X, test_y), 2)
        lost_list.append(re)
        print('k:', k, ', test error:', re)
    fig = px.bar(x=list(range(1, 11)), y=lost_list,
                 title=f"Loss of the polynomial model of degree k over the test set",
                 labels={"x": f"K", "y": "Loss"})
    fig.write_image("israel_4.png")


def qes_6():
    global fig
    polynomial_model = PolynomialFitting(5)
    losses = []
    polynomial_model.fit(israel_data['DayOfYear'].to_numpy(), israel_data['Temp'].to_numpy())
    countries = ['Israel', 'South Africa', 'Jordan', 'The Netherlands']
    for country in countries:

        country_data = df[df['Country'] == country]
        days = country_data['DayOfYear'].values
        temps = country_data['Temp'].values
        mse_loss = polynomial_model.loss(days, temps)
        losses.append(mse_loss)
    fig = px.bar(
        x=countries[1:],
        y=losses[1:],
        title="Loss of the Polynomial Model of Degree 5 Over the Country Set",
        labels={"x": "Country", "y": "Loss"}
    )
    fig.write_image("israel_5.png")


if __name__ == '__main__':
    # Question 2 - Load and preprocessing of city temperature dataset
    np.random.seed(0)
    df = load_data("city_temperature.csv")


    # Question 3 - Exploring data for specific country
    qes_3()
    # Question 4 - Exploring differences between countries

    qes_4()

    # Question 5 - Fitting model for different values of `k`
    qe_5()
    # Question 6 - Evaluating fitted model on different countries
    qes_6()
    pass
