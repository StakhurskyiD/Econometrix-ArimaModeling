import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, acorr_ljungbox
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

def import_and_clean_data(file_path):
    df = pd.read_csv(file_path, delimiter=';', decimal=',')
    df.replace({'\xa0': ''}, regex=True, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y', errors='coerce')
    df.set_index('Date', inplace=True)
    return df


def visualize_data(data):
    for column in data.columns:
        plt.figure(figsize=(10, 6))
        data[column].plot(title=f'Time Series for {column}')
        plt.show()


def adf_test(data):
    for column in data.columns:
        result = adfuller(data[column].dropna())
        print(f'ADF Test for {column}:\nStatistic: {result[0]}\np-value: {result[1]}')
        for key, value in result[4].items():
            print(f'Critical Value {key}: {value}')


def plot_correlogram(data):
    for column in data.columns:
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plot_acf(data[column].dropna(), ax=plt.gca())
        plt.title(f'ACF for {column}')

        plt.subplot(122)
        plot_pacf(data[column].dropna(), ax=plt.gca())
        plt.title(f'PACF for {column}')

        plt.tight_layout()
        plt.show()


def make_series_stationary(data, column, order=1):
    differenced_data = data[column].diff(periods=order).dropna()
    differenced_data.plot(title=f'{column} Differenced {order} Times')
    plt.show()
    return differenced_data

# Initialize the script
file_path = 'data.csv'  # Replace with your file path
cleaned_data = import_and_clean_data(file_path)
visualize_data(cleaned_data)
adf_test(cleaned_data)
plot_correlogram(cleaned_data)

# Differencing and statistical tests for each variable
for column in cleaned_data.columns:
    print(f'Processing {column}')
    stationary_data = make_series_stationary(cleaned_data, column)
    adf_test({column: stationary_data})
    plot_correlogram({column: stationary_data})

    # Model fitting: Example with AutoReg
    if len(stationary_data) > 0:
        model = AutoReg(stationary_data, lags=12).fit()
        print(model.summary())

        # Conduct Ljung-Box test to check for autocorrelation in residuals
        lb_test = acorr_ljungbox(model.resid, lags=[12], return_df=True)
        print(lb_test)

        # ARIMA modeling as needed
        arima_model = ARIMA(stationary_data, order=(2, 0, 2)).fit()
        print(arima_model.summary())
