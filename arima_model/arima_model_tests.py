import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, acorr_breusch_godfrey
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller


def plot_correlogram(data, column):
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plot_acf(data[column].dropna(), ax=plt.gca())
    plt.title(f'ACF for {column}')

    plt.subplot(122)
    plot_pacf(data[column].dropna(), ax=plt.gca())
    plt.title(f'PACF for {column}')

    plt.tight_layout()
    plt.show()


def adf_test(series, column_name):
    """
    Perform the Augmented Dickey-Fuller test on a series and return the results.

    Parameters:
        series (pd.Series): Time series data on which to perform the test.
        column_name (str): The name of the series or column being tested.

    Returns:
        dict: A dictionary containing the ADF statistic, p-value, number of lags used,
              critical values, and a message indicating if the series is stationary.
    """
    result = adfuller(series.dropna())
    adf_output = {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Number of lags used': result[2],
        'Number of observations used': result[3],
        'Critical values': result[4],
        'Is stationary': result[1] < 0.05
    }

    print(f"ADF Test Results for {column_name}:")
    print(f"ADF Statistic: {adf_output['ADF Statistic']}")
    print(f"p-value: {adf_output['p-value']}")
    for key, value in adf_output['Critical values'].items():
        print(f"Critical Value ({key}): {value}")

    if adf_output['Is stationary']:
        print(f"The series '{column_name}' is stationary.")
    else:
        print(f"The series '{column_name}' is not stationary.")

    return adf_output


def show_adf_test_reslut(stationary_data):
    for column, info in stationary_data.items():
        series = info['series']
        order = info['order_of_differencing']

        # Perform the ADF test and process the results within the same iteration
        adf_results = adf_test(series, column)
        print(f"Order of Differencing applied: {order}\n")

def hannan_rissanen_procedure(data, max_ar_order=15):
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    high_order_ar_model = AutoReg(data, lags=max_ar_order).fit()
    residuals = high_order_ar_model.resid

    # Make sure residuals are in a pandas Series for shifting
    residuals = pd.Series(residuals)

    exog = residuals.shift(1).dropna()
    data_aligned, exog_aligned = data.align(exog, join='inner')

    best_aic = np.inf
    best_bic = np.inf
    best_order = None
    best_model = None

    for p in range(1, max_ar_order + 1):
        try:
            arma_model = ARIMA(data_aligned, order=(p, 0, 0), exog=exog_aligned).fit()
            if arma_model.aic < best_aic or arma_model.bic < best_bic:
                best_aic = arma_model.aic
                best_bic = arma_model.bic
                best_order = p
                best_model = arma_model
        except ValueError as e:
            print(f"Encountered a value error with AR order {p}: {e}")
            continue

    if best_model is not None:
        ljung_pvalue = acorr_ljungbox(best_model.resid, lags=[best_order], return_df=True)['lb_pvalue']
        return {
            'best_order': best_order,
            'best_model': best_model,
            'best_aic': best_aic,
            'best_bic': best_bic,
            'ljung_pvalue': ljung_pvalue[best_order]
        }
    else:
        return None


def show_hannan_rissanen_procedure_results(stationary_data):
    for column, info in stationary_data.items():
        series = info['series']
        order = info['order_of_differencing']

        # Perform the ADF test and process the results within the same iteration
        adf_results = adf_test(series, column)

        # Applying Hannan-Rissanen procedure
        hr_results = hannan_rissanen_procedure(series)
        print(hr_results)
        if hr_results:
            print(f"Results for {column}:")
            print(f"Best AR Order: {hr_results['best_order']}")
            print(f"Best AIC: {hr_results['best_aic']}")
            print(f"Best BIC: {hr_results['best_bic']}")
            print(f"Ljung-Box p-value: {hr_results['ljung_pvalue']}\n")
        else:
            print(f"No valid model found for {column}\n")


def perform_bg_test(model, lags):
    """
    Perform the Breusch-Godfrey serial correlation LM test on the residuals of an ARIMA model.

    Parameters:
        model (ARIMAResults): The fitted ARIMA model.
        lags (int): Number of lags to use for the test.

    Returns:
        tuple: Test statistic, p-value, observed R-squared, and p-value for the R-squared.
    """
    bg_test = acorr_breusch_godfrey(model, nlags=lags)
    print("\nBreusch-Godfrey Serial Correlation LM Test Results:")
    print(f"F-statistic: {bg_test[0]}, p-value: {bg_test[1]}")
    print(f"Obs*R-squared: {bg_test[2]}, p-value: {bg_test[3]}")
    return bg_test