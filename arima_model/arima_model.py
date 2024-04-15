import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm


def import_and_clean_data(file_path):
    # Read the CSV file using the semicolon delimiter and set the first row as the header
    df = pd.read_csv(file_path, delimiter=';', header=0, skiprows=[0], decimal=',')

    # Replace non-breaking spaces in numeric columns and convert them to numeric type
    df = df.apply(lambda x: pd.to_numeric(x.astype(str).str.replace('\xa0', ''), errors='coerce'))

    # Parse the 'Date' column to datetime format, assuming the date is in the format YYYY
    df['Date'] = pd.to_datetime(df['Date'], format='%Y', errors='coerce')

    # Set 'Date' as the DataFrame index
    df.set_index('Date', inplace=True)

    return df


def make_series_stationary(data, column, order=1):
    """
    Applies differencing to a specified column in the DataFrame to make it stationary.

    Parameters:
    data: pandas DataFrame containing the time series data.
    column: String name of the column to be differenced.
    order: The order of differencing (1 for first difference, 2 for second difference, etc.)

    Returns:
    stationary_data: pandas DataFrame with the differenced data.
    """
    # Differencing the data
    stationary_data = data[column].diff(periods=order).dropna()

    # Visualize the differenced data
    plt.figure(figsize=(10, 6))
    plt.plot(stationary_data, label=f'{column} - {order} order differenced')
    plt.title(f'{column} - {order} order Differenced Data')
    plt.legend()
    plt.show()

    return stationary_data


def visualize_data(data, columns):
    for column in columns:
        data[column].plot(title=f'Time Series of {column}')
        plt.show()


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


def adf_test(data, column):
    result = adfuller(data[column].dropna())
    print(f'ADF Statistic for {column}: {result[0]}')
    print(f'p-value for {column}: {result[1]}')
    for key, value in result[4].items():
        print(f'Critical Value ({key}): {value}')


file_path = 'data.csv'

cleaned_data = import_and_clean_data(file_path)
stationary_gdp = make_series_stationary(cleaned_data, 'GDP')


# Visualize the time series data
visualize_data(cleaned_data, cleaned_data.select_dtypes(include='number').columns)

# Plot ACF and PACF correlograms for each numerical column
for column in cleaned_data.select_dtypes(include='number').columns:
    plot_correlogram(cleaned_data, column)

# Perform ADF test on each numerical column
for column in cleaned_data.select_dtypes(include='number').columns:
    print(f'\nAugmented Dickey-Fuller Test on "{column}"')
    adf_test(cleaned_data, column)


def difference_data(data, column, order=1):
    differenced_data = data[column].diff(periods=order).dropna()
    differenced_data.plot(title=f'{column} - {order} order differenced')
    plt.show()
    return differenced_data

stationary_gdp = make_series_stationary(cleaned_data, 'GDP')

# Apply first differencing to the non-stationary variables
differenced_inflation = difference_data(cleaned_data, 'Inflation', order=1)
differenced_exchange_rate = difference_data(cleaned_data, 'Exchange rate', order=1)
differenced_total_export = difference_data(cleaned_data, 'Total Export', order=1)
differenced_unemployment_rate = difference_data(cleaned_data, 'Unenployment Rate', order=1)
# Usage

print(cleaned_data.head())


def adf_test(differenced_data, column_name):
    print(f'Augmented Dickey-Fuller Test on "{column_name}"')
    result = adfuller(differenced_data)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print(f'Critical Value ({key}): {value}')
    print('\n')

# Perform ADF test on the differenced data
adf_test(differenced_inflation, 'Differenced Inflation')
adf_test(differenced_exchange_rate, 'Differenced Exchange rate')
adf_test(differenced_total_export, 'Differenced Total Export')
adf_test(differenced_unemployment_rate, 'Differenced Unenployment Rate')


def hannan_rissanen_procedure(data, max_ar_order=15):
    # Fit a high-order AR model to estimate residuals
    high_order_ar_model = AutoReg(data, lags=max_ar_order).fit()
    residuals = high_order_ar_model.resid

    # Prepare the external regressor by shifting and dropping NaNs, then align
    exog = residuals.shift(1).dropna()
    data_aligned, exog_aligned = data.align(exog, join='inner')

    best_aic = np.inf
    best_bic = np.inf
    best_order = None
    best_model = None

    for p in range(1, max_ar_order + 1):
        try:
            arma_model = ARIMA(data_aligned, order=(p, 0, 0), exog=exog_aligned).fit()
            # Compare AIC and BIC for model selection
            if arma_model.aic < best_aic or arma_model.bic < best_bic:
                best_aic = arma_model.aic
                best_bic = arma_model.bic
                best_order = p
                best_model = arma_model
        except ValueError as e:
            print(f"Encountered a value error with AR order {p}: {e}")
            continue

    if best_model is not None:
        # Validation using Ljung-Box test
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


results_inflation = hannan_rissanen_procedure(differenced_inflation)
print("Differenced Inflation")
print(f"Best AR order: {results_inflation['best_order']}")
print(f"Best model AIC: {results_inflation['best_aic']}")
print(f"Best model BIC: {results_inflation['best_bic']}")
print(f"Ljung-Box p-value: {results_inflation['ljung_pvalue']}")


results_inflation = hannan_rissanen_procedure(differenced_exchange_rate)
print("Exchange Rate")
print(f"Best AR order: {results_inflation['best_order']}")
print(f"Best model AIC: {results_inflation['best_aic']}")
print(f"Best model BIC: {results_inflation['best_bic']}")
print(f"Ljung-Box p-value: {results_inflation['ljung_pvalue']}")


results_inflation = hannan_rissanen_procedure(differenced_total_export)
print("Total Export")
print(f"Best AR order: {results_inflation['best_order']}")
print(f"Best model AIC: {results_inflation['best_aic']}")
print(f"Best model BIC: {results_inflation['best_bic']}")
print(f"Ljung-Box p-value: {results_inflation['ljung_pvalue']}")


results_inflation = hannan_rissanen_procedure(differenced_unemployment_rate)
print("Unemployment rate")
print(f"Best AR order: {results_inflation['best_order']}")
print(f"Best model AIC: {results_inflation['best_aic']}")
print(f"Best model BIC: {results_inflation['best_bic']}")
print(f"Ljung-Box p-value: {results_inflation['ljung_pvalue']}")

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_correlograms(data):
    # Calculate the maximum number of lags allowed (50% of the sample size)
    max_lags = len(data) // 2 - 1  # Ensures it's less than 50% of the sample size

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plot_acf(data, ax=plt.gca(), lags=min(40, max_lags))
    plt.title('Autocorrelation Function')

    plt.subplot(122)
    plot_pacf(data, ax=plt.gca(), lags=min(40, max_lags))
    plt.title('Partial Autocorrelation Function')

    plt.tight_layout()
    plt.show()


# Adjusted call to use a safe number of lags based on your data's length
# plot_correlograms(differenced_inflation)  # Use your actual differenced series here


plot_correlograms(differenced_inflation)
plot_correlograms(differenced_exchange_rate)
plot_correlograms(differenced_total_export)
plot_correlograms(differenced_unemployment_rate)


# Fit and summarize an AR(12) model for differenced inflation
model_diff_inflation = sm.tsa.AutoReg(differenced_inflation, lags=12).fit()
print("Differenced Inflation AR(12) Model Summary:")
print(model_diff_inflation.summary())

# Fit and summarize an AR(12) model for differenced exchange rate
model_diff_exchange_rate = sm.tsa.AutoReg(differenced_exchange_rate, lags=12).fit()
print("\nDifferenced Exchange Rate AR(12) Model Summary:")
print(model_diff_exchange_rate.summary())

# Fit and summarize an AR(12) model for differenced total export
model_diff_total_export = sm.tsa.AutoReg(differenced_total_export, lags=12).fit()
print("\nDifferenced Total Export AR(12) Model Summary:")
print(model_diff_total_export.summary())

# Fit and summarize an AR(12) model for differenced unemployment rate
model_diff_unemployment_rate = sm.tsa.AutoReg(differenced_unemployment_rate, lags=12).fit()
print("\nDifferenced Unemployment Rate AR(12) Model Summary:")
print(model_diff_unemployment_rate.summary())


# Fit ARIMA models for each entity with the estimated parameters

# Differenced Inflation ARIMA(p=2, d=1, q=2)
arima_diff_inflation = ARIMA(differenced_inflation, order=(2, 1, 2)).fit()

# Differenced Exchange Rate ARIMA(p=1, d=1, q=1)
arima_diff_exchange_rate = ARIMA(differenced_exchange_rate, order=(1, 1, 1)).fit()

# Differenced Total Export ARIMA(p=2, d=1, q=2)
arima_diff_total_export = ARIMA(differenced_total_export, order=(2, 1, 2)).fit()

# Differenced Unemployment Rate ARIMA(p=2, d=1, q=2)
arima_diff_unemployment_rate = ARIMA(differenced_unemployment_rate, order=(2, 1, 2)).fit()

# Output the summaries of the fitted ARIMA models
arima_model_summaries = {
    "Differenced Inflation ARIMA(2,1,2)": arima_diff_inflation.summary(),
    "Differenced Exchange Rate ARIMA(1,1,1)": arima_diff_exchange_rate.summary(),
    "Differenced Total Export ARIMA(2,1,2)": arima_diff_total_export.summary(),
    "Differenced Unemployment Rate ARIMA(2,1,2)": arima_diff_unemployment_rate.summary()
}

print(arima_model_summaries)

