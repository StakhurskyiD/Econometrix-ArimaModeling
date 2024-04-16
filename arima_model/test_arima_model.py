import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, acorr_breusch_godfrey
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='Non-invertible starting MA parameters found.*')
warnings.filterwarnings('ignore', category=ValueWarning, message='A date index has been provided, but it has no associated frequency information.*')
warnings.filterwarnings('ignore', category=ConvergenceWarning, message='Maximum Likelihood optimization failed to converge.*')


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


def make_series_stationary(data, column):
    """
    Attempts to make a series stationary through differencing.
    Assumes data is a pandas DataFrame and column is the column to be processed.
    Returns the stationary series and the order of differencing that achieved stationarity.
    """
    print(f"Processing column: {column}")
    series = data[column].dropna()  # Ensure no NaN values
    original_series = series.copy()

    # Check initial stationarity
    result = adfuller(series)
    if result[1] <= 0.05:
        print(f"{column} is already stationary.")
        return series, 0

    # Attempt to difference the series up to 3 times
    for d in range(1, 4):
        series = original_series.diff(periods=d).dropna()
        result = adfuller(series)
        if result[1] <= 0.05:
            print(f"{column} becomes stationary after {d} order differencing.")
            plt.figure(figsize=(12, 6))
            plt.plot(series, label=f'{column} - Order {d} Differenced')
            plt.title(f'Stationary Series for {column} after {d} order differencing')
            plt.legend()
            plt.show()
            return series, d

    print(f"{column} did not become stationary after 3 differencings.")
    return None, None  # None or original series could be returned based on preference


def process_all_columns(data):
    """
    Process all columns in the DataFrame to make them stationary.
    """
    stationary_data = {}
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):  # Process only numeric columns
            stationary_series, order = make_series_stationary(data, column)
            if stationary_series is not None:
                stationary_data[column] = {'series': stationary_series, 'order_of_differencing': order}
    return stationary_data


# Example usage assuming 'df' is your DataFrame loaded with data
file_path = 'data4.csv'
cleaned_data = import_and_clean_data(file_path)
stationary_results = process_all_columns(cleaned_data)



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


# Example usage within a loop processing stationary data results
for column, info in stationary_results.items():
    series = info['series']
    order = info['order_of_differencing']

    # Perform the ADF test and process the results within the same iteration
    adf_results = adf_test(series, column)
    print(f"Order of Differencing applied: {order}\n")


def hannan_rissanen_procedure(data, max_ar_order=15):
    # Ensure the input data is a pandas Series
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


for column, info in stationary_results.items():
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



import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_breusch_godfrey

class TimeSeriesModel:
    def __init__(self, series, column_name):
        self.series = series
        self.column_name = column_name
        self.best_model = None
        self.models = []

    def analyze_correlograms(self):
        max_lags = min(40, int(len(self.series) / 2))
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plot_acf(self.series.dropna(), ax=plt.gca(), lags=max_lags)
        plt.title('Autocorrelation Function')
        plt.subplot(122)
        plot_pacf(self.series.dropna(), ax=plt.gca(), lags=max_lags)
        plt.title('Partial Autocorrelation Function')
        plt.tight_layout()
        plt.show()

    def fit_models(self):
        for p in range(1, 13):  # Example range for AR terms
            for q in range(1, 13):  # Example range for MA terms
                try:
                    model = ARIMA(self.series, order=(p, 1, q)).fit()
                    self.models.append((p, q, model.aic, model.bic, model))
                except Exception as e:
                    print(f"Failed to fit model (p={p}, q={q}): {str(e)}")
                    continue

    def evaluate_models(self):
        if not self.models:
            print("No models have been fitted.")
            return
        sorted_models = sorted(self.models, key=lambda x: (x[2], x[3]))  # Sort by AIC then BIC
        self.best_model = sorted_models[0][4]
        print(self.best_model.summary())

    def test_residuals(self):
        if self.best_model is None:
            print("No best model has been selected.")
            return
        bg_test = acorr_breusch_godfrey(self.best_model, nlags=len(self.best_model.arparams))
        jb_test = jarque_bera(self.best_model.resid)
        print(f"Breusch-Godfrey test: F-stat={bg_test[0]}, p-value={bg_test[1]}")
        print(f"Jarque-Bera test: JB statistic={jb_test[0]}, p-value={jb_test[1]}")

    def forecast_and_plot(self, periods=10):
        if self.best_model is None:
            print("No best model has been selected for forecasting.")
            return

        # Get the last date from the historical data index
        last_date = self.series.index[-1]

        # Create a date range for the forecast
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods,
                                       freq=self.series.index.freq)

        # Get the forecast result and set the index to the forecast dates
        forecast_result = self.best_model.get_forecast(steps=periods)
        forecast = forecast_result.predicted_mean
        forecast.index = forecast_dates

        # Get the confidence interval and set the index to the forecast dates
        forecast_conf_int = forecast_result.conf_int()
        forecast_conf_int.index = forecast_dates

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(self.series, label='Historical Data')
        plt.plot(forecast.index, forecast, label='Forecast')
        plt.fill_between(forecast_conf_int.index,
                         forecast_conf_int.iloc[:, 0],
                         forecast_conf_int.iloc[:, 1],
                         color='gray', alpha=0.3)
        plt.legend()
        plt.show()


    # Example usage
for column, info in stationary_results.items():
    ts_model = TimeSeriesModel(info['series'], column)
    ts_model.analyze_correlograms()
    ts_model.fit_models()
    ts_model.evaluate_models()
    ts_model.test_residuals()
    ts_model.forecast_and_plot(10)  # Forecast 10 periods ahead

    print(f"\nProcessing complete for {column}\n")
