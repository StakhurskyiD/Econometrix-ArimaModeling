import matplotlib.pyplot as plt
from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import warnings


def ignore_warnings():
    # Ignore warnings related to non-invertible starting MA parameters
    warnings.filterwarnings('ignore', category=UserWarning, message='Non-invertible starting MA parameters found.*')

    # Ignore warnings about non-stationary starting autoregressive parameters
    warnings.filterwarnings('ignore', category=UserWarning,
                            message='Non-stationary starting autoregressive parameters found.*')

    # Ignore warnings related to date index with no frequency information
    warnings.filterwarnings('ignore', category=ValueWarning,
                            message='A date index has been provided, but it has no associated frequency information.*')

    # Ignore warnings about optimization failures in maximum likelihood calculations
    warnings.filterwarnings('ignore', category=ConvergenceWarning,
                            message='Maximum Likelihood optimization failed to converge.*')


def import_and_clean_data(file_path):
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


def get_stationary_data(data):
    """
    Process all columns in the DataFrame to make them stationary and return a dictionary with the results.
    """
    stationary_data = {}
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):  # Process only numeric columns
            stationary_series, order = make_series_stationary(data, column)
            if stationary_series is not None:
                stationary_data[column] = {
                    'series': stationary_series,
                    'order_of_differencing': order
                }
    return stationary_data


def show_stationary_data(stationary_data):
    """
    Display the stationary data for each column.
    """
    for column, data_info in stationary_data.items():
        series = data_info['series']
        order = data_info['order_of_differencing']
        print(f"Column: {column}")
        print(f"Order of Differencing: {order}")
        series.plot(title=f"Stationary Series for {column}")
        plt.show()
