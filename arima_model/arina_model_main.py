from arima_model_tests import perform_bg_test, show_adf_test_reslut, show_hannan_rissanen_procedure_results
from arima_time_series_model import ArimaTimeSeriesModel
from model_data_preprocesing import get_stationary_data, show_stationary_data, ignore_warnings, \
    import_and_clean_data


def process_time_series(file_path):
    # Step 1: Ignore specific warnings for cleaner output
    ignore_warnings()

    # Step 2: Load and clean data
    data = import_and_clean_data(file_path)

    # Step 3: Process each column to get stationary data
    stationary_data = get_stationary_data(data)

    # Step 4: Display the stationary series
    show_stationary_data(stationary_data)

    show_adf_test_reslut(stationary_data)

    show_hannan_rissanen_procedure_results(stationary_data)

    # Step 5: Process each series for ARIMA modeling and analysis
    for column, info in stationary_data.items():
        series = info['series']

        # Initialize the time series model class
        ts_model = ArimaTimeSeriesModel(series, column)

        # Analyze correlograms
        ts_model.analyze_correlograms()

        # Fit ARIMA models
        ts_model.fit_models()

        # Evaluate models to find the best one
        ts_model.evaluate_models()

        # Test residuals of the best model
        ts_model.test_residuals()

        # Forecast future values and plot the results
        ts_model.forecast_and_plot(10)

        # Optional: Breusch-Godfrey test for serial correlation if needed
        if ts_model.best_model:
            perform_bg_test(ts_model.best_model, lags=len(ts_model.best_model.arparams))

# Assuming the file path is correctly set
file_path = 'data4.csv'
process_time_series(file_path)
