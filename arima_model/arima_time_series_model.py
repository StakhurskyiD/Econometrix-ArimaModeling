import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_breusch_godfrey


class ArimaTimeSeriesModel:
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

