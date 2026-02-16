from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import pandas as pd

def check_stationarity(series):
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")

    if result[1] <= 0.05:
        print(f"{series.name} is stationary. Null hypothesis rejected")
        return "stationary"
    else:
        print(f"{series.name} is non-stationary. Failed to reject null hypothesis")
        return "non-stationary"

def plot_data(X, y, x_f1, x_f2, pos_label="y=1", neg_label="y=0"):
    positive = y == 1
    negative = y == 0
    
    plt.plot(X.loc[positive, x_f1], X.loc[positive, x_f2], 'k+', label=pos_label)
    plt.plot(X.loc[negative, x_f1], X.loc[negative, x_f2], 'yo', label=neg_label)