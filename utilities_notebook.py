from statsmodels.tsa.stattools import adfuller
import talib as ta
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

def zscore_normalize_features(X: pd.DataFrame, columns: list):
    """
    computes Z-score normalized features by column
    
    Args:
      X (DataFrame (m,n)) : input data, m examples, n features
      columns (list (p))  : names of features to be normalized
      
    Returns:
      X_norm (DataFrame (m,n)) : input with selected columns normalized
      mu (Series (p,1))     : mean of each normalized feature
      sigma (Series (p,1))  : standard deviation of each normalized feature
    """
    
    X_norm = X.copy()
    mu = {}
    sigma = {}

    for feature in columns:
        mu[feature] = X[feature].mean()
        sigma[feature] = X[feature].std()
        X_norm[feature] = (X[feature] - mu[feature]) / sigma[feature]
    
    mu = pd.Series(mu)
    sigma = pd.Series(sigma)
    
    return X_norm, mu, sigma

def apply_zscore_normalization(X: pd.DataFrame, mu: pd.Series, sigma: pd.Series, columns: list):
    """
    Applies pre-computed Z-score normalization to specific columns.
    
    Args:
      X (DataFrame)       : input data to be normalized (e.g., X_test)
      mu (Series)         : pre-computed means from the training set
      sigma (Series)      : pre-computed standard deviations from the training set
      columns (list)      : names of features to be normalized
      
    Returns:
      X_norm (DataFrame)  : dataframe with selected columns standardized
    """
    
    X_norm = X.copy()

    for feature in columns:
        # Standardize using the training parameters: (x - mu_train) / sigma_train
        X_norm[feature] = (X[feature] - mu[feature]) / sigma[feature]
    
    return X_norm

def get_engineered_data(df, period=10):
    """
    Combines technical indicator generation and polynomial expansion.
    Input: Raw OHLCV DataFrame with capitalized headers.
    Output: DataFrame with base features, squared terms, and interactions.
    """
    data = df.copy()

    # 1. Standardize Column Names
    data.columns = data.columns.str.lower()

    # 3. Base Feature Calculations
    data['future_1d_returns'] = data['close'].pct_change().shift(-1).dropna()
    data['pct_change_1d'] = data['close'].pct_change()
    data['rsi'] = ta.RSI(data['close'], timeperiod=period)
    data['adx'] = ta.ADX(data['high'], data['low'], data['close'], timeperiod=period)
    
    sma = data['close'].rolling(window=period).mean()
    data['corr'] = data['close'].rolling(window=period).corr(sma)
    data['volatility'] = data['pct_change_1d'].rolling(window=period).std() * 100

    # 4. Cleanup NaNs before expansion
    data.dropna(inplace=True)

    # 5. Define Base Features to Expand
    base_features = ['volume', 'pct_change_1d', 'rsi', 'adx', 'corr', 'volatility']
    
    # Isolate the data for expansion
    X = data[base_features]
    cols = X.columns
    poly_df = X.copy()
    
    # 6. Polynomial Expansion (Squares & Interactions)
    for i in range(len(cols)):
        # Squared terms
        poly_df[f"{cols[i]}^2"] = X[cols[i]] ** 2
        
        # Interaction terms
        for j in range(i + 1, len(cols)):
            poly_df[f"{cols[i]}*{cols[j]}"] = X[cols[i]] * X[cols[j]]

    assert len(data['future_1d_returns']) == len(poly_df)
    
    return data['future_1d_returns'], poly_df