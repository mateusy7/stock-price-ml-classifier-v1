from statsmodels.tsa.stattools import adfuller
import talib as ta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix_at_thresholds, precision_recall_curve, roc_curve, roc_auc_score

def load_ml_data(base_path, folder_name):
    """
    Loads X and y train/test CSV files from a specific directory.
    """
    # Example usage: X_train, X_test, y_train, y_test = load_ml_data("../data", "linear")

    path = f"{base_path}/{folder_name}"
    load_args = {'index_col': 0, 'parse_dates': True}

    if folder_name == "linear":
        file_type = ""
    elif folder_name == "poly":
        file_type = "poly_"
    
    # Load the datasets
    X_train = pd.read_csv(f"{path}/X_{file_type}train.csv", **load_args)
    y_train = pd.read_csv(f"{path}/y_{file_type}train.csv", **load_args)
    X_test  = pd.read_csv(f"{path}/X_{file_type}test.csv", **load_args)
    y_test  = pd.read_csv(f"{path}/y_{file_type}test.csv", **load_args)
    
    # Print shapes for immediate verification
    print(f"Loaded data from: {folder_name}")
    print(f"X_{file_type}train: {X_train.shape}, y_{file_type}train: {y_train.shape}")
    print(f"X_{file_type}test : {X_test.shape}, y_{file_type}test : {y_test.shape}\n")
    
    return X_train, X_test, y_train, y_test

## Data pre-processing functions

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

## Model Training Functions ##

# sigmoid function
def sigmoid(X):
    X = np.clip(X, -500, 500) # protect against overflow
    return 1/(1+np.exp(-X))

# compute cost function

def compute_cost(X, y, w, b, lambda_=0):
    """
    Computes cost

    Args:
        X (ndarray (m,n)) : Data, m examples with n features
        y (ndarray (m,))  : target values
        w (ndarray (n,))  : model parameters
        b (scalar)        : model parameter
    
    Returns:
        cost (scalar)     : cost
    """
    m = X.shape[0]
    z = X @ w + b
    f_wb = sigmoid(z)
    
    # Clip f_wb to be between a tiny epsilon and 1-epsilon
    epsilon = 1e-15
    f_wb = np.clip(f_wb, epsilon, 1 - epsilon)
    
    loss = -y * np.log(f_wb) - (1-y) * np.log(1-f_wb)
    regularized_cost = lambda_ / (2*m) * np.sum(w**2)

    return np.sum(loss) / m + regularized_cost

def compute_gradient(X, y, w, b, lambda_=0):
    """
    Computes the gradient for logistic regression 

    Args:
      X (ndarray (m,n)) : data, m examples by n features
      y (ndarray (m,))  : target value 
      w (ndarray (n,))  : values of parameters of the model
      b (scalar)        : value of bias parameter of the model

    Returns
      dj_dw (ndarray (n,)) : The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)       : The gradient of the cost w.r.t. the parameter b. 
    """
    m = X.shape[0]

    # Linear combination and activation
    z = np.dot(X, w) + b
    f_wb = sigmoid(z)
    err = f_wb - y

    # Gradient calculation using matrix multiplication
    dj_dw = np.dot(X.T, err) / m + lambda_ / m * w
    dj_db = np.sum(err) / m

    return dj_dw, dj_db

def run_gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_, display=True):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m, n))      : data, m examples by n features
      y (ndarray (m,))        : target value 
      w_in (ndarray (n,))     : Initial values of parameters of the model
      b_in (scalar)           : Initial value of parameter of the model
      cost_function           : function to compute cost
      gradient_function       : function to compute gradient
      alpha (float)           : Learning rate
      num_iters (int)         : number of iterations to run gradient descent
      lambda_ (scalar, float) : regularization constant
      
    Returns:
      w (ndarray (n,))        : Updated values of parameters of the model
      b (scalar)              : Updated value of parameter of the model
    """
    J_history = [] # save the cost J at each iteration for plotting
    w = np.copy(w_in)
    b = b_in

    for i in range(num_iters):
        # Calculate gradient and update parameters
        dj_dw, dj_db = gradient_function(X, y, w, b, lambda_)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        # Save cost J at each iteration
        cost = cost_function(X, y, w, b, lambda_)
        J_history.append(cost)

        # print cost
        if display:
            # Print cost at intervals 10% of num_iters
            if i % (num_iters // 10) == 0 or i == (num_iters - 1):
                print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.5f}")
    
    if display:
        # Plot the cost J at each iteration
        plt.figure(figsize=(6,4))
        plt.plot(J_history)
        plt.title('Cost over time')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.show()
    
    return w, b

def predict(X, w, b, p=0.5):
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w
    
    Args:
      X (ndarray (m,n)) : data, m examples by n features
      w (ndarray (n,))  : values of parameters of the model      
      b (scalar)        : value of bias parameter of the model

    Returns:
      f_wb (ndarray (m,)) : The predicted probabilities of samples in X being the positive class=1
      p (ndarray (m,))    : The predictions for samples in X using a threshold at 0.5
    """
    f_wb = sigmoid(X @ w + b)

    pred = np.where(f_wb >= p, 1, 0)

    return f_wb, pred

## Model Evaluation Functions

def get_confusion_matrix(y_test, true_col='signal', pred_col='pred'):
    """
    Calculates the TN, FP, FN, TP from a (m, 2) DataFrame where the first column
    are the true labels, and the second column are the predicted labels, and m is
    the number of samples
    """
    positive = y_test[y_test[true_col] == 1] # filter dataframe by true positive labels
    negative = y_test[y_test[true_col] == 0] # filter dataframe by true negative labels

    # True Negatives: # y_true=0, y_pred=0
    TN = len(negative[negative[pred_col] == 0])
    # False Positives: # y_true=0, y_pred=1
    FP = len(negative[negative[pred_col] == 1])
    # False Negatives: # y_true=1, y_pred=0
    FN = len(positive[positive[pred_col] == 0])
    # True Positives: # y_true=1, y_pred=1
    TP = len(positive[positive[pred_col] == 1])

    return TN, FP, FN, TP

def evaluate_classification_performance(y_test, p, f_wb, threshold, title):
    # --- 1. Data Preparation ---
    y_test['pred'] = p.reshape((-1, 1))
    y_true = y_test['signal']
    y_prob = f_wb

    # --- 2. Metric Calculation ---
    TN, FP, FN, TP = get_confusion_matrix(y_test)

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
    NPV = TN / (TN + FN) if (TN + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0 

    # --- 3. Plotting Setup ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3) # Space for metrics and labels

    # Plot 1: Confusion Matrix
    axes[0, 0].set_title("Confusion Matrix")
    ConfusionMatrixDisplay.from_predictions(y_true, y_test['pred'], ax=axes[0, 0], cmap='Blues', colorbar=False)

    # Metrics Text (Placed to the right of Confusion Matrix)
    metrics_text = (
        fr" $\mathbf{{Threshold ({threshold})}}$" + "\n"
        f"Precision: {precision:.3f}\n"
        f"Recall:    {TPR:.3f}\n"
        f"TNR:       {TNR:.3f}\n"
        f"NPV:       {NPV:.3f}"
    )
    axes[0, 0].text(1.05, 0.5, metrics_text, transform=axes[0, 0].transAxes,
                    fontsize=8, verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='white'))

    # Plot 2: Counts vs Thresholds
    tns, fps, fns, tps, thresholds_m = confusion_matrix_at_thresholds(y_true, y_prob)
    axes[0, 1].plot(thresholds_m, tns, label="TN")
    axes[0, 1].plot(thresholds_m, fps, label="FP")
    axes[0, 1].plot(thresholds_m, fns, label="FN")
    axes[0, 1].plot(thresholds_m, tps, label="TP")
    axes[0, 1].set_title("TNs, FPs, FNs and TPs vs Thresholds")
    axes[0, 1].legend()

    # Plot 3: Precision-Recall vs Threshold
    precisions, recalls, thresholds_pr = precision_recall_curve(y_true, y_prob)
    axes[1, 0].plot(thresholds_pr, precisions[:-1], 'b--', label='Precision')
    axes[1, 0].plot(thresholds_pr, recalls[:-1], 'g--', label='Recall')
    axes[1, 0].set_title("Precision-Recall vs Threshold")
    axes[1, 0].legend()

    # Plot 4: ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    axes[1, 1].plot(fpr, tpr)
    axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    axes[1, 1].set_title(f"ROC Curve (AUC Score: {auc_score:0.3f})")
    axes[1, 1].legend()

    fig.suptitle(title, fontsize=16)
    plt.show()