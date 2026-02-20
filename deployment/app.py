import gradio as gr
import pandas as pd
from datetime import timedelta
from utilities_app import load_model, get_yfinance_data, get_engineered_data, apply_zscore_normalization, get_prediction

# Load model once on startup
model = load_model('model.pkl')

# The 27 features in the exact order the model was trained on
ALL_TRAINING_FEATURES = [
    'volume', 'pct_change_1d', 'rsi', 'adx', 'corr', 'volatility',
    'volume^2', 'pct_change_1d^2', 'rsi^2', 'adx^2', 'corr^2', 'volatility^2',
    'volume*pct_change_1d', 'volume*rsi', 'volume*adx', 'volume*corr', 
    'volume*volatility', 'pct_change_1d*rsi', 'pct_change_1d*adx', 
    'pct_change_1d*corr', 'pct_change_1d*volatility', 'rsi*adx', 'rsi*corr', 
    'rsi*volatility', 'adx*corr', 'adx*volatility', 'corr*volatility'
]

def predict_signal():
    ticker = '5296.KL'
    data = get_yfinance_data(ticker)
    
    if data is None or len(data) < 35:
        return "N/A", "N/A", "Error: Insufficient Data"
    
    data.columns = data.columns.str.lower()

    # 1. Feature Engineering
    data_fe = get_engineered_data(data)
    latest_row = data_fe.tail(1)
    
    # 2. Extract Display Info (Input Date, Price, and Target Date)
    input_date = latest_row.index[0]
    # Ensure we get a single float value for the price
    raw_price = data.loc[input_date, 'close']
    input_price = float(raw_price.iloc[0] if hasattr(raw_price, 'iloc') else raw_price)
    
    input_display = f"{input_date.strftime('%d/%m/%Y')} (Close: RM{input_price:.2f})"
    target_date = (input_date + timedelta(days=1)).strftime('%d/%m/%Y')

    # 3. Normalization (Scaling 23 features)
    data_scaled = apply_zscore_normalization(
        latest_row, 
        model['mu'], 
        model['sigma'], 
        model['features_to_scale']
    )

    # 4. Final Alignment and Prediction
    X_final = data_scaled[ALL_TRAINING_FEATURES].values.flatten() 
    prediction = get_prediction(X_final, model['weights'], model['bias'])
    
    result = "UP" if prediction == 1 else "DOWN/SAME"
    return input_display, target_date, result

with gr.Blocks(title="MR.DIY Predictor") as demo:
    gr.Markdown("# 📈 MR.DIY (5296.KL) Directional Signal")
    
    with gr.Row():
        btn = gr.Button("Generate Prediction for Next Close Price", variant="primary")
    
    with gr.Row():
        with gr.Column():
            input_info = gr.Textbox(label="Based on Candle Date", interactive=False)
            target_info = gr.Textbox(label="Prediction for Day", interactive=False)
        with gr.Column():
            result_output = gr.Label(label="Signal")

    btn.click(
        fn=predict_signal, 
        inputs=None, 
        outputs=[input_info, target_info, result_output]
    )

if __name__ == "__main__":
    demo.launch()