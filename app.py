import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import plotly.graph_objects as go
import seaborn as sns

st.set_page_config(
    page_title="RNN Forecaster",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(10, 18, 30) 0%, rgb(20, 24, 40) 90%);
        color: #f1f5f9;
    }
    
    .header-container {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .title-gradient {
        background: linear-gradient(135deg, #34d399 0%, #059669 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.2rem;
        font-weight: 300;
    }
    
    .card {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class='header-container'>
        <div class='title-gradient'>RNN Time-Series Forecaster</div>
        <div class='subtitle'>Predicting Airline Passenger Trends using RNN Networks</div>
    </div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('rnn_model.keras')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model or scaler: {e}. Run the Jupyter Notebook first.")
    st.stop()

# Load seaborn flights dataset for demo
df = sns.load_dataset('flights')
all_passengers = df['passengers'].values.astype(float)
dates = [f"{row['year']}-{row['month']}" for _, row in df.iterrows()]

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("<div class='card'><h4>🔮 Autoregressive Multi-Month Forecast</h4>", unsafe_allow_html=True)
    forecast_months = st.slider("Select months to forecast into future:", 1, 24, 12)
    
    # Predict future
    input_seq = all_passengers[-12:]
    scaled_seq = scaler.transform(input_seq.reshape(-1, 1)).flatten()
    
    predictions = []
    curr_seq = list(scaled_seq)
    
    for _ in range(forecast_months):
        x_in = np.array(curr_seq[-12:]).reshape(1, 12, 1)
        pred = model.predict(x_in)[0][0]
        predictions.append(pred)
        curr_seq.append(pred)
        
    predictions_unscaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    
    st.markdown(f"<h5>Next Month Forecasted Value:</h5>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='color: #10b981; margin-top:-10px;'>{int(predictions_unscaled[0])} passengers</h2>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'><h4>📈 Historical Trend & Future Forecast Projection</h4>", unsafe_allow_html=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates[-24:], 
        y=all_passengers[-24:],
        mode='lines+markers',
        name='Historical Passengers (Last 2 Years)',
        line=dict(color='#64748b', width=3)
    ))
    
    future_dates = [f"Future month {i+1}" for i in range(forecast_months)]
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predictions_unscaled,
        mode='lines+markers',
        name='Forecasted Passengers',
        line=dict(color='#10b981', width=3, dash='dash')
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1'),
        margin=dict(l=40, r=40, t=20, b=40),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        legend=dict(x=0.01, y=0.99),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
