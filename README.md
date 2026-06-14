# Topic 3: Recurrent Neural Network (RNN)

This folder contains a complete assignment on building, training, and deploying a Simple Recurrent Neural Network (RNN) for time-series forecasting of airline passengers.

## 📊 Dataset
*   **Source**: Seaborn's `flights` dataset (monthly international airline passenger numbers from 1949 to 1960).
*   **Task**: Forecast passenger numbers for a future horizon based on past sequences.

## 🧠 Model Architecture
*   **Framework**: TensorFlow/Keras
*   **Layers**:
    *   `SimpleRNN` Layer (32 units, tanh activation)
    *   `Dense` Output Layer (1 unit)
*   **Optimizer**: Adam
*   **Loss**: Mean Squared Error (MSE)
*   **Sequence Length**: 12 months (lookback window).

## 📂 Directory Files
*   `rnn.ipynb`: Jupyter notebook showcasing time-series windowing, scaling, training, autoregressive forecasting, and plotting.
*   `rnn_model.keras`: Saved Keras model file.
*   `app.py`: Streamlit application displaying passenger historical trends vs. future forecasts on Plotly.
*   `requirements.txt`: Python package requirements.

## 🚀 How to Run

1.  Navigate to this directory:
    ```bash
    cd "c:\Users\bsais\OneDrive\Desktop\aiml-ds\week 21\assignment\03_rnn"
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
