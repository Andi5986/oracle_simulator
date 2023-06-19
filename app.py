import streamlit as st
import numpy as np
from utils import simulate_dapp_oracle  # Import the function from utils.py
from model_neural import create_and_train_model as create_and_train_model_nn
from model_svr import create_and_train_model as create_and_train_model_svr
from model_gb import create_and_train_model_gb
from model_ridge import create_and_train_model_ridge



def app():
    st.sidebar.write("**Simulator**")
    size = st.sidebar.slider('Size of the dataset (days)', min_value=30, max_value=365*5, value=365*3) # 3 years by default
    
    st.title('dApp Oracle Simulation')

    st.write('Simulating data...')
    df = simulate_dapp_oracle(size)

    models = [
        ('Neural Network', create_and_train_model_nn),
        ('Support Vector Regressor', create_and_train_model_svr),
        ('Gradient Boosting Regressor', create_and_train_model_gb),
        ('Ridge Regression', create_and_train_model_ridge)
    ]

# Inside your app function in app.py
    for model_name, model_func in models:
        st.write(f'Training {model_name}...')
        model = model_func(df)
        df[f'Predicted ETH ({model_name})'] = model.predict(df[['Tasks', 'User Base', 'Ether to USD', 'Ether Volatility']])

    # Display time charts
    st.subheader('Time Charts')

    # Random data used
    st.line_chart(df.set_index('Date')['User Base'])
    st.line_chart(df.set_index('Date')['Tasks'])

    # Evolution of Ether vs USD
    st.line_chart(df.set_index('Date')['Ether to USD'])

    # Evolution of ETH
    columns_to_chart = ['ETH'] + [f'Predicted ETH ({model_name})' for model_name, _ in models]
    st.line_chart(df.set_index('Date')[columns_to_chart])
    
    df['USD from ETH'] = np.log1p(df['ETH'] * df['Ether to USD'])
    st.line_chart(df.set_index('Date')['USD from ETH'])

    st.sidebar.write('''
    **Variables:** ***ETH*** as the dependent variable, ***Task*** as the independent variable, 
    ***Users*** as the mediating variable, and ***Currency (Ether to USD)*** as the moderating variable.

    **Models:** Support Vector Regressor (SVR), Multilayer Perceptron (MLP), Gradient Boosting Regressor (GBR), and Ridge Regressor.

    ''')

# Run the app
if __name__ == '__main__':
    app()
