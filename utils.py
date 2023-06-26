import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

def fetch_ether_to_usd(size):
    url = "https://api.coingecko.com/api/v3/coins/ethereum/market_chart"
    params = {"vs_currency": "usd", "days": size}  
    response = requests.get(url, params=params)
    data = response.json()

    df = pd.DataFrame(data['prices'], columns=['time', 'price'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')

    return df['price']  # Returns a pandas Series instead of a numpy array

def generate_stochastic_user_growth(size):
    user_base = np.empty(size)
    user_base[0] = 1000
    growth = np.random.normal(loc=0.001, scale=0.02, size=size-1)  
    user_base[1:] = user_base[0] * np.cumprod(1 + growth)
    return user_base

def generate_stochastic_tasks(size):
    tasks = np.empty(size)
    tasks[0] = np.random.randint(1, 1000)
    growth = np.random.normal(loc=0.01, scale=0.1, size=size-1)
    tasks[1:] = tasks[0] * np.cumprod(1 + growth)
    return tasks

def generate_date_range(size):
    start_date = datetime.now() - timedelta(days=size)
    return pd.date_range(start_date, periods=size).tolist()

def calculate_eth(task, user_base, currency_relation, currency_volatility, net_eth):
    hedge = 1 + currency_volatility**2  # Increase hedge when volatility increases
    currency_relation = 1 / currency_relation 
    eth_limit = 15_000_000
    scaling_factor = 1 - net_eth / eth_limit
    scaling_factor = max(scaling_factor, 0)  # Ensure the scaling factor is not negative
    task = task * scaling_factor
    user_base = user_base * scaling_factor
    eth = np.abs(currency_relation * hedge) * np.log1p(np.abs(task**2)/np.abs(user_base**2))
    eth = np.log1p(eth) 
    return eth


def calculate_volatility(data, window=10):
    percent_change = data.pct_change()
    return percent_change.rolling(window).std()

def simulate_dapp_oracle(size):
    dates = generate_date_range(size)
    tasks = pd.Series(generate_stochastic_tasks(size), index=range(size))  # Convert to pandas Series with proper index
    user_base = pd.Series(generate_stochastic_user_growth(size), index=range(size))  # Convert to pandas Series with proper index
    ether_to_usd = pd.Series(fetch_ether_to_usd(size), index=range(size))  # Convert to pandas Series with proper index
    ether_volatility = calculate_volatility(ether_to_usd).fillna(0)
    net_eth = 0
    eth_values = []

    for i in range(size):
        eth = calculate_eth(tasks[i], user_base[i], ether_to_usd[i], ether_volatility[i], net_eth)
        net_eth += eth
        if i % 10 == 0:  # Example condition: Ether is bought back every 10 days
            eth_bought_back = eth / 2  # Example: Half of the Ether calculated is bought back
            net_eth -= eth_bought_back

        eth_values.append(eth)
    return pd.DataFrame(
        {
            'Date': dates,
            'Tasks': tasks,
            'User Base': user_base,
            'Ether to USD': ether_to_usd,
            'Ether Volatility': ether_volatility,
            'ETH': eth_values,
        }
    )
