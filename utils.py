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

    return df['price'].values[:size]

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

def calculate_eth(task, user_base, currency_relation):
    eth = np.abs(task * user_base) / currency_relation
    eth = np.log1p(eth)  # apply a log transformation, adding 1 to avoid log(0)
    return eth

def simulate_dapp_oracle(size):
    dates = generate_date_range(size)
    tasks = generate_stochastic_tasks(size)
    user_base = generate_stochastic_user_growth(size)
    ether_to_usd = fetch_ether_to_usd(size)
    eth = calculate_eth(tasks, user_base, ether_to_usd)
    return pd.DataFrame(
        {
            'Date': dates,
            'Tasks': tasks,
            'User Base': user_base,
            'Ether to USD': ether_to_usd,
            'ETH': eth,
        }
    )
