import streamlit as st
import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

st.title('Oracle Function Simulation')

# Initialize weights for Oracle function
weights = [random.random() for _ in range(7)]

# Oracle function
def oracle(task_complexity, ether_price, active_users, solved_tasks, unsolved_tasks, user_kpis, service_level_agreements):
    return (
        weights[0] * task_complexity
        + weights[1] * (-1) * np.log(ether_price + 1)  # add 1 to avoid log(0)
        + weights[2] * active_users
        + weights[3] * solved_tasks
        + weights[4] * unsolved_tasks
        + weights[5] * user_kpis
        + weights[6] * service_level_agreements
    )

# Get historical data for Ether
url = "https://api.coingecko.com/api/v3/coins/ethereum/market_chart"
params = {"vs_currency": "usd", "days": "1095"}  # 1095 days is approximately 3 years
response = requests.get(url, params=params)
data = response.json()

# Convert the price data to a Pandas DataFrame
df = pd.DataFrame(data['prices'], columns=['time', 'price'])
df['time'] = pd.to_datetime(df['time'], unit='ms')

# Generate mock data for the oracle function and simulate the last 3 years
oracle_outputs = []
variables = {'task_complexity': [], 'ether_price': [], 'active_users': [], 'solved_tasks': [], 'unsolved_tasks': [], 'user_kpis': [], 'service_level_agreements': []}
for _ in range(len(df)):
    task_complexity = random.randint(1, 10)
    active_users = random.randint(1, 10000)
    solved_tasks = random.randint(1, 1000)
    unsolved_tasks = random.randint(1, 1000)
    user_kpis = random.uniform(0.1, 1)
    service_level_agreements = random.uniform(0.1, 1)
    ether_price = df.iloc[_]['price']
    oracle_outputs.append(oracle(task_complexity, ether_price, active_users, solved_tasks, unsolved_tasks, user_kpis, service_level_agreements))
    variables['task_complexity'].append(task_complexity)
    variables['ether_price'].append(ether_price)
    variables['active_users'].append(active_users)
    variables['solved_tasks'].append(solved_tasks)
    variables['unsolved_tasks'].append(unsolved_tasks)
    variables['user_kpis'].append(user_kpis)
    variables['service_level_agreements'].append(service_level_agreements)

# Convert oracle_outputs into DataFrame and add to the df DataFrame
df['oracle_output'] = oracle_outputs

# Set 'time' as the index
df.set_index('time', inplace=True)

# Resample the price and oracle output data to monthly data and calculate average price and oracle output for each month
monthly_df = df.resample('M').mean()

# Display a line chart of the Oracle output and Ether price over time
st.subheader('Oracle Output and Ether Price Over Time')
st.line_chart(monthly_df[['oracle_output', 'price']])

# Display a scatter plot with linear relation between Oracle output and Ether price
st.subheader('Oracle output vs Ether price')
plt.figure(figsize=(8,6))
plt.scatter(monthly_df['oracle_output'], monthly_df['price'])
m, b = np.polyfit(monthly_df['oracle_output'], monthly_df['price'], 1)
plt.plot(monthly_df['oracle_output'], m*monthly_df['oracle_output'] + b, color='red')
plt.xlabel('Oracle Output')
plt.ylabel('Ether Price')
st.pyplot(plt)

# Display tables showing average values of the variables over time
st.subheader('Average Values of the Variables Over Time')
for var in variables:
    st.write(f"{var}: {sum(variables[var])/len(variables[var])}")
