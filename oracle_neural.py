import streamlit as st
import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import seaborn as sns

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
ether_price_prev = df['price'].iloc[0]  # initial Ether price
active_users_prev = 1000  # initial active users

oracle_outputs = []
variables = {'task_complexity': [], 'ether_price': [], 'active_users': [], 'solved_tasks': [], 'unsolved_tasks': [], 'user_kpis': [], 'service_level_agreements': []}
for i in range(len(df)):
    task_complexity = random.randint(1, 10)
    ether_price = df['price'].iloc[i]  # Ether price based on historical data
    active_users = active_users_prev * random.uniform(0.9, 1.1)  
    solved_tasks = int(active_users * (1 - np.exp(-i/365)) * 0.8)  # 80% of active users solve tasks
    unsolved_tasks = active_users - solved_tasks  # Remaining users did not solve tasks
    user_kpis = random.uniform(0.75, 1)
    service_level_agreements = random.uniform(0.95, 0.99)

    oracle_outputs.append(oracle(task_complexity, ether_price, active_users, solved_tasks, unsolved_tasks, user_kpis, service_level_agreements))
    variables['task_complexity'].append(task_complexity)
    variables['ether_price'].append(ether_price)
    variables['active_users'].append(active_users)
    variables['solved_tasks'].append(solved_tasks)
    variables['unsolved_tasks'].append(unsolved_tasks)
    variables['user_kpis'].append(user_kpis)
    variables['service_level_agreements'].append(service_level_agreements)

    active_users_prev = active_users

# Convert variables and oracle outputs into a DataFrame
df_variables = pd.DataFrame(variables)
oracle_outputs = pd.DataFrame(oracle_outputs, columns=['oracle_output'])

# Normalize variables and oracle outputs
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
df_variables_scaled = pd.DataFrame(scaler_x.fit_transform(df_variables), columns=df_variables.columns)
oracle_outputs_scaled = pd.DataFrame(scaler_y.fit_transform(oracle_outputs), columns=oracle_outputs.columns)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df_variables_scaled, oracle_outputs_scaled, test_size=0.2, random_state=42)

# Build a neural network model
model = keras.models.Sequential()
model.add(keras.layers.Dense(128, activation='relu', input_dim=7))  # 7 input features
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(1))  # output layer: single continuous output

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32)

# Predicting with the trained model
predictions_scaled = model.predict(df_variables_scaled)
predictions = scaler_y.inverse_transform(predictions_scaled)  # inverse scaling to get real-world predictions

# Add the predicted outputs into the df DataFrame
df['predicted_output'] = predictions

# Set 'time' as the index
df.set_index('time', inplace=True)

# Resample the price and predicted output data to monthly data and calculate average price and predicted output for each month
monthly_df = df.resample('M').mean()

# Display a line chart of the Oracle output and Ether price over time
st.subheader('Oracle Output and Ether Price Over Time')
st.line_chart(monthly_df[['predicted_output', 'price']])

# Display a scatter plot with linear relation between Oracle output and Ether price
st.subheader('Predicted output vs Ether price')
plt.figure(figsize=(8,6))
plt.scatter(monthly_df['predicted_output'], monthly_df['price'])
m, b = np.polyfit(monthly_df['predicted_output'], monthly_df['price'], 1)
plt.plot(monthly_df['predicted_output'], m*monthly_df['predicted_output'] + b, color='red')
plt.xlabel('Predicted Output')
plt.ylabel('Ether Price')
st.pyplot(plt)

df['task_complexity'] = df_variables['task_complexity'].values

# Display a pairplot of the variables
st.subheader('Pairplot of the Variables')
df = pd.DataFrame(variables)  # create a DataFrame from the variables dictionary
df['predicted_output'] = predictions 
fig = sns.pairplot(df, hue='task_complexity')
st.pyplot(fig)

# Display a correlation matrix of the variables
st.subheader('Correlation Matrix of the Variables')
corr = df.corr()
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Display tables showing average values of the variables over time
st.subheader('Average Values of the Variables Over Time')
for var in variables:
    st.write(f"{var}: {sum(variables[var])/len(variables[var])}")
