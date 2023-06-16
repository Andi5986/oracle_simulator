import streamlit as st
import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import random
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import seaborn as sns

st.title('Oracle Function Simulation')

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

active_users_prev = 1000  # initial active users

dai_price = 1

# Initialize weights for Oracle function
weights_oracle = [random.random() for _ in range(4)]
weights_reward = [random.random() for _ in range(3)]

# Oracle function
def oracle(ether_price, active_users, solved_tasks, unsolved_tasks):
    return (
        weights_oracle[0] * (-1) * np.log(ether_price + 1)  # add 1 to avoid log(0)
        + weights_oracle[1] * (-1) * np.log(active_users + 1)  # add 1 to avoid log(0)
        + weights_oracle[2] * (-1) * np.log(solved_tasks + 1)  # add 1 to avoid log(0)
        + weights_oracle[3] * unsolved_tasks
    )

def reward(task_complexity, user_kpis, service_level_agreements, active_users, ether_price, dai_price):
    return (
        weights_reward[0] * task_complexity
        + weights_reward[1] * user_kpis
        + weights_reward[2] * service_level_agreements
    ) / active_users * (dai_price / ether_price)

oracle_outputs = []
reward_outputs = []
variables = {'ether_price': [], 'active_users': [], 'solved_tasks': [], 'unsolved_tasks': [], 'task_complexity': [], 'user_kpis': [], 'service_level_agreements': []}
for i in range(len(df)):
    task_complexity = random.randint(1, 10)
    ether_price = df['price'].iloc[i]  # Ether price based on historical data

    growth_rate = random.uniform(0.98, 1.02)  # random growth rate between 98% and 102% - stochastic
    active_users = active_users_prev * growth_rate
    active_users_prev = active_users  # update active_users_prev for next iteration

    solved_tasks = int(active_users * (1 - np.exp(-i/365)) * 0.8)  # 80% of active users solve tasks
    unsolved_tasks = active_users - solved_tasks  # Remaining users did not solve tasks
    user_kpis = random.uniform(0.75, 1)
    service_level_agreements = random.uniform(0.95, 0.99)

    oracle_outputs.append(oracle(ether_price, active_users, solved_tasks, unsolved_tasks))
    reward_outputs.append(reward(task_complexity, user_kpis, service_level_agreements, active_users, ether_price, dai_price))
    variables['ether_price'].append(ether_price)
    variables['active_users'].append(active_users)
    variables['solved_tasks'].append(solved_tasks)
    variables['unsolved_tasks'].append(unsolved_tasks)
    variables['task_complexity'].append(task_complexity)
    variables['user_kpis'].append(user_kpis)
    variables['service_level_agreements'].append(service_level_agreements)

# Convert variables, oracle and reward outputs into DataFrames
df_variables = pd.DataFrame(variables)
df_oracle_outputs = pd.DataFrame(oracle_outputs, columns=['oracle_output'])
df_reward_outputs = pd.DataFrame(reward_outputs, columns=['reward_output'])

# Normalize variables, oracle and reward outputs
scaler_x = MinMaxScaler()
scaler_y_oracle = MinMaxScaler()
scaler_y_reward = MinMaxScaler()
df_variables_scaled = pd.DataFrame(scaler_x.fit_transform(df_variables), columns=df_variables.columns)
df_oracle_outputs_scaled = pd.DataFrame(scaler_y_oracle.fit_transform(df_oracle_outputs), columns=df_oracle_outputs.columns)
df_reward_outputs_scaled = pd.DataFrame(scaler_y_reward.fit_transform(df_reward_outputs), columns=df_reward_outputs.columns)

# Split the data into train and test sets
X_train_oracle, X_test_oracle, y_train_oracle, y_test_oracle = train_test_split(df_variables_scaled, df_oracle_outputs_scaled, test_size=0.2, random_state=42)
X_train_reward, X_test_reward, y_train_reward, y_test_reward = train_test_split(df_variables_scaled, df_reward_outputs_scaled, test_size=0.2, random_state=42)

# Build a neural network model
model_oracle = keras.models.Sequential()
model_oracle.add(keras.layers.Dense(128, activation='relu', input_dim=7))  # 7 input features
model_oracle.add(keras.layers.Dense(64, activation='relu'))
model_oracle.add(keras.layers.Dense(1))  # output layer: single continuous output
model_oracle.compile(optimizer='adam', loss='mean_squared_error')

model_reward = keras.models.Sequential()
model_reward.add(keras.layers.Dense(128, activation='relu', input_dim=7))  # 7 input features
model_reward.add(keras.layers.Dense(64, activation='relu'))
model_reward.add(keras.layers.Dense(1))  # output layer: single continuous output
model_reward.compile(optimizer='adam', loss='mean_squared_error')

history_oracle = model_oracle.fit(X_train_oracle, y_train_oracle, validation_data=(X_test_oracle, y_test_oracle), epochs=100, batch_size=32)
history_reward = model_reward.fit(X_train_reward, y_train_reward, validation_data=(X_test_reward, y_test_reward), epochs=100, batch_size=32)

# Predicting with the trained model
predictions_scaled_oracle = model_oracle.predict(df_variables_scaled)
predictions_oracle = scaler_y_oracle.inverse_transform(predictions_scaled_oracle)  # inverse scaling to get real-world predictions

predictions_scaled_reward = model_reward.predict(df_variables_scaled)
predictions_reward = scaler_y_reward.inverse_transform(predictions_scaled_reward)  # inverse scaling to get real-world predictions

# Add the predicted outputs into the df DataFrame
df['predicted_output_oracle'] = predictions_oracle
df['predicted_output_reward'] = predictions_reward
df['active_users'] = df_variables['active_users'].values
df['solved_tasks'] = df_variables['solved_tasks'].values
df['unsolved_tasks'] = df_variables['unsolved_tasks'].values

# Set 'time' as the index
df.set_index('time', inplace=True)

# Resample the price and predicted output data to monthly data and calculate average price and predicted output for each month
monthly_df = df.resample('M').mean()

# Display line charts of the Oracle and Reward output over time
st.subheader('Oracle Output, Reward Output and Ether Price Over Time')
st.line_chart(monthly_df[['predicted_output_oracle', 'predicted_output_reward', 'price']])

# Display line chart for active users, solved and unsolved tasks over the period
st.subheader('Active Users, Solved and Unsolved Tasks Over Time')
st.line_chart(monthly_df[['active_users', 'solved_tasks', 'unsolved_tasks']])

# Display a line chart of the Reward Output over time
st.subheader('Oracle Output Over Time')
st.line_chart(monthly_df['predicted_output_oracle'])

# Display a line chart of the Reward Output over time
st.subheader('Reward Output Over Time')
st.line_chart(monthly_df['predicted_output_reward'])

# Calculate the reward in USD
monthly_df['reward_output_usd'] = monthly_df['predicted_output_reward'] * monthly_df['price']

# Display a line chart of the Reward Output in USD over time
st.subheader('Reward Output in USD Over Time')
st.line_chart(monthly_df['reward_output_usd'])

# Calculate the R-squared value for the Oracle model
y_pred_oracle = np.poly1d(np.polyfit(monthly_df['predicted_output_oracle'], monthly_df['price'], 3))(monthly_df['predicted_output_oracle'])
r2_oracle = r2_score(monthly_df['price'], y_pred_oracle)

# Calculate the R-squared value for the Reward model
y_pred_reward = np.poly1d(np.polyfit(monthly_df['predicted_output_reward'], monthly_df['price'], 3))(monthly_df['predicted_output_reward'])
r2_reward = r2_score(monthly_df['price'], y_pred_reward)

st.subheader('R-squared values')
st.write(f"For Oracle model: {r2_oracle}")
st.write(f"For Reward model: {r2_reward}")

# Display a pairplot of the variables
st.subheader('Pairplot of the Variables')
df = pd.DataFrame(variables)  # create a DataFrame from the variables dictionary
df['predicted_output_oracle'] = predictions_oracle
df['predicted_output_reward'] = predictions_reward
fig = sns.pairplot(df)
st.pyplot(fig)

# Display a correlation matrix of the variables
st.subheader('Correlation Matrix of the Variables')
corr = df.corr()
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Display tables showing average values of the variables over time
st.subheader('Average Values of the Variables Over Time')
for var, weight in zip(variables, weights_oracle):
    avg_value = np.mean(variables[var])
    st.write(f"{var}: {avg_value} and Weight: {weight}")
    
st.subheader('Assumptions & Explanations')

st.write("""
================================================================
         
The logarithmic function exhibits a diminishing returns effect due to its curve shape. 
This means that significant variations in the input variables (such as ether_price, active_users, and solved_tasks) 
will impact the outcome less as their values increase. Consequently, this stabilizes the function's output, making it more 
resistant to large swings as these variables change, provided other factors remain constant.""")

st.write("""
================================================================
         
In this code, active_users grow at a random rate between -2% and 2% at each time step, forming a random walk. 
This is a common model for stochastic processes across numerous fields. The variability of the growth rate can be adjusted 
by altering the values passed to the random.uniform function.""")

st.write("""
=================================================================
         
R-squared, or the coefficient of determination, is a statistical measure reflecting the appropriateness of fit for a 
statistical model.

R-squared represents the proportion of the variance for a dependent variable explained by independent variables in a 
regression model. In simpler terms, it quantifies the variance in the dependent variable attributable to the variation in 
independent variables.

R-squared values range from 0 to 1:

- An R-squared of 100 percent signifies that changes in the dependent variable are entirely explained by changes in the 
independent variable(s).
- Conversely, an R-squared of 0 percent indicates that the changes in the dependent variable cannot be accounted for by 
changes in the independent variable(s).

A higher R-squared value implies that the model accounts for more variability, while a lower R-squared value indicates 
the model explains less of the variability. However, a high R-squared does not necessarily mean that the model is a good fit. 
It could indicate overfitting, meaning that while the model fits the sample data well, it may not generalize well to new data.

R-squared does not validate whether a regression model is suitable. It's possible to have a low R-squared value for a 
good model, or a high R-squared value for a model that doesn't fit the data!

R-squared is a useful tool for comparing models, but it shouldn't be the only tool you use. Other statistical measures 
and visual inspections of the data and residuals can provide essential information about the model's fit.
         """)

st.sidebar.title("Explanation and Controls")

st.sidebar.markdown("""
This application simulates an Oracle Function. The Oracle Function calculates its output based on several variables:
- **Task Complexity**: A measure of task difficulty, ranging from 1 to 10.
- **Ether Price**: The price of Ether. A higher Ether price reduces the Oracle Function output.
- **Active Users**: The number of active users. A higher number of active users reduces the Oracle Function output.
- **Solved Tasks**: The number of tasks users solved. A higher number of solved tasks reduces the Oracle Function output.
- **Unsolved Tasks**: The number of tasks users didn't solve. More unsolved tasks increase the Oracle Function output.
- **User KPIs**: A measure of user performance. A higher value is better.
- **Service Level Agreements (SLAs)**: A measure of the service level. A higher value is better.

The output of the Oracle Function is used to simulate outcomes over the past three years. The output is normalized and 
utilized to train a neural network model, which predicts future outcomes.
""")

# Control parameters
task_complexity = st.sidebar.slider('Task Complexity', min_value=1, max_value=10, value=5, step=1)
ether_price = st.sidebar.number_input('Ether Price', min_value=0.0, value=2000.0, step=100.0)
active_users = st.sidebar.number_input('Active Users', min_value=1000, value=10000, step=100)

