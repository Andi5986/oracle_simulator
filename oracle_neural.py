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

# Initialize weights for Oracle function
weights = [random.random() for _ in range(7)]
#weights = [0.1, 0.5, 0.2, 0.05, 0.05, 0.1, 0.1]
#weights = [1, 1, 1, 1, 1, 1, 1]

# Oracle function
def oracle(task_complexity, ether_price, active_users, solved_tasks, unsolved_tasks, user_kpis, service_level_agreements):
    return (
        weights[0] * task_complexity
        + weights[1] * (-1) * np.log(ether_price + 1)  # add 1 to avoid log(0)
        + weights[2] * (-1) * np.log(active_users + 1) # add 1 to avoid log(0)
        + weights[3] * (-1) * np.log(solved_tasks + 1) # add 1 to avoid log(0)
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

active_users_prev = 1000  # initial active users

oracle_outputs = []
variables = {'task_complexity': [], 'ether_price': [], 'active_users': [], 'solved_tasks': [], 'unsolved_tasks': [], 'user_kpis': [], 'service_level_agreements': []}
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

    oracle_outputs.append(oracle(task_complexity, ether_price, active_users, solved_tasks, unsolved_tasks, user_kpis, service_level_agreements))
    variables['task_complexity'].append(task_complexity)
    variables['ether_price'].append(ether_price)
    variables['active_users'].append(active_users)
    variables['solved_tasks'].append(solved_tasks)
    variables['unsolved_tasks'].append(unsolved_tasks)
    variables['user_kpis'].append(user_kpis)
    variables['service_level_agreements'].append(service_level_agreements)

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

# Display a scatter plot with the relation between Oracle output and Ether price
st.subheader('Predicted output vs Ether price')
plt.figure(figsize=(8,6))
plt.scatter(monthly_df['predicted_output'], monthly_df['price'])
z = np.polyfit(monthly_df['predicted_output'], monthly_df['price'], 3)
p = np.poly1d(z)
xs = np.linspace(monthly_df['predicted_output'].min(), monthly_df['predicted_output'].max(), 100)
plt.plot(xs, p(xs), color='red')
plt.xlabel('Predicted Output')
plt.ylabel('Ether Price')
st.pyplot(plt)

# Calculate the y values of the polynomial for the original x values
y_pred = p(monthly_df['predicted_output'])

# Calculate the R-squared value for the model
r2 = r2_score(monthly_df['price'], y_pred)

st.subheader('R-squared value')
st.write(r2)


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

for var, weight in zip(variables, weights):
    avg_value = sum(variables[var]) / len(variables[var])
    st.write(f"{var}: {avg_value} and Weight: {weight}")
    
st.subheader('Assumptions & some Explanations')

st.write("""================================================================
         
The logarithm function has a diminishing returns effect (due to its shape), 
it means that big changes in the input variables (like ether_price, active_users, and solved_tasks) 
will have less and less impact on the outcome as their values increase. 
This could make the function's output more stable and resistant to large swings as these variables change, 
assuming other factors are held constant.""")

st.write("""================================================================
         
This code will result in active_users growing with a random rate between -2% and 2% at each time step. 
This creates a random walk, which is a common model of stochastic processes in many fields. 
Adjust the values passed to random.uniform to control the variability of the growth rate. """)

st.write("""=================================================================
         
The R-squared value, also known as the coefficient of determination, is a statistical measure that reflects the 
goodness of fit of a statistical model.

Specifically, R-squared represents the proportion of the variance for a dependent variable that's explained by an 
independent variable or variables in a regression model. In other words, it quantifies the amount of variance in the dependent variable that can be attributed to the variation in independent variables.

R-squared values range from 0 to 1:

An R-squared of 100 percent indicates that all changes in the dependent variable are completely explained by changes 
in the independent variable(s).
An R-squared of 0 percent indicates that none of the changes in the dependent variable can be explained by changes in the
independent variable(s).
Therefore, a higher R-squared value suggests that the model explains more of the variability, while a lower R-squared value 
suggests the model explains less of the variability.
However, a high R-squared does not necessarily indicate that the model has a good fit. A high R-squared can be a result of 
overfitting the model to the data. This means that while the model fits the sample data very well, it might not generalize 
well to new data.

R-squared does not indicate whether a regression model is appropriate. You can have a low R-squared value for a good model, 
or a high R-squared value for a model that does not fit the data!

So while R-squared can be a useful tool for comparing models, it should not be the only one you use. Other statistical measures,
as well as visual inspection of the data and the residuals, can provide valuable information about the fit of your model.
         """)

st.sidebar.title("Explanation and Controls")

st.sidebar.markdown("""
This application simulates an Oracle Function. The Oracle Function calculates its output based on several variables:
- **Task Complexity**: A measure of how complex the task is, ranging from 1 to 10.
- **Ether Price**: The price of Ether. A higher price of Ether reduces the output of the Oracle Function.
- **Active Users**: The number of active users. More active users reduce the output of the Oracle Function.
- **Solved Tasks**: The number of tasks solved by users. More solved tasks reduce the output of the Oracle Function.
- **Unsolved Tasks**: The number of tasks not solved by users. More unsolved tasks increase the output of the Oracle Function.
- **User KPIs**: A measure of user performance, where higher is better.
- **Service Level Agreements (SLAs)**: A measure of the level of service, where higher is better.

The Oracle Function's output is then used to simulate the outcome over the last 3 years. The output is normalized and used to train a neural network model, which is used to predict future outcomes.
""")

# Control parameters
task_complexity = st.sidebar.slider('Task Complexity', min_value=1, max_value=10, value=5, step=1)
ether_price = st.sidebar.number_input('Ether Price', min_value=0.0, value=2000.0, step=100.0)
active_users = st.sidebar.number_input('Active Users', min_value=1000, value=10000, step=100)
