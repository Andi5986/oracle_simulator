import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_and_train_model(df):
    # Define the model
    model = Sequential([
        Dense(32, activation='relu', input_shape=(3,)),  # Input layer
        Dense(32, activation='relu'),  # Hidden layer
        Dense(1, activation='relu')  # Output layer with ReLU activation
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Preprocess target variable (ETH) to avoid negative predictions
    target = df['ETH'].apply(lambda x: max(x, 0))  # Set negative values to 0

    # Fit the model
    model.fit(df[['Tasks', 'User Base', 'Ether to USD']], target, epochs=100)

    return model

