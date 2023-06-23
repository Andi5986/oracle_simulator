# dApp Oracle Simulation

This is a Streamlit web application that simulates a dApp (decentralized application) oracle. The oracle predicts the value of ETH (Ether) based on various factors.

$$
ETH = \log \left(1 + \left| \frac{1}{C} \cdot (1 + V) \cdot \log \left(1 + \frac{|T|}{|U|} \right) \right| \right)
$$

- `ETH` is the calculated result.
- `T` represents the `task`.
- `U` represents the `user_base`.
- `C` represents the `currency_relation`.
- `V` represents the `currency_volatility`.
- `log` is the natural logarithm function.
- `|x|` denotes the absolute value of `x`.

## Variables

- **ETH** (dependent variable): The value of ETH, which is the target variable to predict.
- **Task** (independent variable): Represents the number of tasks related to the dApp. It influences the value of ETH.
- **Users** (mediating variable): Users are considered as a mediating variable that explains the relationship between Task and ETH. It can modify the relationship between them.
- **Currency (Ether to USD)** (moderating variable): Represents the exchange rate between Ether and USD. It can interact with the relationship between Task and ETH, influencing the strength or direction of the relationship.

## Models

The following machine learning models are used for prediction:

- **Neural Network**: A multilayer perceptron (MLP) model that captures complex relationships between variables.
- **Support Vector Regressor (SVR)**: A machine learning algorithm designed to capture complex relationships between features and the target variable.
- **Gradient Boosting Regressor**: An ensemble method that builds a sequence of weak learners to make accurate predictions.
- **Ridge Regression**: A type of linear regression that includes a penalty on the size of coefficients to prevent overfitting.

## Usage

1. Select the size of the dataset (in days) using the sidebar slider.
2. The application will simulate the data and train the models on the dataset.
3. Time charts will be displayed, showing the evolution of the user base, tasks, Ether vs USD, and the predicted values of ETH from each model.

Enjoy exploring the dApp Oracle Simulation!

## Installation

1. Clone the repository:

```
git clone https://github.com/your-username/oracle-function-simulation.git
```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

possible issues with sklearn. See https://towardsdatascience.com/scikit-learn-vs-sklearn-6944b9dc1736#:~:text=scikit%2Dlearn%20and%20sklearn%20both,using%20the%20skikit%20%2Dlearn%20identifier

## Running the Application

3. Navigate to the project directory:

```
cd oracle-function-simulation
```

4. Launch the Streamlit application:

```
streamlit run app.py
```

5. Access the application in your web browser at [http://localhost:8501](http://localhost:8501).

## Data Sources

The historical price data for Ether is obtained from the Coingecko API ([Coingecko](https://coingecko.com/)).

## Contributing

Contributions are welcome! If you would like to contribute to this project, please follow these steps:

7. Fork the repository.

8. Create a new branch:

```
git checkout -b feature/new-feature
```

9. Make your changes and commit them:

```
git commit -m "Add new feature"
```

10. Push your changes to your forked repository:

```
git push origin feature/new-feature
```

11. Open a pull request on the original repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.