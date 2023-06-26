# dApp Oracle Simulation

This is a web application built with Streamlit that emulates a decentralized application (dApp) oracle. A dApp is a decentralized application that runs on a blockchain network in a public, open-source, peer-to-peer environment. An oracle, in the context of blockchains and smart contracts, is an agent that finds and verifies real-world occurrences and submits this information to a blockchain to be used by smart contracts.

The oracle in this application forecasts the value of ETH (the unit of payment in Ether) by considering network effects from crowdsourcing agents who are involved in completing network tasks. 

## Underlying Calculation

The core of this application is a function that calculates the amount of Ethereum (ETH) needed based on several parameters: `task`, `user_base`, `currency_relation`, and `currency_volatility`. The mathematical relationship between these variables is encapsulated in the following formula:

$$
ETH = \log \left(1 + \left| \frac{1}{C} \cdot (1 + V^2) \cdot \log \left(1 + \frac{|T \cdot f(ETH)|^2}{|U \cdot f(ETH)|^2} \right) \right| \right)
$$

Where:

- `ETH` is the amount of Ethereum calculated.
- `T` represents the `task`, indicating the workload of tasks.
- `U` represents the `user_base`, a base value associated with the users in the network.
- `C` represents the `currency_relation`, the exchange rate between Ethereum and USD.
- `V` represents the `currency_volatility`, accounting for the volatility of the currency.
- `log` is the natural logarithm function.
- `|x|` denotes the absolute value of `x`.
- `f(ETH)` represents the scaling factor, calculated as:

$$
f(ETH) = 1 - \frac{ETH}{ETH_{limit}}
$$

Where:

- `ETH` is the current net amount of Ether in the system.
- `ETH_{limit}` is the limit of 15 million Ether.

This scaling factor reduces from 1 to 0 as the net amount of Ether approaches the limit. It is applied to the `task` and `user_base` variables in the formula for `ETH`, reducing their impact on the calculated Ether as the limit is approached.

This calculation forms the basis for the predictions made by the machine learning models in the application, which include a Neural Network, Support Vector Regressor (SVR), Gradient Boosting Regressor, and Ridge Regression.

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
git clone https://github.com/Andi5986/oracle_simulator.git
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