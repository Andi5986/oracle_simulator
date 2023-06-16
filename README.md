# Oracle Function Simulation

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

## Installation

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/your-username/oracle-function-simulation.git
   ```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

## Running the Application

3. Navigate to the project directory:

```
cd oracle-function-simulation
```

4. Launch the Streamlit application:

```
streamlit run main.py
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