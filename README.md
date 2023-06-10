# Oracle Function Simulation

This project is a Python script that simulates an oracle function. The function includes several parameters: task complexity, Ether price, the number of active users, the number of solved tasks, the number of unsolved tasks, user KPIs, and service level agreements.

## Setup

1. Clone the repository

git clone https://github.com/Andi5986/oracle_simulator.git

2. Navigate to the project directory

cd oracle_simulator

3. Create a virtual environment (optional, but recommended)

python3 -m venv env

4. Activate the virtual environment
    - On Windows:
    ```
    .\env\Scripts\activate
    ```
    - On Unix or MacOS:
    ```
    source env/bin/activate
    ```
5. Install the dependencies

pip install -r requirements.txt

## Usage

To run the script, run through steamlit:

streamlit run oracle.py


The script fetches the historical price data for Ether and uses this data to generate and plot the output of the oracle function over time.

## Built With

* [Streamlit](https://streamlit.io/) - The web framework used for visualizations
* [Pandas](https://pandas.pydata.org/) - Data manipulation library
* [Requests](https://requests.readthedocs.io/) - HTTP library for Python, used to fetch Ether price data
* [Matplotlib](https://matplotlib.org/) - Visualization library
* [Scikit-learn](https://scikit-learn.org/) - Machine Learning library, used for linear regression

## Author

* **Andi** - [Andi5986](https://github.com/Andi5986)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


