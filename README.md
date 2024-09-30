
# Stock Price Prediction using Linear Regression

This project implements a stock price prediction model using **Linear Regression** on historical data of **Bank Nifty** stock prices. The model is built using **Python** and libraries such as **Pandas**, **NumPy**, **Scikit-learn**, and **Matplotlib**. It predicts future stock prices based on past trends.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)

## Project Overview

This project uses historical data from Bank Nifty to predict future stock prices using a simple Linear Regression model. The data is split into training and testing datasets, the model is trained, and predictions are evaluated using Mean Squared Error (MSE). The results are visualized with graphs for a better understanding of the performance.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/stock-price-prediction.git
   cd stock-price-prediction
   ```

2. Install the required Python libraries using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

   Required libraries:
   - Pandas
   - NumPy
   - Scikit-learn
   - Matplotlib

3. Download the historical stock data and save it in the `data` folder. The file should be named `nsebank.csv`.

## Usage

1. Make sure you have the dataset (`nsebank.csv`) in the correct folder.
2. Run the Jupyter notebook or Python script to train the model and generate predictions.
   - For Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - For Python script:
     ```bash
     python stock_prediction.py
     ```

3. The model will generate predictions for future stock prices, and the results will be visualized using Matplotlib.

## Technologies Used

- **Python**: Primary programming language.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical operations and array handling.
- **Scikit-learn**: Machine learning algorithms and model evaluation.
- **Matplotlib**: Visualization of stock prices and predictions.


## Contributing

If you want to contribute to this project:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

