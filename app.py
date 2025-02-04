import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    if not stock_data.empty:
        stock_data = stock_data['Adj Close']
        return stock_data
    return pd.Series()  # Return an empty Series if data is not found

def evaluate_simulation(simulated_prices, actual_prices):
    """
    Evaluates the Monte Carlo simulation performance by computing Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
    """
    mae = np.mean(np.abs(simulated_prices - actual_prices))
    rmse = np.sqrt(np.mean((simulated_prices - actual_prices) ** 2))
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

def monte_carlo(prices, num_simulations, num_days):
    if len(prices) < 2:
        return np.zeros((num_days, num_simulations)) 
    
    returns = prices.pct_change().dropna()
    mean_return = returns.mean()
    std_dev = returns.std()
    
    simulations = np.zeros((num_days, num_simulations))
    
    for i in range(num_simulations):
        simulated_prices = [prices.iloc[-1]]
        for _ in range(num_days):
            simulated_price = simulated_prices[-1] * np.exp(
                (mean_return - 0.5 * std_dev**2) + std_dev * np.random.normal()
            )
            simulated_prices.append(simulated_price)
        
        simulation_data = np.array(simulated_prices[1:num_days+1]).flatten()
        simulations[:, i] = simulation_data
    
    return simulations

def plot_simulation(simulations, training_prices, validation_prices, ticker):
    plt.figure(figsize=(12, 7))
    
    # Plotting training data
    plt.plot(training_prices.index, training_prices.values, color='blue', label='Training Prices')
    
    # Plotting validation data
    plt.plot(validation_prices.index, validation_prices.values, color='green', label='Validation Prices')
    
    # Plotting Monte Carlo simulations
    plt.plot(simulations, alpha=0.1, color='red')
    
    plt.title(f"Monte Carlo Simulation for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

def main():
    ticker = input("Enter Stock Ticker: ").strip().upper()
    start_date = input("Enter the start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter the end date (YYYY-MM-DD): ").strip()
    num_simulations = int(input("Enter the number of simulations: "))
    num_days_to_simulate = int(input("Enter the number of days to simulate: "))
    
    # Fetch historical data
    prices = fetch_stock_data(ticker, start_date, end_date)
    
    if prices.empty:
        print("No data found. Please check the ticker and dates.")
        return
    
    if len(prices) < 2:
        print("Insufficient data for Monte Carlo simulation. Need at least two price points.")
        return
    
    # Split data into training and validation sets
    split_ratio = 0.8  # 80% for training, 20% for validation
    split_index = int(len(prices) * split_ratio)
    training_prices = prices.iloc[:split_index]
    validation_prices = prices.iloc[split_index:]
    
    # Calculate the number of days to simulate (based on validation data length)
    num_days = len(validation_prices)
    
    print(f"Number of total price data points: {len(prices)}")
    print(f"Number of training data points: {len(training_prices)}")
    print(f"Number of validation data points: {len(validation_prices)}")
    
    # Run Monte Carlo simulation
    simulations = monte_carlo(training_prices, num_simulations, num_days)
    
    # Extract the path for evaluation (e.g., the mean simulation path)
    mean_simulation = np.mean(simulations, axis=1)
    
    # Evaluate simulation performance
    evaluate_simulation(mean_simulation, validation_prices.values)
    
    # Plot the results
    plot_simulation(simulations, training_prices, validation_prices, ticker)

if __name__ == "__main__":
    main()