## Monte Carlo Simulation - JPMorgan Chase & Co. Stock Price Prediction

I used Monte Carlo simulations to predict the future performance of the JPMorgan Chase & Co stock price, applying Monte Carlo simulations to JPMorgan Chase's stock price and evaluating the model's performance using statistical error metrics (MSE and MAE).

By generating multiple random samples based on historical data, I modeled potential future price paths for JPMorgan Chase & Co's stock.

---

### **Core Math - Monte Carlo Simulation**

1. **Log-Normal Assumption**: Stock prices follow a geometric Brownian motion (discretized version):

$$
S_{t+1} = S_t \cdot e^{(\mu - 0.5 \sigma^2) + \sigma Z}
$$

- **S**: Stock price at time **t**
- **μ**: Mean return of stock prices
- **σ**: Standard deviation of returns
- **Z**: Random normal variable ($N(0, 1)$)
- **t**: Time step (e.g., day, month)

2. **Simulation Process**:
   - Compute daily returns: 

$$
R_t = \frac{P_{t+1} - P_t}{P_t}
$$

- **R**: Daily return at time **t**
- **P**: Stock price at time **t**
- **t**: Time step (e.g., day, month)
- Calculate **μ** (mean) and **σ** (standard deviation) from historical returns.
- For each day in the future, simulate a price based on the log-normal equation.
- Repeat the process for multiple simulations to create a distribution of potential price paths.

---

### **Code**

1. **Fetching Data**:
   - Historical stock price data is retrieved using the Yahoo Finance API.
   - The adjusted closing price is used for accurate simulations.

2. **Splitting Data**:
   - The dataset is split into:
     - **Training Data**: Used to calculate historical mean and standard deviation of returns.
     - **Validation Data**: Used to evaluate simulation accuracy.

3. **Monte Carlo Simulation**:
   - Simulate $N$ price paths for $D$ days using the formula described above.
   - Store all simulated paths in a matrix.

4. **Evaluation Metrics**:
   - **Mean Absolute Error (MAE)**:

$$
MAE = \frac{1}{n} \sum |S_{simulated} - S_{actual}|
$$

   - **Root Mean Squared Error (RMSE)**:

$$
RMSE = \sqrt{\frac{1}{n} \sum (S_{simulated} - S_{actual})^2}
$$

5. **Visualization**:
   - Training and validation prices are plotted alongside simulated paths for visual comparison.

---

### **Results and Analysis**

#### **Simulation Accuracy**:
- **MAE (14.3945):** The model's predictions are, on average, off by 14.39. This is a reasonable error margin for stock prices, indicating that the Monte Carlo simulation captures the general price trend well, though there are still some deviations.

- **RMSE (15.1511):** The RMSE accounts for larger errors, and at 15.15, it suggests that while the model tracks the overall price direction, there's variability in larger price movements. This indicates room for improvement in reducing those larger errors.

- Overall, both metrics show the simulation provides a good approximation of JPMorgan's stock price trajectory, but there's always still potential to reduce errors.
