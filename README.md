# Linear_Regression

📊 Simple Linear Regression
Simple Linear Regression is a statistical method used to model the relationship between two variables:

One independent variable (X) — the predictor

One dependent variable (Y) — the response or outcome

It assumes a linear relationship between the variables, represented by the equation:

y=mx+c
Where:
- y is the predicted output
- x is the input feature
- m is the slope of the line
- c is the y-intercept

✅ Assumptions:
Linearity – the relationship between X and Y is linear

Independence – the residuals are independent

Homoscedasticity – constant variance of residuals

Normality – residuals should be normally distributed

🔍 Example Use Cases:
Predicting house prices based on area

Estimating sales based on advertising spend

Forecasting temperature based on time

###  simple Python example using scikit-learn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Feature (input)
y = np.array([2, 4, 5, 4, 5])                # Target (output)

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Get the slope (m) and intercept (c)
m = model.coef_[0]
c = model.intercept_

print(f"Line equation: y = {m:.2f}x + {c:.2f}")

# Predicting values
y_pred = model.predict(X)

# Plotting
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()
