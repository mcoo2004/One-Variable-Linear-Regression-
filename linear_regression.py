import numpy as np
import matplotlib.pyplot as plt

# Used for plotting graphs and visualizing data


# Function to load data from a text file (10 points)
def load_data(file_path):
  data = np.loadtxt(file_path, delimiter=',')
  X = data[:, 0]
  y = data[:, 1]
  return X, y


# Function to plot the data points (10 points)
def plot_data(X, y):
  plt.scatter(X, y, c='red', marker='x', label='Profit vs. Population')
  plt.xlabel('Population in 10,000s')
  plt.ylabel('Profit in $10,000s')
  plt.title('Data Plot')
  plt.legend()
  plt.show()


#compute the cost of using theta as the parameter for linear regression
def compute_cost(X, y, theta):
  m = len(y)
  predictions = X.dot(theta)
  return (1 / (2 * m)) * np.sum(np.square(predictions - y))


# Function to perform gradient descent to learn theta (40 points)
def gradient_descent(X, y, theta, alpha, num_iterations):
  m = len(y)
  J_history = []
  for i in range(num_iterations):
    predictions = X.dot(theta)
    errors = np.dot(X.transpose(), (predictions - y))
    theta -= (alpha / m) * errors
    J_history.append(compute_cost(X, y, theta))
  return theta, J_history


def main():
  X, y = load_data('data.txt')
  m = len(y)
  X = np.stack([np.ones(m), X], axis=1)
  y = y[:, np.newaxis]
  theta = np.zeros((2, 1))
  alpha = 0.01
  iterations = 1500

  # Run Gradient Descent
  theta, J_history = gradient_descent(X, y, theta, alpha, iterations)

  # Plotting the regression line (10 points)
  plt.scatter(X[:, 1], y, c='red', marker='x', label='Training data')
  plt.plot(X[:, 1], np.dot(X, theta), label='Linear regression')
  plt.legend()
  plt.xlabel('Population in 10,000s')
  plt.ylabel('Profit in $10,000s')
  plt.title('Linear Regression Fit')
  plt.show()

  # Predict values for population sizes of 35,000 and 70,000 (10 points)
  predict1 = np.dot([1, 3.5], theta)
  predict2 = np.dot([1, 7], theta)
  print(f"Predicted profit for a population of 35,000: {predict1*10000:.2f}")
  print(f"Predicted profit for a population of 70,000: {predict2*10000:.2f}")


# copilot's code. skips importing txt file otherwise? not sure why. won't display the graph without it.
if __name__ == "__main__":
  main()
