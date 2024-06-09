import numpy as np
import theta
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

housing = fetch_california_housing()
#print(len(housing))
x,y = housing.data, housing.target
scaler = StandardScaler()
x = scaler.fit_transform(x)
#print("Length and width of x before adding any column", x.shape)
x = np.c_[np.ones(x.shape[0]), x]

# Calculate the mean of each column
means = np.nanmean(x, axis=0)

# Replace NaN values with column means
x = np.where(np.isnan(x), means, x)

# Calculate the mean of finite values
finite_mean = np.nanmean(x[np.isfinite(x)])

# Replace infinite values with the mean of finite values
x = np.where(np.isfinite(x), x, finite_mean)

#x = np.c_[np.full((x.shape[0]), 5), x]
#print("Length and width of x after adding new column", x.shape)

#print("Length of y ", len(y))


# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

def compute_cost(x, y, theta):
    n = len(y)
    predictions = x.dot(theta)
    cost = (1/(2*n)) * np.sum(np.square(predictions - y))
    return cost

def gradient_descent(x, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        for j in range(m):
            prediction = np.dot(x[j], theta)
            error = prediction - y[j]
            gradient = x[j].reshape(1, -1).T.dot(error)
            theta -= learning_rate * gradient.reshape(-1)
        cost_history[i] = compute_cost(x, y, theta)
    return theta, cost_history


def batch_gradient_descent(x, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        predictions = x.dot(theta)
        errors = predictions - y
        gradient = (1/m) * x.T.dot(errors)
        theta -= learning_rate * gradient
        cost_history[i] = compute_cost(x, y, theta)
    return theta, cost_history


def mini_batch_gradient_descent(x, y, theta, learning_rate, iterations, batch_size):
    m = len(y)
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        # Randomly shuffle the data
        idx = np.random.permutation(m)
        x_shuffled, y_shuffled = x[idx], y[idx]
        for j in range(0, m, batch_size):
            x_batch, y_batch = x_shuffled[j:j+batch_size], y_shuffled[j:j+batch_size]
            predictions = x_batch.dot(theta)
            errors = predictions - y_batch
            gradient = (1/batch_size) * x_batch.T.dot(errors)
            theta -= learning_rate * gradient
        cost_history[i] = compute_cost(x, y, theta)
    return theta, cost_history


# Initialize theta with zeros
theta = np.zeros(x_train.shape[1])

# Set hyperparameters
learning_rate = 0.01
iterations = 1000
batch_size = 16

# Run batch gradient descent
theta_batch, cost_history_batch = batch_gradient_descent(x_train, y_train, theta, learning_rate, iterations)
theta_gradient, cost_history_gradient = gradient_descent(x_train, y_train, theta.copy(), learning_rate, iterations)

# Run mini-batch gradient descent
theta_mini_batch, cost_history_mini_batch = mini_batch_gradient_descent(x_train, y_train, theta, learning_rate, iterations, batch_size)



plt.plot(range(1, iterations + 1), cost_history_gradient, label='Gradient Descent', color='green')
plt.plot(range(1, iterations + 1), cost_history_batch, label='Batch GD', color='blue')
plt.plot(range(1, iterations + 1), cost_history_mini_batch, label='Mini-batch GD', color='red')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Batch vs. Mini-batch Gradient Descent: Cost vs. Iterations')
plt.legend()
plt.show()

# Predictions on test set for gradient descent
y_pred_gradient = x_test.dot(theta_gradient)

# Calculate mean squared error for gradient descent
mse_gradient = np.mean((y_pred_gradient - y_test)**2)
print("Mean Squared Error (Gradient Descent):", mse_gradient)



# Predictions on test set for batch gradient descent
y_pred_batch = x_test.dot(theta_batch)

# Calculate mean squared error for batch gradient descent
mse_batch = np.mean((y_pred_batch - y_test)**2)
print("Mean Squared Error (Batch GD):", mse_batch)

# Predictions on test set for mini-batch gradient descent
y_pred_mini_batch = x_test.dot(theta_mini_batch)

# Calculate mean squared error for mini-batch gradient descent
mse_mini_batch = np.mean((y_pred_mini_batch - y_test)**2)
print("Mean Squared Error (Mini-batch GD):", mse_mini_batch)


































#for i, column in enumerate(x.T):
#   column_name = f"Column {i + 1}"
#  print("Column name is ::", column_name, " Column type is ::", type(column), " and value is :: ", column, " size is :: ", len(column))
#print("Data key are :: ", housing.keys())
#print(housing.get("data"))
#for item in housing.get("data"):


#for key, value in housing.items():
#   size = "size null" if value is None else len(value)
#  print("The Key is : [{", key, "}] and its Type is : [{", type(key), "}] and the Value type is :", type(value),
#       " Size of value is :", size)
#print("------------------------------------------------")
#if isinstance(value, list):
#    print("The Key is :", key)
#   for val in value:
#      print("Its values are :: ", val)
#if isinstance(value, str):
# print("Its values are :: ", value)
#  count = 0
#   print("The Key is:", key)
# for word in value.split():
#    if count >= 5:
#       break
#  print("Its values are:", word)
# count += 1

