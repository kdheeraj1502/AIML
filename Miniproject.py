import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
dataset = pd.read_csv('/Users/dheerajkumar/Documents/AIML-Classes/assignments/assignment2/Dataset - Mini Project.csv')

# Step 1: Exploratory Data Analysis (EDA)
print(dataset.describe())
print(dataset.info())

# Step 2: Plotting graphs & correlations
plt.figure(figsize=(10, 6))
sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Pairplot
sns.pairplot(dataset)
plt.show()

# Step 3: Model Building using Multiple Linear Regression

# Prepare the data
X = dataset.drop('Cost', axis=1)
y = dataset['Cost']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3.1 Using Stochastic Gradient Descent (SGD)
sgd_reg = SGDRegressor(max_iter=5000, tol=1e-3, random_state=42)
sgd_reg.fit(X_train_scaled, y_train)
y_pred_sgd = sgd_reg.predict(X_test_scaled)

# 3.2 Using Mini Batch Gradient Descent
mini_batch_reg = SGDRegressor(max_iter=5000, tol=1e-3, random_state=42, learning_rate='adaptive', eta0=0.01)
mini_batch_reg.fit(X_train_scaled, y_train)
y_pred_mini_batch = mini_batch_reg.predict(X_test_scaled)

# 3.3 Using Gradient Descent (Manual Implementation)
X_train_bias = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
X_test_bias = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]

learning_rate = 0.01
n_iterations = 1000
m = X_train_bias.shape[0]

theta = np.random.randn(X_train_bias.shape[1])

for iteration in range(n_iterations):
    gradients = 2/m * X_train_bias.T.dot(X_train_bias.dot(theta) - y_train)
    theta = theta - learning_rate * gradients

y_pred_gd = X_test_bias.dot(theta)

# 3.4 Using Normal Equation (sklearn library)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin_reg = lin_reg.predict(X_test)

# Step 4: Evaluation Metrics
def print_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f'{model_name} - MSE: {mse}, RMSE: {rmse}, R-squared: {r2}')

print_metrics(y_test, y_pred_sgd, 'SGD Regressor')
print_metrics(y_test, y_pred_mini_batch, 'Mini Batch Gradient Descent Regressor')
print_metrics(y_test, y_pred_gd, 'Gradient Descent (Manual Implementation)')
print_metrics(y_test, y_pred_lin_reg, 'Linear Regression (Normal Equation)')
