import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For creating attractive and informative statistical graphics
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and testing sets
from sklearn.preprocessing import StandardScaler  # For standardizing the features
from sklearn.linear_model import SGDRegressor, LinearRegression  # For linear regression models
from sklearn.metrics import mean_squared_error, r2_score  # For evaluation metrics
import numpy as np  # For numerical operations

# 1. Analyze the dataset and do EDA
# Load the dataset
dataset = pd.read_csv('/Users/dheerajkumar/Documents/AIML-Classes/assignments/assignment2/Dataset - Mini Project.csv')

# Display basic statistics of the dataset
print(dataset.describe())

# Display information about the dataset
print(dataset.info())

# 2. Plotting of various graphs & correlations
# Create a heatmap to show correlations between features
plt.figure(figsize=(10, 6))
sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Create a pairplot to show relationships between features
sns.pairplot(dataset)
plt.show()

# 3. Model Building using Multiple Linear Regression
# Prepare the data for model training
X = dataset.drop('Cost', axis=1)  # Features (independent variables)
y = dataset['Cost']  # Target variable (dependent variable)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (mean=0, variance=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform the training set
X_test_scaled = scaler.transform(X_test)  # Transform the testing set based on training set parameters

# 3.1 Using Stochastic Gradient Descent (SGD)
# Initialize and train the SGD Regressor
sgd_reg = SGDRegressor(max_iter=5000, tol=1e-3, random_state=42)
sgd_reg.fit(X_train_scaled, y_train)

# Predict the target variable using the trained SGD model
y_pred_sgd = sgd_reg.predict(X_test_scaled)

# 3.2 Using Mini Batch Gradient Descent
# Initialize and train the Mini Batch Gradient Descent Regressor
mini_batch_reg = SGDRegressor(max_iter=5000, tol=1e-3, random_state=42, learning_rate='adaptive', eta0=0.01)
mini_batch_reg.fit(X_train_scaled, y_train)

# Predict the target variable using the trained Mini Batch model
y_pred_mini_batch = mini_batch_reg.predict(X_test_scaled)

# 3.3 Using Gradient Descent (Manual Implementation)
# Add bias term (intercept) to the features
X_train_bias = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
X_test_bias = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]

# Initialize parameters for gradient descent
learning_rate = 0.01
n_iterations = 1000
m = X_train_bias.shape[0]

# Initialize theta (coefficients)
theta = np.random.randn(X_train_bias.shape[1])

# Perform gradient descent
for iteration in range(n_iterations):
    gradients = 2/m * X_train_bias.T.dot(X_train_bias.dot(theta) - y_train)
    theta = theta - learning_rate * gradients

# Predict the target variable using the manually implemented Gradient Descent model
y_pred_gd = X_test_bias.dot(theta)

# 3.4 Using Normal Equation (sklearn library)
# Initialize and train the Linear Regression model using normal equation
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predict the target variable using the trained Linear Regression model
y_pred_lin_reg = lin_reg.predict(X_test)

# 4. Calculating the R squared, RMSE, and MSE for the model
# Function to calculate and print metrics for model evaluation
def print_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)  # Calculate Mean Squared Error
    rmse = np.sqrt(mse)  # Calculate Root Mean Squared Error
    r2 = r2_score(y_true, y_pred)  # Calculate R-squared (coefficient of determination)
    print(f'{model_name} - MSE: {mse}, RMSE: {rmse}, R-squared: {r2}')

# Print evaluation metrics for each model
print_metrics(y_test, y_pred_sgd, 'SGD Regressor')
print_metrics(y_test, y_pred_mini_batch, 'Mini Batch Gradient Descent Regressor')
print_metrics(y_test, y_pred_gd, 'Gradient Descent (Manual Implementation)')
print_metrics(y_test, y_pred_lin_reg, 'Linear Regression (Normal Equation)')
