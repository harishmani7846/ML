# Importing pre-built modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("C:/Users/Admin/Desktop/ml/simple.csv.csv")

# Preview the data
print(data.head())
print(data.tail())

# Extract features and target
x = data.iloc[:, :-1].values  # All rows, all columns except last (Mileage)
print("Features (Mileage):\n", x)

y = data.iloc[:, 1].values    # All rows, only second column (Selling Price)
print("Target (Selling Price):\n", y)

# Split into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Create and train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict test data
ot = model.predict(x_test)
print("Predicted Selling Prices on Test Data:\n", ot)

# Predict for a specific mileage
predicted_price = model.predict([[1000]])
print("Predicted Selling Price for 1000 mileage:", predicted_price[0])

# Plotting the results
plt.scatter(x_train, y_train, color="red", label="Training Data")
plt.plot(x_train, model.predict(x_train), color="blue", label="Regression Line")
plt.title("Mileage vs Selling Price")
plt.xlabel("Mileage")
plt.ylabel("Selling Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.scatter(x_test, y_test, color="red", label="Training Data")
plt.plot(x_test, model.predict(x_test), color="blue", label="Regression Line")
plt.title("Mileage vs Selling Price")
plt.xlabel("Mileage")
plt.ylabel("Selling Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()