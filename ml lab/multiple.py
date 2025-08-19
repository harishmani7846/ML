import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer
import seaborn as sns

# Load the dataset
ad = pd.read_csv("/home/student/ex2.csv")

ad.head()
ad.tail()

# Separate features and target
x = ad[['Bedrooms', 'Size', 'Age', 'Zipcode']]
y = ad['Selling price']

# Apply OneHotEncoding to Zipcode
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['Zipcode'])], remainder='passthrough')
x_encoded = ct.fit_transform(x)

# Split the data
xtr, xte, ytr, yte = train_test_split(x_encoded, y, test_size=0.2, random_state=0)

# Train the model
model = LinearRegression()
model.fit(xtr, ytr)

# Predict
ot = model.predict(xte)

# Model details
coefficents = model.coef_
intercept = model.intercept_
print("coefficents:", coefficents)
print("Intercept:", intercept)

# Plot: Actual vs Predicted Selling Price
plt.figure(figsize=(8,6))
plt.scatter(yte, ot, color="blue", s=100)
plt.plot([min(yte), max(yte)], [min(yte), max(yte)], 'r--')
plt.title("Actual vs Predicted Selling Price")
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot: Feature Correlation Heatmap (numeric only)
plt.figure(figsize=(6,5))
sns.heatmap(ad[['Bedrooms', 'Size', 'Age']].corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()