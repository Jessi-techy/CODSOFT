import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
sales_dataset = pd.read_csv("C:/Users/JAYA SREE/Downloads/advertising.csv")
print(sales_dataset.head())
X=sales_dataset.drop(columns=['Sales'],axis=1)
y=sales_dataset['Sales']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
model = LinearRegression()
model.fit(X_train,y_train)
training_prediction = model.predict(X_train)
testing_prediction = model.predict(X_test)

print("Mean Squared Error (Train):", mean_squared_error(y_train, training_prediction))
print("Mean Absolute Error (Train):", mean_absolute_error(y_train, training_prediction))
print("R² Score (Train):", r2_score(y_train, training_prediction))

print("\nMean Squared Error (Test):", mean_squared_error(y_test, testing_prediction))
print("Mean Absolute Error (Test):", mean_absolute_error(y_test, testing_prediction))
print("R² Score (Test):", r2_score(y_test, testing_prediction))

plt.figure(figsize=(8, 6))
plt.scatter(y_test, testing_prediction, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()
