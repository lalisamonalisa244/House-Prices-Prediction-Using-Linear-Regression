import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

#using the cleaned data to train the model
df = pd.read_csv('clean_housing_data.csv')

#dropping the location column as it is not a numerical value and cannot be used in the linear regression model. We can use one-hot encoding to convert the location column into numerical values, but for simplicity, we will drop it for now.
df1 = df.drop('location', axis=1)

#Splitting the data into features and target variable  
X = df1.drop('price', axis=1)
y = df1['price']

#print(X.head(10))
#print(y.head(10))

#print(X.head())

#spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=50)

#calling the linear regression model and fitting the data to the model
RegModel = linear_model.LinearRegression()
RegModel.fit(X_train, y_train)
print(RegModel.predict(X_test))

#using random data to test the model    
input_data = np.array([2052, 4.0, 2, 2045]).reshape(1, -1)
print("========Output prediction========")
print(RegModel.predict(input_data))

print(RegModel.predict(input_data))

#checking the metrics of the model using mean absolute error and mean squared error
y_pred = RegModel.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)   
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
