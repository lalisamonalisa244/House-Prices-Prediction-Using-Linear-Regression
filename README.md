# House-Prices-Prediction-Using-Linear-Regression

This project is mainly based on predicting house prices with a moderate dataset using the linear regression model on supervised learning for training machines. 

Taking a senerio, there are a lot of houses with different facilities like bathroom, bedroom, kitchen, garage and os on. In order to predict the prices of those houses using a machine, you have to train the machine on a dataset containing the values of those facilities then using it to make a prediction using a linear regression model or any other models. 

For example, you have a house at Los Angeles and another house at Saudi Arabia with a specific number of facilities, you will want to predict the price of the houses based on the number of the facilities, the location and how big the house is and instead of doing it manually, you do it with a machine and that's what this project is for. 

First of all, I got a messy data from kaggle. The data contains six columns which are area_sqft, bedrooms, bathrooms, year_built, price and the location. I cleaned the messy dataset using pandas then named it clean_housing_data.csv. After that, I imported the csv file in another python file on vscode to use it to train the model. Now moving to the prediction part, I started with importing the neccessary packages and importing the dataset I cleaned using pandas. I dropped the "location" column that's not a numerical value for a better accuracy in the prediction, also, I dropped it because the type of supervised learning model I'm using is linear regression and it deals mainly with numerical values. 

Splitting the data into target and featured variables was the next thing I did by dropping the targetted value in the dataset I"m assigning to the featured variable and then assigned it as the target variable. After that, I split the dataset into training set and testing set using the train-test split and using only 20% of the data to test the model. I called the linear regression model and fit my data in it to make predictions based on the data. Then, I used random set data to test the model to get a prediction.

Lastly, I checked the metrics of the model using the mean absolute error and the mean squared error to check how accurate the prediction is by comparing it with the target variables. A limitation of this work dropping of the "location" column which could influence the price prediction of a house. In the version two of this work, the "location" column will be transformed with "One-hot Encoding", to assess the effect on the MAE and MSE.
