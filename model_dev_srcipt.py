import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_data():
	# load the dataset
	data = pd.read_csv('data/sample_submission.csv')

	# display basic information and first few rows
	print("Data Info:", data.info())
	print("Data Head:", data.head())

	# check for missing values
	missing_values = data.isnull().sum().sort_values(ascending=False)
	print("Missing Values:", missing_values[missing_values > 0])

	# Descriptive statistics
	print("Data Describe:", data.describe())

	# visualize correlations with a heatmap
	plt.figure(figsize=(12, 8))
	sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
	plt.show()

	return data


def preprocess_data(data):
	# separate features and target variable
	x = data.drop('SalePrice', axis=1)
	y = data['SalePrice']

	# Handle missing values and encode categorical variables
	numeric_features = x.select_dtypes(include=[np.number]).columns
	categorical_features = x.select_dtypes(include=[object]).columns

	# create preprocessing pipelines
	numeric_transformer = Pipeline(steps=[
		('imputer', SimpleImputer(strategy='median')),
		('scaler', StandardScaler())
	])

	categorical_transformer = Pipeline(steps=[
		('imputer', SimpleImputer(strategy='most_frequent')),
		('onehot', OneHotEncoder(handle_unknown='ignore'))
	])

	# combine preprocessing steps
	preprocessor = ColumnTransformer(
		transformers=[
			('num', numeric_transformer, numeric_features),
			('cat', categorical_transformer, categorical_features)
		])

	# preprocess the data
	x_preprocessed = preprocessor.fit_transform(x)

	return x_preprocessed, y


def evaluate_model(model, x_test, y_test):
	y_pred = model.predict(x_test)
	rmse = np.sqrt(mean_squared_error(y_test, y_pred))
	r2 = r2_score(y_test, y_pred)
	return rmse, r2


def main():
	data = load_data()
	x, y = preprocess_data(data)

	# split data into train and test sets
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

	# train Linear Regression model
	linear_model = LinearRegression()
	linear_model.fit(x_train, y_train)

	# train Random Forest Regressor
	rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
	rf_model.fit(x_train, y_train)

	# evaluate Linear Regression
	linear_rmse, linear_r2 = evaluate_model(linear_model, x_test, y_test)
	print(f'Linear Regression - RMSE: {linear_rmse}, R2: {linear_r2}')

	# evaluate Random Forest Regressor
	rf_rmse, rf_r2 = evaluate_model(rf_model, x_test, y_test)
	print(f'Random Forest - RMSE: {rf_rmse}, R2: {rf_r2}')

	# find the best model
	if rf_rmse < linear_rmse:
		model_name = 'Random Forest Regressor'
		best_model = rf_model
		best_rmse, best_r2 = rf_rmse, rf_r2
	else:
		model_name = 'Linear Regression'
		best_model = linear_model
		best_rmse, best_r2 = linear_rmse, linear_r2

	print(f'Best Model: {model_name} - RMSE: {best_rmse}, R2: {best_r2}')


if __name__ == '__main__':
	main()
