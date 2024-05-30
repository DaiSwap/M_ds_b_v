
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

delhi_file_path = '/home/pv/home-data-for-house-delhi/train.csv'

home_data = pd.read_csv(delhi_file_path)
y = home_data.SalePrice
print("Column names:")
print(home_data.columns)
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]


delhi_model = DecisionTreeRegressor()
# Fit Model
delhi_model.fit(X, y)

print("First in-sample predictions:", delhi_model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())

#import train test split function, arguments X and y , random_sate=1
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

#specify model and fit with training data
delhi_model = DecisionTreeRegressor(random_state=1)
delhi_model.fit(train_X, train_y)

val_predictions = delhi_model.predict(val_X)

#calculate mean absolute error in validation data
from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_y, val_predictions)
print(val_mae)
