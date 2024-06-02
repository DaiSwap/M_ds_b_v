import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

melbourne_file_path = '/home/pv/ml_model/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# Filter rows with missing price values
filtered_melbourne_data = melbourne_data.dropna(axis=0)

# Choose target and features
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                      'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]

melbourne_model = DecisionTreeRegressor()

# Fit model on the entire dataset (in-sample evaluation)
melbourne_model.fit(X, y)

predicted_home_prices = melbourne_model.predict(X)

# Calculate the Mean Absolute Error for in-sample predictions
in_sample_mae = mean_absolute_error(y, predicted_home_prices)
print("In-sample Mean Absolute Error:")
print(in_sample_mae)
print("This is an in-sample score, does not work for new data only training data")

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(train_X, train_y)

val_predictions = melbourne_model.predict(val_X)


validation_mae = mean_absolute_error(val_y, val_predictions)
print("Validation Mean Absolute Error:")
print(validation_mae)
