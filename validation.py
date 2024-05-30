import pandas as pd

# Load data
melbourne_file_path = '/home/pv/ml_model/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
# Filter rows with missing price values
filtered_melbourne_data = melbourne_data.dropna(axis=0)
# Choose target and features
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]

from sklearn.tree import DecisionTreeRegressor
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(X, y)


from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
r = mean_absolute_error(y, predicted_home_prices)
print("mean:")
print(r)
'''this is an insample score'''