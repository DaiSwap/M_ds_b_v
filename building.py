'''The steps to building and using a model are:

    Define: What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
    Fit: Capture patterns from provided data. This is the heart of modeling.
    Predict: Just what it sounds like
    Evaluate: Determine how accurate the model's predictions are.
'''
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

melbourne_file_path = '/home/pv/ml_model/melb_data.csv' 
melbourne_data = pd.read_csv(melbourne_file_path)
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea','YearBuilt']
X = melbourne_data[melbourne_features]
y = melbourne_data.Price
# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)

'''

Many machine learning models allow some randomness in model training. 
Specifying a number for random_state ensures you get the same results in each run. 
This is considered a good practice. You use any number, and model quality won't depend meaningfully on exactly what value you choose.
We now have a fitted model that we can use to make predictions.
In practice, you'll want to make predictions for new houses coming on the market rather than the houses we already have prices for. 
But we'll make predictions for the first few rows of the training data to see how the predict function works.
'''

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))

