import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error
import shap
import matplotlib.pyplot as plt

melbourne_file_path = '/home/pv/ml_model/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
# important to filter
filtered_melbourne_data = melbourne_data.dropna(subset=['Price'])
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                      'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]

# important to handle missing datA
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
train_X, val_X, train_y, val_y = train_test_split(X_scaled, y, random_state=0)
model = DecisionTreeRegressor(random_state=0)

# Cross-validation
cv_scores = cross_val_score(model, train_X, train_y, cv=5, scoring='neg_mean_absolute_error')
print("Cross-validated MAE: ", -cv_scores.mean())

# Hyperparameter tuning with Grid Search
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}
grid_search = GridSearchCV(DecisionTreeRegressor(random_state=0), param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(train_X, train_y)
print("Best parameters: ", grid_search.best_params_)
print("Best CV MAE: ", -grid_search.best_score_)

# Train the best model
best_model = grid_search.best_estimator_
best_model.fit(train_X, train_y)

# Evaluate on validation set
val_predictions = best_model.predict(val_X)
validation_mae = mean_absolute_error(val_y, val_predictions)
print("Validation Mean Absolute Error: ", validation_mae)

# Model interpretation with SHAP values
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(train_X)
shap.summary_plot(shap_values, train_X, feature_names=melbourne_features)

# finally visualise
plt.figure()
plt.title("Feature Importance")
plt.barh(melbourne_features, best_model.feature_importances_)
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()
