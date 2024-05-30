import pandas as pd


melbourne_file_path = '/home/pv/ml_model/melb_data.csv'  

# Read the data and store data in DataFrame titled melbourne_data
try:
    melbourne_data = pd.read_csv(melbourne_file_path)
    print("File read successfully.")
except FileNotFoundError:
    print(f"File not found at {melbourne_file_path}")
    exit()

# Print a summary of the data in Melbourne data
print("Data summary:")
print(melbourne_data.describe())
# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0) 

# to print all the column names
print("Column names:")
print(melbourne_data.columns)

# save filepath to variable for easier access
# melbourne_file_path = '.path_to_file_eg/melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
# melbourne_data = pd.read_csv(melbourne_file_path) 
# print a summary of the data in Melbourne data
# melbourne_data.describe() /to print summary statistics

# selected only required columns and displayed
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
print(X.describe())
print(X.head()) # to show top few rows