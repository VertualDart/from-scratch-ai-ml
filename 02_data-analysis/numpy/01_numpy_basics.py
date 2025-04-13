import numpy as np
import pandas as pd

vector = np.array([1, 2, 3, 4, 5, 6]) #Vector - 1D array
matrix = np.array([[1, 2], [3, 4], [5, 6]])
print("1D array: ", vector)
print("2D array, matrix:\n ", matrix)

# Slicing
print("First element of array vector: ", vector[0])
print("Last Row of matrix: ", matrix[-1])

# Operations on matrix
#Mean of matrix
mean_val = np.mean(matrix)
print("Mean of matrix", mean_val)

#Variance of matrix
variance_val = np.var(matrix)
print("Variance of matrix: ", variance_val)

# Standard deviation of matrix:
std_dev_val = np.std(matrix)
print("Standard deviation of Matrix", std_dev_val)



data = {
    'Name': ['John Smith','Doe John', 'Ben Dover', 'Anna'],
    'Age': [21, 24, 32, 41],
    'City': ['NYC', 'Ohio', 'Berlin', 'Tokyo']
}

df = pd.DataFrame(data)
print("\nDataFrame\n", df)

# Calculate mean of dataframe (column)
mean_age = df['Age'].mean()
print(f"\nAverage Age: {mean_age}")

# Handling missing data
# Adding a salary column with NaN (missing) values
df['Salary'] = [50000, np.nan, 60000, 70000]
print("\nDataFrame with NaN in Salary:\n", df)

# Fill missing values with mean of salary
df['Salary'].fillna(df['Salary'].mean(), inplace=True)
print("\nDataFrame after filling NaN values in Salary:\n", df)
 
# np.nan is used to introduce missing values and 
# fillna() is used to fill in missing values with other values (mean here for example)