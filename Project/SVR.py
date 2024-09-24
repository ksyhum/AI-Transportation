import pandas as pd
# import data, skiprows=1 karena data di csv mulai dari baris ke-2
dataset = pd.read_csv("Dataset-PT.csv",skiprows=1)

dataset.head()

dataset.columns

df = dataset[['arrival_delay', 'dwell_time', 'travel_time_for_previous_section',
       'scheduled_travel_time', 'upstream_stop_delay', 'origin_delay',
       'previous_bus_delay', 'previous_trip_travel_time', 'recurrent_delay']]

df.head()

corr_matrix = df.corr()
corr_matrix

corr_matrix['arrival_delay'].sort_values(ascending=False)

x = df.drop(['arrival_delay'], axis=1)
x.head()

x.columns

y = df['arrival_delay']
y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

# Define a parameter grid for hyperparameter tuning
param_grid = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 1, 10]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(SVR(), param_grid, cv=5, verbose=2)

# Fit the grid search to the scaled training data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

print("Best Parameters:", best_params)
print("Best Score:", grid_search.best_score_)

# Create an SVR model with the best parameters from the grid search
best_svr = SVR(kernel=best_params['kernel'], C=best_params['C'], epsilon=best_params['epsilon'])
best_svr.fit(X_train, y_train)

y_pred_SVR = best_svr.predict(X_test)
mae_SVR = mean_absolute_error(y_test, y_pred_SVR)
mse_SVR = mean_squared_error(y_test, y_pred_SVR)
r2_SVR = r2_score(y_test, y_pred_SVR)
print(f"Mean Absolute Error Model SVR: {mae_SVR}")
print(f"Mean Squared Error Model SVR: {mse_SVR}")
print(f"R-squared Model SVR: {r2_SVR}")