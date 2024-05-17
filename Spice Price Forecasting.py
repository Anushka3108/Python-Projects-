# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:16:30 2024

@author: Administrator
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engine.outliers import Winsorizer
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv(r"C:\Users\Administrator\Downloads\Project\Spices_Data.csv")

data.head()

# # Credentials to connect to Database
user = 'root'  # user name
pw = 'Anushka1234'  # password
db = 'Std'  # database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# # to_sql() - function to push the dataframe onto a SQL table.
data.to_sql('spices_final_table', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

sql = 'select * from spices_final_table;'
df = pd.read_sql_query(sql, engine)


df.info()

df.describe()

#Checking for the missing values
df.isnull().any()


df.duplicated().sum()
df.shape

pd.unique(df['Spices'])

# Convert 'Month&Year' column to datetime
df['Month&Year'] = pd.to_datetime(df['Month&Year'], format='%d %B %Y').dt.date

# Set 'Month&Year' column as index
df.set_index('Month&Year', inplace=True)

print(df)
df=df.drop('Grade', axis=1)
# Save the modified DataFrame to a CSV file
df.to_csv('spices_data_modified.csv', index=True)

print("Data saved to spices_data_modified.csv")

#Mean
print('Mean of prices with respect to each spice')
mean = df.groupby('Spices')['Price'].mean().reset_index()
print(mean)

#Median
median = df.groupby('Spices')['Price'].median().reset_index()
print(median)

#Variance
variance = df.groupby('Spices')['Price'].var().reset_index()
print(variance)

#Standard Deviation
stdev = df.groupby('Spices')['Price'].std().reset_index()
print(stdev)

skewness = df.groupby('Spices')['Price'].skew().reset_index()
print(skewness)


kurtosis = df.groupby('Spices')['Price'].apply(lambda x: x.kurt()).reset_index()
print(kurtosis)


### Data Visualization 
spices=df['Spices'].unique()

# Define the number of rows and columns for the subplot grid
num_rows = 3
num_cols = len(spices) // num_rows + (len(spices) % num_rows > 0)

# Create subplots for each spice
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 8), sharey=True)

for i, spice in enumerate(spices):
    row = i // num_cols
    col = i % num_cols
    
    subset = df[df['Spices'] == spice]
    
    sns.histplot(subset['Price'], color='skyblue', bins=10, kde=True, ax=axes[row, col])
    axes[row, col].set_title(f'Histogram for {spice} Prices', fontsize=10)
    axes[row, col].set_xlabel('Price', fontsize=8)
    axes[row, col].tick_params(axis='both', labelsize=6)

# Adjust layout
plt.tight_layout()
plt.show()
    
### Data Preprocessing 
#Imputation of missing values in the Price column
df['Price'] = df['Price'].fillna(df['Price'].median())
print(df.isnull().sum())

# Impute missing values in the 'location' column with the most frequent value (mode)
mode_location = df['Location'].mode()[0]
df['Location'] = df['Location'].fillna(mode_location)



spices = df['Spices'].unique()
# Set up a grid layout for subplots
fig, axes = plt.subplots(nrows=len(spices), ncols=1, figsize=(8, 4 * len(spices)))

# Create separate box plots for each spice
for ax, spice in zip(axes, spices):
    sns.boxplot(x='Price', data=df[df['Spices'] == spice], ax=ax)
    ax.set_title(spice)
    ax.set_xlabel('Price')

# Adjust layout to prevent overlapping titles
plt.tight_layout()

# Set a common title for all subplots
plt.suptitle('Box Plots for Each Spice', y=1.02)

plt.show()


treated_df = pd.DataFrame()

# Treating outliers individually for each spice
for spice in spices:
    spice_data = df[df['Spices'] == spice].copy()  # Ensure to create a copy
    
    # Create Winsorizer object
    winsor_iqr = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['Price'])
    
    # Treat outliers for each spice
    spice_data['Price'] = winsor_iqr.fit_transform(spice_data[['Price']])
    
    # Append the treated data to the new DataFrame
    treated_df = pd.concat([treated_df, spice_data], ignore_index=False)
    
    

#Plotting the graph after the outlier treatment

spices = treated_df['Spices'].unique()
# Set up a grid layout for subplots
fig, axes = plt.subplots(nrows=len(spices), ncols=1, figsize=(8, 4 * len(spices)))

# Create separate box plots for each spice
for ax, spice in zip(axes, spices):
    sns.boxplot(x='Price', data=treated_df[treated_df['Spices'] == spice], ax=ax)
    ax.set_title(spice)
    ax.set_xlabel('Price')

# Adjust layout to prevent overlapping titles
plt.tight_layout()

# Set a common title for all subplots
plt.suptitle('Box Plots for Each Spice', y=1.02)

plt.show()



from statsmodels.tsa.stattools import adfuller, kpss

# Function to perform ADF and KPSS tests and print the results
def test_stationarity(timeseries, spice_name):
    # ADF test
    adf_result = adfuller(timeseries, autolag='AIC')
    
    # KPSS test
    kpss_result = kpss(timeseries)
    
    # Print results
    print(f'\nResults for {spice_name}:')
    print(f'ADF Test p-value: {adf_result[1]}')
    print(f'KPSS Test p-value: {kpss_result[1]}')

    # Check for stationarity
    if adf_result[1] < 0.05 and kpss_result[1] > 0.05:
        print(f'{spice_name} is likely stationary.')
    else:
        print(f'{spice_name} is likely non-stationary.')

# Assuming 'df' is your DataFrame
for spice_name in treated_df['Spices'].unique():
    spice_data = treated_df[treated_df['Spices'] == spice_name]['Price']
    
    # Check for stationarity using ADF and KPSS tests
    test_stationarity(spice_data, spice_name)


##### Test for cheking Random walk
import statsmodels.api as sm

# Create a random time series data
np.random.seed(42)
date_rng = pd.date_range(start='2021-01-01', end='2023-12-31', freq='M')
spices = treated_df['Spices'].unique()
df = pd.DataFrame(date_rng, columns=['Month&Year'])
df['Price'] = np.cumsum(np.random.randn(len(date_rng)))

# Generating random values for spice columns
for spice in spices:
    df[spice] = np.cumsum(np.random.randn(len(date_rng)))

# Plot the time series data for each spice
for spice in spices:
    df[spice].plot(figsize=(12, 6), title=f'Time Series Data - {spice}')
    plt.show()

    # Fit an autoregressive model (AR(1)) for each spice
    model = sm.tsa.AutoReg(df[spice], lags=1)
    results = model.fit()

    # Get the AR(1) coefficient for each spice
    ar1_coefficient = results.params[1]

    # Generate the random walk using the AR(1) coefficient for each spice
    random_walk = np.zeros(len(df[spice]))
    random_walk[0] = df[spice].iloc[0]

    for i in range(1, len(df[spice])):
        random_walk[i] = random_walk[i-1] + ar1_coefficient * (df[spice].iloc[i-1] - random_walk[i-1])

    # Plot the original time series data and the random walk for each spice
    plt.figure(figsize=(12, 6))
    plt.plot(df[spice], label=f'Original Data - {spice}')
    plt.plot(df.index, random_walk, label=f'Random Walk - {spice}', linestyle='dashed')
    plt.title(f'Original Data vs Random Walk - {spice}')
    plt.legend()
    plt.show()


### Log transformation 
#Log Transformation to make the data stationary
# Iterate through each unique spice and print the log-transformed values
for spice in treated_df['Spices'].unique():
    spice_data = treated_df[treated_df['Spices'] == spice]
    
    # Perform log transformation on the price of the spice and print the results
    treated_df['log_price'] = np.log(spice_data['Price'])

print(treated_df)    


#############Model Building 
#Train & Test split

for spice in treated_df['Spices'].unique():
    spice_data = treated_df[treated_df['Spices'] == spice]['Price']
    
    # Sort the DataFrame by the index 
    treated_df.sort_index(inplace=True)

    # Split the DataFrame into training (first 24 months) and testing (last 12 months)
    train_df = treated_df.iloc[:-12]  # Select all rows except the last 12
    test_df = treated_df.iloc[-12:]  # Select the last 12 rows

############### ARIMA ##################
from statsmodels.tsa.arima.model import ARIMA

# Create a dictionary to store MAPE values
error_dict = {'Spice': [], 'MAPE': [], 'RMSE': []}

# Iterate through each unique spice
for spice in train_df['Spices'].unique():
    spice_data = train_df[train_df['Spices'] == spice]['Price']
    
    # Fit the ARIMA model
    order = (1, 1, 1)  
    model = ARIMA(spice_data, order=order)
    res = model.fit()

    # Make predictions on the training set
    predictions = res.predict(start=0, end=len(spice_data)-1, dynamic=False)
    
    # Calculate Mean Absolute Percentage Error (MAPE)
    actual_values = spice_data.values
    absolute_percentage_error = np.abs((actual_values - predictions) / actual_values)
    mape = np.mean(absolute_percentage_error) * 100
    
    # Calculate Root Mean Squared Error (RMSE)
    squared_errors = (actual_values - predictions) ** 2
    mean_squared_error = np.mean(squared_errors)
    rmse = np.sqrt(mean_squared_error)
    
    print(f"\nResults for {spice}:")
    print(res.summary())
    print(f"MAPE: {mape:.2f}%")
    print(f"RMSE: {rmse:.2f}")
    print('************************************************************************************************************') 
    # Store MAPE value in the dictionary
    error_dict['Spice'].append(spice)
    error_dict['MAPE'].append(mape)
    error_dict['RMSE'].append(rmse)
    
# Convert the dictionary to a DataFrame
error_df = pd.DataFrame(error_dict)

# Display the MAPE DataFrame
print(error_df)


############### SARIMA ##########################
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Create a dictionary to store MAPE values
error_dict = {'Spice': [], 'MAPE': [], 'RMSE': []}

# Iterate through each unique spice
for spice in train_df['Spices'].unique():
    spice_data = train_df[train_df['Spices'] == spice]['Price']
    
    # Fit the SARIMA model
    model=SARIMAX(spice_data,order=(1, 1, 0),seasonal_order=(1, 1, 0, 12))
    result = model.fit()

    # Make predictions on the training set
    predictions = result.predict(start=0, end=len(spice_data)-1, dynamic=False)
    
    # Calculate Mean Absolute Percentage Error (MAPE)
    actual_values = spice_data.values
    absolute_percentage_error = np.abs((actual_values - predictions) / actual_values)
    mape = np.mean(absolute_percentage_error) * 100
    
    # Calculate Root Mean Squared Error (RMSE) manually
    squared_errors = (actual_values - predictions) ** 2
    mean_squared_error = np.mean(squared_errors)
    rmse = np.sqrt(mean_squared_error)
    
    print(f"\nResults for {spice}:")
    print(result.summary())
    print(f"MAPE: {mape:.2f}%")
    print(f"RMSE: {rmse:.2f}")
    print('************************************************************************************************************') 
    # Store MAPE value in the dictionary
    error_dict['Spice'].append(spice)
    error_dict['MAPE'].append(mape)
    error_dict['RMSE'].append(rmse)
    
# Convert the dictionary to a DataFrame
error_df = pd.DataFrame(error_dict)

# Display the MAPE DataFrame
print(error_df)


######################### SIMPLE EXPONENTIAL SMOOTHING ###############3333
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# Iterate through each unique spice
for spice in train_df['Spices'].unique():
    spice_data = train_df[train_df['Spices'] == spice]['Price']
    
    ses_model = SimpleExpSmoothing(spice_data).fit()
    predictions = ses_model.predict(start=0, end=len(spice_data)-1)
    
    # Calculate Mean Absolute Percentage Error (MAPE)
    actual_values = spice_data.values
    absolute_percentage_error = np.abs((actual_values - predictions) / actual_values)
    mape = np.mean(absolute_percentage_error) * 100
    
    # Calculate Root Mean Squared Error (RMSE) manually
    squared_errors = (actual_values - predictions) ** 2
    mean_squared_error = np.mean(squared_errors)
    rmse = np.sqrt(mean_squared_error)
    
    print(f"\nResults for {spice}:")
    print(ses_model.summary())
    print(f"MAPE: {mape:.2f}%")
    print(f"RMSE: {rmse:.2f}")
    print('************************************************************************************************************') 
    # Store MAPE value in the dictionary
    error_dict['Spice'].append(spice)
    error_dict['MAPE'].append(mape)
    error_dict['RMSE'].append(rmse)
    
# Convert the dictionary to a DataFrame
error_df = pd.DataFrame(error_dict)

print(error_df)
################## HOLT'S SMOOTHING #####################
from statsmodels.tsa.holtwinters import Holt

# Iterate through each unique spice
for spice in train_df['Spices'].unique():
    spice_data = train_df[train_df['Spices'] == spice]['Price']
   
    hw_model = Holt(spice_data).fit()
    predictions = hw_model.predict(start=0, end=len(spice_data)-1)
    
   # Calculate Mean Absolute Percentage Error (MAPE)
    actual_values = spice_data.values
    absolute_percentage_error = np.abs((actual_values - predictions) / actual_values)
    mape = np.mean(absolute_percentage_error) * 100
    
    # Calculate Root Mean Squared Error (RMSE) manually
    squared_errors = (actual_values - predictions) ** 2
    mean_squared_error = np.mean(squared_errors)
    rmse = np.sqrt(mean_squared_error)
    
    print(f"\nResults for {spice}:")
    print(hw_model.summary())
    print(f"MAPE: {mape:.2f}%")
    print(f"RMSE: {rmse:.2f}")
    print('************************************************************************************************************') 
    # Store MAPE value in the dictionary
    error_dict['Spice'].append(spice)
    error_dict['MAPE'].append(mape)
    error_dict['RMSE'].append(rmse)   
    
# Convert the dictionary to a DataFrame
error_df = pd.DataFrame(error_dict)

# Display the MAPE DataFrame
print(error_df)
    

#################### HOLT'S WINTER EXPONENTIAL SMOOTHING ###################
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Iterate through each unique spice
for spice in train_df['Spices'].unique():
    spice_data = train_df[train_df['Spices'] == spice]['Price']
    
    hwe_model_add = ExponentialSmoothing(spice_data, seasonal = "add", trend = "add", seasonal_periods = 11).fit()
    predictions = hwe_model_add.predict(start=0, end=len(spice_data)-1)
    
    # Calculate Mean Absolute Percentage Error (MAPE)
    actual_values = spice_data.values
    absolute_percentage_error = np.abs((actual_values - predictions) / actual_values)
    mape = np.mean(absolute_percentage_error) * 100
    
    # Calculate Root Mean Squared Error (RMSE) manually
    squared_errors = (actual_values - predictions) ** 2
    mean_squared_error = np.mean(squared_errors)
    rmse = np.sqrt(mean_squared_error)
    
    print(f"\nResults for {spice}:")
    print(hwe_model_add.summary())
    print(f"MAPE: {mape:.2f}%")
    print(f"RMSE: {rmse:.2f}")
    print('************************************************************************************************************') 
    # Store MAPE value in the dictionary
    error_dict['Spice'].append(spice)
    error_dict['MAPE'].append(mape)
    error_dict['RMSE'].append(rmse)
    
# Convert the dictionary to a DataFrame
error_df = pd.DataFrame(error_dict)

# Display the MAPE DataFrame
print(error_df)
    


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tabulate import tabulate

# Initialize an empty dictionary to store models and predictions for each spice
spice_models = {}
spice_predictions = {}
spice_actual_values = {}
# Initialize a dictionary to store MAPE values for each spice
mape_values = {}
# Loop through each spice
for spice in spices:
    # Filter data for the current spice
    spice_data = train_df[train_df['Spices'] == spice]
    
    # Ensure the DataFrame is sorted by 'Month&Year'
    spice_data.sort_index(inplace=True)
    
    # Get the price values for the current spice
    price_values = spice_data['Price'].values.reshape(-1, 1)
    
    # Normalize the data using Min-Max scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(price_values)
    
    # Choose the number of time steps (look-back) for creating sequences
    n_steps = 12
    
    def create_sequences(data, n_steps):
        X, y = [],[]
        for i in range(len(data)):
            end_ix = i + n_steps
            if end_ix > len(data)-1:
                break
            seq_x, seq_y = data[i:end_ix], data[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)
    
    # Create sequences and labels
    X, y = create_sequences(scaled_data, n_steps)
    
    # Reshape data for LSTM (samples, time steps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(n_steps, 1)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse')
    
    # Train the model
    model.fit(X, y, epochs=100, batch_size=32, verbose=0)
    
    # Save the model for the current spice
    spice_models[spice] = model
    
    # Extract the last 12 months data for prediction
    last_12_months = scaled_data[-n_steps:]
    
    # Reshape data for prediction
    last_12_months = last_12_months.reshape((1, n_steps, 1))
    
    # Make predictions
    predicted_values = model.predict(last_12_months)
    
    # Inverse transform to get the original scale
    predicted_values = scaler.inverse_transform(predicted_values)
    
    # Save the predictions for the current spice
    spice_predictions[spice] = predicted_values.flatten()
    
    # Save the actual values for the current spice
    actual_values = spice_data['Price'].values[-n_steps:]
    spice_actual_values[spice] = actual_values
    
    # Calculate MAPE for the current spice
    actual_values = spice_actual_values[spice]
    forecasted_values = spice_predictions[spice]
    
    # Ensure the lengths of actual and forecasted values are the same
    min_length = min(len(actual_values), len(forecasted_values))
    actual_values = actual_values[:min_length]
    forecasted_values = forecasted_values[:min_length]
    
    # Calculate MAPE
    mape = np.mean(np.abs((actual_values - forecasted_values) / actual_values)) * 100
    mape_values[spice] = mape

# Display the forecasted and actual prices in a table
table_data = []
for spice in spices:
    table_data.append([spice, spice_predictions[spice], spice_actual_values[spice]])

headers = ["Spice", "Forecasted Prices", "Actual Prices"]
table = tabulate(table_data, headers, tablefmt="pretty")

print(table)
# Display MAPE values for each spice
for spice, mape in mape_values.items():
    print(f"MAPE for {spice}: {mape:.2f}%")
    
    
# Plot the forecasted and actual prices for each spice
for spice in spices:
    plt.figure(figsize=(10, 5))
    plt.plot(spice_actual_values[spice], label='Actual Prices', marker='o')
    plt.plot(spice_predictions[spice], label='Forecasted Prices', marker='o')
    plt.title(f"{spice} - Forecasted vs Actual Prices")
    plt.xlabel("Months")
    plt.ylabel("Price")
    plt.legend()
    plt.show()    
    
