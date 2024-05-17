import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from feature_engine.outliers import Winsorizer
from sklearn.impute import SimpleImputer
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt

def calculate_mape(actual_values, forecast_values):
    if len(actual_values) == 0 or len(forecast_values) == 0:
        return np.nan  # Return NaN if arrays are empty to avoid division by zero

    # Calculate MAPE
    mape = np.mean(np.abs((actual_values - forecast_values) / actual_values)) * 100
    return mape

def main():
    st.title("Spice Price Forecasting")

    uploadedFile = st.sidebar.file_uploader("Choose a file", type=['csv', 'xlsx'], accept_multiple_files=False, key="fileUploader")

    if uploadedFile is not None:
        df = pd.read_csv(uploadedFile)
        
        df['Month&Year'] = pd.to_datetime(df['Month&Year'], format='%d %B %Y').dt.date

        df.set_index('Month&Year', inplace=True)

        df.sort_values(by='Month&Year', inplace=True)
        
        # Impute 'Price' column with median
        price_imputer = SimpleImputer(strategy='median')
        df['Price'] = price_imputer.fit_transform(df[['Price']])
        
        # Impute 'Location' column with most frequent value
        location_imputer = SimpleImputer(strategy='most_frequent')
        df['Location'] = location_imputer.fit_transform(df[['Location']])[:, 0]
        
        # Create an empty DataFrame to store the treated data
        data = pd.DataFrame()

        # Create a pipeline for treating outliers using Winsorizer
        outlier_treatment = Winsorizer(capping_method='iqr', tail='both', fold=1.0, variables=['Price'])

        # Treating outliers individually for each spice
        spices=df['Spices'].unique()
        for spice in spices:
            spice_data = df[df['Spices'] == spice].copy()  # Ensure to create a copy
    
            # Apply the outlier treatment pipeline to each spice
            spice_data['Price'] = outlier_treatment.fit_transform(spice_data[['Price']])
    
            # Append the treated data to the new DataFrame
            data = pd.concat([data, spice_data], ignore_index=False)
    
        selected_spice = st.sidebar.selectbox("Select Spice", data['Spices'].unique())
        num_months = st.sidebar.number_input("Number of Months to Predict", min_value=1, max_value=12, value=12)

        if st.button("Predict"):
            spices_list=[]
            models_list=[]
            mape_list=[]
            forecast_results = []

            for spice in [selected_spice]:
                spice_data = data[data['Spices'] == spice]['Price']

                # Store the results of each model in a dictionary
                model_results = {'ARIMA': None, 'SARIMA': None, 'Simple_Exponential _Smoothening': None, 'Holt_model': None, 'Holt_Winter_Exponential_Add': None, 'Holt_Winter_Exponential_Mul': None}

                # ARIMA model
                arima_model = ARIMA(spice_data, order=(1, 1, 1))
                arima_result = arima_model.fit()
                arima_mape = calculate_mape(spice_data, arima_result.predict(start=0, end=len(spice_data)-1, dynamic=False))
                model_results['ARIMA'] = arima_mape
                
                # SARIMA model
                sarima_model = SARIMAX(spice_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                sarima_result = sarima_model.fit()
                sarima_mape = calculate_mape(spice_data, sarima_result.predict(start=0, end=len(spice_data)-1, dynamic=False))
                model_results['SARIMA'] = sarima_mape
                
                # Simple Exponential Smoothing (SES) model
                ses_model = SimpleExpSmoothing(spice_data).fit()
                ses_mape = calculate_mape(spice_data, ses_model.predict(start=0, end=len(spice_data)-1))
                model_results['Simple_Exponential _Smoothening'] = ses_mape
                
                # Holt model
                holt_model = Holt(spice_data).fit()
                holt_mape = calculate_mape(spice_data, holt_model.predict(start=0, end=len(spice_data)-1))
                model_results['Holt_model'] = holt_mape
                    
                # Holt-Winters Exponential Smoothing (HWE) models
                hwea_model = ExponentialSmoothing(spice_data, seasonal="add", trend="add", seasonal_periods=12).fit()
                hwea_mape = calculate_mape(spice_data, hwea_model.predict(start=0, end=len(spice_data)-1))
                model_results[f'Holt_Winter_Exponential_Add'] = hwea_mape
                    
                # Holt-Winters Exponential Smoothing (HWE) models
                hwem_model = ExponentialSmoothing(spice_data, seasonal="add", trend="add", seasonal_periods=12).fit()
                hwem_mape = calculate_mape(spice_data, hwem_model.predict(start=0, end=len(spice_data)-1))
                model_results[f'Holt_Winter_Exponential_Mult'] = hwem_mape
                
                # Filter out None values
                filtered_results = {k: v for k, v in model_results.items() if v is not None}
                
                # Find the model with the minimum MAPE
                if filtered_results:
                    best_model_name, best_model_mape = min(filtered_results.items(), key=lambda x: x[1])
                    
                    # Append the results to the lists
                    spices_list.append(spice)
                    models_list.append(best_model_name)
                    mape_list.append(best_model_mape)

                    # Create a DataFrame from the lists
                    model_df = pd.DataFrame({'Spice': spices_list, 'Model': models_list, 'MAPE': mape_list})
                    
                    # Check if the spice has a corresponding model with MAPE less than 10%
                    if best_model_mape < 10:
                        # Train the best model using the entire dataset (train + test)
                        if best_model_name == 'ARIMA':
                            best_model = ARIMA(spice_data, order=(1, 1, 1))
                        elif best_model_name == 'SARIMA':
                            best_model = SARIMAX(spice_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                        elif best_model_name == 'Simple_Exponential _Smoothening':
                            best_model = SimpleExpSmoothing(spice_data)
                        elif best_model_name == 'Holt_model':
                            best_model = Holt(spice_data)
                        elif best_model_name == 'Holt_Winter_Exponential_Add':
                            best_model = ExponentialSmoothing(spice_data, seasonal="add", trend="add", seasonal_periods=12)
                        elif best_model_name == 'Holt_Winter_Exponential_Mult':
                            best_model = ExponentialSmoothing(spice_data, seasonal="mul", trend="add", seasonal_periods=12)

                        best_model_fit = best_model.fit()

                        # Forecast the next 12 months
                        forecast = best_model_fit.forecast(steps=num_months)
                        
                        # Create a DataFrame for the forecast results
                        forecast_df_spice = pd.DataFrame({'Month&Year': pd.date_range(start=data.index.max(), periods=num_months + 1, freq='M')[1:],
                                                          'Forecast': forecast})
                        
                        # Set 'Month&Year' as index
                        forecast_df_spice.set_index('Month&Year', inplace=True)
                
                        # Append the forecast results to the list
                        forecast_results.append({'Spice': spice, 'Forecast': forecast_df_spice, 'Actual': spice_data})
                    
                        # Display the best model name and forecast table
                        st.write(f"Best Model for {spice}: {best_model_name}")
                        st.write("Forecast Table:")
                        st.write(forecast_df_spice)
                        
                        # Plotting actual and forecasted values for the selected spice
                        for item in forecast_results:
                            spice = item['Spice']
                            actual_data = item['Actual']
                            forecast_data = item['Forecast']
                            
                            plt.figure(figsize=(10, 6))
                            
                            # Plotting actual values
                            plt.plot(actual_data.index, actual_data.values, label=f'Actual {spice}')
                            
                            # Plotting forecasted values
                            plt.plot(forecast_data.index, forecast_data['Forecast'].values, linestyle='dashed', label=f'Forecast {spice}')
                            
                            plt.title(f'Actual vs Forecasted Values for {spice}')
                            plt.xlabel('Month&Year')
                            plt.ylabel('Price')
                            plt.legend()
                            st.pyplot(plt)  # Display the plot in Streamlit

if __name__ == '__main__':
    main()
