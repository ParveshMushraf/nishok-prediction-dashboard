import streamlit as st
import pandas as pd
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from datetime import timedelta
from sklearn.linear_model import ARDRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM, Dense
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import GRU, Dense
import pandas as pd
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

# Function to load data and perform sales predictions using Prophet
def prophet_prediction(df_prophet, date_column, sales_column, forecasting_period):
    df_prophet.columns = ['ds', 'y']

    model_prophet = Prophet()
    model_prophet.fit(df_prophet)

    future_prophet = model_prophet.make_future_dataframe(periods=forecasting_period, freq='D')
    forecast_prophet = model_prophet.predict(future_prophet)

    prophet_output = forecast_prophet[['ds', 'yhat']].tail(forecasting_period)
    prophet_output.columns = ['Date', 'prophet']

    y_true = df_prophet['y'][-forecasting_period:]
    y_pred = forecast_prophet['yhat'][-forecasting_period:]
    mse = mean_squared_error(y_true, y_pred)

    return prophet_output, mse

# Function to perform sales predictions using Multiple Regressor
def regressor_prediction(df_regressor, date_column, sales_column, forecasting_period):
    df_regressor.columns = [date_column, sales_column]
    df_regressor[date_column] = pd.to_datetime(df_regressor[date_column])
    df_regressor['Day'] = df_regressor[date_column].dt.day
    df_regressor['Month'] = df_regressor[date_column].dt.month
    df_regressor['Year'] = df_regressor[date_column].dt.year
    df_regressor.set_index(date_column, inplace=True)

    X_regressor = df_regressor[['Day', 'Month', 'Year']]
    y_regressor = df_regressor[sales_column]

    X_train_regressor, X_test_regressor, y_train_regressor, y_test_regressor = train_test_split(X_regressor, y_regressor, test_size=0.2, random_state=42)

    model_regressor = LinearRegression()
    model_regressor.fit(X_train_regressor, y_train_regressor)

    future_dates_regressor = pd.date_range(start=df_regressor.index[-1] + pd.DateOffset(1), periods=forecasting_period, freq='D')
    future_features_regressor = pd.DataFrame({
        'Day': future_dates_regressor.day,
        'Month': future_dates_regressor.month,
        'Year': future_dates_regressor.year
    })

    future_predictions_regressor = model_regressor.predict(future_features_regressor)

    future_predictions_df_regressor = pd.DataFrame({
        'Date': future_dates_regressor,
        'Mul_Reg': future_predictions_regressor
    })

    y_true = y_regressor[-forecasting_period:]
    y_pred = future_predictions_regressor
    mse = mean_squared_error(y_true, y_pred)

    return future_predictions_df_regressor, mse

# Function to perform sales predictions using Decision Tree
def decision_tree_prediction(df_tree, date_column, sales_column, forecasting_period):
    df_tree.columns = [date_column, sales_column]
    df_tree[date_column] = pd.to_datetime(df_tree[date_column])
    df_tree['Day'] = df_tree[date_column].dt.day
    df_tree['Month'] = df_tree[date_column].dt.month
    df_tree['Year'] = df_tree[date_column].dt.year
    df_tree.set_index(date_column, inplace=True)

    X_tree = df_tree[['Day', 'Month', 'Year']]
    y_tree = df_tree[sales_column]

    X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(X_tree, y_tree, test_size=0.2, random_state=42)

    model_tree = DecisionTreeRegressor()
    model_tree.fit(X_train_tree, y_train_tree)

    future_dates_tree = pd.date_range(start=df_tree.index[-1] + pd.DateOffset(1), periods=forecasting_period, freq='D')
    future_features_tree = pd.DataFrame({
        'Day': future_dates_tree.day,
        'Month': future_dates_tree.month,
        'Year': future_dates_tree.year
    })

    future_predictions_tree = model_tree.predict(future_features_tree)

    future_predictions_df_tree = pd.DataFrame({
        'Date': future_dates_tree,
        'Decision_tree': future_predictions_tree
    })

    y_true = y_tree[-forecasting_period:]
    y_pred = future_predictions_tree
    mse = mean_squared_error(y_true, y_pred)

    return future_predictions_df_tree, mse

# Function to perform sales predictions using K-Means
def kmeans_prediction(df, date_column, sales_column, forecasting_period):
    df.columns = [date_column, sales_column]
    df[date_column] = pd.to_datetime(df[date_column])
    df['Day'] = df[date_column].dt.day
    df['Month'] = df[date_column].dt.month
    df['Year'] = df[date_column].dt.year
    df.set_index(date_column, inplace=True)

    X = df[['Day', 'Month', 'Year']]
    y = df[sales_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train, y_train)

    future_dates_kmeans = pd.date_range(start=df.index[-1] + pd.DateOffset(1), periods=forecasting_period, freq='D')
    future_features_kmeans = pd.DataFrame({
        'Day': future_dates_kmeans.day,
        'Month': future_dates_kmeans.month,
        'Year': future_dates_kmeans.year
    })

    future_predictions_kmeans = model.predict(future_features_kmeans)

    future_predictions_df_kmeans = pd.DataFrame({
        'Date': future_dates_kmeans,
        'K_Means': future_predictions_kmeans
    })

    y_true = y[-forecasting_period:]
    y_pred = future_predictions_kmeans
    mse = mean_squared_error(y_true, y_pred)

    return future_predictions_df_kmeans, mse

# Function to perform sales predictions using Logistic Regression
def logreg_prediction(file_path, date_column, sales_column, forecasting_period):
    # Read the data from the file
    sales_data = pd.read_csv(file_path)

    # Prepare features and target
    X = pd.to_numeric(pd.to_datetime(sales_data[date_column])).values.reshape(-1, 1)
    y = sales_data[sales_column]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Predict sales for the test set
    y_pred = model.predict(X_test_scaled)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)

    # Generate new dates for forecasting
    new_dates = pd.date_range(start=sales_data[date_column].iloc[-1], periods=forecasting_period, freq='D')
    new_dates_numeric = pd.to_numeric(new_dates).values.reshape(-1, 1)
    new_dates_scaled = scaler.transform(new_dates_numeric)

    # Predict sales for the new dates
    new_sales_pred = model.predict(new_dates_scaled)

    # Create a DataFrame to store predictions
    predictions_df_logreg = pd.DataFrame({'Date': new_dates, 'Log_Reg': new_sales_pred})

    return predictions_df_logreg, mse

# Function to perform sales predictions using Support Vector Regression (SVR)
def svr_prediction(file_path, date_column, sales_column, forecasting_period):
    # Read the data from the file
    sales_data = pd.read_csv(file_path)

    # Prepare features and target
    X = pd.to_numeric(pd.to_datetime(sales_data[date_column])).values.reshape(-1, 1)
    y = sales_data[sales_column]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Support Vector Regression model
    model = SVR()
    model.fit(X_train_scaled, y_train)

    # Predict sales for the test set
    y_pred = model.predict(X_test_scaled)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)

    # Generate new dates for forecasting
    new_dates = pd.date_range(start=sales_data[date_column].iloc[-1], periods=forecasting_period, freq='D')
    new_dates_numeric = pd.to_numeric(new_dates).values.reshape(-1, 1)
    new_dates_scaled = scaler.transform(new_dates_numeric)

    # Predict sales for the new dates
    new_sales_pred = model.predict(new_dates_scaled)

    # Create a DataFrame to store predictions
    predictions_df_svr = pd.DataFrame({'Date': new_dates, 'SVR': new_sales_pred})

    return predictions_df_svr, mse

def gbm_prediction(file_path, forecasting_period, date_column, sales_column):
    sales_data = pd.read_csv(file_path)
    X = pd.to_numeric(pd.to_datetime(sales_data[date_column])).values.reshape(-1, 1)
    y = sales_data[sales_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingRegressor()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)

    new_dates = pd.date_range(start=sales_data[date_column].iloc[-1], periods=forecasting_period, freq='D')
    new_dates_numeric = pd.to_numeric(new_dates).values.reshape(-1, 1)
    new_dates_scaled = scaler.transform(new_dates_numeric)

    new_sales_pred = model.predict(new_dates_scaled)

    predictions_df_gbm = pd.DataFrame({'Date': new_dates, 'GBM': new_sales_pred})

    return predictions_df_gbm, mse

# Function to perform sales predictions using Ridge Regression
def ridge_prediction(file_path, forecasting_period, date_column, sales_column):
    df = pd.read_csv(file_path)
    df[date_column] = pd.to_datetime(df[date_column])
    df['Days'] = (df[date_column] - df[date_column].min()).dt.days
    X = df[['Days']]
    y = df[sales_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)

    y_pred = ridge_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)

    last_date = df[date_column].max()
    next_year_dates = pd.date_range(last_date + timedelta(days=1), periods=forecasting_period, freq='D')
    X_next_year = pd.DataFrame({'Days': (next_year_dates - df[date_column].min()).days})  # Modified this line
    X_next_year_scaled = scaler.transform(X_next_year)

    predictions_next_year = ridge_model.predict(X_next_year_scaled)
    next_year_predictions_df_ridge = pd.DataFrame({'Date': next_year_dates, 'Ridge_reg': predictions_next_year})

    return next_year_predictions_df_ridge, mse

# Function to perform sales predictions using Lasso Regression
def lasso_prediction(file_path, forecasting_period, date_column, sales_column):
    df = pd.read_csv(file_path)
    df[date_column] = pd.to_datetime(df[date_column])
    df['Days'] = (df[date_column] - df[date_column].min()).dt.days
    X = df[['Days']]
    y = df[sales_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lasso_model = Lasso(alpha=0.01)
    lasso_model.fit(X_train_scaled, y_train)

    y_pred = lasso_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)

    last_date = df[date_column].max()
    next_year_dates = pd.date_range(last_date + timedelta(days=1), periods=forecasting_period, freq='D')
    X_next_year = pd.DataFrame({'Days': (next_year_dates - df[date_column].min()).days})  # Modified this line
    X_next_year_scaled = scaler.transform(X_next_year)

    predictions_next_year = lasso_model.predict(X_next_year_scaled)
    next_year_predictions_df_lasso = pd.DataFrame({'Date': next_year_dates, 'Lasso_reg': predictions_next_year})

    return next_year_predictions_df_lasso, mse



# Streamlit App
def main():
    st.title("Forecasting Dashboard")

    # File path input
    file_path = st.text_input('Enter the file path for sales data:', 'sales_data.csv')

    # Column name inputs
    date_column = st.text_input('Enter the column name for Date:', )
    sales_column = st.text_input('Enter the column name for Sales:', )

    # Forecasting period selection
    forecasting_period = st.slider('Select forecasting period (in days):', min_value=1, max_value=365, value=10)

    if st.button('Generate Predictions'):
        df = pd.read_csv(file_path)
        prophet_output, prophet_mse = prophet_prediction(df.copy(), date_column, sales_column, forecasting_period)
        regressor_output, regressor_mse = regressor_prediction(df.copy(), date_column, sales_column, forecasting_period)
        decision_tree_output, decision_tree_mse = decision_tree_prediction(df.copy(), date_column, sales_column, forecasting_period)
        kmeans_output, kmeans_mse = kmeans_prediction(df.copy(), date_column, sales_column, forecasting_period)
        logreg_output, logreg_mse = logreg_prediction(file_path, date_column, sales_column, forecasting_period)
        svr_output, svr_mse = svr_prediction(file_path, date_column, sales_column, forecasting_period)
        gbm_output, gbm_mse = gbm_prediction(file_path, forecasting_period, date_column, sales_column)
        ridge_output, ridge_mse = ridge_prediction(file_path, forecasting_period, date_column, sales_column)
        lasso_output, lasso_mse = lasso_prediction(file_path, forecasting_period, date_column, sales_column)

        # Combine all results into a single dataframe
        final_results = pd.merge(prophet_output, regressor_output, on='Date', how='outer')
        final_results = pd.merge(final_results, decision_tree_output, on='Date', how='outer')
        final_results = pd.merge(final_results, kmeans_output, on='Date', how='outer')
        final_results = pd.merge(final_results, logreg_output, on='Date', how='outer')
        final_results = pd.merge(final_results, svr_output, on='Date', how='outer')
        final_results = pd.merge(final_results, gbm_output, on='Date', how='outer')
        final_results = pd.merge(final_results, ridge_output, on='Date', how='outer')
        final_results = pd.merge(final_results, lasso_output, on='Date', how='outer')
        final_results = final_results.round(0)

        # Display results in a single table
        st.write(" Predictions:")
        st.write(final_results.set_index('Date'))

        # Display MSE values
        st.write("Mean Squared Error (MSE):")
        st.write(f"Prophet: {prophet_mse}")
        st.write(f"Regressor: {regressor_mse}")
        st.write(f"Decision Tree: {decision_tree_mse}")
        st.write(f"K-Means: {kmeans_mse}")
        st.write(f"Logistic Regression: {logreg_mse}")
        st.write(f"SVR: {svr_mse}")
        st.write(f"GBM Model: {gbm_mse}")
        st.write(f"Ridge Regression Model: {ridge_mse}")
        st.write(f"Lasso Regression Model: {lasso_mse}")

        # Plot MSE values
        st.write("Mean Squared Error (MSE) Visualization:")
        mse_data = pd.DataFrame({
            'Model': ['Prophet', 'Regressor', 'Decision Tree', 'K-Means', 'Logistic Regression', 'SVR','GBM','Ridge','Lasso'],
            'MSE': [prophet_mse, regressor_mse, decision_tree_mse, kmeans_mse, logreg_mse, svr_mse,gbm_mse,ridge_mse,lasso_mse]
        })
        mse_chart = st.line_chart(mse_data.set_index('Model'))

if __name__ == '__main__':
    main()
