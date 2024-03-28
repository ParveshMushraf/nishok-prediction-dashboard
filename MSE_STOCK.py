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
def prophet_prediction(file_path, forecasting_period):
    df_prophet = pd.read_csv(file_path)
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
def regressor_prediction(file_path, forecasting_period):
    df_regressor = pd.read_csv(file_path)
    df_regressor['Date'] = pd.to_datetime(df_regressor['Date'])
    df_regressor['Day'] = df_regressor['Date'].dt.day
    df_regressor['Month'] = df_regressor['Date'].dt.month
    df_regressor['Year'] = df_regressor['Date'].dt.year
    df_regressor.set_index('Date', inplace=True)

    X_regressor = df_regressor[['Day', 'Month', 'Year']]
    y_regressor = df_regressor['Stock']

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
def decision_tree_prediction(file_path, forecasting_period):
    df_tree = pd.read_csv(file_path)
    df_tree['Date'] = pd.to_datetime(df_tree['Date'])
    df_tree['Day'] = df_tree['Date'].dt.day
    df_tree['Month'] = df_tree['Date'].dt.month
    df_tree['Year'] = df_tree['Date'].dt.year
    df_tree.set_index('Date', inplace=True)

    X_tree = df_tree[['Day', 'Month', 'Year']]
    y_tree = df_tree['Stock']

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
def kmeans_prediction(file_path, forecasting_period):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df.set_index('Date', inplace=True)

    X = df[['Day', 'Month', 'Year']]
    y = df['Stock']

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
def logreg_prediction(file_path, forecasting_period):

    # Read the data from the file
    sales_data = pd.read_csv(file_path)

    # Prepare features and target
    X = pd.to_numeric(pd.to_datetime(sales_data['Date'])).values.reshape(-1, 1)
    y = sales_data['Stock']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the model
    model = LinearRegression()
    model.fit(X_scaled, y)

    # Generate new dates for forecasting
    new_dates = pd.date_range(start=sales_data['Date'].iloc[-1], periods=forecasting_period, freq='D')
    new_dates_numeric = pd.to_numeric(new_dates).values.reshape(-1, 1)
    new_dates_scaled = scaler.transform(new_dates_numeric)

    # Predict sales for the new dates
    new_sales_pred = model.predict(new_dates_scaled)

    # Create a DataFrame to store predictions
    predictions_df_logreg = pd.DataFrame({'Date': new_dates, 'Log_Reg': new_sales_pred})

    # Calculate Mean Squared Error (MSE)
    y_true = y[-forecasting_period:]
    y_pred = model.predict(X_scaled[-forecasting_period:])
    mse = mean_squared_error(y_true, y_pred)

    return predictions_df_logreg, mse

def svm_prediction(file_path, forecasting_period):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['NumericDate'] = df['Date'].dt.dayofyear
    X = df['NumericDate'].values.reshape(-1, 1)
    y = df['Stock'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    svr_model = SVR(kernel='rbf')
    svr_model.fit(X_train_scaled, y_train)
    next_year_dates = pd.date_range(df['Date'].max(), periods=forecasting_period, freq='D')
    next_year_numeric_dates = next_year_dates.dayofyear.values.reshape(-1, 1)
    next_year_dates_scaled = scaler.transform(next_year_numeric_dates)
    sales_predictions = svr_model.predict(next_year_dates_scaled)
    predictions_df_SVM = pd.DataFrame({'Date': next_year_dates, 'SVM': sales_predictions})

    # Calculate Mean Squared Error (MSE)
    y_true = y[-forecasting_period:]
    y_pred = svr_model.predict(X_test_scaled[-forecasting_period:])
    mse = mean_squared_error(y_true, y_pred)

    return predictions_df_SVM, mse

# Function to perform sales predictions using Multi-layer Perceptron
def mlp_prediction(file_path, forecasting_period):
    # Load data
    df = pd.read_csv(file_path)

    # Convert 'Date' to numerical representation
    df['Date'] = pd.to_datetime(df['Date'])
    df['NumericDate'] = df['Date'].dt.dayofyear  # You can choose a different representation based on your needs

    # Assuming you have NumericDate and Sales columns
    X = df['NumericDate'].values.reshape(-1, 1)
    y = df['Stock'].values

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the MLPRegressor model
    mlp_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    mlp_model.fit(X_train_scaled, y_train)

    # Generate dates for the next year
    next_year_dates = pd.date_range(df['Date'].max(), periods=forecasting_period, freq='D')
    next_year_numeric_dates = next_year_dates.dayofyear.values.reshape(-1, 1)

    # Standardize next year's dates
    next_year_dates_scaled = scaler.transform(next_year_numeric_dates)

    # Make predictions
    sales_predictions = mlp_model.predict(next_year_dates_scaled)

    # Create a DataFrame with dates and predictions
    predictions_df_MLP = pd.DataFrame({'Date': next_year_dates, 'MLP': sales_predictions})

    # Calculate Mean Squared Error (MSE)
    y_true = y[-forecasting_period:]
    y_pred = mlp_model.predict(X_test_scaled[-forecasting_period:])
    mse = mean_squared_error(y_true, y_pred)

    return predictions_df_MLP, mse

# Function to perform sales predictions using Automatic Relevance Determination
def ard_prediction(file_path, forecasting_period):
    # Load data
    df = pd.read_csv(file_path)

    # Convert 'Date' to numerical representation
    df['Date'] = pd.to_datetime(df['Date'])
    df['NumericDate'] = df['Date'].dt.dayofyear  # You can choose a different representation based on your needs

    # Assuming you have NumericDate and Sales columns
    X = df['NumericDate'].values.reshape(-1, 1)
    y = df['Stock'].values

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the ARDRegression model
    ard_model = ARDRegression()
    ard_model.fit(X_train_scaled, y_train)

    # Generate dates for the next year
    next_year_dates = pd.date_range(df['Date'].max(), periods=forecasting_period, freq='D')
    next_year_numeric_dates = next_year_dates.dayofyear.values.reshape(-1, 1)

    # Standardize next year's dates
    next_year_dates_scaled = scaler.transform(next_year_numeric_dates)

    # Make predictions
    sales_predictions = ard_model.predict(next_year_dates_scaled)

    # Create a DataFrame with dates and predictions
    predictions_df_ARD = pd.DataFrame({'Date': next_year_dates, 'ARD': sales_predictions})

    # Calculate Mean Squared Error (MSE)
    y_true = y[-forecasting_period:]
    y_pred = ard_model.predict(X_test_scaled[-forecasting_period:])
    mse = mean_squared_error(y_true, y_pred)

    return predictions_df_ARD, mse

def stack_prediction(file_path, forecasting_period):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['NumericDate'] = df['Date'].dt.dayofyear
    X = df['NumericDate'].values.reshape(-1, 1)
    y = df['Stock'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    linear_reg = LinearRegression()
    mlp_reg = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    svr_reg = SVR(kernel='rbf')
    stacked_model = StackingRegressor(
        estimators=[('linear', linear_reg), ('mlp', mlp_reg), ('svr', svr_reg)],
        final_estimator=LinearRegression()
    )
    stacked_model.fit(X_train_scaled, y_train)

    next_year_dates = pd.date_range(df['Date'].max(), periods=forecasting_period, freq='D')
    next_year_numeric_dates = next_year_dates.dayofyear.values.reshape(-1, 1)
    next_year_dates_scaled = scaler.transform(next_year_numeric_dates)
    sales_predictions = stacked_model.predict(next_year_dates_scaled)
    predictions_df_stack = pd.DataFrame({'Date': next_year_dates, 'STACK': sales_predictions})

    y_true = y_test
    y_pred = stacked_model.predict(X_test_scaled)
    mse = mean_squared_error(y_true, y_pred)

    return predictions_df_stack, mse
def linear_reg_prediction(file_path, forecasting_period):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['NumericDate'] = df['Date'].dt.dayofyear
    X = df['NumericDate'].values.reshape(-1, 1)
    y = df['Stock'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)

    next_year_dates = pd.date_range(df['Date'].max(), periods=forecasting_period, freq='D')
    next_year_numeric_dates = next_year_dates.dayofyear.values.reshape(-1, 1)
    next_year_dates_scaled = scaler.transform(next_year_numeric_dates)
    sales_predictions = linear_model.predict(next_year_dates_scaled)
    predictions_df_linear = pd.DataFrame({'Date': next_year_dates, 'Linear': sales_predictions.flatten()})

    y_true = y_test
    y_pred = linear_model.predict(X_test_scaled)
    mse = mean_squared_error(y_true, y_pred)

    return predictions_df_linear, mse


# New function for Poisson Regression prediction
def poisson_reg_prediction(file_path, forecasting_period):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['NumericDate'] = df['Date'].dt.dayofyear
    X = df['NumericDate'].values.reshape(-1, 1)
    y = df['Stock'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    poisson_model = PoissonRegressor()
    poisson_model.fit(X_train_scaled, y_train)

    next_year_dates = pd.date_range(df['Date'].max(), periods=forecasting_period, freq='D')
    next_year_numeric_dates = next_year_dates.dayofyear.values.reshape(-1, 1)
    next_year_dates_scaled = scaler.transform(next_year_numeric_dates)
    sales_predictions = poisson_model.predict(next_year_dates_scaled)
    predictions_df_poisson = pd.DataFrame({'Date': next_year_dates, 'Poisson': sales_predictions})

    y_true = y_test
    y_pred = poisson_model.predict(X_test_scaled)
    mse = mean_squared_error(y_true, y_pred)

    return predictions_df_poisson, mse

    # New function for Gaussian Process Regression prediction
def gaussian_prediction(file_path, forecasting_period):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['NumericDate'] = df['Date'].dt.dayofyear
    X = df['NumericDate'].values.reshape(-1, 1)
    y = df['Stock'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    gp_model.fit(X_train_scaled, y_train)
    next_year_dates = pd.date_range(df['Date'].max(), periods=forecasting_period, freq='D')
    next_year_numeric_dates = next_year_dates.dayofyear.values.reshape(-1, 1)
    next_year_dates_scaled = scaler.transform(next_year_numeric_dates)
    sales_predictions, sigma = gp_model.predict(next_year_dates_scaled, return_std=True)

    predictions_df_gaussian = pd.DataFrame({'Date': next_year_dates, 'Gaussian': sales_predictions})

    y_true = df['Stock'].values[-forecasting_period:]
    y_pred = sales_predictions
    mse = mean_squared_error(y_true, y_pred)

    return predictions_df_gaussian, mse

# New function for ARIMA prediction
def arima_prediction(file_path, forecasting_period):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    model = ARIMA(df['Stock'], order=(5, 1, 2))
    results = model.fit()
    next_month_forecast = results.get_forecast(steps=forecasting_period)
    next_month_index = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=forecasting_period)
    next_month_sales = next_month_forecast.predicted_mean.values
    rounded_sales = pd.Series(next_month_sales).round().astype(int)
    rounded_sales_df_arima = pd.DataFrame({'Date': next_month_index, 'ARIMA': rounded_sales})

    y_true = df['Stock'].values[-forecasting_period:]
    y_pred = next_month_sales
    mse = mean_squared_error(y_true, y_pred)

    return rounded_sales_df_arima, mse

# New function for Partial Least Squares (PLS) Regression prediction
def pls_prediction(file_path, forecasting_period):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['NumericDate'] = df['Date'].dt.dayofyear
    X = df['NumericDate'].values.reshape(-1, 1)
    y = df['Stock'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)
    next_year_dates = pd.date_range(df['Date'].max(), periods=forecasting_period, freq='D')
    next_year_numeric_dates = next_year_dates.dayofyear.values.reshape(-1, 1)
    next_year_dates_scaled = scaler.transform(next_year_numeric_dates)
    sales_predictions = linear_model.predict(next_year_dates_scaled)
    predictions_df_PLS = pd.DataFrame({'Date': next_year_dates, 'PLS': sales_predictions.flatten()})

    y_true = df['Stock'].values[-forecasting_period:]
    y_pred = sales_predictions
    mse = mean_squared_error(y_true, y_pred)

    return predictions_df_PLS, mse

# New function for SARIMA forecasting
def sarima_forecasting(file_path, forecasting_period):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
    model_sarima = SARIMAX(df['Stock'], order=order, seasonal_order=seasonal_order)
    results_sarima = model_sarima.fit(disp=False)
    next_year_forecast = results_sarima.get_forecast(steps=forecasting_period)
    next_year_index = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=forecasting_period)
    next_year_sales = next_year_forecast.predicted_mean
    rounded_sales = pd.Series(next_year_sales).round().astype(int)
    rounded_sales_df = pd.DataFrame({'Date': next_year_index, 'Sarima': rounded_sales})

    y_true = df['Stock'].values[-forecasting_period:]
    y_pred = next_year_sales
    mse = mean_squared_error(y_true, y_pred)

    return rounded_sales_df, mse


def exponential_smoothing_forecasting(file_path, forecasting_period):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    model = ExponentialSmoothing(df['Stock'], seasonal='add', seasonal_periods=12)
    result = model.fit()

    next_period_start = df.index[-1] + pd.DateOffset(days=1)
    next_period_end = next_period_start + pd.DateOffset(days=forecasting_period)
    expon = result.predict(start=next_period_start, end=next_period_end)

    output_df = pd.DataFrame({'Date': expon.index, 'Expo': expon.values})
    output_df.to_csv("expo_output.csv", index=False)

    y_true = df['Stock'][-forecasting_period:]
    y_pred = expon[-forecasting_period:]
    mse = mean_squared_error(y_true, y_pred)

    return output_df, mse

# Function for Random Forest regression
def random_forest(file_path, forecasting_period):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])

    X = df[['Date']]
    y = df['Stock']

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    last_date = df['Date'].max()
    next_year_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecasting_period, freq='D')

    next_year_df = pd.DataFrame({'Date': next_year_dates})

    next_year_predictions = rf_model.predict(next_year_df[['Date']])

    next_year_df['Random_forest'] = next_year_predictions

    next_year_df.to_csv('random_output.csv', index=False)

    y_true = y[-forecasting_period:]
    y_pred = next_year_predictions
    mse = mean_squared_error(y_true, y_pred)

    return next_year_df, mse

# Function for elastic net regression
def elastic_net(file_path, forecasting_period):
    # Load your training data
    train_data = pd.read_csv(file_path)

    # Ensure that the 'Date' column is in the correct date format
    train_data['Date'] = pd.to_datetime(train_data['Date'])

    # Feature engineering for the training data (similar to previous code)
    train_data['year'] = train_data['Date'].dt.year
    train_data['month'] = train_data['Date'].dt.month
    train_data['day'] = train_data['Date'].dt.day
    train_data['day_of_week'] = train_data['Date'].dt.dayofweek
    train_data['day_of_year'] = train_data['Date'].dt.dayofyear
    train_data['week_of_year'] = train_data['Date'].dt.isocalendar().week

    # Drop the original 'Date' column and target 'Sales' column
    X_train = train_data.drop(['Date', 'Stock'], axis=1)
    y_train = train_data['Stock']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Initialize the Elastic Net model
    alpha = 0.5  # L1 regularization term
    l1_ratio = 0.5  # Elastic Net mixing parameter
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model using MSE
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error on test set: {mse}')

    # Generate a DataFrame for the next year
    next_year_dates = pd.date_range(start=train_data['Date'].max() + timedelta(days=1), periods=forecasting_period, freq='D')
    next_year_data = pd.DataFrame({'Date': next_year_dates})

    # Feature engineering for the next year data
    next_year_data['year'] = next_year_data['Date'].dt.year
    next_year_data['month'] = next_year_data['Date'].dt.month
    next_year_data['day'] = next_year_data['Date'].dt.day
    next_year_data['day_of_week'] = next_year_data['Date'].dt.dayofweek
    next_year_data['day_of_year'] = next_year_data['Date'].dt.dayofyear
    next_year_data['week_of_year'] = next_year_data['Date'].dt.isocalendar().week

    # Use the trained model to predict sales for the next year
    next_year_predictions = model.predict(next_year_data.drop('Date', axis=1))

    # Add the predicted values to the next year DataFrame
    next_year_data['Elastic_net'] = next_year_predictions

    # Print the DataFrame with 'Date' and 'predicted_sales' columns, removing unwanted columns
    output_columns = ['Date', 'Elastic_net']
    next_year_data_output = next_year_data[output_columns]
    print(next_year_data_output)

    # Save the next year data to a CSV file
    next_year_data_output.to_csv('elasticnet_output.csv', index=False)
    return next_year_data_output,mse


def lightgbm(file_path, forecasting_period):
    # Load your training data
    train_data = pd.read_csv(file_path)

    # Ensure that the 'Date' column is in the correct date format
    train_data['Date'] = pd.to_datetime(train_data['Date'])

    # Feature engineering for the training data (similar to previous code)
    train_data['year'] = train_data['Date'].dt.year
    train_data['month'] = train_data['Date'].dt.month
    train_data['day'] = train_data['Date'].dt.day
    train_data['day_of_week'] = train_data['Date'].dt.dayofweek
    train_data['day_of_year'] = train_data['Date'].dt.dayofyear
    train_data['week_of_year'] = train_data['Date'].dt.isocalendar().week

    # Drop the original 'Date' column and target 'Sales' column
    X_train = train_data.drop(['Date', 'Stock'], axis=1)
    y_train = train_data['Stock']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Initialize the LightGBM model
    model = LGBMRegressor()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model using MSE
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error on test set: {mse}')

    # Generate a DataFrame for the next year
    next_year_dates = pd.date_range(start=train_data['Date'].max() + timedelta(days=1), periods=forecasting_period, freq='D')
    next_year_data = pd.DataFrame({'Date': next_year_dates})

    # Feature engineering for the next year data
    next_year_data['year'] = next_year_data['Date'].dt.year
    next_year_data['month'] = next_year_data['Date'].dt.month
    next_year_data['day'] = next_year_data['Date'].dt.day
    next_year_data['day_of_week'] = next_year_data['Date'].dt.dayofweek
    next_year_data['day_of_year'] = next_year_data['Date'].dt.dayofyear
    next_year_data['week_of_year'] = next_year_data['Date'].dt.isocalendar().week

    # Use the trained model to predict sales for the next year
    next_year_predictions = model.predict(next_year_data.drop('Date', axis=1))

    # Add the predicted values to the next year DataFrame
    next_year_data['LIGHT_GBM'] = next_year_predictions

    # Print the DataFrame with 'Date' and 'predicted_sales' columns, removing unwanted columns
    output_columns = ['Date', 'LIGHT_GBM']
    next_year_data_output = next_year_data[output_columns]
    print(next_year_data_output)

    # Save the next year data to a CSV file
    next_year_data_output.to_csv('lightgbm_output.csv', index=False)
    return next_year_data_output,mse

def catboost(file_path, forecasting_period):
    # Load your training data
    train_data = pd.read_csv(file_path)

    # Ensure that the 'Date' column is in the correct date format
    train_data['Date'] = pd.to_datetime(train_data['Date'])

    # Feature engineering for the training data (similar to previous code)
    train_data['year'] = train_data['Date'].dt.year
    train_data['month'] = train_data['Date'].dt.month
    train_data['day'] = train_data['Date'].dt.day
    train_data['day_of_week'] = train_data['Date'].dt.dayofweek
    train_data['day_of_year'] = train_data['Date'].dt.dayofyear
    train_data['week_of_year'] = train_data['Date'].dt.isocalendar().week

    # Drop the original 'Date' column and target 'Sales' column
    X_train = train_data.drop(['Date', 'Stock'], axis=1)
    y_train = train_data['Stock']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Initialize the CatBoost model
    model = CatBoostRegressor()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model using MSE
    mse = mean_squared_error(y_test, predictions)

    # Generate a DataFrame for the next year
    next_year_dates = pd.date_range(start=train_data['Date'].max() + timedelta(days=1), periods=forecasting_period, freq='D')
    next_year_data = pd.DataFrame({'Date': next_year_dates})

    # Feature engineering for the next year data
    next_year_data['year'] = next_year_data['Date'].dt.year
    next_year_data['month'] = next_year_data['Date'].dt.month
    next_year_data['day'] = next_year_data['Date'].dt.day
    next_year_data['day_of_week'] = next_year_data['Date'].dt.dayofweek
    next_year_data['day_of_year'] = next_year_data['Date'].dt.dayofyear
    next_year_data['week_of_year'] = next_year_data['Date'].dt.isocalendar().week

    # Use the trained model to predict sales for the next year
    next_year_predictions = model.predict(next_year_data.drop('Date', axis=1))

    # Add the predicted values to the next year DataFrame
    next_year_data['CATBOOST'] = next_year_predictions

    # Print the DataFrame with 'Date' and 'predicted_sales' columns, removing unwanted columns
    output_columns = ['Date', 'CATBOOST']
    next_year_data_output = next_year_data[output_columns]
    print(next_year_data_output)

    # Save the next year data to a CSV file
    next_year_data_output.to_csv('catboost_output.csv', index=False)
    return next_year_data_output, mse

def ridge_prediction(file_path, forecasting_period):

    # Read the data from the file
    df = pd.read_csv(file_path)

    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Calculate the number of days since the earliest date
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    X = df[['Days']]
    y = df['Stock']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize Ridge Regression model
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)

    # Predict sales for the test set
    y_pred = ridge_model.predict(X_test_scaled)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error on Test Set: {mse}')

    # Determine the last date in the dataset
    last_date = df['Date'].max()

    # Generate dates for the next year
    next_year_dates = pd.date_range(last_date + timedelta(days=1), periods=forecasting_period, freq='D')
    X_next_year = pd.DataFrame({'Days': (next_year_dates - df['Date'].min()).days})

    # Scale the features for the next year
    X_next_year_scaled = scaler.transform(X_next_year)

    # Predict sales for the next year
    predictions_next_year = ridge_model.predict(X_next_year_scaled)

    # Create a DataFrame to store predictions for the next year
    next_year_predictions_df_ridge = pd.DataFrame({'Date': next_year_dates, 'Ridge_reg': predictions_next_year})

    return next_year_predictions_df_ridge,mse

def lasso_prediction(file_path, forecasting_period):


    # Read the data from the file
    df = pd.read_csv(file_path)

    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Calculate the number of days since the earliest date
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    X = df[['Days']]
    y = df['Stock']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize Lasso Regression model
    lasso_model = Lasso(alpha=0.01)
    lasso_model.fit(X_train_scaled, y_train)

    # Predict sales for the test set
    y_pred = lasso_model.predict(X_test_scaled)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error on Test Set: {mse}')

    # Determine the last date in the dataset
    last_date = df['Date'].max()

    # Generate dates for the next year
    next_year_dates = pd.date_range(last_date + timedelta(days=1), periods=forecasting_period, freq='D')
    X_next_year = pd.DataFrame({'Days': (next_year_dates - df['Date'].min()).days})

    # Scale the features for the next year
    X_next_year_scaled = scaler.transform(X_next_year)

    # Predict sales for the next year
    predictions_next_year = lasso_model.predict(X_next_year_scaled)

    # Create a DataFrame to store predictions for the next year
    next_year_predictions_df_lasso = pd.DataFrame({'Date': next_year_dates, 'Lasso_reg': predictions_next_year})

    return next_year_predictions_df_lasso,mse


def svr_prediction(file_path, forecasting_period):

    # Read the data from the file
    sales_data = pd.read_csv(file_path)

    # Prepare features and target
    X = pd.to_numeric(pd.to_datetime(sales_data['Date'])).values.reshape(-1, 1)
    y = sales_data['Stock']

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
    print(f"Mean Squared Error: {mse}")

    # Generate new dates for forecasting
    new_dates = pd.date_range(start=sales_data['Date'].iloc[-1], periods=forecasting_period, freq='D')
    new_dates_numeric = pd.to_numeric(new_dates).values.reshape(-1, 1)
    new_dates_scaled = scaler.transform(new_dates_numeric)

    # Predict sales for the new dates
    new_sales_pred = model.predict(new_dates_scaled)

    # Create a DataFrame to store predictions
    predictions_df_svr = pd.DataFrame({'Date': new_dates, 'SVR': new_sales_pred})
    return predictions_df_svr,mse


def gbm_prediction(file_path, forecasting_period):

    # Read the data from the file
    sales_data = pd.read_csv(file_path)

    # Prepare features and target
    X = pd.to_numeric(pd.to_datetime(sales_data['Date'])).values.reshape(-1, 1)
    y = sales_data['Stock']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Gradient Boosting Regressor model
    model = GradientBoostingRegressor()
    model.fit(X_train_scaled, y_train)

    # Predict sales for the test set
    y_pred = model.predict(X_test_scaled)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Generate new dates for forecasting
    new_dates = pd.date_range(start=sales_data['Date'].iloc[-1], periods=forecasting_period, freq='D')
    new_dates_numeric = pd.to_numeric(new_dates).values.reshape(-1, 1)
    new_dates_scaled = scaler.transform(new_dates_numeric)

    # Predict sales for the new dates
    new_sales_pred = model.predict(new_dates_scaled)

    # Create a DataFrame to store predictions
    predictions_df_gbm = pd.DataFrame({'Date': new_dates, 'GBM': new_sales_pred})
    return predictions_df_gbm,mse    


def pca(file_path, forecasting_period):
    # Load your training data
    train_data = pd.read_csv(file_path)

    # Ensure that the 'Date' column is in the correct date format
    train_data['Date'] = pd.to_datetime(train_data['Date'])

    # Feature engineering for the training data (similar to previous code)
    train_data['year'] = train_data['Date'].dt.year
    train_data['month'] = train_data['Date'].dt.month
    train_data['day'] = train_data['Date'].dt.day
    train_data['day_of_week'] = train_data['Date'].dt.dayofweek
    train_data['day_of_year'] = train_data['Date'].dt.dayofyear
    train_data['week_of_year'] = train_data['Date'].dt.isocalendar().week

    # Drop the original 'Date' column and target 'Sales' column
    X = train_data.drop(['Date', 'Stock'], axis=1)
    y = train_data['Stock']

    # Apply PCA to reduce dimensionality
    n_components = 5  # Choose the number of principal components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Initialize a Linear Regression model
    model = LinearRegression()

    # Train the model on the entire dataset
    model.fit(X_pca, y)

    # Generate a DataFrame for the next year
    next_year_dates = pd.date_range(start=train_data['Date'].max() + timedelta(days=1), periods=forecasting_period, freq='D')
    next_year_data = pd.DataFrame({'Date': next_year_dates})

    # Feature engineering for the next year data
    next_year_data['year'] = next_year_data['Date'].dt.year
    next_year_data['month'] = next_year_data['Date'].dt.month
    next_year_data['day'] = next_year_data['Date'].dt.day
    next_year_data['day_of_week'] = next_year_data['Date'].dt.dayofweek
    next_year_data['day_of_year'] = next_year_data['Date'].dt.dayofyear
    next_year_data['week_of_year'] = next_year_data['Date'].dt.isocalendar().week

    # Apply PCA to the next year data
    next_year_X = next_year_data.drop('Date', axis=1)
    next_year_data_pca = pca.transform(next_year_X)

    # Use the trained model to predict sales for the next year
    next_year_predictions = model.predict(next_year_data_pca)

    # Add the predicted values to the next year DataFrame
    next_year_data['PCA'] = next_year_predictions

    # Calculate Mean Squared Error (MSE)
    y_true = train_data['Stock']
    y_pred = model.predict(X_pca)
    mse = mean_squared_error(y_true, y_pred)
    print('MSE on training data:', mse)

    # Print the DataFrame with 'Date' and 'predicted_sales' columns, removing unwanted columns
    output_columns = ['Date', 'PCA']
    next_year_data_output = next_year_data[output_columns]
    print(next_year_data_output)

    # Save the next year data to a CSV file
    next_year_data_output.to_csv('pca_output.csv', index=False)

    # Return the MSE along with the predicted sales
    return next_year_data_output, mse
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import timedelta

def ica(file_path, forecasting_period):
    # Load your training data
    train_data = pd.read_csv(file_path)

    # Ensure that the 'Date' column is in the correct date format
    train_data['Date'] = pd.to_datetime(train_data['Date'])

    # Feature engineering for the training data (similar to previous code)
    train_data['year'] = train_data['Date'].dt.year
    train_data['month'] = train_data['Date'].dt.month
    train_data['day'] = train_data['Date'].dt.day
    train_data['day_of_week'] = train_data['Date'].dt.dayofweek
    train_data['day_of_year'] = train_data['Date'].dt.dayofyear
    train_data['week_of_year'] = train_data['Date'].dt.isocalendar().week

    # Drop the original 'Date' column and target 'Sales' column
    X = train_data.drop(['Date', 'Stock'], axis=1)
    y = train_data['Stock']

    # Apply ICA to reduce dimensionality
    n_components = 5  # Choose the number of independent components
    ica = FastICA(n_components=n_components, random_state=42)
    X_ica = ica.fit_transform(X)

    # Initialize a Linear Regression model
    model = LinearRegression()

    # Train the model on the entire dataset
    model.fit(X_ica, y)

    # Generate a DataFrame for the next year
    next_year_dates = pd.date_range(start=train_data['Date'].max() + timedelta(days=1), periods=forecasting_period, freq='D')
    next_year_data = pd.DataFrame({'Date': next_year_dates})

    # Feature engineering for the next year data
    next_year_data['year'] = next_year_data['Date'].dt.year
    next_year_data['month'] = next_year_data['Date'].dt.month
    next_year_data['day'] = next_year_data['Date'].dt.day
    next_year_data['day_of_week'] = next_year_data['Date'].dt.dayofweek
    next_year_data['day_of_year'] = next_year_data['Date'].dt.dayofyear
    next_year_data['week_of_year'] = next_year_data['Date'].dt.isocalendar().week

    # Apply ICA to the next year data
    next_year_X = next_year_data.drop('Date', axis=1)
    next_year_data_ica = ica.transform(next_year_X)

    # Use the trained model to predict sales for the next year
    next_year_predictions = model.predict(next_year_data_ica)

    # Calculate Mean Squared Error (MSE)
    train_predictions = model.predict(X_ica)
    mse = mean_squared_error(y, train_predictions)
    print(f"Train MSE: {mse}")

    # Add the predicted values to the next year DataFrame
    next_year_data['ICA'] = next_year_predictions

    # Print the DataFrame with 'Date' and 'predicted_sales' columns, removing unwanted columns
    output_columns = ['Date', 'ICA']
    next_year_data_output = next_year_data[output_columns]
    print(next_year_data_output)

    # Save the next year data to a CSV file
    next_year_data_output.to_csv('ica_output.csv', index=False)

    return next_year_data_output, mse

def svm_prediction(file_path, forecasting_period):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['NumericDate'] = df['Date'].dt.dayofyear
    X = df['NumericDate'].values.reshape(-1, 1)
    y = df['Stock'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    svr_model = SVR(kernel='rbf')
    svr_model.fit(X_train_scaled, y_train)
    next_year_dates = pd.date_range(df['Date'].max(), periods=forecasting_period, freq='D')
    next_year_numeric_dates = next_year_dates.dayofyear.values.reshape(-1, 1)
    next_year_dates_scaled = scaler.transform(next_year_numeric_dates)
    sales_predictions = svr_model.predict(next_year_dates_scaled)
    predictions_df_SVM = pd.DataFrame({'Date': next_year_dates, 'SVM': sales_predictions})

    # Calculate Mean Squared Error (MSE)
    y_true = df['Stock']
    X_all_scaled = scaler.transform(X)
    y_pred_all = svr_model.predict(X_all_scaled)
    mse = mean_squared_error(y_true, y_pred_all)


    return predictions_df_SVM, mse

def mlp_prediction(file_path, forecasting_period):
    # Load data
    df = pd.read_csv(file_path)

    # Convert 'Date' to numerical representation
    df['Date'] = pd.to_datetime(df['Date'])
    df['NumericDate'] = df['Date'].dt.dayofyear  # You can choose a different representation based on your needs

    # Assuming you have NumericDate and Sales columns
    X = df['NumericDate'].values.reshape(-1, 1)
    y = df['Stock'].values

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the MLPRegressor model
    mlp_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    mlp_model.fit(X_train_scaled, y_train)

    # Generate dates for the next year
    next_year_dates = pd.date_range(df['Date'].max(), periods=forecasting_period, freq='D')
    next_year_numeric_dates = next_year_dates.dayofyear.values.reshape(-1, 1)

    # Standardize next year's dates
    next_year_dates_scaled = scaler.transform(next_year_numeric_dates)

    # Make predictions
    sales_predictions = mlp_model.predict(next_year_dates_scaled)

    # Create a DataFrame with dates and predictions
    predictions_df_MLP = pd.DataFrame({'Date': next_year_dates, 'MLP': sales_predictions})

    # Calculate Mean Squared Error (MSE)
    y_true = df['Stock']
    X_all_scaled = scaler.transform(X)
    y_pred_all = mlp_model.predict(X_all_scaled)
    mse = mean_squared_error(y_true, y_pred_all)

    # Save predictions to a CSV file
    predictions_df_MLP.to_csv('mlp_output.csv', index=False)

    return predictions_df_MLP, mse





import streamlit as st
import pandas as pd

# Streamlit App
def main():
    st.title("Stock Forecasting Dashboard")

    # File path input
    file_path = st.text_input('Enter the file path for sales data:', 'stock_data.csv')

    # Forecasting period selection
   # forecasting_period = st.number_input('Select forecasting period (in days):', min_value=1, max_value=365, value=10)
    # Forecasting period selection(slider)
    forecasting_period = st.slider('Select forecasting period (in days):', min_value=1, max_value=365, value=10)

    if st.button('Generate Predictions'):
        # Perform predictions using each model
        prophet_output, prophet_mse = prophet_prediction(file_path, forecasting_period)
        regressor_output, regressor_mse = regressor_prediction(file_path, forecasting_period)
        decision_tree_output, decision_tree_mse = decision_tree_prediction(file_path, forecasting_period)
        kmeans_output, kmeans_mse = kmeans_prediction(file_path, forecasting_period)
        logreg_output, logreg_mse = logreg_prediction(file_path, forecasting_period)
        svr_output,svr_mse=svr_prediction(file_path,forecasting_period)
        ard_output, ard_mse = ard_prediction(file_path, forecasting_period)
        stack_output, stack_mse = stack_prediction(file_path, forecasting_period)
        linear_output, linear_mse = linear_reg_prediction(file_path, forecasting_period)
        poisson_output, poisson_mse = poisson_reg_prediction(file_path, forecasting_period)
        gaussian_output, gaussian_mse = gaussian_prediction(file_path, forecasting_period)
        arima_output, arima_mse = arima_prediction(file_path, forecasting_period)
        pls_output, pls_mse = pls_prediction(file_path, forecasting_period)
        sarima_output, sarima_mse = sarima_forecasting(file_path, forecasting_period)
        expo_output, expo_mse = exponential_smoothing_forecasting(file_path, forecasting_period)
        random_output, random_mse = random_forest(file_path, forecasting_period)
        elasticnet_output, elasticnet_mse = elastic_net(file_path, forecasting_period)
        lightgbm_output, lightgbm_mse = lightgbm(file_path, forecasting_period)
        catboost_output, catboost_mse = catboost(file_path, forecasting_period)
        ridge_output,ridge_mse=ridge_prediction(file_path, forecasting_period)
        lasso_output,lasso_mse=lasso_prediction(file_path,forecasting_period)
        gbm_output,gbm_mse=gbm_prediction(file_path,forecasting_period)
        pca_output,pca_mse=pca(file_path, forecasting_period)
        ica_output,ica_mse=ica(file_path, forecasting_period)
        svm_output,svm_mse=svm_prediction(file_path, forecasting_period)
        mlp_output,mlp_mse=mlp_prediction(file_path, forecasting_period)


        # Combine all results into a single dataframe
        final_results = pd.merge(prophet_output, regressor_output, on='Date', how='outer')
        final_results = pd.merge(final_results, decision_tree_output, on='Date', how='outer')
        final_results = pd.merge(final_results, kmeans_output, on='Date', how='outer')
        final_results = pd.merge(final_results, logreg_output, on='Date', how='outer')
        final_results = pd.merge(final_results, svr_output, on='Date', how='outer')
        final_results = pd.merge(final_results, ard_output, on='Date', how='outer')
        final_results = pd.merge(final_results, stack_output, on='Date', how='outer')
        final_results = pd.merge(final_results, linear_output, on='Date', how='outer')
        final_results = pd.merge(final_results, poisson_output, on='Date', how='outer')
        final_results = pd.merge(final_results, gaussian_output, on='Date', how='outer')
        final_results = pd.merge(final_results, arima_output, on='Date', how='outer')
        final_results = pd.merge(final_results, pls_output, on='Date', how='outer')
        final_results = pd.merge(final_results, sarima_output, on='Date', how='outer')
        final_results = pd.merge(final_results, expo_output, on='Date', how='outer')
        final_results = pd.merge(final_results, random_output, on='Date', how='outer')
        final_results = pd.merge(final_results, elasticnet_output, on='Date', how='outer')
        final_results = pd.merge(final_results, lightgbm_output, on='Date', how='outer')
        final_results = pd.merge(final_results, catboost_output, on='Date', how='outer')
        final_results = pd.merge(final_results, ridge_output, on='Date', how='outer')
        final_results = pd.merge(final_results, lasso_output, on='Date', how='outer')
        final_results = pd.merge(final_results, gbm_output, on='Date', how='outer')
        final_results = pd.merge(final_results, pca_output, on='Date', how='outer')
        final_results = pd.merge(final_results, ica_output, on='Date', how='outer')
        final_results = pd.merge(final_results, svm_output, on='Date', how='outer')
        final_results = pd.merge(final_results, mlp_output, on='Date', how='outer')
      


        final_results = final_results.round(0)

        # Display results in a single table
        st.write("Stock Predictions:")
        st.write(final_results.set_index('Date'))

 # Create a DataFrame to store MSE values
        mse_data = pd.DataFrame({
            'Model': ['Prophet', 'Regressor', 'Decision Tree', 'K-Means', 'Logistic Regression',
                      'ARD', 'Stacked Model', 'Linear Regression', 'Poisson Regression',
                      'Gaussian', 'ARIMA', 'PLS', 'SARIMA', 'EXPO', 'RandomForest', 'ElasticNet', 'CatBoost','Ridge','Lasso','Lightgbm','SVR','GBM','PCA','ICA','SVM','MLP'],
            'MSE': [prophet_mse, regressor_mse, decision_tree_mse, kmeans_mse, logreg_mse, ard_mse,
                    stack_mse, linear_mse, poisson_mse, gaussian_mse, arima_mse, pls_mse, sarima_mse, expo_mse,
                    random_mse, elasticnet_mse, catboost_mse,ridge_mse,lasso_mse,lightgbm_mse,svr_mse,gbm_mse,pca_mse,ica_mse,svm_mse,mlp_mse]
        })
        st.write(mse_data.set_index('Model'))

        # Sort the MSE values in ascending order
        sorted_mse_data = mse_data.sort_values(by='MSE', ascending=True)

        # Get the top 5 models with the lowest MSE values
        best_models = sorted_mse_data.head(5)

        # Display the 5 best models with their MSE values
        st.write("Top 5 Models with the Lowest MSE:")
        st.write(best_models.set_index('Model'))


        # Plot MSE values
        st.write("Mean Squared Error (MSE) Chart:")
        mse_chart = st.line_chart(mse_data.set_index('Model'))

if __name__ == '__main__':
    main()



