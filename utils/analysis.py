import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller

# Try to import Prophet but provide fallback if not available
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not available. Using other models as fallback.")

def generate_forecast(series, forecast_period, freq='M', model_type='SARIMAX', confidence_level=95):
    """Generate forecast using SARIMAX as ARIMA replacement"""
    try:
        # Basic validation
        if len(series) < 12:
            raise ValueError("Insufficient data (minimum 12 periods required)")
        
        if series.isnull().any():
            series = series.interpolate()

        # Seasonal configuration
        seasonal_periods = 12 if freq in ['M', 'MS'] else 4 if freq in ['Q', 'QS'] else 1
        has_seasonality = len(series) > 2 * seasonal_periods

        # Model selection
        if model_type == 'SARIMAX':
            # SARIMAX configuration (1,1,1)x(1,1,1,12) as default
            if has_seasonality:
                model = SARIMAX(series,
                              order=(1,1,1),
                              seasonal_order=(1,1,1,seasonal_periods),
                              enforce_stationarity=False,
                              enforce_invertibility=False)
            else:
                model = SARIMAX(series,
                              order=(1,1,1),
                              enforce_stationarity=False,
                              enforce_invertibility=False)
            
            fitted_model = model.fit(disp=False)
            forecast = fitted_model.get_forecast(steps=forecast_period)
            forecast_values = forecast.predicted_mean
            conf_int = forecast.conf_int(alpha=1-confidence_level/100)
            lower, upper = conf_int.iloc[:,0], conf_int.iloc[:,1]

        elif model_type == 'Prophet' and PROPHET_AVAILABLE:
            prophet_df = pd.DataFrame({
                'ds': series.index,
                'y': series.values
            })
            model = Prophet(
                yearly_seasonality=has_seasonality,
                interval_width=confidence_level/100
            )
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=forecast_period, freq=freq)
            forecast_df = model.predict(future)
            forecast_values = forecast_df['yhat'][-forecast_period:].values
            lower = forecast_df['yhat_lower'][-forecast_period:].values
            upper = forecast_df['yhat_upper'][-forecast_period:].values
            
        else:  # Exponential Smoothing fallback
            if model_type == 'Prophet' and not PROPHET_AVAILABLE:
                print("Warning: Prophet not available, using Exponential Smoothing")
                
            if has_seasonality:
                model = ExponentialSmoothing(
                    series,
                    trend='add',
                    seasonal='add',
                    seasonal_periods=seasonal_periods,
                    damped_trend=True
                ).fit()
            else:
                model = ExponentialSmoothing(
                    series,
                    trend='add',
                    seasonal=None,
                    damped_trend=True
                ).fit()
            forecast_values = model.forecast(forecast_period)
            
            # Calculate confidence intervals
            std_error = np.std(series - model.fittedvalues)
            z_score = {'95': 1.96, '90': 1.645, '80': 1.28}.get(str(confidence_level), 1.96
            lower = forecast_values - z_score * std_error
            upper = forecast_values + z_score * std_error

        # Cross-validation for metrics
        train_size = max(int(len(series) * 0.8), 12)
        train, test = series[:train_size], series[train_size:]
        
        try:
            if model_type == 'SARIMAX':
                val_model = SARIMAX(train,
                                  order=(1,1,1),
                                  seasonal_order=(1,1,1,seasonal_periods) if has_seasonality else (0,0,0,0),
                                  enforce_stationarity=False)
                val_fitted = val_model.fit(disp=False)
                val_forecast = val_fitted.get_forecast(steps=len(test))
                val_pred = val_forecast.predicted_mean
                
            elif model_type == 'Prophet' and PROPHET_AVAILABLE:
                val_prophet_df = pd.DataFrame({
                    'ds': train.index,
                    'y': train.values
                })
                val_model = Prophet(yearly_seasonality=has_seasonality)
                val_model.fit(val_prophet_df)
                val_future = val_model.make_future_dataframe(periods=len(test), freq=freq)
                val_pred = val_model.predict(val_future)['yhat'][-len(test):].values
                
            else:
                val_model = ExponentialSmoothing(
                    train,
                    trend='add',
                    seasonal='add' if has_seasonality else None,
                    seasonal_periods=seasonal_periods if has_seasonality else None
                ).fit()
                val_pred = val_model.forecast(len(test))
            
            # Calculate metrics
            mae = mean_absolute_error(test, val_pred)
            rmse = np.sqrt(mean_squared_error(test, val_pred))
            r2 = r2_score(test, val_pred)
            
        except Exception as e:
            print(f"Validation error: {str(e)}")
            mae, rmse, r2 = None, None, None
        
        return {
            'forecast': forecast_values,
            'lower': lower,
            'upper': upper,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'model_used': 'Prophet' if model_type == 'Prophet' and PROPHET_AVAILABLE else model_type
        }
        
    except Exception as e:
        print(f"Forecast error: {str(e)}")
        return None

def analyze_time_series(df, date_col, value_col, forecast_period=12, model_type='SARIMAX', confidence_interval=95):
    """Complete time series analysis function"""
    results = {
        'mean': None,
        'std': None,
        'trend': None,
        'has_seasonality': False,
        'stationarity': {
            'test_statistic': None,
            'p_value': None,
            'critical_values': {
                '1%': None,
                '5%': None,
                '10%': None
            },
            'is_stationary': None,
            'error': None
        },
        'decomposition_fig': None,
        'forecast_fig': None,
        'diagnostics_fig': None,
        'forecast_df': None,
        'mae': None,
        'rmse': None,
        'r2': None,
        'model_used': None
    }
    
    try:
        # Pre-processing
        df = df.sort_values(date_col).dropna(subset=[date_col, value_col])
        series = df.set_index(date_col)[value_col].astype(float)
        freq = pd.infer_freq(df[date_col]) or 'M'
        
        # Basic statistics
        results.update({
            'mean': series.mean(),
            'std': series.std(),
            'trend': 'Increasing' if series.pct_change().mean() > 0 else 'Decreasing'
        })
        
        # Stationarity test
        try:
            adf_result = adfuller(series.dropna())
            if len(adf_result) >= 5:
                results['stationarity'].update({
                    'test_statistic': adf_result[0],
                    'p_value': adf_result[1],
                    'critical_values': {
                        '1%': adf_result[4]['1%'],
                        '5%': adf_result[4]['5%'],
                        '10%': adf_result[4]['10%']
                    },
                    'is_stationary': adf_result[1] <= 0.05,
                    'error': None
                })
        except Exception as e:
            results['stationarity']['error'] = f"Stationarity test error: {str(e)}"
        
        # Seasonal decomposition
        if len(series) >= 24:
            try:
                decomposition = seasonal_decompose(
                    series.asfreq(freq),
                    model='additive',
                    period=12 if freq in ['M', 'MS'] else 4
                )
                seasonal_ratio = decomposition.seasonal.std() / decomposition.observed.std()
                results['has_seasonality'] = seasonal_ratio > 0.1
                
                # Decomposition plot
                decomp_df = pd.DataFrame({
                    'Observed': decomposition.observed,
                    'Trend': decomposition.trend,
                    'Seasonal': decomposition.seasonal,
                    'Residual': decomposition.resid
                }).reset_index()
                
                results['decomposition_fig'] = px.line(
                    decomp_df.melt(id_vars=[date_col], var_name='Component'),
                    x=date_col,
                    y='value',
                    color='Component',
                    facet_row='Component',
                    height=800,
                    title="Time Series Decomposition"
                )
            except Exception as e:
                print(f"Decomposition error: {str(e)}")
        
        # Forecasting
        forecast_result = generate_forecast(
            series, 
            forecast_period, 
            freq, 
            model_type, 
            confidence_interval
        )
        
        if forecast_result:
            results.update({
                'mae': forecast_result['mae'],
                'rmse': forecast_result['rmse'],
                'r2': forecast_result['r2'],
                'model_used': forecast_result['model_used']
            })
            
            # Forecast plot data
            last_date = df[date_col].iloc[-1]
            future_dates = pd.date_range(
                start=last_date,
                periods=forecast_period+1,
                freq=freq
            )[1:]
            
            forecast_df = pd.DataFrame({
                date_col: future_dates,
                value_col: forecast_result['forecast'],
                'lower': forecast_result['lower'],
                'upper': forecast_result['upper'],
                'type': 'Forecast'
            })
            
            results['forecast_df'] = forecast_df
            
            # Create forecast plot
            original_df = df[[date_col, value_col]].copy()
            original_df['type'] = 'Historical'
            
            fig = px.line(
                original_df,
                x=date_col,
                y=value_col,
                color='type',
                title=f"Time Series Forecast ({forecast_result['model_used']})"
            )
            
            fig.add_scatter(
                x=forecast_df[date_col],
                y=forecast_df[value_col],
                mode='lines',
                name='Forecast',
                line=dict(dash='dot', color='red')
            )
            
            if forecast_result['lower'] is not None:
                fig.add_scatter(
                    x=forecast_df[date_col],
                    y=forecast_df['lower'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                )
                
                fig.add_scatter(
                    x=forecast_df[date_col],
                    y=forecast_df['upper'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.2)',
                    name=f'{confidence_interval}% Confidence'
                )
            
            results['forecast_fig'] = fig
    
    except Exception as e:
        print(f"Analysis error: {str(e)}")
    
    return results