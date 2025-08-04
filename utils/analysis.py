import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
import streamlit as st

# Try to import Prophet but provide fallback if not available
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("Prophet not available. Using other models as fallback.")

def generate_forecast(series, forecast_period, freq='M', model_type='SARIMAX', confidence_level=95):
    """Generate forecast using selected model"""
    try:
        # Basic validation
        if len(series) < 12:
            raise ValueError("Insufficient data (minimum 12 periods required)")
        
        if series.isnull().any():
            series = series.interpolate()

        # Seasonal configuration
        seasonal_periods = 12 if freq in ['M', 'MS'] else 4 if freq in ['Q', 'QS'] else 1
        has_seasonality = len(series) > 2 * seasonal_periods

        # Model implementation
        if model_type == 'SARIMAX':
            if has_seasonality:
                model = SARIMAX(series,
                              order=(1,1,1),
                              seasonal_order=(1,1,1,seasonal_periods),
                              enforce_stationarity=False)
            else:
                model = SARIMAX(series,
                              order=(1,1,1),
                              enforce_stationarity=False)
            
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
                st.warning("Prophet not available, using Exponential Smoothing")
                
            model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal='add' if has_seasonality else None,
                seasonal_periods=seasonal_periods if has_seasonality else None
            ).fit()
            forecast_values = model.forecast(forecast_period)
            
            # Calculate confidence intervals
            std_error = np.std(series - model.fittedvalues)
            z_score = {'95': 1.96, '90': 1.645, '80': 1.28}.get(str(confidence_level), 1.96)
            lower = forecast_values - z_score * std_error
            upper = forecast_values + z_score * std_error

        # Cross-validation for metrics
        train_size = max(int(len(series) * 0.8), 12)
        train, test = series[:train_size], series[train_size:]
        
        try:
            if model_type == 'SARIMAX':
                val_model = SARIMAX(train,
                                  order=(1,1,1),
                                  seasonal_order=(1,1,1,seasonal_periods) if has_seasonality else (0,0,0,0))
                val_fitted = val_model.fit(disp=False)
                val_pred = val_fitted.get_forecast(steps=len(test)).predicted_mean
                
            elif model_type == 'Prophet' and PROPHET_AVAILABLE:
                val_model = Prophet(yearly_seasonality=has_seasonality)
                val_model.fit(pd.DataFrame({'ds': train.index, 'y': train.values}))
                val_pred = val_model.predict(
                    val_model.make_future_dataframe(periods=len(test), freq=freq)
                )['yhat'][-len(test):].values
                
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
            st.warning(f"Validation error: {str(e)}")
            mae, rmse, r2 = None, None, None
        
        return {
            'forecast': forecast_values,
            'lower': lower,
            'upper': upper,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'model_used': model_type
        }
        
    except Exception as e:
        st.error(f"Forecast error: {str(e)}")
        return None

def analyze_time_series(df, date_col, value_col, forecast_period=12, model_type='SARIMAX', confidence_interval=95):
    """Complete time series analysis function"""
    results = {
        'mean': None,
        'std': None,
        'trend': None,
        'has_seasonality': False,
        'stationarity': None,
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
            results['stationarity'] = {
                'test_statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] <= 0.05
            }
        except Exception as e:
            st.warning(f"Stationarity test error: {str(e)}")
        
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
                
                # Create decomposition plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df[date_col], y=decomposition.observed, name='Observed'))
                fig.add_trace(go.Scatter(x=df[date_col], y=decomposition.trend, name='Trend'))
                fig.add_trace(go.Scatter(x=df[date_col], y=decomposition.seasonal, name='Seasonal'))
                fig.add_trace(go.Scatter(x=df[date_col], y=decomposition.resid, name='Residual'))
                results['decomposition_fig'] = fig
                
            except Exception as e:
                st.warning(f"Decomposition error: {str(e)}")
        
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
            
            # Prepare forecast data
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
                'upper': forecast_result['upper']
            })
            results['forecast_df'] = forecast_df
            
            # Create forecast plot
            fig = px.line(df, x=date_col, y=value_col, title="Historical Data")
            fig.add_scatter(
                x=forecast_df[date_col],
                y=forecast_df[value_col],
                mode='lines',
                name='Forecast',
                line=dict(color='red', dash='dot')
            )
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
                name=f'{confidence_interval}% CI'
            )
            results['forecast_fig'] = fig
    
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
    
    return results