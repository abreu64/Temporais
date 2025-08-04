import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller

# Try to import Prophet but provide fallback if not available
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet não disponível. Usando outros modelos como fallback.")

def generate_forecast(series, forecast_period, freq='M', model_type='AutoARIMA', confidence_level=95):
    """Gera previsão robusta com tratamento de erros e múltiplos modelos"""
    try:
        # Verificação básica de dados
        if len(series) < 12:
            raise ValueError("Dados insuficientes (mínimo 12 períodos)")
        
        if series.isnull().any():
            series = series.interpolate()
        
        # Configuração sazonal
        seasonal_periods = 12 if freq in ['M', 'MS'] else 4 if freq in ['Q', 'QS'] else 1
        has_seasonality = len(series) > 2 * seasonal_periods
        
        # Seleção do modelo
        if model_type == 'AutoARIMA':
            if has_seasonality:
                model = ARIMA(series, order=(1,1,1), seasonal_order=(1,1,1,12))
            else:
                model = ARIMA(series, order=(1,1,1))
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=forecast_period)
            lower, upper = None, None
            
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
            forecast = forecast_df['yhat'][-forecast_period:].values
            lower = forecast_df['yhat_lower'][-forecast_period:].values
            upper = forecast_df['yhat_upper'][-forecast_period:].values
            
        else:  # Exponential Smoothing (padrão)
            if model_type == 'Prophet' and not PROPHET_AVAILABLE:
                print("Aviso: Prophet não disponível, usando Exponential Smoothing como alternativa")
                
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
            forecast = model.forecast(forecast_period)
            lower, upper = None, None
        
        # Validação cruzada para métricas
        train_size = max(int(len(series) * 0.8), 12)
        train, test = series[:train_size], series[train_size:]
        
        try:
            if model_type == 'AutoARIMA':
                if has_seasonality:
                    val_model = ARIMA(train, order=(1,1,1), seasonal_order=(1,1,1,12))
                else:
                    val_model = ARIMA(train, order=(1,1,1))
                val_fitted = val_model.fit()
                val_forecast = val_fitted.forecast(steps=len(test))
                
            elif model_type == 'Prophet' and PROPHET_AVAILABLE:
                val_prophet_df = pd.DataFrame({
                    'ds': train.index,
                    'y': train.values
                })
                val_model = Prophet(yearly_seasonality=has_seasonality)
                val_model.fit(val_prophet_df)
                val_future = val_model.make_future_dataframe(periods=len(test), freq=freq)
                val_forecast = val_model.predict(val_future)['yhat'][-len(test):].values
                
            else:
                val_model = ExponentialSmoothing(
                    train,
                    trend='add',
                    seasonal='add' if has_seasonality else None,
                    seasonal_periods=seasonal_periods if has_seasonality else None
                ).fit()
                val_forecast = val_model.forecast(len(test))
            
            # Cálculo de métricas
            mae = mean_absolute_error(test, val_forecast)
            rmse = np.sqrt(mean_squared_error(test, val_forecast))
            r2 = r2_score(test, val_forecast)
            
            # Intervalo de confiança para modelos não-Prophet
            if model_type != 'Prophet' or not PROPHET_AVAILABLE:
                std_error = np.std(test - val_forecast)
                z_score = {'95': 1.96, '90': 1.645, '80': 1.28}.get(str(confidence_level), 1.96)
                lower = forecast - z_score * std_error
                upper = forecast + z_score * std_error
                
        except Exception as e:
            print(f"Erro na validação: {str(e)}")
            mae, rmse, r2 = None, None, None
            lower, upper = None, None
        
        return {
            'forecast': forecast,
            'lower': lower,
            'upper': upper,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'model_used': 'Prophet' if model_type == 'Prophet' and PROPHET_AVAILABLE else model_type
        }
        
    except Exception as e:
        print(f"Erro na previsão: {str(e)}")
        return None

def analyze_time_series(df, date_col, value_col, forecast_period=12, model_type='AutoARIMA', confidence_interval=95):
    """Função principal de análise com múltiplos modelos e métricas"""
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
        # Pré-processamento seguro
        df = df.sort_values(date_col).dropna(subset=[date_col, value_col])
        series = df.set_index(date_col)[value_col].astype(float)
        freq = pd.infer_freq(df[date_col]) or 'M'
        
        # Estatísticas básicas
        results.update({
            'mean': series.mean(),
            'std': series.std(),
            'trend': 'Crescente' if series.pct_change().mean() > 0 else 'Decrescente'
        })
        
        # Teste de estacionariedade corrigido
        try:
            adf_result = adfuller(series.dropna())
            if len(adf_result) >= 5:  # Verifica se tem todos os componentes
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
            else:
                raise ValueError("Resultado do teste ADF incompleto")
        except Exception as e:
            results['stationarity']['error'] = f"Erro no teste de estacionariedade: {str(e)}"
            print(results['stationarity']['error'])
        
        # Decomposição sazonal (apenas se tiver dados suficientes)
        if len(series) >= 24:
            try:
                decomposition = seasonal_decompose(
                    series.asfreq(freq),
                    model='additive',
                    period=12 if freq in ['M', 'MS'] else 4
                )
                seasonal_ratio = decomposition.seasonal.std() / decomposition.observed.std()
                results['has_seasonality'] = seasonal_ratio > 0.1
                
                # Plot de decomposição
                decomp_df = pd.DataFrame({
                    'Observado': decomposition.observed,
                    'Tendência': decomposition.trend,
                    'Sazonalidade': decomposition.seasonal,
                    'Resíduo': decomposition.resid
                }).reset_index()
                
                results['decomposition_fig'] = px.line(
                    decomp_df.melt(id_vars=[date_col], var_name='Componente'),
                    x=date_col,
                    y='value',
                    color='Componente',
                    facet_row='Componente',
                    height=800,
                    title="Decomposição da Série Temporal"
                )
            except Exception as e:
                print(f"Erro na decomposição: {str(e)}")
        
        # Previsão
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
            
            # Criação do gráfico de previsão
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
                'type': 'Previsão'
            })
            
            results['forecast_df'] = forecast_df
            
            original_df = df[[date_col, value_col]].copy()
            original_df['type'] = 'Original'
            
            fig = px.line(
                original_df,
                x=date_col,
                y=value_col,
                color='type',
                title=f"Série Temporal com Previsão ({forecast_result['model_used']})"
            )
            
            fig.add_scatter(
                x=forecast_df[date_col],
                y=forecast_df[value_col],
                mode='lines',
                name='Previsão',
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
                    name=f'Intervalo {confidence_interval}%'
                )
            
            results['forecast_fig'] = fig
    
    except Exception as e:
        print(f"Erro na análise: {str(e)}")
    
    return results