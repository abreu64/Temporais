import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import load_csv_data
from utils.analysis import analyze_time_series
import datetime
from typing import Tuple, Optional

# Page configuration
st.set_page_config(
    page_title="Advanced Time Series Analysis",
    page_icon="📈",
    layout="wide"
)

def validate_date_range(df: pd.DataFrame, date_col: str, min_date: datetime.date, max_date: datetime.date) -> pd.DataFrame:
    """Validate and filter data based on date range."""
    filtered_df = df[(df[date_col] >= pd.to_datetime(min_date)) & 
                    (df[date_col] <= pd.to_datetime(max_date))]
    
    if filtered_df.empty:
        raise ValueError(f"No data available for the selected period ({min_date} to {max_date})")
    
    return filtered_df

def display_analysis_results(analysis_results: dict, df: pd.DataFrame, date_col: str, value_col: str) -> None:
    """Display all analysis results in tabs."""
    st.header("📈 Data Visualization")
    tab1, tab2, tab3, tab4 = st.tabs(["Time Series", "Decomposition", "Forecast", "Diagnostics"])
    
    # Tab 1: Time Series
    with tab1:
        fig = px.line(
            df, 
            x=date_col, 
            y=value_col,
            title=f"Time Series - {value_col}",
            labels={value_col: "Value", date_col: "Date"},
            template="plotly_white"
        )
        fig.update_layout(hovermode="x unified", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("🔍 View Raw Data"):
            st.dataframe(df.sort_values(date_col, ascending=False))
    
    # Tab 2: Decomposition
    with tab2:
        if analysis_results.get('decomposition_fig'):
            st.plotly_chart(analysis_results['decomposition_fig'], use_container_width=True)
            st.markdown("""
            **Decomposition Legend:**
            - **Observed**: Original data
            - **Trend**: Long-term pattern
            - **Seasonal**: Repeating patterns
            - **Residual**: Unexplained variation
            """)
        else:
            st.warning("""
            Decomposition not available. Possible reasons:
            - Insufficient data (minimum 24 periods recommended)
            - No identifiable temporal pattern
            """)
    
    # Tab 3: Forecast
    with tab3:
        if analysis_results.get('forecast_fig'):
            st.plotly_chart(analysis_results['forecast_fig'], use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Absolute Error (MAE)", f"{analysis_results.get('mae', 0):.2f}")
            col2.metric("Root Mean Squared Error (RMSE)", f"{analysis_results.get('rmse', 0):.2f}")
            col3.metric("R² Score", f"{analysis_results.get('r2', 0):.4f}")
            
            with st.expander("🔍 Forecast Details"):
                st.dataframe(analysis_results.get('forecast_df', pd.DataFrame()))
        else:
            st.error("""
            Forecast could not be generated. Possible reasons:
            - Insufficient data (minimum 12 periods required)
            - No identifiable pattern
            - Issue with selected model
            """)
    
    # Tab 4: Diagnostics
    with tab4:
        diagnostics = analysis_results.get('diagnostics_fig')
        if diagnostics:
            st.plotly_chart(diagnostics, use_container_width=True)
            st.markdown("""
            **Diagnostics Interpretation:**
            - **ACF/PACF**: Show correlation between observations at different lags
            - **Residuals**: Should be random and uncorrelated
            - **QQ-Plot**: Checks residual normality
            """)
    
    # Metrics and Statistics
    st.header("📊 Metrics and Statistics")
    
    cols = st.columns(4)
    metrics = [
        ("Mean", f"{analysis_results.get('mean', 0):.2f}"),
        ("Std Dev", f"{analysis_results.get('std', 0):.2f}"),
        ("Trend", analysis_results.get('trend', 'Undetermined')),
        ("Seasonality", "Yes" if analysis_results.get('has_seasonality', False) else "No")
    ]
    
    for col, (name, value) in zip(cols, metrics):
        col.metric(name, value)
    
    # Additional Analysis
    st.header("🔍 Detailed Analysis")
    with st.expander("📌 Stationarity Analysis"):
        if analysis_results.get('stationarity'):
            st.write(f"""
            **Augmented Dickey-Fuller Test:**
            - Test Statistic: {analysis_results['stationarity']['test_statistic']:.4f}
            - Critical Value (5%): {analysis_results['stationarity']['critical_values']['5%']:.4f}
            - p-value: {analysis_results['stationarity']['p_value']:.4f}
            """)
            if analysis_results['stationarity']['is_stationary']:
                st.success("Data is stationary")
            else:
                st.warning("Data is not stationary")
        else:
            st.warning("Stationarity analysis not available")

# Sidebar - Upload and settings
with st.sidebar:
    st.header("📤 Upload & Settings")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="File must contain at least one date/datetime column and one numeric column"
    )
    
    if uploaded_file:
        # Advanced settings
        st.header("⚙️ Analysis Settings")
        
        # Forecast period
        forecast_period = st.slider(
            "Forecast Period",
            min_value=1,
            max_value=36,
            value=6,
            help="Select how many periods to forecast"
        )
        
        # Confidence interval
        confidence_interval = st.slider(
            "Confidence Interval",
            min_value=80,
            max_value=99,
            value=95,
            step=1,
            help="Confidence level for predictions"
        )
        
        # Model selection
        model_type = st.selectbox(
            "Modelo de Previsão",
            options=["SARIMAX", "Prophet", "Exponencial"],
            index=0,
            help="Selecione o modelo de previsão"
        )
        
        # Date filter
        st.header("🗓 Date Filter")
        min_date = st.date_input(
            "Start Date",
            value=datetime.date(2020, 1, 1),
            help="Select start date for analysis"
        )
        max_date = st.date_input(
            "End Date",
            value=datetime.date.today(),
            help="Select end date for analysis"
        )

# Main page
st.title("📊 Advanced Time Series Analysis")
st.markdown("""
Upload a CSV file containing temporal data for complete analysis.
The file must contain at least:
- One date column in YYYY-MM-DD format
- One numeric value column
""")

# Data processing and analysis
if not uploaded_file:
    st.info("👈 Please upload a CSV file in the sidebar to begin")
    st.stop()

try:
    # Load and validate data
    data = load_csv_data(uploaded_file)
    if not data or len(data) != 3:
        raise ValueError("Unexpected return format from load_csv_data")
    
    df, date_col, value_col = data
    
    # Apply date filter
    df = validate_date_range(df, date_col, min_date, max_date)
    
    # Show data summary
    with st.expander("📋 Dataset Summary", expanded=True):
        st.write(f"**Total Rows:** {len(df)}")
        st.write(f"**Date Range:** {df[date_col].min().date()} to {df[date_col].max().date()}")
        st.write(f"**Frequency:** {pd.infer_freq(df[date_col]) or 'Irregular'}")
    
    # Analyze data
    with st.spinner("🔍 Analyzing time series data..."):
        analysis_results = analyze_time_series(
            df, 
            date_col, 
            value_col, 
            forecast_period=forecast_period,
            model_type=model_type,
            confidence_interval=confidence_interval
        )
    
    # Display results
    display_analysis_results(analysis_results, df, date_col, value_col)

except ValueError as e:
    st.error(f"⚠️ Validation Error: {str(e)}")
    st.info("Please check your data format and try again")
except Exception as e:
    st.error(f"🚨 Processing Error: {str(e)}")
    st.info("""
    Troubleshooting Tips:
    1. Verify the file has at least one date and one numeric column
    2. Check date format is YYYY-MM-DD
    3. Ensure numeric values don't contain text/special characters
    4. Make sure there's enough data for selected period
    """)