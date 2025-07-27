from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from data_loader import get_firestore_docs
from data_offline import filtered_data

class EnhancedProphetModel:
    def __init__(self, 
                 changepoint_prior_scale=0.1,
                 seasonality_prior_scale=10,
                 holidays_prior_scale=10,
                 seasonality_mode='multiplicative',
                 interval_width=0.8,
                 mcmc_samples=0):
        """
        Enhanced Prophet model with better parameter tuning
        
        Args:
            changepoint_prior_scale: Controls flexibility of trend changes (higher = more flexible)
            seasonality_prior_scale: Controls strength of seasonality (higher = stronger seasonality)
            holidays_prior_scale: Controls strength of holiday effects
            seasonality_mode: 'additive' or 'multiplicative' seasonality
            interval_width: Width of uncertainty intervals
            mcmc_samples: Number of MCMC samples for uncertainty estimation (0 = MAP estimation)
        """
        self.model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            seasonality_mode=seasonality_mode,
            interval_width=interval_width,
            mcmc_samples=mcmc_samples,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        self.df = None
        self.is_fitted = False
        
    def prepare_data(self, data):
        """
        Enhanced data preparation with daily aggregation and outlier detection
        """
        df = pd.DataFrame(data=data)
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        
        # Rename columns for Prophet
        df = df.rename(columns={
            "consume": "y",
            "time": "ds"
        })
        
        # Sort by date
        df = df.sort_values('ds').reset_index(drop=True)
        
        # Aggregate to daily level - sum consumption per day
        df['date'] = df['ds'].dt.date
        daily_df = df.groupby('date').agg({
            'y': 'sum',  # Sum daily consumption
            'ds': 'first'  # Keep first timestamp of the day
        }).reset_index()
        
        # Use only the date part for daily forecasting
        daily_df['ds'] = pd.to_datetime(daily_df['date'])
        daily_df = daily_df.drop('date', axis=1)
        
        # Sort by date
        daily_df = daily_df.sort_values('ds').reset_index(drop=True)
        
        # Handle missing values
        daily_df['y'] = daily_df['y'].fillna(daily_df['y'].median())
        
        # Outlier detection and handling using IQR method
        Q1 = daily_df['y'].quantile(0.25)
        Q3 = daily_df['y'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers instead of removing them to preserve data continuity
        daily_df['y'] = daily_df['y'].clip(lower=lower_bound, upper=upper_bound)
        
        # Add custom regressors for daily patterns
        daily_df = self._add_custom_features(daily_df)
        
        print(f"Daily data prepared: {len(daily_df)} days from {daily_df['ds'].min().date()} to {daily_df['ds'].max().date()}")
        print(f"Daily consumption range: {daily_df['y'].min():.2f} to {daily_df['y'].max():.2f}")
        print(f"Average daily consumption: {daily_df['y'].mean():.2f}")
        
        return daily_df
    
    def _add_custom_features(self, df):
        """
        Add custom features for daily patterns
        """
        # Day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df['ds'].dt.dayofweek
        
        # Month (for seasonal patterns)
        df['month'] = df['ds'].dt.month
        
        # Quarter
        df['quarter'] = df['ds'].dt.quarter
        
        # Is weekend
        df['is_weekend'] = (df['ds'].dt.dayofweek >= 5).astype(int)
        
        # Is weekday
        df['is_weekday'] = (df['ds'].dt.dayofweek < 5).astype(int)
        
        # Day of month
        df['day_of_month'] = df['ds'].dt.day
        
        # Days since start (time-based trend)
        df['days_since_start'] = (df['ds'] - df['ds'].min()).dt.days
        
        return df
    
    def add_custom_seasonalities(self):
        """
        Add custom seasonalities for daily forecasting
        """
        # Weekly seasonality (7-day cycle) - most important for daily data
        self.model.add_seasonality(name='weekly', period=7, fourier_order=10)
        
        # Monthly seasonality (30.44-day cycle)
        self.model.add_seasonality(name='monthly', period=30.44, fourier_order=8)
        
        # Quarterly seasonality (91.25-day cycle)
        self.model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
        
        # Bi-weekly seasonality (14-day cycle)
        self.model.add_seasonality(name='biweekly', period=14, fourier_order=6)
    
    def add_regressors(self, df):
        """
        Add external regressors for daily consumption patterns
        """
        # Add day of week patterns
        self.model.add_regressor('day_of_week')
        self.model.add_regressor('is_weekend')
        self.model.add_regressor('is_weekday')
        self.model.add_regressor('days_since_start')
        
        return df
    
    def fit(self, data):
        """
        Fit the enhanced Prophet model
        """
        self.df = self.prepare_data(data)
        
        # Add custom seasonalities
        self.add_custom_seasonalities()
        
        # Add regressors
        self.df = self.add_regressors(self.df)
        
        # Fit the model
        self.model.fit(self.df)
        self.is_fitted = True
        
        print("Model fitted successfully!")
        
        # Print changepoints detected by Prophet
        changepoints = self.model.changepoints
        if len(changepoints) > 0:
            print(f"Detected {len(changepoints)} trend changepoints:")
            for cp in changepoints[-5:]:  # Show last 5 changepoints
                print(f"  - {cp}")
    
    def predict(self, periods):
        """
        Generate daily forecasts
        
        Args:
            periods (int): Number of future days to forecast
            
        Returns:
            pandas.DataFrame: Forecast results for future days only
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Create future dataframe for daily predictions
            future = self.model.make_future_dataframe(periods=periods, freq='D')
            
            # Add the same custom features for future periods
            future['day_of_week'] = future['ds'].dt.dayofweek
            future['month'] = future['ds'].dt.month
            future['quarter'] = future['ds'].dt.quarter
            future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)
            future['is_weekday'] = (future['ds'].dt.dayofweek < 5).astype(int)
            future['day_of_month'] = future['ds'].dt.day
            future['days_since_start'] = (future['ds'] - self.df['ds'].min()).dt.days
            
            # Generate forecast
            forecast = self.model.predict(future)
            
            # Get only future predictions
            original_data_length = len(self.df)
            future_forecast = forecast.iloc[original_data_length:].reset_index(drop=True)
            
            return future_forecast
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None
    
    def cross_validate_model(self, initial='60 days', period='14 days', horizon='7 days'):
        """
        Perform time series cross-validation for daily forecasting
        """
        from prophet.diagnostics import cross_validation, performance_metrics
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before cross-validation")
        
        try:
            df_cv = cross_validation(self.model, initial=initial, period=period, horizon=horizon)
            df_performance = performance_metrics(df_cv)
            
            print("Cross-validation performance (daily forecasting):")
            print(df_performance[['horizon', 'mape', 'rmse', 'mae']].round(4))
            
            return df_cv, df_performance
            
        except Exception as e:
            print(f"Error in cross-validation: {e}")
            return None, None
    
    def plot_forecast(self, forecast, periods):
        """
        Plot the daily forecast with components
        """
        # Plot main forecast
        fig1 = self.model.plot(forecast)
        plt.title(f'Daily Energy Consumption Forecast - {periods} days ahead')
        plt.xlabel('Date')
        plt.ylabel('Daily Consumption')
        plt.show()
        
        # Plot components
        fig2 = self.model.plot_components(forecast)
        plt.show()
    
    def get_model_summary(self):
        """
        Get summary statistics of the model
        """
        if not self.is_fitted:
            return "Model not fitted yet"
        
        summary = {
            'data_points': len(self.df),
            'date_range': f"{self.df['ds'].min()} to {self.df['ds'].max()}",
            'consumption_stats': {
                'mean': self.df['y'].mean(),
                'std': self.df['y'].std(),
                'min': self.df['y'].min(),
                'max': self.df['y'].max()
            },
            'changepoints_detected': len(self.model.changepoints)
        }
        
        return summary

# Usage example with improved parameters
def create_enhanced_model():
    """
    Create and train an enhanced Prophet model
    """
    # Load data
    data = filtered_data  # or get_firestore_docs()
    
    # Create enhanced model with optimized parameters
    enhanced_model = EnhancedProphetModel(
        changepoint_prior_scale=0.05,  # More conservative trend changes
        seasonality_prior_scale=15,    # Stronger seasonality
        seasonality_mode='multiplicative',  # Better for energy consumption
        interval_width=0.95  # Wider confidence intervals
    )
    
    # Fit the model
    enhanced_model.fit(data)
    
    # Print model summary
    print("\nModel Summary:")
    summary = enhanced_model.get_model_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    return enhanced_model

def model_predict(periods, enhanced_model=None):
    """
    Generate daily forecasts using the enhanced model
    
    Args:
        periods (int): Number of future days to forecast
        enhanced_model: Pre-trained enhanced model (optional)
        
    Returns:
        pandas.DataFrame: Forecast results for future days only
    """
    if enhanced_model is None:
        enhanced_model = create_enhanced_model()
    
    forecast = enhanced_model.predict(periods)
    
    if forecast is not None:
        # Display key forecast metrics
        print(f"\nDaily Forecast Summary for next {periods} days:")
        print(f"Predicted mean daily consumption: {forecast['yhat'].mean():.2f}")
        print(f"Prediction range: {forecast['yhat'].min():.2f} to {forecast['yhat'].max():.2f}")
        print(f"Confidence interval width: {(forecast['yhat_upper'] - forecast['yhat_lower']).mean():.2f}")
        
        # Show trend direction
        trend_change = forecast['yhat'].iloc[-1] - forecast['yhat'].iloc[0]
        trend_direction = "increasing" if trend_change > 0 else "decreasing"
        print(f"Overall trend: {trend_direction} ({trend_change:+.2f})")
        
        # Show weekday vs weekend predictions
        forecast_with_weekday = forecast.copy()
        forecast_with_weekday['day_of_week'] = pd.to_datetime(forecast['ds']).dt.dayofweek
        forecast_with_weekday['is_weekend'] = forecast_with_weekday['day_of_week'] >= 5
        
        weekday_avg = forecast_with_weekday[~forecast_with_weekday['is_weekend']]['yhat'].mean()
        weekend_avg = forecast_with_weekday[forecast_with_weekday['is_weekend']]['yhat'].mean()
        
        print(f"Average weekday consumption: {weekday_avg:.2f}")
        print(f"Average weekend consumption: {weekend_avg:.2f}")
        print(f"Weekend vs Weekday difference: {weekend_avg - weekday_avg:+.2f}")
    
    return forecast

# Initialize the enhanced model
if __name__ == "__main__":
    enhanced_model = create_enhanced_model()
    
    # Example prediction for 7 days ahead
    forecast = model_predict(7, enhanced_model)  # 7 days ahead
    
    # Optional: Plot the forecast
    if forecast is not None:
        enhanced_model.plot_forecast(
            enhanced_model.model.predict(
                enhanced_model.model.make_future_dataframe(periods=7, freq='D')
            ), 
            7
        )