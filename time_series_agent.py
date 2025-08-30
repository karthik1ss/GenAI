#!/usr/bin/env python3
"""
Time Series Forecasting Agent - Requests Per Second Analysis
Specialized agent for analyzing API traffic patterns using CloudWatch Logs Insights data
and predicting future usage patterns with Lag-Llama model
"""

import json
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from strands import Agent, tool
from strands.models import BedrockModel
from strands_tools import memory
import os
import warnings
warnings.filterwarnings('ignore')

# Try importing time series libraries with proper error handling
try:
    import torch
    from lag_llama.gluon.estimator import LagLlamaEstimator
    from gluonts.dataset.pandas import PandasDataset
    from gluonts.dataset.split import split
    HAS_LAG_LLAMA = False
except ImportError:
    print("⚠️  Lag-Llama dependencies not installed. Using statistical forecasting as fallback.")
    HAS_LAG_LLAMA = False
    # Create placeholder classes to avoid NameError
    class PandasDataset:
        pass

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    HAS_STATSMODELS = True
except ImportError:
    print("⚠️  Statsmodels not available. Using simple moving average.")
    HAS_STATSMODELS = False

try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    print("⚠️  Boto3 not available. Some features may be limited.")
    HAS_BOTO3 = False

class TimeSeriesProcessor:
    """Process CloudWatch Logs Insights data into time series for forecasting"""
    
    def __init__(self):
        self.processed_data = {}

    def parse_cloudwatch_logs(self, log_data: List[Dict]) -> pd.DataFrame:
        """
        Parse CloudWatch Logs Insights data and convert to requests per second

        Expected input format from CloudWatch Logs Insights:
        [
            {
                "hour": "2023-10-01 00:00:00",
                "count(*)": 216,
                "application": "amzn1.sellerapps.app.ade9e8f4-0b14-4f0d-ac7d-f6615922e35f",
                "SellerId": "A13YO9XQO12E5U",  # Changed from partyId
                "resourcePath": "/listings/2021-08-01/items/{sellerId}/{sku}"
            }
        ]

        Args:
            log_data: List of dictionaries containing hourly aggregated data
                     from CloudWatch Logs Insights

        Returns:
            DataFrame with timestamp, requests_per_sec, and metadata columns
        """
        parsed_logs = []

        for log_entry in log_data:
            try:
                # Handle CloudWatch Logs Insights format
                if 'hour' in log_entry and 'count(*)' in log_entry:
                    hourly_requests = int(log_entry.get('count(*)', 0))
                    requests_per_sec = round(hourly_requests / 3600, 4)  # Convert to per-second rate

                    parsed_entry = {
                        'timestamp': pd.to_datetime(log_entry['hour']),
                        'application_id': log_entry.get('application', ''),
                        'seller_id': log_entry.get('partyId', ''),
                        'resource_path': log_entry.get('resourcePath', ''),
                        'requests_per_sec': requests_per_sec,
                        'hourly_requests': hourly_requests,
                        'data_source': 'cloudwatch_insights'
                    }
                    
                # Handle individual log entry format (fallback)
                elif 'request_time' in log_entry:
                    timestamp = pd.to_datetime(log_entry['request_time'])
                    
                    parsed_entry = {
                        'timestamp': timestamp,
                        'application_id': log_entry.get('application', ''),
                        'seller_id': log_entry.get('partyId', ''),  # Changed from party_id
                        'resource_path': log_entry.get('resource_path', ''),
                        'requests_per_sec': 1.0 / 3600,  # Single request converted to per-second
                        'hourly_requests': 1,
                        'data_source': 'individual_logs'
                    }
                else:
                    print(f"Unknown log format: {log_entry}")
                    continue
                    
                parsed_logs.append(parsed_entry)

            except Exception as e:
                print(f"Error parsing log entry: {e}")
                print(f"Problematic entry: {log_entry}")
                continue

        if not parsed_logs:
            return pd.DataFrame()

        df = pd.DataFrame(parsed_logs)
        df = df.sort_values('timestamp')

        return df

    def create_time_series(self, df: pd.DataFrame,
                           application_id: str = None,
                           seller_id: str = None,  # Changed from party_id
                           resource_path: str = None,
                           granularity: str = '1H') -> pd.DataFrame:
        """
        Create time series from CloudWatch Logs Insights data

        Args:
            df: DataFrame with parsed log data
            application_id: Optional filter for application ID
            seller_id: Optional filter for seller ID  # Changed from party_id
            resource_path: Optional filter for resource path
            granularity: Time series granularity (default: '1H')

        Returns:
            Time series DataFrame with requests per second aggregated by time period
        """
        # Apply filters
        filtered_df = df.copy()

        if application_id:
            filtered_df = filtered_df[filtered_df['application_id'] == application_id]
        if seller_id:  # Changed from party_id
            filtered_df = filtered_df[filtered_df['seller_id'] == seller_id]
        if resource_path:
            filtered_df = filtered_df[filtered_df['resource_path'] == resource_path]

        if filtered_df.empty:
            return pd.DataFrame()

        # For CloudWatch data, requests_per_sec is already calculated per hour
        # For individual logs, we need to aggregate by time period
        if filtered_df['data_source'].iloc[0] == 'cloudwatch_insights':
            # Data is already aggregated by hour
            time_series = filtered_df[['timestamp', 'requests_per_sec', 'hourly_requests']].copy()
        else:
            # Aggregate individual requests by time period
            time_series = filtered_df.groupby(pd.Grouper(key='timestamp', freq=granularity)).agg({
                'requests_per_sec': 'sum',  # Sum up all requests in the period
                'hourly_requests': 'sum'
            }).reset_index()

        # Fill missing time intervals with 0
        if not time_series.empty:
            time_range = pd.date_range(
                start=time_series['timestamp'].min(),
                end=time_series['timestamp'].max(),
                freq=granularity
            )

            time_series = time_series.set_index('timestamp').reindex(time_range, fill_value=0)
            time_series.index.name = 'timestamp'
            time_series = time_series.reset_index()

        return time_series


class LagLlamaForecaster:
    """Lag-Llama based time series forecasting for requests per second"""
    
    def __init__(self):
        self.model = None
        self.is_fitted = False
        
    def prepare_data_for_lag_llama(self, time_series: pd.DataFrame) -> Union[Any, None]:
        """Prepare requests per second data for Lag-Llama"""
        
        if not HAS_LAG_LLAMA:
            return None
            
        # Use requests_per_sec as the target variable
        ts_data = time_series.set_index('timestamp')['requests_per_sec']
        
        # Create dataset
        dataset = PandasDataset(
            dataframes=ts_data,
            target="requests_per_sec",
            timestamp="timestamp"
        )
        
        return dataset
    
    def train_and_forecast(self, time_series: pd.DataFrame, 
                          forecast_horizon: int = 168) -> Tuple[np.array, Dict]:
        """Train Lag-Llama model and generate forecasts for requests per second"""
        
        values = time_series['requests_per_sec'].values
        
        if len(values) < 5:
            avg_value = np.mean(values) if len(values) > 0 else 0
            forecast = np.full(forecast_horizon, avg_value)
            return forecast, {
                'mean': forecast,
                'lower_bound': forecast * 0.8,
                'upper_bound': forecast * 1.2,
                'model_type': 'Simple Average (Insufficient Data)'
            }
        
        if HAS_LAG_LLAMA and len(values) >= 24:
            try:
                return self._lag_llama_forecast(time_series, forecast_horizon)
            except Exception as e:
                print(f"Lag-Llama forecasting failed: {e}")
        
        if HAS_STATSMODELS and len(values) >= 3:  # Reduced minimum requirement
            try:
                return self._statistical_forecast(values, forecast_horizon)
            except Exception as e:
                print(f"Statistical forecasting failed: {e}")
        
        return self._simple_forecast(values, forecast_horizon)
    
    def _lag_llama_forecast(self, time_series: pd.DataFrame, forecast_horizon: int):
        """Lag-Llama forecasting implementation for requests per second"""
        if not HAS_LAG_LLAMA:
            raise ImportError("Lag-Llama dependencies not available")
            
        dataset = self.prepare_data_for_lag_llama(time_series)
        
        estimator = LagLlamaEstimator(
            prediction_length=forecast_horizon,
            context_length=min(168, len(time_series)),
            scaling=True,
            time_feat=True,
            trainer_kwargs={"max_epochs": 5}
        )
        
        train_data, test_data = split(dataset, offset=-forecast_horizon)
        predictor = estimator.train(train_data)
        forecasts = list(predictor.predict(test_data))
        
        forecast_values = forecasts[0].mean if forecasts else np.zeros(forecast_horizon)
        
        return forecast_values, {
            'mean': forecast_values,
            'lower_bound': forecasts[0].quantile(0.1) if forecasts else forecast_values * 0.8,
            'upper_bound': forecasts[0].quantile(0.9) if forecasts else forecast_values * 1.2,
            'model_type': 'Lag-Llama'
        }
    
    def _statistical_forecast(self, values: np.array, forecast_horizon: int):
        """Statistical forecasting using ARIMA for requests per second - FIXED VERSION"""
        if not HAS_STATSMODELS:
            raise ImportError("Statsmodels not available")
        
        try:
            # Ensure we have enough data for ARIMA
            if len(values) < 3:
                # Fall back to simple forecast if not enough data
                return self._simple_forecast(values, forecast_horizon)
            
            # Create ARIMA model with simpler parameters for small datasets
            if len(values) < 10:
                # Use simpler ARIMA model for small datasets
                model = ARIMA(values, order=(1, 0, 1))
            else:
                model = ARIMA(values, order=(1, 1, 1))
            
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=forecast_horizon)
            
            # Get confidence intervals - FIXED: handle different data types
            forecast_result = fitted_model.get_forecast(steps=forecast_horizon)
            conf_int = forecast_result.conf_int()
            
            # Convert to numpy arrays properly - FIX FOR THE ILOC ERROR
            if hasattr(conf_int, 'iloc'):
                # If it's a DataFrame, use iloc
                lower_bound = conf_int.iloc[:, 0].values
                upper_bound = conf_int.iloc[:, 1].values
            elif hasattr(conf_int, 'values'):
                # If it's already a numpy array with .values attribute
                lower_bound = conf_int.values[:, 0]
                upper_bound = conf_int.values[:, 1]
            else:
                # If it's a plain numpy array
                lower_bound = conf_int[:, 0]
                upper_bound = conf_int[:, 1]
            
            # Ensure forecast is a numpy array
            if hasattr(forecast, 'values'):
                forecast_values = forecast.values
            else:
                forecast_values = np.array(forecast)
            
            # Ensure non-negative forecasts
            forecast_values = np.maximum(forecast_values, 0)
            lower_bound = np.maximum(lower_bound, 0)
            upper_bound = np.maximum(upper_bound, 0)
            
            return forecast_values, {
                'mean': forecast_values,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'model_type': 'ARIMA'
            }
            
        except Exception as e:
            print(f"ARIMA model failed: {e}")
            # Fall back to simple forecast if ARIMA fails
            return self._simple_forecast(values, forecast_horizon)
    
    def _simple_forecast(self, values: np.array, forecast_horizon: int):
        """Simple moving average forecast with trend for requests per second - ENHANCED VERSION"""
        try:
            # Ensure values is a numpy array
            values = np.array(values)
            
            if len(values) == 0:
                # If no values, return zeros
                forecast = np.zeros(forecast_horizon)
                return forecast, {
                    'mean': forecast,
                    'lower_bound': forecast,
                    'upper_bound': forecast,
                    'model_type': 'Zero Forecast (No Data)'
                }
            
            if len(values) == 1:
                # If only one value, use it as constant
                forecast = np.full(forecast_horizon, values[0])
                return forecast, {
                    'mean': forecast,
                    'lower_bound': forecast * 0.8,
                    'upper_bound': forecast * 1.2,
                    'model_type': 'Constant Forecast'
                }
            
            # Calculate window size for moving average
            window = min(24, len(values) // 2)
            if window < 2:
                window = len(values)
            
            # Calculate moving average
            moving_avg = np.mean(values[-window:])
            
            # Calculate trend
            if len(values) > 1:
                trend_window = min(window, len(values))
                trend = (values[-1] - values[-trend_window]) / trend_window
            else:
                trend = 0
            
            # Generate forecast with trend
            forecast = np.array([moving_avg + trend * i for i in range(1, forecast_horizon + 1)])
            
            # Ensure non-negative values
            forecast = np.maximum(forecast, 0)
            
            # Create confidence bounds
            volatility = np.std(values) if len(values) > 1 else moving_avg * 0.1
            lower_bound = forecast - 1.5 * volatility
            upper_bound = forecast + 1.5 * volatility
            
            # Ensure non-negative bounds
            lower_bound = np.maximum(lower_bound, 0)
            upper_bound = np.maximum(upper_bound, forecast * 0.5)  # At least 50% of forecast
            
            return forecast, {
                'mean': forecast,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'model_type': 'Moving Average with Trend'
            }
            
        except Exception as e:
            print(f"Simple forecast failed: {e}")
            # Ultimate fallback - constant forecast
            if len(values) > 0:
                avg_value = np.mean(values)
            else:
                avg_value = 0
            
            forecast = np.full(forecast_horizon, avg_value)
            return forecast, {
                'mean': forecast,
                'lower_bound': forecast * 0.8,
                'upper_bound': forecast * 1.2,
                'model_type': 'Fallback Constant'
            }


# Initialize global instances
ts_processor = TimeSeriesProcessor()
forecaster = LagLlamaForecaster()


@tool
def parse_cloudwatch_logs_data(log_data_json: str = None) -> str:
    """
    Parse CloudWatch Logs Insights data into structured time series format for forecasting.
    
    This tool processes CloudWatch Logs Insights aggregated data and converts it into 
    time series data suitable for traffic pattern analysis and rate limit forecasting.
    
    Expected input format:
    [
        {
            "hour": "2023-10-01 00:00:00",
            "count(*)": 216,
            "application": "amzn1.sellerapps.app.ade9e8f4-0b14-4f0d-ac7d-f6615922e35f",
            "SellerId": "A13YO9XQO12E5U",  # Changed from partyId
            "resourcePath": "/listings/2021-08-01/items/{sellerId}/{sku}"
        }
    ]
    
    Args:
        log_data_json: JSON string containing CloudWatch Logs Insights aggregated data
                      with hour, count(*), application, SellerId, resourcePath fields
    
    Returns:
        JSON string with parsed time series data, summary statistics, and requests per second metrics.
    """
    try:
        # Validate input parameters
        if not log_data_json:
            return json.dumps({
                "status": "error",
                "message": "log_data_json must be provided",
                "timestamp": datetime.now().isoformat()
            })

        if isinstance(log_data_json, str):
            try:
                log_data = json.loads(log_data_json)
            except json.JSONDecodeError:
                return json.dumps({
                    "status": "error",
                    "message": "Invalid JSON format in input string",
                    "timestamp": datetime.now().isoformat()
                })
        else:
            log_data = log_data_json
        
        # Validate log data format
        if not isinstance(log_data, list):
            return json.dumps({
                "status": "error",
                "message": "CloudWatch data must be a list of log entries",
                "timestamp": datetime.now().isoformat()
            })
        
        if len(log_data) == 0:
            return json.dumps({
                "status": "error",
                "message": "CloudWatch data list is empty",
                "timestamp": datetime.now().isoformat()
            })
        
        # Check if the data has the expected format
        first_entry = log_data[0]
        if not ('hour' in first_entry and 'count(*)' in first_entry):
            return json.dumps({
                "status": "error",
                "message": "CloudWatch data does not have the expected format. Each entry should have 'hour' and 'count(*)' fields.",
                "timestamp": datetime.now().isoformat()
            })
        
        df = ts_processor.parse_cloudwatch_logs(log_data)
        
        if df.empty:
            return json.dumps({
                "status": "error",
                "message": "No valid log entries found",
                "timestamp": datetime.now().isoformat()
            })
        
        # Calculate summary statistics
        total_hourly_requests = df['hourly_requests'].sum()
        avg_requests_per_sec = df['requests_per_sec'].mean()
        max_requests_per_sec = df['requests_per_sec'].max()
        
        summary = {
            "status": "success",
            "total_data_points": len(df),
            "unique_applications": df['application_id'].nunique(),
            "unique_sellers": df['seller_id'].nunique(),  # Changed from unique_parties
            "unique_resources": df['resource_path'].nunique(),
            "time_range": {
                "start": df['timestamp'].min().isoformat(),
                "end": df['timestamp'].max().isoformat(),
                "duration_hours": (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
            },
            "traffic_metrics": {
                "total_hourly_requests": int(total_hourly_requests),
                "avg_requests_per_sec": round(avg_requests_per_sec, 4),
                "max_requests_per_sec": round(max_requests_per_sec, 4),
                "peak_hourly_requests": int(df['hourly_requests'].max())
            },
            "top_applications": df['application_id'].value_counts().head().to_dict(),
            "top_resources": df['resource_path'].value_counts().head().to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Store processed data for later use
        ts_processor.processed_data['latest'] = df
        
        return json.dumps(summary, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to parse CloudWatch logs data: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })


@tool
def analyze_rate_limit_compliance_rps(application_id: str, seller_id: str,  # Changed from party_id
                                     resource_path: str, current_rate_limit_rps: float) -> str:
    """
    Analyze rate limit compliance using requests per second data.
    
    This tool examines usage patterns for specific application/seller/resource
    combinations and checks if they exceed current rate limits measured in requests per second.
    
    Args:
        application_id: Application identifier to analyze
        seller_id: Seller identifier  # Changed from party_id
        resource_path: API resource path to analyze
        current_rate_limit_rps: Current rate limit in requests per second (e.g., 1.0)
    
    Returns:
        JSON string with compliance analysis and violation details based on requests per second.
    """
    try:
        if 'latest' not in ts_processor.processed_data:
            return json.dumps({
                "status": "error", 
                "message": "No CloudWatch logs data available. Please parse logs first using parse_cloudwatch_logs_data tool.",
                "timestamp": datetime.now().isoformat()
            })
        
        df = ts_processor.processed_data['latest']
        time_series = ts_processor.create_time_series(
            df, application_id, seller_id, resource_path, granularity='1H'  # Changed from party_id
        )
        
        if time_series.empty:
            return json.dumps({
                "status": "no_data",
                "message": f"No data found for application: {application_id}, seller: {seller_id}, resource: {resource_path}",  # Changed from party
                "timestamp": datetime.now().isoformat()
            })
        
        # Analyze violations based on requests per second
        violations = time_series[time_series['requests_per_sec'] > current_rate_limit_rps].copy()
        
        total_hours = len(time_series)
        violation_hours = len(violations)
        
        max_requests_per_sec = time_series['requests_per_sec'].max()
        avg_requests_per_sec = time_series['requests_per_sec'].mean()
        
        compliance_analysis = {
            "status": "success",
            "application_id": application_id,
            "seller_id": seller_id,  # Changed from party_id
            "resource_path": resource_path,
            "current_rate_limit_rps": current_rate_limit_rps,
            "analysis_period": {
                "total_hours_analyzed": total_hours,
                "start_time": time_series['timestamp'].min().isoformat(),
                "end_time": time_series['timestamp'].max().isoformat()
            },
            "usage_statistics": {
                "max_requests_per_sec": round(max_requests_per_sec, 4),
                "avg_requests_per_sec": round(avg_requests_per_sec, 4),
                "total_hourly_requests": int(time_series['hourly_requests'].sum()),
                "peak_hourly_requests": int(time_series['hourly_requests'].max())
            },
            "compliance_summary": {
                "rate_limit_violations": {
                    "count": violation_hours,
                    "percentage": round((violation_hours / total_hours) * 100, 2) if total_hours > 0 else 0,
                    "max_violation_rps": round(violations['requests_per_sec'].max(), 4) if not violations.empty else 0,
                    "max_exceeded_by": round((violations['requests_per_sec'].max() - current_rate_limit_rps), 4) if not violations.empty else 0
                },
                "is_compliant": violation_hours == 0,
                "compliance_score": round(((total_hours - violation_hours) / total_hours) * 100, 2) if total_hours > 0 else 100
            },
            "violation_details": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Add violation details
        if not violations.empty:
            violation_details = []
            for _, violation in violations.head(10).iterrows():  # Limit to 10 most recent
                violation_details.append({
                    "timestamp": violation['timestamp'].isoformat(),
                    "requests_per_sec": round(violation['requests_per_sec'], 4),
                    "exceeded_by_rps": round(violation['requests_per_sec'] - current_rate_limit_rps, 4),
                    "hourly_requests": int(violation['hourly_requests']),
                    "severity": "high" if violation['requests_per_sec'] > current_rate_limit_rps * 2 else "medium"
                })
            
            compliance_analysis["violation_details"] = violation_details
        
        return json.dumps(compliance_analysis, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to analyze rate limit compliance: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })


@tool
def forecast_traffic_patterns_rps(application_id: str, seller_id: str, resource_path: str,  # Changed from party_id
                                 forecast_periods: str = "1,7,15,30") -> str:
    """
    Generate time series forecasts for API usage patterns in requests per second using Lag-Llama model.
    
    This tool uses advanced time series forecasting to predict future API usage
    patterns measured in requests per second for specified periods.
    
    Args:
        application_id: Application identifier to forecast
        seller_id: Seller identifier  # Changed from party_id
        resource_path: API resource path to forecast
        forecast_periods: Comma-separated forecast periods in days (e.g., "1,7,15,30")
    
    Returns:
        JSON string with forecasted usage patterns and recommended rate limits in requests per second.
    """
    try:
        periods = [int(p.strip()) for p in forecast_periods.split(',')]
        
        if 'latest' not in ts_processor.processed_data:
            return json.dumps({
                "status": "error",
                "message": "No CloudWatch logs data available. Please parse logs first using parse_cloudwatch_logs_data tool.",
                "timestamp": datetime.now().isoformat()
            })
        
        df = ts_processor.processed_data['latest']
        time_series = ts_processor.create_time_series(
            df, application_id, seller_id, resource_path, granularity='1H'  # Changed from party_id
        )
        
        if time_series.empty or len(time_series) < 5:
            return json.dumps({
                "status": "insufficient_data",
                "message": f"Insufficient data for forecasting. Need at least 5 data points, got {len(time_series)}",
                "timestamp": datetime.now().isoformat()
            })
        
        forecasts = {}
        
        for period_days in periods:
            forecast_horizon = period_days * 24  # Convert days to hours
            
            try:
                forecast_values, forecast_stats = forecaster.train_and_forecast(
                    time_series, forecast_horizon
                )
                
                # Calculate recommended rate limits with safety margin
                max_forecast_rps = np.max(forecast_stats['upper_bound'])
                avg_forecast_rps = np.mean(forecast_stats['mean'])
                
                # Recommend rate limits with 20% safety margin
                recommended_rate_rps = round(max_forecast_rps * 1.2, 4)
                recommended_burst_rps = round(max_forecast_rps * 1.5, 4)  # 50% higher for burst
                
                forecasts[f"{period_days}_days"] = {
                    "period_days": period_days,
                    "forecast_horizon_hours": forecast_horizon,
                    "model_used": forecast_stats['model_type'],
                    "predictions_rps": {
                        "max_requests_per_sec": round(np.max(forecast_stats['mean']), 4),
                        "avg_requests_per_sec": round(np.mean(forecast_stats['mean']), 4),
                        "peak_requests_per_sec": round(np.max(forecast_stats['upper_bound']), 4),
                        "min_requests_per_sec": round(np.max(0, np.min(forecast_stats['lower_bound'])), 4)
                    },
                    "predictions_hourly": {
                        "max_hourly_requests": int(np.max(forecast_stats['mean']) * 3600),
                        "avg_hourly_requests": int(np.mean(forecast_stats['mean']) * 3600),
                        "peak_hourly_requests": int(np.max(forecast_stats['upper_bound']) * 3600)
                    },
                    "recommended_rate_limits": {
                        "tier": "PREDICTED",
                        "rate_rps": recommended_rate_rps,
                        "burst_rps": recommended_burst_rps,
                        "confidence": "high" if forecast_stats['model_type'] == 'Lag-Llama' else "medium"
                    },
                    "trend_analysis": {
                        "is_increasing": forecast_values[-1] > forecast_values[0],
                        "growth_rate_percent": round(((forecast_values[-1] - forecast_values[0]) / forecast_values[0] * 100), 2) if forecast_values[0] > 0 else 0,
                        "volatility": round(np.std(forecast_values), 4)
                    }
                }
                
            except Exception as forecast_error:
                forecasts[f"{period_days}_days"] = {
                    "period_days": period_days,
                    "status": "error",
                    "message": f"Forecasting failed: {str(forecast_error)}"
                }
        
        forecast_summary = {
            "status": "success",
            "application_id": application_id,
            "seller_id": seller_id,  # Changed from party_id
            "resource_path": resource_path,
            "historical_data": {
                "data_points": len(time_series),
                "time_range": {
                    "start": time_series['timestamp'].min().isoformat(),
                    "end": time_series['timestamp'].max().isoformat()
                },
                "current_max_rps": round(time_series['requests_per_sec'].max(), 4),
                "current_avg_rps": round(time_series['requests_per_sec'].mean(), 4),
                "current_max_hourly": int(time_series['hourly_requests'].max()),
                "current_avg_hourly": int(time_series['hourly_requests'].mean())
            },
            "forecasts": forecasts,
            "model_capabilities": {
                "lag_llama_available": HAS_LAG_LLAMA,
                "statistical_models_available": HAS_STATSMODELS,
                "installation_note": "Lag-Llama requires: git clone https://github.com/time-series-foundation-models/lag-llama/ && cd lag-llama && pip install -r requirements.txt" if not HAS_LAG_LLAMA else None
            },
            "recommendations": {
                "immediate_action": _generate_immediate_recommendations_rps(forecasts),
                "long_term_strategy": _generate_long_term_strategy_rps(forecasts)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return json.dumps(forecast_summary, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to generate traffic forecasts: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })


@tool
def comprehensive_rps_analysis(application_id: str, seller_id: str, resource_path: str,  # Changed from party_id
                              current_rate_limit_rps: float) -> str:
    """
    Perform comprehensive traffic analysis for requests per second data combining current 
    compliance and future forecasts.
    
    This tool automatically loads CloudWatch data from file and performs
    complete analysis including compliance checking and forecasting.
    
    Args:
        application_id: Application identifier to analyze
        seller_id: Seller identifier  # Changed from party_id
        resource_path: API resource path to analyze
        current_rate_limit_rps: Current rate limit in requests per second (e.g., 1.0)
    
    Returns:
        JSON string with comprehensive analysis and recommendations based on requests per second metrics.
    """
    try:
        # Validate required parameters
        if not all([application_id, seller_id, resource_path, current_rate_limit_rps]):  # Changed from party_id
            return json.dumps({
                "status": "error",
                "message": "Missing required parameters. Please provide application_id, seller_id, resource_path, and current_rate_limit_rps.",
                "timestamp": datetime.now().isoformat()
            })
            
        # Convert current_rate_limit_rps to float if it's not already
        try:
            current_rate_limit_rps = float(current_rate_limit_rps)
        except (ValueError, TypeError):
            return json.dumps({
                "status": "error",
                "message": f"Invalid rate limit value: {current_rate_limit_rps}. Please provide a valid number.",
                "timestamp": datetime.now().isoformat()
            })
        
        # EMBEDDED FILE PATH - Change this to your actual CloudWatch JSON file path
        CLOUDWATCH_DATA_FILE = "logs-insights-results.json"
        
        # Step 1: Load CloudWatch data directly from file
        from direct_loader import load_cloudwatch_data
        data_result = load_cloudwatch_data(CLOUDWATCH_DATA_FILE)
        
        if data_result.get('status') != 'success':
            return json.dumps({
                "status": "error",
                "message": f"Failed to load CloudWatch data: {data_result.get('message', 'Unknown error')}",
                "file_info": data_result.get('file_info', {}),
                "timestamp": datetime.now().isoformat()
            })
        
        log_data = data_result.get('data')
        file_info = data_result.get('file_info')
        
        # Step 2: Parse the CloudWatch logs data
        parse_result = parse_cloudwatch_logs_data(log_data_json=json.dumps(log_data))
        parse_data = json.loads(parse_result)
        
        if parse_data.get('status') != 'success':
            return json.dumps({
                "status": "error",
                "message": f"Failed to parse CloudWatch data: {parse_data.get('message', 'Unknown parsing error')}",
                "file_info": file_info,
                "timestamp": datetime.now().isoformat()
            })
        
        # Step 3: Analyze current compliance
        compliance_result = analyze_rate_limit_compliance_rps(
            application_id, seller_id, resource_path, current_rate_limit_rps  # Changed from party_id
        )
        compliance_data = json.loads(compliance_result)
        
        # Step 4: Generate forecasts
        forecast_result = forecast_traffic_patterns_rps(
            application_id, seller_id, resource_path  # Changed from party_id
        )
        forecast_data = json.loads(forecast_result)
        
        # Step 5: Combine all analyses
        comprehensive_analysis = {
            "status": "success",
            "analysis_type": "comprehensive_rps_traffic_analysis",
            "data_source": "direct_file",
            "data_source_details": CLOUDWATCH_DATA_FILE,
            "file_info": file_info,
            "target": {
                "application_id": application_id,
                "seller_id": seller_id,  # Changed from party_id
                "resource_path": resource_path
            },
            "current_rate_limit_rps": current_rate_limit_rps,
            
            # Parsed log summary
            "cloudwatch_data_summary": parse_data,
            
            # Current compliance analysis
            "compliance_analysis": compliance_data,
            
            # Future forecasts
            "forecast_analysis": forecast_data,
            
            # Integrated recommendations
            "integrated_recommendations": _generate_integrated_recommendations_rps(
                compliance_data, forecast_data, current_rate_limit_rps
            ),
            
            "timestamp": datetime.now().isoformat()
        }
        
        return json.dumps(comprehensive_analysis, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Comprehensive RPS analysis failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })


def _generate_immediate_recommendations_rps(forecasts: Dict) -> List[str]:
    """Generate immediate action recommendations based on RPS forecasts"""
    recommendations = []
    
    if '1_days' in forecasts and forecasts['1_days'].get('status') != 'error':
        day1 = forecasts['1_days']
        max_rps = day1['predictions_rps']['max_requests_per_sec']
        
        if max_rps > 0.1:  # More than 0.1 requests per second
            recommendations.append(f"High traffic predicted: {max_rps:.4f} req/sec - consider upgrading rate limits within 24 hours")
        
        if day1['trend_analysis']['growth_rate_percent'] > 50:
            recommendations.append("Rapid growth detected - monitor closely and prepare for scaling")
    
    if '7_days' in forecasts and forecasts['7_days'].get('status') != 'error':
        week1 = forecasts['7_days']
        peak_rps = week1['predictions_rps']['peak_requests_per_sec']
        
        if peak_rps > 0.5:  # More than 0.5 requests per second
            recommendations.append(f"Plan capacity upgrades for next week - peak {peak_rps:.4f} req/sec expected")
    
    if not recommendations:
        recommendations.append("Current rate limits appear adequate for predicted traffic")
    
    return recommendations


def _generate_long_term_strategy_rps(forecasts: Dict) -> List[str]:
    """Generate long-term strategy recommendations based on RPS forecasts"""
    strategy = []
    
    if '30_days' in forecasts and forecasts['30_days'].get('status') != 'error':
        month1 = forecasts['30_days']
        
        if month1['trend_analysis']['is_increasing']:
            strategy.append("Implement auto-scaling policies to handle growing traffic patterns")
            
        if month1['trend_analysis']['volatility'] > 0.1:
            strategy.append("High volatility detected - consider implementing adaptive rate limiting")
            
        peak_rps = month1['predictions_rps']['peak_requests_per_sec']
        if peak_rps > 1.0:  # More than 1 request per second
            strategy.append(f"Enterprise-grade infrastructure recommended for sustained high traffic ({peak_rps:.4f} req/sec)")
    
    strategy.append("Regular forecast updates recommended as traffic patterns evolve")
    
    return strategy


def _generate_integrated_recommendations_rps(compliance_data: Dict, forecast_data: Dict, 
                                           current_rate_limit_rps: float) -> Dict:
    """Generate integrated recommendations based on compliance and forecast analysis for RPS data"""
    
    recommendations = {
        "priority": "medium",
        "action_required": False,
        "immediate_actions": [],
        "short_term_actions": [],
        "long_term_actions": [],
        "rate_limit_recommendations_rps": {},
        "capacity_planning": {}
    }
    
    # Analyze current compliance
    if compliance_data.get('status') == 'success':
        violations = compliance_data.get('compliance_summary', {}).get('rate_limit_violations', {}).get('count', 0)
        max_violation_rps = compliance_data.get('compliance_summary', {}).get('rate_limit_violations', {}).get('max_violation_rps', 0)
        
        if violations > 0:
            recommendations["priority"] = "high"
            recommendations["action_required"] = True
            recommendations["immediate_actions"].append(
                f"Rate limit violated {violations} times with peak at {max_violation_rps:.4f} req/sec - immediate upgrade required"
            )
    
    # Analyze forecasts
    if forecast_data.get('status') == 'success':
        forecasts = forecast_data.get('forecasts', {})
        
        # Check 1-day forecast
        if '1_days' in forecasts and forecasts['1_days'].get('status') != 'error':
            day1_rec = forecasts['1_days'].get('recommended_rate_limits', {})
            predicted_rate_rps = day1_rec.get('rate_rps', 0)
            
            if predicted_rate_rps > current_rate_limit_rps:
                recommendations["short_term_actions"].append(
                    f"Upgrade rate limit to {predicted_rate_rps:.4f} req/sec within 24 hours"
                )
                recommendations["rate_limit_recommendations_rps"]["immediate"] = day1_rec
        
        # Check 30-day forecast
        if '30_days' in forecasts and forecasts['30_days'].get('status') != 'error':
            month1_rec = forecasts['30_days'].get('recommended_rate_limits', {})
            long_term_rate_rps = month1_rec.get('rate_rps', 0)
            
            recommendations["rate_limit_recommendations_rps"]["monthly_target"] = month1_rec
            
            if long_term_rate_rps > current_rate_limit_rps * 2:
                recommendations["long_term_actions"].append(
                    "Consider infrastructure scaling for projected long-term growth"
                )
                recommendations["capacity_planning"]["scaling_needed"] = True
                recommendations["capacity_planning"]["projected_increase"] = f"{long_term_rate_rps / current_rate_limit_rps:.1f}x"
                
            # Add specific RPS-based recommendations
            if long_term_rate_rps > 1.0:
                recommendations["capacity_planning"]["infrastructure_tier"] = "enterprise"
                recommendations["long_term_actions"].append(
                    f"Enterprise infrastructure recommended for {long_term_rate_rps:.4f} req/sec sustained load"
                )
            elif long_term_rate_rps > 0.1:
                recommendations["capacity_planning"]["infrastructure_tier"] = "standard"
                recommendations["long_term_actions"].append(
                    f"Standard scaling sufficient for {long_term_rate_rps:.4f} req/sec projected load"
                )
    
    return recommendations


def create_rps_forecasting_agent() -> Agent:
    """Create the Time Series Forecasting Agent specialized for requests per second analysis"""

    # Enhanced system prompt for RPS analysis
    system_prompt = """
    You are an advanced Time Series Forecasting Agent specialized in API traffic analysis 
    using requests per second (RPS) metrics from CloudWatch Logs Insights data.
    
    Your specialized capabilities include:
    
    📊 CLOUDWATCH LOGS INSIGHTS PROCESSING:
    1. Parse aggregated hourly data with format: hour, count(*), application, SellerId, resourcePath
    2. Convert hourly request counts to requests per second for accurate rate limit analysis
    3. Handle time series gaps and normalize data for forecasting
    4. Direct file loading for CloudWatch data analysis
    
    🔍 REQUESTS PER SECOND ANALYSIS:
    1. Analyze current rate limit compliance using precise RPS measurements
    2. Identify violations with exact RPS values and timing
    3. Calculate compliance scores based on RPS thresholds
    4. Support for seller-based analysis (SellerId instead of partyId)
    
    🔮 ADVANCED FORECASTING:
    1. Use Lag-Llama model for state-of-the-art time series forecasting (when available)
    2. Predict future traffic patterns for 1 day, 7 days, 15 days, and 30 days
    3. Generate RPS-based rate limit recommendations with safety margins
    4. Provide confidence intervals and trend analysis
    
    🎯 RATE LIMIT OPTIMIZATION:
    1. Recommend optimal rate limits in requests per second
    2. Calculate burst capacity recommendations
    3. Provide scaling strategies based on predicted growth
    4. Account for traffic volatility and seasonal patterns
    
    🔄 WORKFLOW FOR RPS ANALYSIS:
    When analyzing CloudWatch data, systematically:
    1. Use comprehensive_rps_analysis with the required parameters (application_id, seller_id, resource_path, current_rate_limit_rps)
    2. The tool automatically loads data from the embedded file path
    3. Provides complete analysis including compliance checking and forecasting
    4. Direct file loading ensures the most up-to-date data is used
    
    📋 RESPONSE STRATEGY:
    - Always express rate limits in requests per second (RPS) for precision
    - Provide both RPS and hourly request equivalents for context
    - Include confidence levels for all predictions
    - Explain forecasting methodology and model selection
    - Give specific, actionable recommendations with timelines
    - Use seller_id terminology instead of party_id
    
    Be precise with RPS calculations, thorough in analysis, and clear in recommendations.
    Always specify units (req/sec) and provide context for decision-making.
    """

    try:
        # Configure Bedrock model with fallback options
        model_options = [
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        ]

        regions_to_try = [
            os.environ.get('AWS_REGION', 'us-west-2'),
            'us-west-2',
        ]

        bedrock_model = None
        successful_config = None

        for region in regions_to_try:
            for model_id in model_options:
                try:
                    bedrock_model = BedrockModel(
                        model_id=model_id,
                        region_name=region,
                        temperature=0.1,  # Low temperature for analytical precision
                        max_tokens=4096,
                        streaming=True
                    )
                    successful_config = f"{model_id} in {region}"
                    print(f"✓ RPS Forecasting Agent configured: {successful_config}")
                    break
                except Exception:
                    continue
            if bedrock_model:
                break

        if not bedrock_model:
            raise Exception("No accessible Claude models found. Please enable model access in Bedrock console.")

        # Create specialized RPS forecasting agent
        agent = Agent(
            model=bedrock_model,
            system_prompt=system_prompt,
            tools=[
                comprehensive_rps_analysis,
                memory  # For storing analysis results
            ]
        )

        return agent

    except Exception as e:
        print(f"❌ Failed to create RPS Forecasting Agent: {e}")
        raise


def main():
    """Main function to run the Time Series Forecasting Agent for RPS analysis"""

    print("=" * 80)
    print("TIME SERIES FORECASTING AGENT - REQUESTS PER SECOND")
    print("CloudWatch Logs Insights • Lag-Llama Forecasting • Rate Limit Optimization")
    print("With Memory Caching and SellerId Support")
    print("Powered by Amazon Bedrock Claude")
    print("=" * 80)

    # Check capabilities
    print("Checking forecasting capabilities...")

    if HAS_LAG_LLAMA:
        print("✓ Lag-Llama model available for advanced forecasting")
    else:
        print("⚠ Lag-Llama not available, using statistical forecasting")
        print("  Install with:")
        print("    git clone https://github.com/time-series-foundation-models/lag-llama/")
        print("    cd lag-llama")
        print("    pip install -r requirements.txt")
        print("    # Download model weights:")
        print("    huggingface-cli download time-series-foundation-models/Lag-Llama lag-llama.ckpt --local-dir ./")
        print("    # Or install dependencies separately:")
        print("    pip install torch gluonts pytorch-lightning")

    if HAS_STATSMODELS:
        print("✓ Statistical models (ARIMA) available")
    else:
        print("⚠ Install statsmodels for enhanced forecasting: pip install statsmodels")

    print("✓ Memory caching enabled for CloudWatch data")

    try:
        # Check AWS Bedrock access
        if HAS_BOTO3:
            try:
                bedrock_client = boto3.client('bedrock', region_name=os.environ.get('AWS_REGION', 'us-west-2'))
                bedrock_client.list_foundation_models()
                print("✓ Amazon Bedrock connection verified")
            except Exception as e:
                print(f"⚠ Bedrock access warning: {e}")
        else:
            print("⚠ Boto3 not available - some features may be limited")

        # Create the RPS forecasting agent
        agent = create_rps_forecasting_agent()
        print("✓ RPS Forecasting Agent initialized successfully")

        print("\nAVAILABLE TOOLS")
        print("=" * 80)

        tools_description = {
            "🎯 comprehensive_rps_analysis": "End-to-end analysis with recommendations (auto-caching enabled)",
            "🧠 memory": "Store and retrieve analysis results"
        }

        for tool, description in tools_description.items():
            print(f"  ✓ {tool}: {description}")

        print("\n" + "=" * 80)
        print("USAGE EXAMPLES")
        print("=" * 80)
        print("🎯 Comprehensive Analysis:")
        print('   "Perform RPS analysis for application ABC, seller XYZ123, resource /api/endpoint, current limit 1.0 req/sec"')

        print("\nINTERACTIVE MODE")
        print("=" * 80)
        print("🔮 Enter your RPS analysis requests. The agent loads data from logs-insights-results.json with caching")
        print("Type 'quit' to exit, 'help' for examples, or 'cache' to check cache status.")

        # Interactive loop
        while True:
            try:
                user_input = input("\n📊 Enter your RPS analysis request: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if user_input.lower() == 'help':
                    print("\n📚 HELP - Example Requests for RPS Analysis:")
                    print("• 'Analyze RPS for application amzn1.sellerapps.app.test-123, seller A13YO9XQO12E5U, resource /api/endpoint, limit 1.0 req/sec'")
                    print("• 'Check compliance and forecast for app XYZ, seller ABC123, resource /feeds, current limit 0.5 req/sec'")
                    print("• 'Comprehensive analysis for high traffic application'")
                    print("• 'What RPS limit should I set for this endpoint?'")
                    continue

                if user_input.lower() == 'cache':
                    from direct_loader import load_cloudwatch_data
                    file_info = load_cloudwatch_data("logs-insights-results.json")
                    print("\n📊 File Status:")
                    print(json.dumps(file_info, indent=2, default=str))
                    continue

                if not user_input:
                    continue

                print(f"\n📊 Processing RPS analysis...")
                print(f"Input: {user_input}")
                print("-" * 40)

                # Process with the RPS forecasting agent
                response = agent(user_input)

                print(f"\n✅ RPS Analysis Complete:")
                print(response)
                print("\n" + "-" * 80)

            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error processing request: {e}")
                print("Please try again with a different request.")

    except Exception as e:
        print(f"❌ Failed to initialize RPS Forecasting Agent: {e}")
        print("\nTroubleshooting steps:")
        print("1. Check AWS credentials and Bedrock model access")
        print("2. Install dependencies:")
        print("   For Lag-Llama (advanced forecasting):")
        print("     git clone https://github.com/time-series-foundation-models/lag-llama/")
        print("     cd lag-llama")
        print("     pip install -r requirements.txt")
        print("     huggingface-cli download time-series-foundation-models/Lag-Llama lag-llama.ckpt --local-dir ./")
        print("   For basic dependencies:")
        print("     pip install torch gluonts statsmodels pandas numpy boto3")
        print("3. Verify Claude model availability in your region")
        print("4. Ensure CloudWatch Logs Insights data format is correct")
        return 1

    return 0


if __name__ == "__main__":
    # Set up environment variables
    if not os.environ.get('AWS_REGION'):
        os.environ['AWS_REGION'] = 'us-west-2'

    print("=" * 60)
    print("TIME SERIES FORECASTING AGENT")
    print("Specialized for Requests Per Second Analysis")
    print("With SellerId Support and Memory Caching")
    print("=" * 60)
    print("\nKey Features:")
    print("✓ SellerId instead of partyId throughout")
    print("✓ Memory caching for CloudWatch data")
    print("✓ Automatic cache refresh on file changes")
    print("✓ Enhanced performance for repeated requests")
    print("\nDependency Installation Instructions:")
    print("Basic dependencies: pip install pandas numpy statsmodels boto3")
    print("For Lag-Llama (advanced forecasting):")
    print("  git clone https://github.com/time-series-foundation-models/lag-llama/")
    print("  cd lag-llama")
    print("  pip install -r requirements.txt")
    print("  huggingface-cli download time-series-foundation-models/Lag-Llama lag-llama.ckpt --local-dir ./")
    print("For basic PyTorch forecasting: pip install torch gluonts")
    print("-" * 60)

    sys.exit(main())