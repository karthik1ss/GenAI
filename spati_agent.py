#!/usr/bin/env python3
"""
SPATI (SP API Traffic Intelligence) AI Agent
Uses Amazon Bedrock to analyze API traffic patterns
"""

import json
import sys
import re
from typing import Dict, Any, Optional, List
from strands import Agent, tool
from strands_tools import use_llm, memory
from strands.models import BedrockModel
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config as BotocoreConfig
import os
from datetime import datetime

class DynamoDBClient:
    """DynamoDB client for tier and throttling data"""

    def __init__(self, region_name: str = 'us-west-2'):
        self.dynamodb = boto3.resource('dynamodb', region_name=region_name)
        self.table1 = self.dynamodb.Table('APICurrentTier')
        self.table2 = self.dynamodb.Table('ThrottlingTiers')

    def query_api_current_tier(self, app_id: str, seller_id: str, api_path: str) -> Dict[str, Any]:
        """Get current tier assignment for an app"""
        try:
            # Primary key: AppId + SellerId
            primary_key = f"{app_id}#{seller_id}"
            
            print(f"Querying with key: {primary_key}, APIPath: {api_path}")  # Debug log
            
            response = self.table1.get_item(
                Key={
                    'AppId_SellerId': primary_key,
                    'APIPath': api_path
                }
            )
            
            if 'Item' in response:
                item = response['Item']
                print(f"Found item: {item}")  # Debug log
                return item
            else:
                print(f"No item found for key: {primary_key}, APIPath: {api_path}")  # Debug log
                return {}
                
        except ClientError as e:
            print(f"Error querying APICurrentTier: {e}")
            return {}

    def query_throttling_tiers(self, method: str, api_path: str) -> Dict[str, Any]:
        """Get throttling configuration for an API endpoint"""
        try:
            primary_key = f"{method}{api_path}"
            response = self.table2.get_item(
                Key={
                    'Method_APIPath': primary_key
                }
            )
            return response.get('Item', {})
        except ClientError as e:
            print(f"Error querying ThrottlingTiers: {e}")
            return {}


# Initialize DynamoDB client
db_client = DynamoDBClient()


@tool
def query_api_current_tier(app_id: str, seller_id: str, api_path: str) -> str:
    """
    Get current tier for an application and API path
    
    Args:
        app_id: Application ID
        seller_id: Seller ID
        api_path: API path
    
    Returns:
        JSON string with tier information
    """
    result = db_client.query_api_current_tier(app_id, seller_id, api_path)
    return json.dumps(result, indent=2, default=str)


@tool
def query_throttling_tiers(method: str, api_path: str) -> str:
    """
    Get throttling configuration for an API endpoint
    
    Args:
        method: HTTP method (GET, POST, etc.)
        api_path: API path
    
    Returns:
        JSON string with throttling tiers
    """
    result = db_client.query_throttling_tiers(method, api_path)
    return json.dumps(result, indent=2, default=str)


@tool
def parse_traffic_analysis_request(user_input: str) -> str:
    """
    Extract application details from natural language request
    
    Args:
        user_input: Natural language request
    
    Returns:
        JSON string with extracted parameters
    """
    # Extract AppId (handle both Appid: and appId: formats)
    app_id_match = re.search(r'[Aa]pp[Ii]d:\s*([^,\s]+)', user_input)
    app_id = app_id_match.group(1) if app_id_match else None

    # Extract Seller ID (handle both sellerId: and seller Id: formats)
    seller_id_match = re.search(r'[Ss]eller\s*[Ii]d[:(\s]+([^,\s)]+)', user_input)
    seller_id = seller_id_match.group(1) if seller_id_match else None

    # Extract API path with quotes
    api_match = re.search(r'API\s+path\s+(GET|POST|PUT|DELETE)\s+["\']([^"\']+)["\']', user_input)
    if not api_match:
        # Try without quotes
        api_match = re.search(r'API\s+path\s+(GET|POST|PUT|DELETE)\s+(\S+)', user_input)
    
    method = api_match.group(1) if api_match else None
    api_path = api_match.group(2) if api_match else None

    # Format API path to match DynamoDB format
    if method and api_path:
        api_path = f"{method} {api_path}"

    print(f"Parsed values: AppId={app_id}, SellerId={seller_id}, Method={method}, Path={api_path}")  # Debug log

    result = {
        "app_id": app_id,
        "seller_id": seller_id,
        "api_path": api_path,
        "method": method,
        "original_request": user_input,
        "parsed_timestamp": datetime.now().isoformat()
    }

    return json.dumps(result, indent=2)


@tool
def analyze_traffic_patterns(app_id: str, seller_id: str, method: str, api_path: str) -> str:
    """
    Analyze traffic patterns and provide recommendations
    
    Args:
        app_id: Application ID
        seller_id: Seller ID
        method: HTTP method
        api_path: API path
    
    Returns:
        JSON string with analysis and recommendations
    """
    current_tier_data = db_client.query_api_current_tier(app_id, seller_id, api_path)
    throttling_data = db_client.query_throttling_tiers(method, api_path)

    analysis = {
        "request_details": {
            "app_id": app_id,
            "seller_id": seller_id,
            "method": method,
            "api_path": api_path,
            "analysis_timestamp": datetime.now().isoformat()
        },
        "current_tier_assignment": current_tier_data,
        "available_usage_plans": throttling_data,
        "analysis_summary": {},
        "recommendations": []
    }

    current_tier = current_tier_data.get('tier', 'Unknown')
    usage_plans = throttling_data.get('usagePlans', [])
    current_plan = next((plan for plan in usage_plans if plan.get('tier') == current_tier), None)

    analysis["analysis_summary"] = {
        "current_tier": current_tier,
        "current_rate_limit": current_plan.get('rate') if current_plan else 'Unknown',
        "current_burst_limit": current_plan.get('burst') if current_plan else 'Unknown',
        "available_tiers": [plan.get('tier') for plan in usage_plans],
        "highest_tier_available": max(usage_plans, key=lambda x: x.get('rate', 0)) if usage_plans else None
    }

    if current_plan and usage_plans:
        max_rate_plan = max(usage_plans, key=lambda x: x.get('rate', 0))
        if current_plan.get('rate', 0) < max_rate_plan.get('rate', 0):
            analysis["recommendations"].append({
                "type": "tier_upgrade",
                "message": f"Consider upgrading from {current_tier} to {max_rate_plan.get('tier')} for higher throughput",
                "potential_improvement": f"Rate limit: {current_plan.get('rate')} → {max_rate_plan.get('rate')}"
            })

    if not current_tier_data:
        analysis["recommendations"].append({
            "type": "configuration_missing",
            "message": "No tier assignment found",
            "action": "Check APICurrentTier table configuration"
        })

    if not throttling_data:
        analysis["recommendations"].append({
            "type": "throttling_config_missing",
            "message": "No throttling configuration found",
            "action": "Verify ThrottlingTiers table configuration"
        })

    return json.dumps(analysis, indent=2, default=str)


def create_spati_agent() -> Agent:
    """Create SPATI AI agent with Bedrock integration and time series forecasting capabilities"""
    system_prompt = """
    You analyze SP API traffic patterns and provide optimization recommendations.
    
    Your capabilities include:
    
    1. TRAFFIC PATTERN ANALYSIS:
       - Analyze current API usage patterns
       - Identify tier assignments and rate limits
       - Compare against available tiers
       - Provide upgrade recommendations
       
    2.  📊 CLOUDWATCH LOGS INSIGHTS PROCESSING:
        - Parse aggregated hourly data with format: hour, count(*), application, SellerId, resourcePath
        - Convert hourly request counts to requests per second for accurate rate limit analysis
        - Handle time series gaps and normalize data for forecasting
        - Direct file loading for CloudWatch data analysis
    
    3. REQUESTS PER SECOND ANALYSIS:
        - Analyze current rate limit compliance using precise RPS measurements
        - Identify violations with exact RPS values and timing
        - Calculate compliance scores based on RPS thresholds
        - Support for seller-based analysis (SellerId instead of partyId)
    
    4. ADVANCED FORECASTING:
        - Use Lag-Llama/ARIMA model for state-of-the-art time series forecasting (when available)
        - Predict future traffic patterns for 1 day, 7 days, 15 days, and 30 days
        - Generate RPS-based rate limit recommendations with safety margins
        - Provide confidence intervals and trend analysis
    
    5. RATE LIMIT OPTIMIZATION:
        - Recommend optimal rate limits in requests per second
        - Calculate burst capacity recommendations
        - Provide scaling strategies based on predicted growth
        - Account for traffic volatility and seasonal patterns
    
    Keep responses concise and focused:
    1. Show current configuration status
    2. List available tiers with rates
    3. Provide clear recommendations with forecasting insights
    4. Use bullet points for better readability
    5. Avoid verbose explanations
    
    Format output as:
    
    # Traffic Analysis
    - App: [ID]
    - API: [Method + Path]
    
    ## Current Status
    - Tier: [Current/Not Assigned]
    - Rate: [X req/sec]
    
    ## Available Tiers
    - Tier1: [rate/burst]
    - Tier2: [rate/burst]
    
    ## Traffic Forecast
    - Current usage: [X req/sec]
    - 7-day forecast: [Y req/sec]
    - 30-day forecast: [Z req/sec]
    
    ## Recommendation
    - [Simple, actionable recommendation]
    
    Use available tools systematically and be direct in analysis.
    """

    try:
        boto_config = BotocoreConfig(
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=10,
            read_timeout=60,
            max_pool_connections=50
        )

        bedrock_model = BedrockModel(
            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
            region_name=os.environ.get('AWS_REGION', 'us-west-2'),
            temperature=0.3,
            max_tokens=4096,
            top_p=0.8,
            streaming=True,
            cache_prompt="default",
            cache_tools="default",
            boto_client_config=boto_config
        )

        print("✓ BedrockModel configured")
        
        # Import time series forecasting tools
        try:
            from time_series_agent import comprehensive_rps_analysis, forecast_traffic_patterns_rps
            has_forecasting = True
            print("✓ Time series forecasting tools imported")
        except ImportError as e:
            has_forecasting = False
            print(f"⚠ Time series forecasting tools not available: {e}")
        
        # Create tool list with base tools
        tools = [
            parse_traffic_analysis_request,
            analyze_traffic_patterns,
            comprehensive_rps_analysis,
            memory
        ]

        agent = Agent(
            model=bedrock_model,
            system_prompt=system_prompt,
            tools=tools
        )

        print("✓ Agent created with " + ("forecasting capabilities" if has_forecasting else "basic capabilities"))
        return agent

    except Exception as e:
        print(f"❌ Error creating agent: {e}")
        print(f"Error type: {type(e).__name__}")

        print("Attempting fallback configuration...")
        try:
            fallback_model = BedrockModel(
                model_id="us.anthropic.claude-sonnet-4-20250514-v1:0"
            )
            
            # Create tool list with base tools for fallback
            tools = [
                parse_traffic_analysis_request,
                analyze_traffic_patterns,
                memory
            ]
            
            # Try to add forecasting tools if available
            try:
                from time_series_agent import comprehensive_rps_analysis, forecast_traffic_patterns_rps
                tools.extend([
                    comprehensive_rps_analysis,
                    forecast_traffic_patterns_rps
                ])
                print("✓ Time series forecasting tools added to fallback configuration")
            except ImportError:
                print("⚠ Fallback configuration without forecasting tools")

            agent = Agent(
                model=fallback_model,
                system_prompt=system_prompt,
                tools=tools
            )

            print("✓ Agent created with fallback config")
            return agent

        except Exception as fallback_error:
            print(f"❌ Fallback failed: {fallback_error}")
            raise


def main():
    """Run the SPATI agent with time series forecasting capabilities"""
    print("=" * 80)
    print("SPATI - SP API TRAFFIC INTELLIGENCE")
    print("With Time Series Forecasting & Rate Limit Optimization")
    print("Powered by Amazon Bedrock")
    print("=" * 80)

    try:
        # Check for time series forecasting capabilities
        try:
            from time_series_agent import HAS_LAG_LLAMA, HAS_STATSMODELS
            print("Checking forecasting capabilities...")
            
            if HAS_LAG_LLAMA:
                print("✓ Lag-Llama model available for advanced forecasting")
            else:
                print("⚠ Lag-Llama not available, using statistical forecasting")
                print("  Install with: git clone https://github.com/time-series-foundation-models/lag-llama/")
            
            if HAS_STATSMODELS:
                print("✓ Statistical models (ARIMA) available")
            else:
                print("⚠ Install statsmodels for enhanced forecasting: pip install statsmodels")
        except ImportError:
            print("⚠ Time series forecasting module not available")
        
        # Verify AWS credentials
        try:
            bedrock_client = boto3.client('bedrock', region_name=os.environ.get('AWS_REGION', 'us-west-2'))
            bedrock_client.list_foundation_models()
            print("✓ Bedrock connection verified")
        except Exception as e:
            print(f"⚠ Warning: Bedrock access issue: {e}")
            print("  Check Bedrock model access in console")

        agent = create_spati_agent()

        print("✓ Agent initialized")
        print("✓ Model: Claude 4 Sonnet")
        print(f"✓ Region: {os.environ.get('AWS_REGION', 'us-west-2')}")
        print("✓ Tools loaded:")
        for tool_name in agent.tool_names:
            print(f"  - {tool_name}")

        print("\n" + "=" * 80)
        print("READY FOR INPUT")
        print("=" * 80)
        print("Enter traffic analysis requests. Type 'quit' to exit.")
        print("\nExample:")
        print('tell me the traffic pattern analysis of Application(Appid:amzn1.sellerapps.app.ade9e8f4-0b14-4f0d-ac7d-f6615922e35f), with seller Id(A13YO9XQO12E5U), API path GET "/listings/2021-08-01/restrictions"')
        print("\nFor time series forecasting:")
        print('analyze traffic forecast for application amzn1.sellerapps.app.ade9e8f4-0b14-4f0d-ac7d-f6615922e35f, seller A13YO9XQO12E5U, resource /listings/2021-08-01/items, current limit 1.0 req/sec')
        print("\n" + "-" * 80)

        while True:
            try:
                user_input = input("\n🤖 Enter request: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if not user_input:
                    continue

                print(f"\n📝 Processing...")
                print(f"Input: {user_input}")
                print("-" * 40)

                response = agent(user_input)

                print(f"\n✅ Analysis Complete:")
                print(response)
                print("\n" + "-" * 80)

            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again.")

    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check AWS credentials")
        print("2. Verify DynamoDB tables")
        print("3. Enable Claude models in Bedrock")
        print("4. Check bedrock:InvokeModel permissions")
        print("5. Ensure time_series_agent.py is in the same directory")
        return 1

    return 0


if __name__ == "__main__":
    # Set required environment variables
    if not os.environ.get('AWS_REGION'):
        os.environ['AWS_REGION'] = 'us-west-2'
    
    # Set knowledge base ID
    os.environ['STRANDS_KNOWLEDGE_BASE_ID'] = 'BKNFNGM6PD'

    sys.exit(main())
