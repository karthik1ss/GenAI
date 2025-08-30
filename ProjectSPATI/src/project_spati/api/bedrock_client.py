import boto3
import json
from typing import Dict, List, Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)

class BedrockClient:
    def __init__(self, region: str = "us-west-2"):
        """Initialize Bedrock client"""
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = "anthropic.claude-v2"  # Default model

    def invoke_model(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Invoke Bedrock model with prompt

        Args:
            prompt (str): Input prompt
            max_tokens (int): Maximum tokens in response

        Returns:
            str: Model response
        """
        try:
            body = json.dumps({
                "prompt": prompt,
                "max_tokens_to_sample": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
            })

            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body
            )

            response_body = json.loads(response.get("body").read())
            return response_body.get("completion", "")

        except Exception as e:
            logger.error(f"Error invoking Bedrock model: {str(e)}")
            raise

    def analyze_traffic_pattern(self, metrics_data: Dict) -> Dict:
        """
        Analyze SP API traffic patterns

        Args:
            metrics_data (Dict): Traffic metrics data

        Returns:
            Dict: Analysis results
        """
        prompt = f"""
        Analyze the following SP API traffic metrics and provide insights:
        {json.dumps(metrics_data, indent=2)}

        Please provide:
        1. Traffic pattern analysis
        2. Potential issues or anomalies
        3. Recommendations for optimization
        4. Risk assessment
        """

        response = self.invoke_model(prompt)
        return {
            "analysis": response,
            "raw_metrics": metrics_data
        }
