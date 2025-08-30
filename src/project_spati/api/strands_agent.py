from strands_agents import Agent, AgentConfig
from typing import Dict, List, Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)

class TrafficAnalysisAgent:
    def __init__(self, agent_id: str = "spati-agent"):
        """Initialize Strands agent for traffic analysis"""
        self.config = AgentConfig(
            agent_id=agent_id,
            model="anthropic.claude-v2",
            temperature=0.7,
            max_tokens=2000
        )
        self.agent = Agent(self.config)

    def analyze_traffic(self, context: Dict) -> Dict:
        """
        Analyze traffic patterns using Strands agent

        Args:
            context (Dict): Context data including metrics and tickets

        Returns:
            Dict: Analysis results
        """
        try:
            # Prepare prompt with context
            prompt = self._build_analysis_prompt(context)

            # Get agent response
            response = self.agent.complete(prompt)

            return {
                "analysis": response.text,
                "context": context,
                "confidence": response.metadata.get("confidence", 0.0)
            }

        except Exception as e:
            logger.error(f"Error in traffic analysis: {str(e)}")
            raise

    def get_recommendations(self, analysis_result: Dict) -> Dict:
        """
        Get recommendations based on analysis

        Args:
            analysis_result (Dict): Previous analysis results

        Returns:
            Dict: Recommendations
        """
        try:
            prompt = self._build_recommendation_prompt(analysis_result)
            response = self.agent.complete(prompt)

            return {
                "recommendations": response.text,
                "analysis": analysis_result,
                "confidence": response.metadata.get("confidence", 0.0)
            }

        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            raise

    def _build_analysis_prompt(self, context: Dict) -> str:
        """Build prompt for traffic analysis"""
        return f"""
        Analyze the SP API traffic patterns and issues based on this context:

        Metrics:
        {context.get('metrics', {})}

        Related Tickets:
        {context.get('tickets', [])}

        Please provide:
        1. Traffic pattern analysis
        2. Identified anomalies or issues
        3. Correlation with tickets
        4. Risk assessment
        """

    def _build_recommendation_prompt(self, analysis: Dict) -> str:
        """Build prompt for recommendations"""
        return f"""
        Based on this traffic analysis:
        {analysis.get('analysis', '')}

        Please provide:
        1. Recommended throttling adjustments
        2. Scaling suggestions
        3. Preventive measures
        4. Monitoring recommendations
        """
