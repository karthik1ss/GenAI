import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from typing import Dict, List
from ..api.strands_agent import TrafficAnalysisAgent

# Page config
st.set_page_config(
    page_title="SP API Traffic Intelligence",
    page_icon="📊",
    layout="wide"
)

class SPATIApp:
    def __init__(self):
        """Initialize SPATI Streamlit app"""
        self.agent = TrafficAnalysisAgent()

    def run(self):
        """Run the Streamlit app"""
        st.title("SP API Traffic Intelligence 📊")

        # Sidebar
        self._render_sidebar()

        # Main content
        col1, col2 = st.columns([2, 1])

        with col1:
            self._render_metrics_section()

        with col2:
            self._render_analysis_section()

    def _render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.header("Settings")

            # Time range selector
            st.subheader("Time Range")
            hours = st.slider(
                "Hours of data",
                min_value=1,
                max_value=168,
                value=24,
                help="Select how many hours of data to analyze"
            )

            # Analysis options
            st.subheader("Analysis Options")
            include_tickets = st.checkbox(
                "Include ticket context",
                value=True,
                help="Include related tickets in analysis"
            )

            confidence_threshold = st.slider(
                "Confidence threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                help="Minimum confidence level for recommendations"
            )

            if st.button("Run Analysis"):
                self._run_analysis(hours, include_tickets, confidence_threshold)

    def _render_metrics_section(self):
        """Render metrics and visualizations"""
        st.header("Traffic Metrics")

        # Metrics tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "API Calls",
            "Throttling",
            "Errors",
            "Latency"
        ])

        if "metrics" in st.session_state:
            metrics = st.session_state.metrics

            with tab1:
                self._plot_metric(
                    metrics.get("api_calls", []),
                    "API Calls per Hour",
                    "Number of Calls"
                )

            with tab2:
                self._plot_metric(
                    metrics.get("throttling", []),
                    "Throttled Requests per Hour",
                    "Number of Throttled Requests"
                )

            with tab3:
                self._plot_metric(
                    metrics.get("errors", []),
                    "Error Rate per Hour",
                    "Number of Errors"
                )

            with tab4:
                self._plot_metric(
                    metrics.get("latency", []),
                    "Average Latency per Hour",
                    "Latency (ms)"
                )

    def _render_analysis_section(self):
        """Render analysis and recommendations"""
        st.header("Analysis & Recommendations")

        if "analysis" in st.session_state:
            analysis = st.session_state.analysis

            # Analysis results
            st.subheader("Traffic Pattern Analysis")
            st.write(analysis.get("analysis", "No analysis available"))

            # Confidence score
            confidence = analysis.get("confidence", 0.0)
            st.metric(
                "Analysis Confidence",
                f"{confidence:.2%}",
                delta=None,
                delta_color="normal"
            )

            # Recommendations
            if "recommendations" in st.session_state:
                st.subheader("Recommendations")
                recs = st.session_state.recommendations
                st.write(recs.get("recommendations", "No recommendations available"))

                # Action items
                st.subheader("Action Items")
                for i, action in enumerate(recs.get("action_items", []), 1):
                    st.checkbox(
                        action,
                        key=f"action_{i}",
                        help="Mark action as completed"
                    )

    def _plot_metric(self, data: List[Dict], title: str, y_axis_title: str):
        """Plot metric data using Plotly"""
        if not data:
            st.write("No data available")
            return

        fig = go.Figure()

        timestamps = [d["timestamp"] for d in data]
        values = [d["value"] for d in data]

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=values,
                mode="lines+markers",
                name=title
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title=y_axis_title,
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    def _run_analysis(self, hours: int, include_tickets: bool, confidence_threshold: float):
        """Run traffic analysis"""
        try:
            # Get context data
            context = {
                "metrics": self._get_metrics_data(hours),
                "tickets": self._get_ticket_data() if include_tickets else []
            }

            # Run analysis
            analysis = self.agent.analyze_traffic(context)

            if analysis["confidence"] >= confidence_threshold:
                # Get recommendations
                recommendations = self.agent.get_recommendations(analysis)

                # Update session state
                st.session_state.metrics = context["metrics"]
                st.session_state.analysis = analysis
                st.session_state.recommendations = recommendations

                st.success("Analysis completed successfully!")
            else:
                st.warning(
                    f"Analysis confidence ({analysis['confidence']:.2%}) "
                    f"below threshold ({confidence_threshold:.2%})"
                )

        except Exception as e:
            st.error(f"Error running analysis: {str(e)}")

    def _get_metrics_data(self, hours: int) -> Dict:
        """Get metrics data (placeholder)"""
        # TODO: Replace with actual metrics data
        return {
            "api_calls": [],
            "throttling": [],
            "errors": [],
            "latency": []
        }

    def _get_ticket_data(self) -> List[Dict]:
        """Get ticket data (placeholder)"""
        # TODO: Replace with actual ticket data
        return []

if __name__ == "__main__":
    app = SPATIApp()
    app.run()
