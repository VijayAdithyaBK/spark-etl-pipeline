"""
Fraud Analytics Dashboard.
Interactive Streamlit dashboard for exploring fraud detection insights.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import pandas as pd


# Page config
st.set_page_config(
    page_title="Fraud Analytics Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: #f8fafc;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e2e8f0;
    }
    .risk-high { color: #dc2626; font-weight: 600; }
    .risk-medium { color: #d97706; font-weight: 600; }
    .risk-low { color: #059669; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


def load_analytics_data():
    """Load analytics data from JSON report."""
    report_path = Path("data/processed/analytics_report.json")
    
    if not report_path.exists():
        st.error("Analytics report not found. Please run the pipeline first: `uv run python main.py`")
        return None
    
    with open(report_path) as f:
        return json.load(f)


def main():
    st.title("🔍 Fraud Analytics Dashboard")
    st.caption("Interactive exploration of fraud detection insights")
    
    # Load data
    data = load_analytics_data()
    
    if data is None:
        return
    
    sections = data.get("sections", {})
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Executive Summary Section
    exec_data = sections.get("executive_summary", {}).get("data", [{}])[0]
    
    st.header("Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Transactions",
            f"{exec_data.get('total_transactions', 0):,}"
        )
    
    with col2:
        st.metric(
            "Fraud Cases",
            f"{exec_data.get('fraud_transactions', 0):,}",
            delta=f"{exec_data.get('fraud_rate_percent', 0):.2f}%",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Amount at Risk",
            f"${exec_data.get('fraud_amount', 0):,.0f}"
        )
    
    with col4:
        st.metric(
            "Avg Fraud Amount",
            f"${exec_data.get('avg_fraud_transaction', 0):,.2f}"
        )
    
    st.divider()
    
    # Category Analysis
    st.header("Fraud by Category")
    
    category_data = sections.get("fraud_by_category", {}).get("data", [])
    
    if category_data:
        # Filter controls
        col1, col2 = st.columns([1, 3])
        
        with col1:
            min_txns = st.number_input("Min Transactions", value=100, step=100)
        
        filtered_categories = [c for c in category_data if c.get("total_transactions", 0) >= min_txns]
        
        cat_df = pd.DataFrame(filtered_categories)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                cat_df.sort_values("fraud_rate_percent", ascending=True),
                x="fraud_rate_percent",
                y="category",
                orientation="h",
                title="Fraud Rate by Category",
                labels={"fraud_rate_percent": "Fraud Rate (%)", "category": ""},
                color="fraud_rate_percent",
                color_continuous_scale="Reds"
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                cat_df.sort_values("fraud_amount", ascending=True),
                x="fraud_amount",
                y="category",
                orientation="h",
                title="Fraud Amount by Category ($)",
                labels={"fraud_amount": "Fraud Amount ($)", "category": ""},
                color="fraud_amount",
                color_continuous_scale="Blues"
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        with st.expander("View Category Data"):
            st.dataframe(cat_df, use_container_width=True)
    
    st.divider()
    
    # Time Analysis
    st.header("Temporal Patterns")
    
    time_data = sections.get("time_trends", {}).get("data", [{}])[0]
    hourly = time_data.get("hourly_distribution", [])
    daily = time_data.get("daily_distribution", [])
    
    col1, col2 = st.columns(2)
    
    with col1:
        if hourly:
            hourly_df = pd.DataFrame(hourly)
            fig = px.line(
                hourly_df,
                x="hour",
                y="fraud_rate",
                title="Hourly Fraud Rate",
                labels={"hour": "Hour", "fraud_rate": "Fraud Rate (%)"},
                markers=True
            )
            fig.update_traces(fill="tozeroy", line_color="#2563eb")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if daily:
            daily_df = pd.DataFrame(daily)
            colors = ["#dc2626" if r == daily_df["fraud_rate"].max() else "#94a3b8" 
                     for r in daily_df["fraud_rate"]]
            fig = px.bar(
                daily_df,
                x="day",
                y="fraud_rate",
                title="Daily Fraud Rate",
                labels={"day": "Day", "fraud_rate": "Fraud Rate (%)"}
            )
            fig.update_traces(marker_color=colors)
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Risk Patterns
    st.header("Transaction Risk Patterns")
    
    patterns = sections.get("high_risk_patterns", {}).get("data", [])
    
    if patterns:
        pattern_df = pd.DataFrame(patterns)
        
        # Color code risk levels
        def risk_color(level):
            return {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}.get(level, "⚪")
        
        pattern_df["Risk"] = pattern_df["risk_level"].apply(risk_color) + " " + pattern_df["risk_level"]
        
        st.dataframe(
            pattern_df[["pattern", "threshold", "occurrences", "fraud_count", "fraud_rate_percent", "Risk"]],
            use_container_width=True,
            hide_index=True
        )
    
    st.divider()
    
    # Anomaly Detection
    st.header("Anomaly Detection")
    
    anomaly_data = sections.get("anomalies", {}).get("data", [{}])[0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Anomalies Detected", f"{anomaly_data.get('total_anomalies', 0):,}")
    
    with col2:
        st.metric(
            "Fraudulent Anomalies",
            f"{anomaly_data.get('fraudulent_anomalies', 0):,}",
            delta=f"{anomaly_data.get('anomaly_fraud_rate_percent', 0):.1f}%",
            delta_color="inverse"
        )
    
    with col3:
        st.metric("Alert Threshold", f"${anomaly_data.get('threshold_amount', 0):,.0f}")
    
    # Donut chart for anomaly distribution
    if anomaly_data.get("total_anomalies", 0) > 0:
        fig = go.Figure(data=[go.Pie(
            labels=["Fraudulent", "Legitimate"],
            values=[
                anomaly_data.get("fraudulent_anomalies", 0),
                anomaly_data.get("total_anomalies", 0) - anomaly_data.get("fraudulent_anomalies", 0)
            ],
            hole=0.6,
            marker_colors=["#dc2626", "#10b981"]
        )])
        fig.update_layout(
            title="Anomaly Composition",
            height=300,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # High Risk Merchants
    st.header("High-Risk Merchants")
    
    merchant_data = sections.get("fraud_by_merchant", {}).get("data", [])
    
    if merchant_data:
        merchant_df = pd.DataFrame(merchant_data[:20])
        
        fig = px.bar(
            merchant_df.sort_values("fraud_rate_percent", ascending=True).tail(10),
            x="fraud_rate_percent",
            y="merchant",
            orientation="h",
            title="Top 10 High-Risk Merchants",
            labels={"fraud_rate_percent": "Fraud Rate (%)", "merchant": ""},
            color="fraud_rate_percent",
            color_continuous_scale="Reds"
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("View All High-Risk Merchants"):
            st.dataframe(merchant_df, use_container_width=True, hide_index=True)
    
    # Footer
    st.divider()
    st.caption(f"Report generated: {data.get('generated_at', 'N/A')}")


if __name__ == "__main__":
    main()
