"""
Streamlit dashboard for AML visualization
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

st.set_page_config(
    page_title="AML Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Anti-Money Laundering Dashboard")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Controls")
    risk_threshold = st.slider("Risk Threshold", 0, 100, 70)
    show_patterns = st.multiselect(
        "Show Patterns",
        ["Smurfing", "Layering", "Round-tripping"],
        default=["Smurfing", "Layering", "Round-tripping"]
    )
    
    st.markdown("---")
    st.info("""
    **Patterns Detected:**
    - 🐟 **Smurfing**: Large sum split into small transactions
    - 🥞 **Layering**: Money through multiple accounts
    - 🔄 **Round-tripping**: Money returns to source
    """)

# Main content
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Transactions", "10,000", "Today")
with col2:
    st.metric("Suspicious Accounts", "47", "↑ 12")
with col3:
    st.metric("High Risk Score (>70)", "23", "⚠️")
with col4:
    st.metric("Patterns Detected", "156", "🔍")

st.markdown("---")

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Transaction Amount Distribution")
    # Sample data
    amounts = [10, 50, 100, 500, 1000, 5000, 10000]
    counts = [1200, 800, 600, 400, 200, 100, 50]
    fig = go.Figure(data=[go.Bar(x=amounts, y=counts)])
    fig.update_layout(xaxis_title="Amount (₹)", yaxis_title="Frequency")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Risk Score Distribution")
    risk_scores = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    frequencies = [500, 400, 300, 250, 200, 150, 100, 50, 30, 20]
    fig = go.Figure(data=[go.Bar(x=risk_scores, y=frequencies, marker_color='red')])
    fig.update_layout(xaxis_title="Risk Score", yaxis_title="Number of Accounts")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Suspicious Transactions Table
st.subheader("🚨 High-Risk Transactions")
transactions_data = {
    'Transaction ID': [f'TX_{i}' for i in range(1, 11)],
    'From Account': [f'ACC_{i}' for i in range(1, 11)],
    'To Account': [f'ACC_{i+5}' for i in range(1, 11)],
    'Amount (₹)': [50000, 25000, 100000, 75000, 30000, 80000, 45000, 60000, 35000, 90000],
    'Risk Score': [95, 88, 92, 75, 82, 91, 78, 85, 79, 94],
    'Pattern': ['Smurfing', 'Layering', 'Round-tripping', 'Smurfing', 'Layering',
                'Round-tripping', 'Smurfing', 'Layering', 'Smurfing', 'Round-tripping']
}

df_high_risk = pd.DataFrame(transactions_data)
st.dataframe(df_high_risk, use_container_width=True)

# Pattern Analysis
st.markdown("---")
st.subheader("📈 Pattern Analysis")

pattern_data = {
    'Pattern': ['Smurfing', 'Layering', 'Round-tripping'],
    'Count': [78, 52, 26],
    'Avg Amount (₹)': [45000, 120000, 80000]
}
df_patterns = pd.DataFrame(pattern_data)

col1, col2 = st.columns(2)

with col1:
    fig = px.pie(df_patterns, values='Count', names='Pattern', title='Pattern Distribution')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.bar(df_patterns, x='Pattern', y='Avg Amount (₹)', 
                 title='Average Amount by Pattern', color='Pattern')
    st.plotly_chart(fig, use_container_width=True)

# Network Graph Placeholder
st.markdown("---")
st.subheader("🌐 Transaction Network Visualization")
st.info("""
**Network Graph Features (Coming Soon):**
- Interactive graph visualization of transactions
- Highlight suspicious nodes and edges
- Filter by risk score and pattern type
- Drill down into specific transactions
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🚀 GNN-based Anti-Money Laundering Detection System</p>
    <p style='font-size: 12px;'>Using Graph Neural Networks to detect Smurfing, Layering, and Round-tripping patterns</p>
</div>
""", unsafe_allow_html=True)