"""
Risk meter component for Streamlit dashboard
"""
import streamlit as st
import plotly.graph_objects as go

def risk_meter(risk_score, title="Risk Score"):
    """Display a gauge chart for risk score"""
    
    # Determine color based on risk
    if risk_score < 30:
        color = "green"
    elif risk_score < 60:
        color = "yellow"
    elif risk_score < 85:
        color = "orange"
    else:
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score,
        title = {'text': title},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "lightyellow"},
                {'range': [60, 85], 'color': "lightsalmon"},
                {'range': [85, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

def risk_badge(risk_score):
    """Display a colored badge for risk level"""
    if risk_score < 30:
        badge = "🟢 Low Risk"
    elif risk_score < 60:
        badge = "🟡 Medium Risk"
    elif risk_score < 85:
        badge = "🟠 High Risk"
    else:
        badge = "🔴 Critical Risk"
    
    return badge

def risk_metrics_dashboard(accounts_data):
    """Display comprehensive risk metrics dashboard"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_risk = sum(a['risk_score'] for a in accounts_data) / len(accounts_data)
        st.metric("Average Risk Score", f"{avg_risk:.1f}", 
                 delta="↑" if avg_risk > 50 else "↓")
    
    with col2:
        high_risk_count = len([a for a in accounts_data if a['risk_score'] > 70])
        st.metric("High Risk Accounts", high_risk_count,
                 delta="⚠️" if high_risk_count > 0 else "✅")
    
    with col3:
        critical_count = len([a for a in accounts_data if a['risk_score'] > 90])
        st.metric("Critical Risk", critical_count,
                 delta="🚨" if critical_count > 0 else "✓")

if __name__ == "__main__":
    # Test the component
    risk_meter(85, "Account Risk Score")