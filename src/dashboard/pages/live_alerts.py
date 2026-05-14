"""
Live alerts page for real-time fraud detection
"""
import streamlit as st
import pandas as pd
import random
import time

st.set_page_config(page_title="Live Alerts", page_icon="🚨", layout="wide")

st.title("🚨 Live Fraud Alerts")
st.markdown("Real-time suspicious activity monitoring")

# Auto-refresh
auto_refresh = st.checkbox("Auto-refresh (every 5 seconds)", value=True)

# Placeholder for alerts
alert_placeholder = st.empty()

def generate_alert():
    """Generate mock alert"""
    patterns = ["Smurfing", "Layering", "Round-tripping", "Velocity Spike", "Round Numbers"]
    amounts = [5000, 10000, 25000, 50000, 100000, 250000]
    accounts = [f"UPI_{random.randint(1000, 9999)}" for _ in range(3)]
    
    return {
        "timestamp": time.strftime("%H:%M:%S"),
        "account": random.choice(accounts),
        "pattern": random.choice(patterns),
        "amount": random.choice(amounts),
        "risk_score": random.randint(70, 100),
        "status": "New" if random.random() > 0.7 else "Investigating"
    }

# Alert history
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

# Main loop
while auto_refresh:
    new_alert = generate_alert()
    st.session_state.alerts.insert(0, new_alert)
    
    # Keep only last 50 alerts
    st.session_state.alerts = st.session_state.alerts[:50]
    
    # Display alerts
    alerts_df = pd.DataFrame(st.session_state.alerts)
    
    with alert_placeholder.container():
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Active Alerts", len([a for a in st.session_state.alerts if a['status'] == 'New']))
        with col2:
            st.metric("Total Today", len(st.session_state.alerts))
        with col3:
            high_risk = len([a for a in st.session_state.alerts if a['risk_score'] > 90])
            st.metric("Critical Risk", high_risk, delta="⚠️")
        with col4:
            st.metric("Avg Risk Score", 
                     int(sum(a['risk_score'] for a in st.session_state.alerts) / len(st.session_state.alerts)) 
                     if st.session_state.alerts else 0)
        
        st.markdown("---")
        
        # Alerts table
        if not alerts_df.empty:
            st.dataframe(
                alerts_df,
                use_container_width=True,
                column_config={
                    "risk_score": st.column_config.ProgressColumn(
                        "Risk Score",
                        format="%d%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "status": st.column_config.SelectboxColumn(
                        "Status",
                        options=["New", "Investigating", "Resolved"],
                    ),
                }
            )
    
    time.sleep(5)
    if not auto_refresh:
        break

if not auto_refresh:
    st.info("Auto-refresh is disabled. Click the checkbox to enable.")
    if st.button("Refresh Now"):
        st.rerun()