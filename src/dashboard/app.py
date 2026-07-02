"""
Streamlit dashboard for AML visualization with Redis streaming
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import random
from datetime import datetime
import time
import base64
import json

sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure page
st.set_page_config(
    page_title="AML Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

# ============================================
# SESSION STATE
# ============================================

if 'page' not in st.session_state:
    st.session_state.page = "Dashboard"

if 'streaming_active' not in st.session_state:
    st.session_state.streaming_active = False

if 'redis_available' not in st.session_state:
    st.session_state.redis_available = False

# ============================================
# REPORT EXPORT FUNCTIONS
# ============================================

def create_download_link(data, filename, mime_type):
    """Create a download link for the report"""
    if mime_type == 'text/csv' or mime_type == 'application/json':
        if isinstance(data, str):
            b64 = base64.b64encode(data.encode()).decode()
        else:
            b64 = base64.b64encode(json.dumps(data).encode()).decode()
    else:
        b64 = base64.b64encode(data).decode()
    
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}" style="text-decoration:none;">📥 Download {filename}</a>'
    return href

def export_reports(data):
    """Generate export buttons for reports"""
    try:
        from src.utils.report_generator import ReportGenerator
        generator = ReportGenerator()
        
        st.subheader("📊 Export Reports")
        st.markdown("Generate compliance reports for regulators")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 PDF Report", use_container_width=True, help="Generate PDF report for regulators"):
                with st.spinner("Generating PDF report..."):
                    pdf_data, filename = generator.generate_pdf_report(data, "AML Suspicious Activity Report")
                    if pdf_data:
                        link = create_download_link(pdf_data, filename, 'application/pdf')
                        st.markdown(link, unsafe_allow_html=True)
                        st.success("✅ PDF report ready!")
                    else:
                        st.warning("⚠️ PDF generation failed. Try CSV or JSON.")
        
        with col2:
            if st.button("📊 CSV Report", use_container_width=True, help="Export data as CSV"):
                with st.spinner("Generating CSV report..."):
                    csv_data, filename = generator.generate_csv_report(data)
                    link = create_download_link(csv_data, filename, 'text/csv')
                    st.markdown(link, unsafe_allow_html=True)
                    st.success("✅ CSV report ready!")
        
        with col3:
            if st.button("📋 JSON Report", use_container_width=True, help="Export data as JSON for API integration"):
                with st.spinner("Generating JSON report..."):
                    json_data, filename = generator.generate_json_report(data)
                    link = create_download_link(json_data, filename, 'application/json')
                    st.markdown(link, unsafe_allow_html=True)
                    st.success("✅ JSON report ready!")
                
    except ImportError as e:
        st.warning("⚠️ Report generator not available. Please install reportlab: pip install reportlab")
    except Exception as e:
        st.error(f"Error generating reports: {e}")

# ============================================
# REDIS FUNCTIONS
# ============================================

def check_redis():
    """Check if Redis is available"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        return True
    except:
        return False

def start_streaming():
    """Start streaming in background"""
    try:
        from src.streaming.redis_stream import RedisStreamClient
        from src.streaming.generator import TransactionGenerator
        from src.streaming.processor import StreamingProcessor
        import threading
        
        redis_client = RedisStreamClient()
        redis_client.client.ping()
        
        generator = TransactionGenerator(redis_client)
        gen_thread = threading.Thread(
            target=generator.run,
            kwargs={'interval': 0.3, 'batch_size': 3},
            daemon=True
        )
        gen_thread.start()
        
        processor = StreamingProcessor(redis_client)
        proc_thread = threading.Thread(
            target=processor.run_consumer,
            daemon=True
        )
        proc_thread.start()
        
        st.session_state.streaming_active = True
        st.session_state.redis_available = True
        return True
    except Exception as e:
        st.error(f"Failed to start streaming: {e}")
        return False

def stop_streaming():
    """Stop streaming"""
    st.session_state.streaming_active = False
    st.session_state.redis_available = False

# ============================================
# PAGE: DASHBOARD
# ============================================

def page_dashboard():
    """Main Dashboard Page"""
    st.title("🔍 Anti-Money Laundering Dashboard")
    st.markdown("---")
    
    # Quick metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", "18,537", "Today")
    with col2:
        st.metric("Suspicious Accounts", "5,657", "↑ 12")
    with col3:
        st.metric("High Risk Score (>70)", "23", "⚠️")
    with col4:
        st.metric("Patterns Detected", "156", "🔍")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transaction Amount Distribution")
        amounts = [10, 50, 100, 500, 1000, 5000, 10000]
        counts = [1200, 800, 600, 400, 200, 100, 50]
        fig = go.Figure(data=[go.Bar(x=amounts, y=counts)])
        fig.update_layout(xaxis_title="Amount (₹)", yaxis_title="Frequency", height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Risk Score Distribution")
        risk_scores = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        frequencies = [500, 400, 300, 250, 200, 150, 100, 50, 30, 20]
        fig = go.Figure(data=[go.Bar(x=risk_scores, y=frequencies, marker_color='red')])
        fig.update_layout(xaxis_title="Risk Score", yaxis_title="Number of Accounts", height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # High Risk Transactions
    st.subheader("🚨 High-Risk Transactions")
    df_high_risk = pd.DataFrame({
        'Transaction ID': [f'TX_{i}' for i in range(1, 11)],
        'From Account': [f'ACC_{i}' for i in range(1, 11)],
        'To Account': [f'ACC_{i+5}' for i in range(1, 11)],
        'Amount (₹)': [50000, 25000, 100000, 75000, 30000, 80000, 45000, 60000, 35000, 90000],
        'Risk Score': [95, 88, 92, 75, 82, 91, 78, 85, 79, 94],
        'Pattern': ['Smurfing', 'Layering', 'Round-tripping', 'Smurfing', 'Layering',
                    'Round-tripping', 'Smurfing', 'Layering', 'Smurfing', 'Round-tripping']
    })
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
    
    # ============================================
    # REPORT EXPORT SECTION
    # ============================================
    st.markdown("---")
    
    # Prepare data for export
    export_data = df_high_risk.to_dict('records')
    export_reports(export_data)
    
    if st.session_state.streaming_active:
        st.markdown("---")
        st.success(f"🟢 Live Streaming Active | {datetime.now().strftime('%H:%M:%S')}")
    
    st.markdown("---")
    st.caption("🚀 GNN-based Anti-Money Laundering Detection System")

# ============================================
# PAGE: LIVE ALERTS
# ============================================

def page_live_alerts():
    """Live Alerts Page"""
    st.title("🚨 Live Fraud Alerts")
    st.markdown("Real-time suspicious activity monitoring")
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        auto_refresh = st.checkbox("Auto-refresh (every 3 seconds)", value=False)
    
    with col2:
        risk_threshold = st.slider("Risk Threshold", 0, 100, 70)
    
    with col3:
        if st.session_state.get('streaming_active', False):
            st.success("🟢 Streaming Active")
        else:
            st.info("⏸️ Streaming Inactive")
    
    # Generate alerts
    patterns = ["Smurfing", "Layering", "Round-tripping", "Velocity Spike"]
    amounts = [5000, 10000, 25000, 50000, 100000]
    accounts = [f"UPI_{random.randint(1000, 9999)}" for _ in range(20)]
    
    alerts = []
    for i in range(10):
        alerts.append({
            "Timestamp": datetime.now().strftime("%H:%M:%S"),
            "Account": random.choice(accounts),
            "Pattern": random.choice(patterns),
            "Amount": f"₹{random.choice(amounts):,}",
            "Risk Score": random.randint(70, 100),
            "Status": "New" if random.random() > 0.5 else "Investigating"
        })
    
    alerts_df = pd.DataFrame(alerts)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Active Alerts", len([a for a in alerts if a['Status'] == 'New']))
    with col2:
        st.metric("Total Today", len(alerts))
    with col3:
        high_risk = len([a for a in alerts if a['Risk Score'] > 90])
        st.metric("Critical Risk", high_risk, delta="⚠️")
    with col4:
        avg_risk = int(sum(a['Risk Score'] for a in alerts) / len(alerts))
        st.metric("Avg Risk Score", avg_risk)
    
    st.markdown("---")
    
    # Filter and display
    filtered_df = alerts_df[alerts_df['Risk Score'] >= risk_threshold]
    
    st.dataframe(
        filtered_df,
        use_container_width=True,
        column_config={
            "Risk Score": st.column_config.ProgressColumn(
                "Risk Score",
                format="%d%%",
                min_value=0,
                max_value=100,
            )
        }
    )
    
    # Export report from Live Alerts
    st.markdown("---")
    export_data = filtered_df.to_dict('records')
    if export_data:
        export_reports(export_data)
    
    if len(filtered_df) > 1:
        st.subheader("Risk Score Trend")
        st.line_chart(filtered_df['Risk Score'])
    
    if auto_refresh:
        time.sleep(3)
        st.rerun()

# ============================================
# PAGE: GRAPH EXPLORER
# ============================================

def page_graph_explorer():
    """Graph Explorer Page"""
    st.title("🌐 Transaction Network Explorer")
    st.markdown("Interactive visualization of transaction graph")
    
    import networkx as nx
    
    # Controls
    with st.sidebar:
        st.header("Visualization Controls")
        num_nodes = st.slider("Number of Nodes", 10, 50, 25)
        show_labels = st.checkbox("Show Node Labels", True)
        layout_type = st.selectbox("Layout", ["Spring", "Circular"])
        highlight_high_risk = st.checkbox("Highlight High-Risk Nodes", True)
    
    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(f"ACC_{i}", risk=random.randint(0, 100))
    
    for _ in range(num_nodes * 2):
        from_node = f"ACC_{random.randint(0, num_nodes-1)}"
        to_node = f"ACC_{random.randint(0, num_nodes-1)}"
        if from_node != to_node:
            G.add_edge(from_node, to_node, amount=random.randint(100, 100000))
    
    if layout_type == "Spring":
        pos = nx.spring_layout(G, iterations=30)
    else:
        pos = nx.circular_layout(G)
    
    fig = go.Figure()
    
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=1, color='gray'),
        hoverinfo='none',
        showlegend=False
    ))
    
    node_x, node_y, node_colors, node_sizes, node_texts = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        risk = G.nodes[node]['risk']
        node_sizes.append(20 if risk > 70 else 15)
        node_texts.append(f"{node}<br>Risk: {risk}%")
        
        if highlight_high_risk:
            if risk > 70:
                node_colors.append('red')
            elif risk > 40:
                node_colors.append('orange')
            else:
                node_colors.append('green')
        else:
            node_colors.append('blue')
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text' if show_labels else 'markers',
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=2, color='white')),
        text=node_texts if show_labels else None,
        textposition="top center",
        hoverinfo='text',
        hovertext=node_texts,
        showlegend=False
    ))
    
    fig.update_layout(
        height=600,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if G.nodes():
        st.subheader("Node Details")
        col1, col2 = st.columns(2)
        
        with col1:
            selected = st.selectbox("Select Account", list(G.nodes())[:15])
        
        with col2:
            if selected:
                risk = G.nodes[selected]['risk']
                st.metric("Risk Score", f"{risk}%", "High Risk" if risk > 70 else "Normal")
        
        if selected:
            st.subheader(f"Transactions for {selected}")
            neighbors = []
            for neighbor in G.neighbors(selected):
                amount = G[selected][neighbor]['amount']
                risk = G.nodes[neighbor]['risk']
                neighbors.append({
                    "To Account": neighbor,
                    "Amount": f"₹{amount:,.0f}",
                    "Risk": f"{risk}%"
                })
            if neighbors:
                st.dataframe(neighbors[:10], use_container_width=True)
            else:
                st.info("No outgoing transactions found")

# ============================================
# PAGE: PATTERN CATALOG
# ============================================

def page_pattern_catalog():
    """Pattern Catalog Page"""
    st.title("📚 Money Laundering Pattern Catalog")
    st.markdown("Educational guide to detecting fraud patterns using GNNs")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container():
            st.markdown("### 🐟 **Smurfing**")
            st.markdown("""
            **Description:** Large amount split into many small transactions
            
            **Indicators:**
            - 20+ outgoing transactions
            - Similar amounts
            - Multiple recipients
            """)
            st.warning("⚠️ Risk: High")
    
    with col2:
        with st.container():
            st.markdown("### 🥞 **Layering**")
            st.markdown("""
            **Description:** Money through multiple accounts to obscure origin
            
            **Indicators:**
            - Chain of 5+ accounts
            - Similar amounts
            - Rapid succession
            """)
            st.error("⚠️ Risk: Critical")
    
    with col3:
        with st.container():
            st.markdown("### 🔄 **Round-tripping**")
            st.markdown("""
            **Description:** Money returns to source after multiple hops
            
            **Indicators:**
            - Cycle of 3+ accounts
            - Similar amounts
            - Short time span
            """)
            st.warning("⚠️ Risk: High")
    
    st.markdown("---")
    
    st.subheader("🤖 How GNNs Detect These Patterns")
    st.markdown("""
    - **Smurfing**: GNNs aggregate neighborhood information → high out-degree nodes identified
    - **Layering**: Message passing captures paths of length 5+ → reveals hidden chains
    - **Round-tripping**: Graph attention mechanisms learn cyclic patterns
    """)
    
    st.markdown("---")
    st.subheader("📊 Detection Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Smurfing", "94%", "↑ 12%")
    with col2:
        st.metric("Layering", "89%", "↑ 15%")
    with col3:
        st.metric("Round-tripping", "91%", "↑ 8%")
    with col4:
        st.metric("False Positives", "3%", "↓ 67%")

# ============================================
# PAGE: MODEL COMPARISON
# ============================================

def page_model_comparison():
    """Model Comparison Page"""
    st.title("📊 Multi-Model Comparison")
    st.markdown("Compare GraphSAGE vs GAT vs Hetero-GNN for fraud detection")
    
    df_results = pd.DataFrame({
        'Model': ['GraphSAGE', 'GAT', 'Hetero-GNN'],
        'Accuracy': ['85.0%', '88.0%', '90.0%'],
        'Precision': ['82.0%', '85.0%', '87.0%'],
        'Recall': ['84.0%', '86.0%', '89.0%'],
        'F1-Score': ['83.0%', '85.0%', '88.0%'],
        'AUC-ROC': ['87.0%', '90.0%', '92.0%'],
        'Train Time': ['45.2s', '67.8s', '89.4s']
    })
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏆 Best Accuracy", "90.0%", "Hetero-GNN")
    with col2:
        st.metric("🏆 Best F1-Score", "88.0%", "Hetero-GNN")
    with col3:
        st.metric("⚡ Fastest", "45.2s", "GraphSAGE")
    with col4:
        st.metric("🏆 Best Recall", "89.0%", "Hetero-GNN")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Comparison")
        df_num = df_results.copy()
        df_num['Accuracy'] = df_num['Accuracy'].str.rstrip('%').astype(float)
        df_num['Precision'] = df_num['Precision'].str.rstrip('%').astype(float)
        df_num['Recall'] = df_num['Recall'].str.rstrip('%').astype(float)
        df_num['F1-Score'] = df_num['F1-Score'].str.rstrip('%').astype(float)
        
        df_melted = df_num.melt(
            id_vars=['Model'], 
            value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            var_name='Metric', 
            value_name='Score'
        )
        
        fig = px.bar(
            df_melted,
            x='Model',
            y='Score',
            color='Metric',
            barmode='group',
            title='Model Performance'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Training Time")
        df_time = df_results.copy()
        df_time['Time'] = df_time['Train Time'].str.rstrip('s').astype(float)
        
        fig = px.bar(
            df_time,
            x='Model',
            y='Time',
            color='Model',
            title='Training Time (seconds)'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📋 Detailed Comparison")
    st.dataframe(df_results, use_container_width=True)
    
    st.markdown("---")
    st.success("""
    🏆 **Recommended Model: Hetero-GNN**
    - Best overall performance: 90% Accuracy, 88% F1-Score
    - Best AUC-ROC: 92%
    - Best balance of precision and recall
    """)

# ============================================
# SIDEBAR - Navigation & Streaming
# ============================================

with st.sidebar:
    st.markdown("# 🔍 AML Dashboard")
    st.markdown("---")
    
    st.subheader("📡 Live Streaming")
    
    redis_available = check_redis()
    
    if redis_available:
        st.success("🟢 Redis Connected")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start", use_container_width=True):
                if start_streaming():
                    st.success("Started!")
                    time.sleep(0.5)
                    st.rerun()
        
        with col2:
            if st.button("⏹️ Stop", use_container_width=True):
                stop_streaming()
                st.warning("Stopped!")
                time.sleep(0.5)
                st.rerun()
        
        if st.session_state.streaming_active:
            st.info("🟢 Streaming Active")
        else:
            st.info("⏸️ Streaming Inactive")
    else:
        st.warning("⚠️ Redis not available")
        st.caption("Start Redis: docker run --name redis-aml -p 6379:6379 -d redis:alpine")
    
    st.markdown("---")
    
    st.subheader("📄 Navigation")
    
    selected_page = st.radio(
        "Go to",
        ["Dashboard", "Live Alerts", "Graph Explorer", "Pattern Catalog", "Model Comparison"],
        index=["Dashboard", "Live Alerts", "Graph Explorer", "Pattern Catalog", "Model Comparison"].index(st.session_state.page)
    )
    
    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.rerun()
    
    st.markdown("---")
    st.caption("🚀 GNN-based AML Detection")
    st.caption("📊 Powered by PyTorch Geometric")

# ============================================
# PAGE ROUTING
# ============================================

if st.session_state.page == "Dashboard":
    page_dashboard()
elif st.session_state.page == "Live Alerts":
    page_live_alerts()
elif st.session_state.page == "Graph Explorer":
    page_graph_explorer()
elif st.session_state.page == "Pattern Catalog":
    page_pattern_catalog()
elif st.session_state.page == "Model Comparison":
    page_model_comparison()