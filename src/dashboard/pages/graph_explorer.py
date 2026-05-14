"""
Interactive graph visualization page
"""
import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import random
import numpy as np

st.set_page_config(page_title="Graph Explorer", page_icon="🌐", layout="wide")

st.title("🌐 Transaction Network Explorer")
st.markdown("Interactive visualization of transaction graph")

# Sidebar controls
with st.sidebar:
    st.header("Visualization Controls")
    num_nodes = st.slider("Number of Nodes", 10, 100, 30)
    show_labels = st.checkbox("Show Node Labels", True)
    layout_type = st.selectbox("Layout", ["Spring", "Circular", "Kamada-Kawai"])
    highlight_high_risk = st.checkbox("Highlight High-Risk Nodes", True)

# Generate sample graph
def create_sample_graph(num_nodes=30):
    """Create sample transaction graph"""
    G = nx.DiGraph()
    
    # Add nodes with risk scores
    for i in range(num_nodes):
        risk = random.randint(0, 100)
        G.add_node(f"ACC_{i}", risk=risk)
    
    # Add edges (transactions)
    for _ in range(num_nodes * 2):
        from_node = f"ACC_{random.randint(0, num_nodes-1)}"
        to_node = f"ACC_{random.randint(0, num_nodes-1)}"
        if from_node != to_node:
            amount = random.randint(100, 100000)
            G.add_edge(from_node, to_node, amount=amount)
    
    return G

# Create graph
G = create_sample_graph(num_nodes)

# Get layout positions
if layout_type == "Spring":
    pos = nx.spring_layout(G)
elif layout_type == "Circular":
    pos = nx.circular_layout(G)
else:
    pos = nx.kamada_kawai_layout(G)

# Create plotly figure
fig = go.Figure()

# Add edges
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    
    # Color based on transaction amount
    amount = G[edge[0]][edge[1]]['amount']
    color = 'lightgray' if amount < 10000 else 'orange' if amount < 50000 else 'red'
    
    fig.add_trace(go.Scatter(
        x=[x0, x1],
        y=[y0, y1],
        mode='lines',
        line=dict(width=1, color=color),
        hoverinfo='text',
        text=f"Amount: ₹{amount:,.0f}",
        showlegend=False
    ))

# Add nodes
node_x = []
node_y = []
node_colors = []
node_texts = []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    
    risk = G.nodes[node]['risk']
    
    # Color based on risk
    if highlight_high_risk:
        if risk > 70:
            node_colors.append('red')
        elif risk > 40:
            node_colors.append('orange')
        else:
            node_colors.append('green')
    else:
        node_colors.append('blue')
    
    node_texts.append(f"{node}<br>Risk: {risk}")

fig.add_trace(go.Scatter(
    x=node_x,
    y=node_y,
    mode='markers+text' if show_labels else 'markers',
    marker=dict(size=15, color=node_colors, line=dict(width=2, color='white')),
    text=node_texts if show_labels else None,
    textposition="top center",
    hoverinfo='text',
    hovertext=node_texts,
    showlegend=False
))

fig.update_layout(
    title="Transaction Network Graph",
    showlegend=False,
    hovermode='closest',
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    height=700
)

st.plotly_chart(fig, use_container_width=True)

# Node details
st.subheader("Node Details")
col1, col2 = st.columns(2)

with col1:
    selected_node = st.selectbox("Select Account", list(G.nodes()))
    
with col2:
    if selected_node:
        risk = G.nodes[selected_node]['risk']
        st.metric("Risk Score", f"{risk}%", 
                 delta="High Risk" if risk > 70 else "Normal")

# Neighbors table
if selected_node:
    st.subheader(f"Transactions for {selected_node}")
    
    neighbors = []
    for neighbor in G.neighbors(selected_node):
        amount = G[selected_node][neighbor]['amount']
        neighbors.append({
            "To Account": neighbor,
            "Amount": f"₹{amount:,.0f}",
            "Risk": f"{G.nodes[neighbor]['risk']}%"
        })
    
    if neighbors:
        st.dataframe(neighbors, use_container_width=True)
    else:
        st.info("No outgoing transactions found")